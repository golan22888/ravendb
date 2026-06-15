using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using Jint;
using Raven.Client;
using Raven.Server.Documents.ETL.Providers.AI.GenAi.Stats;
using Raven.Server.Documents.ETL.Stats;
using Raven.Server.Documents.Patch;
using Raven.Server.Documents.TransactionMerger.Commands;
using Raven.Server.ServerWide.Context;
using Sparrow.Json;
using Sparrow.Json.Parsing;
using Sparrow.Server.Logging;
using PatchRequest = Raven.Server.Documents.Patch.PatchRequest;

namespace Raven.Server.Documents.ETL.Providers.AI.GenAi;

internal sealed class GenAiBatchPatchCommand : DocumentMergedTransactionCommand
{
    private readonly List<GenAiResultItem> _items;
    private readonly PatchRequest _patchRequest;
    private readonly string _taskIdentifier;
    private readonly RavenLogger _logger;
    private readonly EtlProcessStatistics _statistics;
    private readonly GenAiStatsScope _scope;

    public GenAiBatchPatchCommand(
        List<GenAiResultItem> items,
        PatchRequest patchRequest,
        string taskIdentifier,
        RavenLogger logger,
        EtlProcessStatistics statistics, 
        GenAiStatsScope scope)
    {
        _items = items ?? throw new ArgumentException(nameof(items));
        _patchRequest = patchRequest ?? throw new ArgumentException(nameof(patchRequest));
        _logger = logger ?? throw new ArgumentException(nameof(logger));
        _statistics = statistics ?? throw new ArgumentException(nameof(statistics));
        _scope = scope;

        if (string.IsNullOrEmpty(taskIdentifier))
            throw new ArgumentException(nameof(taskIdentifier));
        _taskIdentifier = taskIdentifier;
    }

    protected override long ExecuteCmd(DocumentsOperationContext context)
    {
        var hashes = new Dictionary<string, (Document Doc, List<string> Hashes)>();

        using (var statsScope = _scope.For(GenAiOperations.ApplyUpdateScript))
        {
            using (context.DocumentDatabase.Scripts.GetScriptRunner(_patchRequest, readOnly: false, out var runner))
            {
                foreach (var item in _items)
                {
                    statsScope.NumberOfContextObjects++;

                    if (item.ContextOutput.IsCached)
                        statsScope.TotalCachedContexts++;

                    if (item.UpdateHash == false)
                        continue;
                    
                    ref var tuple = ref CollectionsMarshal.GetValueRefOrAddDefault(hashes, item.DocumentId, out var exists);
                    if (exists is false)
                    {
                        Document document = GetCurrentDocument(context, item.DocumentId);
                        tuple = (document, []);
                    }

                    if (tuple.Doc is null)
                        continue; // document was probably deleted while we talked to the model, skipping this

                    tuple.Hashes.Add(item.ContextOutput.AiHash);

                    if (item.ModelOutput is null)
                        continue;
                    
                    statsScope.TotalUpdates++;

                    var args = CreatePatchArgs(context, item);
                    try
                    {
                        var documentInstance = (BlittableObjectInstance)runner.Translate(context, tuple.Doc).AsObject();
                        using (var scriptResult = runner.Run(context, context, "execute", item.DocumentId, [documentInstance, args]))
                        using (var old = tuple.Doc.Data)
                        {
                            tuple.Doc.Data = scriptResult.TranslateToObject(context);
                        }
                    }
                    catch (Exception e)
                    {
                        // do not update metadata hash, log error, raise alert
                        tuple.Hashes.Remove(item.ContextOutput.AiHash);
                        var msg = $"Failed to apply update script for context in document '{item.DocumentId}'. " +
                                  $"Context was: {item.ContextOutput.Context}{Environment.NewLine}" +
                                  $"Error: {e}";

                        statsScope.UpdateFailures++;
                        _statistics.RecordItemLoadError(msg, item.DocumentId);
                        
                        if (_logger.IsWarnEnabled)
                            _logger.Warn(msg);
                    }
                }
            }

            // update metadata for each doc in same transaction
            foreach (var (id, (doc, allHashes)) in hashes)
            {
                // this indicates that there was an error in the update script
                // and that we should not update this document
                if (allHashes.Count is 0)
                    continue;

                UpdateHashesInMetadata(id, doc.Data, _taskIdentifier, allHashes, context);
            }

            WriteRetries(context);

            return statsScope.TotalUpdates;
        }
    }

    private const int MaxRetryBackoffMinutes = 60;

    private sealed record RetryEntry(string Reason, string Error, int Attempt, DateTime NextRetry);

    private void WriteRetries(DocumentsOperationContext context)
    {
        var byDoc = new Dictionary<string, (List<(string Hash, string Reason, string Error)> Upserts, List<string> Clears)>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in _items)
        {
            if (item.ShouldRetry == false && item.ModelOutput == null)
                continue;

            if (byDoc.TryGetValue(item.DocumentId, out var work) == false)
                byDoc[item.DocumentId] = work = (new List<(string, string, string)>(), new List<string>());

            if (item.ShouldRetry)
                work.Upserts.Add((item.ContextOutput.AiHash, item.RetryReason, item.RetryError));
            else
                work.Clears.Add(item.ContextOutput.AiHash);
        }

        if (byDoc.Count == 0)
            return;

        var now = context.DocumentDatabase.Time.GetUtcNow();
        foreach (var (docId, work) in byDoc)
            WriteRetryForDocument(context, docId, work.Upserts, work.Clears, now);
    }

    private void WriteRetryForDocument(DocumentsOperationContext context, string docId,
        List<(string Hash, string Reason, string Error)> upserts, List<string> clears, DateTime now)
    {
        var doc = context.DocumentDatabase.DocumentsStorage.Get(context, docId);
        if (doc == null)
            return;

        var data = doc.Data.CloneOnTheSameContext();

        var retryByIdentifier = ReadRetry(data);
        if (retryByIdentifier.TryGetValue(_taskIdentifier, out var ours) == false)
            ours = new Dictionary<string, RetryEntry>(StringComparer.Ordinal);

        var changed = false;
        foreach (var hash in clears)
        {
            if (ours.Remove(hash))
                changed = true;
        }
        foreach (var (hash, reason, error) in upserts)
        {
            var attempt = (ours.TryGetValue(hash, out var existing) ? existing.Attempt : 0) + 1;
            ours[hash] = new RetryEntry(reason, error, attempt, now.Add(BackoffFor(attempt)));
            changed = true;
        }

        if (changed == false)
            return;

        if (ours.Count == 0)
            retryByIdentifier.Remove(_taskIdentifier);
        else
            retryByIdentifier[_taskIdentifier] = ours;

        WriteRetryAndRefresh(context, docId, data, retryByIdentifier);
    }

    private static TimeSpan BackoffFor(int attempt)
    {
        var minutes = attempt >= 7 ? MaxRetryBackoffMinutes : Math.Min(MaxRetryBackoffMinutes, 1 << (attempt - 1));
        return TimeSpan.FromMinutes(minutes);
    }

    private static Dictionary<string, Dictionary<string, RetryEntry>> ReadRetry(BlittableJsonReaderObject data)
    {
        var result = new Dictionary<string, Dictionary<string, RetryEntry>>(StringComparer.Ordinal);
        if (data.TryGet(Constants.Documents.Metadata.Key, out BlittableJsonReaderObject metadata) == false ||
            metadata.TryGet(Constants.Documents.Metadata.GenAiRetry, out BlittableJsonReaderObject retry) == false)
            return result;

        foreach (var identifier in retry.GetPropertyNames())
        {
            if (retry.TryGet(identifier, out BlittableJsonReaderObject byHash) == false)
                continue;

            var entries = new Dictionary<string, RetryEntry>(StringComparer.Ordinal);
            foreach (var hash in byHash.GetPropertyNames())
            {
                if (byHash.TryGet(hash, out BlittableJsonReaderObject entry) == false)
                    continue;

                entry.TryGet(GenAiRetryFields.Reason, out string reason);
                entry.TryGet(GenAiRetryFields.Error, out string error);
                entry.TryGet(GenAiRetryFields.Attempt, out int attempt);
                entry.TryGet(GenAiRetryFields.NextRetry, out DateTime nextRetry);
                entries[hash] = new RetryEntry(reason, error, attempt, nextRetry);
            }

            result[identifier] = entries;
        }

        return result;
    }

    private static void WriteRetryAndRefresh(DocumentsOperationContext context, string id, BlittableJsonReaderObject data,
        Dictionary<string, Dictionary<string, RetryEntry>> retryByIdentifier)
    {
        data.TryGet(Constants.Documents.Metadata.Key, out BlittableJsonReaderObject metadata);

        DynamicJsonValue retryJson = null;
        DateTime? earliest = null;
        foreach (var (identifier, entries) in retryByIdentifier)
        {
            if (entries.Count == 0)
                continue;

            var byHash = new DynamicJsonValue();
            foreach (var (hash, e) in entries)
            {
                byHash[hash] = new DynamicJsonValue
                {
                    [GenAiRetryFields.Reason] = e.Reason,
                    [GenAiRetryFields.Error] = e.Error,
                    [GenAiRetryFields.Attempt] = e.Attempt,
                    [GenAiRetryFields.NextRetry] = e.NextRetry
                };

                if (earliest == null || e.NextRetry < earliest.Value)
                    earliest = e.NextRetry;
            }

            (retryJson ??= new DynamicJsonValue())[identifier] = byHash;
        }

        // @refresh is always set to our earliest NextRetry (never kept at an earlier value): it is one-shot and must
        // fire at NextRetry so the context is due when the document is re-fed, otherwise the retry is stranded.
        if (metadata == null)
        {
            var fresh = new DynamicJsonValue();
            if (retryJson != null)
                fresh[Constants.Documents.Metadata.GenAiRetry] = retryJson;
            if (earliest != null)
                fresh[Constants.Documents.Metadata.Refresh] = earliest.Value;

            data.Modifications = new DynamicJsonValue(data) { [Constants.Documents.Metadata.Key] = fresh };
        }
        else
        {
            metadata.Modifications = new DynamicJsonValue(metadata);
            if (retryJson != null)
                metadata.Modifications[Constants.Documents.Metadata.GenAiRetry] = retryJson;
            else
                metadata.Modifications.Remove(Constants.Documents.Metadata.GenAiRetry);

            if (earliest != null)
                metadata.Modifications[Constants.Documents.Metadata.Refresh] = earliest.Value;

            data.Modifications = new DynamicJsonValue(data) { [Constants.Documents.Metadata.Key] = metadata };
        }

        using (var old = data)
        {
            data = context.ReadObject(old, id);
        }

        context.DocumentDatabase.DocumentsStorage.Put(context, id, expectedChangeVector: null, data);
    }

    private static BlittableJsonReaderObject CreatePatchArgs(DocumentsOperationContext context, GenAiResultItem item)
    {
        var djv = new DynamicJsonValue
        {
            ["output"] = item.ModelOutput.Output,
            ["input"] = item.ContextOutput.Context
        };

        return context.ReadObject(djv, item.DocumentId);
    }

    internal static BlittableJsonReaderObject UpdateHashesInMetadata(string id, BlittableJsonReaderObject doc, string taskIdentifier, List<string> allHashes, DocumentsOperationContext context)
    {
        if (doc.TryGet(Constants.Documents.Metadata.Key, out BlittableJsonReaderObject metadata) == false)
        {
            // no metadata at all (shouldn't happen)

            doc.Modifications = new DynamicJsonValue(doc)
            {
                [Constants.Documents.Metadata.Key] = new DynamicJsonValue
                {
                    [Constants.Documents.Metadata.GenAiHashes] = new DynamicJsonValue
                    {
                        [taskIdentifier] = allHashes
                    }
                }
            };
        }

        else if (metadata.TryGet(Constants.Documents.Metadata.GenAiHashes, out BlittableJsonReaderObject hashes) == false)
        {
            // no hashes section

            metadata.Modifications = new DynamicJsonValue(metadata)
            {
                [Constants.Documents.Metadata.GenAiHashes] = new DynamicJsonValue
                {
                    [taskIdentifier] = allHashes
                }
            };
            doc.Modifications = new DynamicJsonValue(doc)
            {
                [Constants.Documents.Metadata.Key] = metadata
            };
        }

        else
        {
            // we already have the hashes section, need to modify it

            if (hashes.TryGet(taskIdentifier, out BlittableJsonReaderArray existingHashes) && existingHashes != null && 
                existingHashes.Length == allHashes.Count)
            {
                bool needToUpdate = false;

                foreach (var hash in existingHashes)
                {
                    if (allHashes.Contains(hash.ToString())) 
                        continue;

                    // we have a new hash that is not in the existing hashes
                    needToUpdate = true;
                    break;
                }

                if (needToUpdate == false)
                    return doc; // we already have the hashes for this task, no need to update
            }

            hashes.Modifications = new DynamicJsonValue(hashes)
            {
                [taskIdentifier] = allHashes
            };

            metadata.Modifications = new DynamicJsonValue(metadata)
            {
                [Constants.Documents.Metadata.GenAiHashes] = hashes
            };

            doc.Modifications = new DynamicJsonValue(doc)
            {
                [Constants.Documents.Metadata.Key] = metadata
            };
        }

        using (var old = doc)
        {
            doc = context.ReadObject(old, id);
        }

        context.DocumentDatabase.DocumentsStorage.Put(context, id, expectedChangeVector: null, doc);

        return doc;
    }

    private Document GetCurrentDocument(DocumentsOperationContext context, string id)
    {
        var originalDocument = context.DocumentDatabase.DocumentsStorage.Get(context, id);

        if (originalDocument != null)
        {
            using (var oldData = originalDocument.Data)
            {
                // we clone it, to keep it safe from defrag due to the patch modifications
                originalDocument.Data = originalDocument.Data?.CloneOnTheSameContext();
            }
        }

        return originalDocument;
    }

    public override IReplayableCommandDto<DocumentsOperationContext, DocumentsTransaction, DocumentMergedTransactionCommand> ToDto(DocumentsOperationContext context)
    {
        throw new NotSupportedException($"Replay not supported for {nameof(GenAiBatchPatchCommand)}");
    }
}

