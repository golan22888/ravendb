using System;
using System.Threading.Tasks;
using JetBrains.Annotations;
using Raven.Client.Documents.Commands.Batches;
using Raven.Client.Documents.Operations;
using Raven.Client.Documents.Queries;
using Raven.Server.Documents.Handlers.Processors.Batches;
using Raven.Server.Documents.Queries;
using Raven.Server.Json;
using Raven.Server.ServerWide.Context;
using Raven.Server.TrafficWatch;
using Sparrow.Json;
using static Raven.Server.NotificationCenter.Notifications.DatabaseStatsChanged;
using static Raven.Server.Utils.MetricCacher.Keys;

namespace Raven.Server.Documents.Handlers.Processors.Queries;

internal abstract class AbstractOperationQueriesHandlerProcessor<TRequestHandler, TOperationContext> : AbstractQueriesHandlerProcessor<TRequestHandler, TOperationContext>
    where TOperationContext : JsonOperationContext
    where TRequestHandler : AbstractDatabaseRequestHandler<TOperationContext>
{
    protected AbstractOperationQueriesHandlerProcessor([NotNull] TRequestHandler requestHandler, QueryMetadataCache queryMetadataCache) : base(requestHandler, queryMetadataCache)
    {
    }

    protected abstract long GetNextOperationId();

    protected abstract IDisposable AllocateContextForAsyncOperation(out TOperationContext asyncOperationContext);

    protected abstract void ScheduleOperation(TOperationContext asyncOperationContext, IDisposable returnAsyncOperationContext, IndexQueryServerSide query, long operationId, QueryOperationOptions options);

    public override async ValueTask ExecuteAsync()
    {
        using (ContextPool.AllocateOperationContext(out JsonOperationContext context))
        using (var tracker = CreateRequestTimeTracker())
        {
            var operationId = RequestHandler.GetLongQueryString("operationId", required: false) ?? GetNextOperationId();
            var options = GetQueryOperationOptions();

            var returnContext = AllocateContextForAsyncOperation(out var asyncOperationContext); // we don't dispose this as operation is async

            try
            {
                var query = await GetIndexQueryAsync(asyncOperationContext, QueryMethod, tracker, addSpatialProperties: false);
                // options.IndexOptions = await GetWaitForIndexesOptionsAsync(asyncOperationContext);
                 // (var query, options.IndexOptions) = await GetPatchOptionsAsync(asyncOperationContext, QueryMethod, tracker, addSpatialProperties: false);

                query.DisableAutoIndexCreation = RequestHandler.GetBoolValueQueryString("disableAutoIndexCreation", false) ?? false;

                if (TrafficWatchManager.HasRegisteredClients)
                    RequestHandler.TrafficWatchQuery(query);

                ScheduleOperation(asyncOperationContext, returnContext, query, operationId, options);

                // await WaitForIndexes(waitForIndexesOptions);

                await using (var writer = new AsyncBlittableJsonTextWriter(context, RequestHandler.ResponseBodyStream()))
                {
                    writer.WriteOperationIdAndNodeTag(context, operationId, ServerStore.NodeTag);
                }
            }
            catch
            {
                returnContext.Dispose();
                throw;
            }
        }
    }

    // private async Task WaitForIndexes(IndexBatchOptions options)
    // {
    //     if (options.WaitForIndexesTimeout.HasValue)
    //     {
    //         long lastDocumentEtag, lastTombstoneEtag;
    //
    //         using (database.DocumentsStorage.ContextPool.AllocateOperationContext(out DocumentsOperationContext context))
    //         using (var tx = context.OpenReadTransaction())
    //         {
    //             lastDocumentEtag = DocumentsStorage.ReadLastDocumentEtag(tx.InnerTransaction);
    //             lastTombstoneEtag = DocumentsStorage.ReadLastTombstoneEtag(tx.InnerTransaction);
    //             modifiedCollections ??= database.DocumentsStorage.GetCollections(context).Select(c => c.Name).ToHashSet(StringComparer.OrdinalIgnoreCase);
    //         }
    //
    //         await BatchHandlerProcessorForBulkDocs.WaitForIndexesAsync(database, options.WaitForIndexesTimeout.Value,
    //             options.SpecifiedIndexesQueryString, options.WaitForIndexThrow,
    //             lastDocumentEtag, lastTombstoneEtag, modifiedCollections, token);
    //     }
    // }

    protected QueryOperationOptions GetQueryOperationOptions()
    {
        var options = new QueryOperationOptions
        {
            AllowStale = RequestHandler.GetBoolValueQueryString("allowStale", required: false) ?? false,
            MaxOpsPerSecond = RequestHandler.GetIntValueQueryString("maxOpsPerSec", required: false),
            StaleTimeout = RequestHandler.GetTimeSpanQueryString("staleTimeout", required: false),
            RetrieveDetails = RequestHandler.GetBoolValueQueryString("details", required: false) ?? false,
            IgnoreMaxStepsForScript = RequestHandler.GetBoolValueQueryString("ignoreMaxStepsForScript", required: false) ?? false,
        };
        var WaitForIndexes = RequestHandler.GetBoolValueQueryString("waitForIndexes", required: false) ?? false;
        var WaitForIndexesTimeout = RequestHandler.GetTimeSpanQueryString("waitForIndexesTimeout", required: false);
        var ThrowOnTimeoutInWaitForIndexes = RequestHandler.GetBoolValueQueryString("ThrowOnTimeoutInWaitForIndexes", required: false) ?? false;
        var WaitForSpecificIndexes = RequestHandler.GetStringValuesQueryString("waitForSpecificIndexes", required: false);
        options.IndexOptions = new IndexBatchOptions()
        {
            WaitForIndexes = WaitForIndexes,
            WaitForIndexesTimeout = WaitForIndexesTimeout,
            WaitForSpecificIndexes = WaitForSpecificIndexes,
            ThrowOnTimeoutInWaitForIndexes = ThrowOnTimeoutInWaitForIndexes,
        };
        return options;
    }

    protected static string GetOperationDescription(IndexQueryServerSide query)
    {
        return query.Metadata.IsDynamic
            ? (query.Metadata.IsCollectionQuery ? AbstractQueryRunner.CollectionIndexPrefix : AbstractQueryRunner.DynamicIndexPrefix) + query.Metadata.CollectionName
            : query.Metadata.IndexName;
    }

    protected static BulkOperationResult.OperationDetails GetDetailedDescription(IndexQueryServerSide query)
    {
        return new BulkOperationResult.OperationDetails
        {
            Query = query.QueryParameters?.Count > 0 ? $"{query.Query}{Environment.NewLine}{query.QueryParameters}" : query.Query
        };
    }
}
