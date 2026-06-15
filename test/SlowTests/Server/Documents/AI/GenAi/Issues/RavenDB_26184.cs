using System;
using System.Linq;
using System.Threading.Tasks;
using FastTests;
using Newtonsoft.Json;
using Raven.Client;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Documents.Operations.ConnectionStrings;
using Raven.Server.Config;
using Raven.Server.Documents.ETL.Providers.AI.GenAi;
using Sparrow.Json;
using Tests.Infrastructure;
using Xunit;

namespace SlowTests.Server.Documents.AI.GenAi.Issues
{
    public class RavenDB_26184(ITestOutputHelper output) : RavenTestBase(output)
    {
        // A context that always times out is scheduled for a deferred retry (@gen-ai-retry + @refresh), not looped or
        // given up, and is NOT written as a success hash (which would stop it being retried).
        [RavenTheory(RavenTestCategory.Etl | RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.OpenAi, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task GenAi_ModelCallThatAlwaysTimesOut_ShouldBeScheduledForRetry_NotLooped(Options options, GenAiConfiguration config)
        {
            const int perCallDelayMs = 6_000;
            const int timeoutInSec = 3;
            const string marker = "ZZZMARKER";

            // Invariant: every call's window expires (and the call is cancelled) before the injected delay elapses,
            // so every attempt times out. Asserted so a future tweak can't silently let calls succeed.
            Assert.True(timeoutInSec * 1000 < perCallDelayMs, "test invariant: timeout < perCallDelay (every call must time out)");

            options.ModifyDatabaseRecord = record =>
                record.Settings[RavenConfiguration.GetKey(x => x.Ai.GenAiSendToModelTimeout)] = timeoutInSec.ToString();

            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            config.Prompt = "Reply with the given text unchanged.";
            config.Collection = "Posts";
            config.SampleObject = JsonConvert.SerializeObject(new { Result = "text" });
            config.UpdateScript = @"const idx = this.Comments.findIndex(c => c.Id == $input.Id);
this.Comments[idx].Result = $output.Result;";
            config.GenAiTransformation = new GenAiTransformation
            {
                Script = "for (const comment of this.Comments) ai.genContext({Text: comment.Text, Id: comment.Id});"
            };
            config.MaxConcurrency = 1;
            config.Identifier = "retry-on-timeout";

            store.Maintenance.Send(new AddGenAiOperation(config));

            // EtlLoader populates the process asynchronously after AddGenAiOperation, so wait for it.
            var db = await GetDatabase(store.Database);
            GenAiTask etlProcess = null;
            Assert.True(await WaitForValueAsync(() =>
            {
                etlProcess = db.EtlLoader.Processes.OfType<GenAiTask>().FirstOrDefault();
                return Task.FromResult(etlProcess != null);
            }, true, timeout: 15_000), "GenAi ETL process was not loaded in time");

            // Make every model call exceed the per-request timeout (delay only on the user-content message - it carries
            // the marker; skip the "AI Agent Parameters:" message which also interpolates it). Assumes no Queries/tools.
            etlProcess.GetChatCompletionClient().ForTestingPurposesOnly().SimulateFailureAsync = async msg =>
            {
                if (msg.Contains(marker) && msg.Contains("AI Agent Parameters:") == false)
                    await Task.Delay(perCallDelayMs);
            };

            const string docId = "posts/1";
            using (var session = store.OpenSession())
            {
                session.Store(new GenAiBasics.Post([
                    new GenAiBasics.Comment($"{marker} hopeless comment", "author") { Id = "1" }
                ], "title", "body"), docId);
                session.SaveChanges();
            }

            // the context times out; the fix schedules a retry (writes @gen-ai-retry + sets @refresh) after the first
            // timeout, so it is not retried every batch or given up. On the buggy code nothing is written (it loops).
            var scheduled = await WaitForValueAsync(async () =>
            {
                using var session = store.OpenAsyncSession();
                var doc = await session.LoadAsync<BlittableJsonReaderObject>(docId);
                if (doc == null || doc.TryGet(Constants.Documents.Metadata.Key, out BlittableJsonReaderObject metadata) == false)
                    return false;
                if (metadata.TryGet(Constants.Documents.Metadata.GenAiRetry, out BlittableJsonReaderObject retry) == false)
                    return false;

                return retry.TryGet(config.Identifier, out BlittableJsonReaderObject taskRetry) && taskRetry.Count >= 1;
            }, true, timeout: 30_000);

            Assert.True(scheduled, "a context that always times out should be scheduled for retry in @gen-ai-retry, not looped or given up");

            // it must NOT also be recorded as a success - otherwise the @refresh re-feed would treat it as done and never retry it
            using (var session = store.OpenAsyncSession())
            {
                var doc = await session.LoadAsync<BlittableJsonReaderObject>(docId);
                Assert.True(doc.TryGet(Constants.Documents.Metadata.Key, out BlittableJsonReaderObject metadata));
                Assert.False(metadata.TryGet(Constants.Documents.Metadata.GenAiHashes, out BlittableJsonReaderObject _),
                    "a timed-out context must not be written as a success hash");
            }
        }

        // Scheduling the retry must set @refresh to OUR NextRetry, overwriting any earlier @refresh already on the
        // document. @refresh is one-shot: if an earlier (e.g. user-set) @refresh were kept, it would fire before
        // NextRetry, be consumed, and leave the retry with no trigger - the context would be stranded (never retried).
        [RavenTheory(RavenTestCategory.Etl | RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.OpenAi, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task GenAi_TimeoutRetry_SetsRefreshToNextRetry_OverwritingAnEarlierExistingRefresh(Options options, GenAiConfiguration config)
        {
            const int perCallDelayMs = 6_000;
            const int timeoutInSec = 3;
            const string marker = "ZZZMARKER";

            options.ModifyDatabaseRecord = record =>
                record.Settings[RavenConfiguration.GetKey(x => x.Ai.GenAiSendToModelTimeout)] = timeoutInSec.ToString();

            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            config.Prompt = "Reply with the given text unchanged.";
            config.Collection = "Posts";
            config.SampleObject = JsonConvert.SerializeObject(new { Result = "text" });
            config.UpdateScript = @"const idx = this.Comments.findIndex(c => c.Id == $input.Id);
this.Comments[idx].Result = $output.Result;";
            config.GenAiTransformation = new GenAiTransformation
            {
                Script = "for (const comment of this.Comments) ai.genContext({Text: comment.Text, Id: comment.Id});"
            };
            config.MaxConcurrency = 1;
            config.Identifier = "retry-on-timeout";

            store.Maintenance.Send(new AddGenAiOperation(config));

            var db = await GetDatabase(store.Database);
            GenAiTask etlProcess = null;
            Assert.True(await WaitForValueAsync(() =>
            {
                etlProcess = db.EtlLoader.Processes.OfType<GenAiTask>().FirstOrDefault();
                return Task.FromResult(etlProcess != null);
            }, true, timeout: 15_000), "GenAi ETL process was not loaded in time");

            etlProcess.GetChatCompletionClient().ForTestingPurposesOnly().SimulateFailureAsync = async msg =>
            {
                if (msg.Contains(marker) && msg.Contains("AI Agent Parameters:") == false)
                    await Task.Delay(perCallDelayMs);
            };

            const string docId = "posts/1";
            using (var session = store.OpenSession())
            {
                var post = new GenAiBasics.Post([
                    new GenAiBasics.Comment($"{marker} hopeless comment", "author") { Id = "1" }
                ], "title", "body");
                session.Store(post, docId);
                // an unrelated, EARLIER @refresh already on the document (Refresh feature is off here, so it just sits)
                session.Advanced.GetMetadataFor(post)[Constants.Documents.Metadata.Refresh] = DateTime.UtcNow.AddSeconds(5).ToString("o");
                session.SaveChanges();
            }

            // wait for the retry to be scheduled
            Assert.True(await WaitForValueAsync(async () =>
            {
                using var session = store.OpenAsyncSession();
                var doc = await session.LoadAsync<BlittableJsonReaderObject>(docId);
                return doc != null &&
                       doc.TryGet(Constants.Documents.Metadata.Key, out BlittableJsonReaderObject metadata) &&
                       metadata.TryGet(Constants.Documents.Metadata.GenAiRetry, out BlittableJsonReaderObject retry) &&
                       retry.TryGet(config.Identifier, out BlittableJsonReaderObject taskRetry) && taskRetry.Count >= 1;
            }, true, timeout: 30_000), "retry was not scheduled");

            // @refresh must now be our NextRetry (well in the future), not the earlier +5s value
            using (var session = store.OpenAsyncSession())
            {
                var doc = await session.LoadAsync<BlittableJsonReaderObject>(docId);
                Assert.True(doc.TryGet(Constants.Documents.Metadata.Key, out BlittableJsonReaderObject metadata));
                Assert.True(metadata.TryGet(Constants.Documents.Metadata.Refresh, out DateTime refreshAt), "@refresh should be set");
                Assert.True(refreshAt > DateTime.UtcNow.AddSeconds(30),
                    $"@refresh should be overwritten to NextRetry (well in the future), not kept at the earlier value. Got {refreshAt:o}");
            }
        }
    }
}
