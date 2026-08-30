using System;
using System.Net;
using System.Reflection;
using System.Threading.Tasks;
using FastTests;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Exceptions;
using Raven.Server.Documents.ETL.Providers.AI.GenAi;
using Tests.Infrastructure;
using Xunit;

namespace SlowTests.Server.Documents.AI
{
    public class GenAiRetryDelaySelectionTests : RavenTestBase
    {
        public GenAiRetryDelaySelectionTests(ITestOutputHelper output) : base(output)
        {
        }

        private static readonly MethodInfo EnterFallbackModeMethod =
            typeof(GenAiTask).GetMethod("EnterFallbackMode", BindingFlags.NonPublic | BindingFlags.Instance);

        private static readonly TimeSpan ProviderFloor = TimeSpan.FromSeconds(1);

        // ---- a provider-supplied delay is honored -----------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task RateLimit_WithDelay_UsesTheProviderDelay()
        {
            Assert.Equal(TimeSpan.FromSeconds(30), await FallbackFor(RateLimit(TimeSpan.FromSeconds(30))));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Overloaded529_WithDelay_UsesTheProviderDelay()
        {
            Assert.Equal(TimeSpan.FromSeconds(7), await FallbackFor(Overloaded(TimeSpan.FromSeconds(7))));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task RateLimit_WithoutDelay_IsFlooredToOneSecond()
        {
            Assert.Equal(ProviderFloor, await FallbackFor(RateLimit(retryAfter: null)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task DelayBelowTheFloor_IsRaisedToIt()
        {
            Assert.Equal(ProviderFloor, await FallbackFor(RateLimit(TimeSpan.FromMilliseconds(200))));
        }

        // ---- the shapes the load phase produces -------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task PartialBatch_FindsTheRetryableFailureAmongUnrelatedOnes()
        {
            var partial = new AggregateException(
                new InvalidOperationException("bad script"),
                new UnsuccessfulAiRequestException("400", HttpStatusCode.BadRequest),
                RateLimit(TimeSpan.FromSeconds(20)));

            Assert.Equal(TimeSpan.FromSeconds(20), await FallbackFor(partial));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task NestedAggregate_IsFlattened()
        {
            var nested = new AggregateException(new AggregateException(RateLimit(TimeSpan.FromSeconds(9))));

            Assert.Equal(TimeSpan.FromSeconds(9), await FallbackFor(nested));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task RateLimit_WinsOverOverloaded529()
        {
            var rateLimitFirst = new AggregateException(RateLimit(TimeSpan.FromSeconds(2)), Overloaded(TimeSpan.FromSeconds(60)));
            var rateLimitLast = new AggregateException(Overloaded(TimeSpan.FromSeconds(60)), RateLimit(TimeSpan.FromSeconds(2)));

            Assert.Equal(TimeSpan.FromSeconds(2), await FallbackFor(rateLimitFirst));
            Assert.Equal(TimeSpan.FromSeconds(2), await FallbackFor(rateLimitLast));
        }

        // ---- everything else uses the generic ETL backoff ---------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Overloaded529_WithoutDelay_UsesTheGenericBackoff()
        {
            var generic = await GenericBackoff();

            Assert.Equal(generic, await FallbackFor(Overloaded(retryAfter: null)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task PermanentAndUnrelatedFailures_UseTheGenericBackoff()
        {
            var generic = await GenericBackoff();

            Assert.Equal(generic, await FallbackFor(new UnsuccessfulAiRequestException("400", HttpStatusCode.BadRequest)));
            Assert.Equal(generic, await FallbackFor(new InvalidOperationException("boom")));
            Assert.Equal(generic, await FallbackFor(new AggregateException(new InvalidOperationException("boom"))));
        }

        // ---- harness ----------------------------------------------------------------------------------------------

        private static RateLimitException RateLimit(TimeSpan? retryAfter)
        {
            var ex = new RateLimitException("429");
            if (retryAfter.HasValue)
                ex.RetryAfter = retryAfter.Value;
            return ex;
        }

        private static UnsuccessfulAiRequestException Overloaded(TimeSpan? retryAfter) =>
            new("529", (HttpStatusCode)529) { RetryAfter = retryAfter };

        private Task<TimeSpan?> GenericBackoff() => FallbackFor(new Exception("not an AI failure"));

        private async Task<TimeSpan?> FallbackFor(Exception e)
        {
            Assert.NotNull(EnterFallbackModeMethod); // a rename must fail loudly, not silently skip every assertion

            using var store = GetDocumentStore();
            var database = await Databases.GetDocumentDatabaseInstanceFor(store);

            var config = new GenAiConfiguration
            {
                Name = "genai-fallback",
                Identifier = "genai-fallback",
                ConnectionStringName = "genai-cs",
                Collection = "Docs",
                Prompt = "unused - no batch runs here",
                SampleObject = "{\"Answer\":\"a\"}",
                TestMode = true, // skips building a chat client, which would need a real connection string
                MaxConcurrency = 1,
                GenAiTransformation = new GenAiTransformation { Script = "ai.genContext(this);" }
            };

            using var task = new GenAiTask(config.Transforms[0], config, database, database.ServerStore);

            EnterFallbackModeMethod.Invoke(task, [e, null]);

            return task.FallbackTime;
        }
    }
}
