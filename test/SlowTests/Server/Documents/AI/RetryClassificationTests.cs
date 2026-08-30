using System;
using System.Net;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using FastTests;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Exceptions;
using Raven.Server.Documents.AI;
using Raven.Server.Documents.AI.Settings;
using Raven.Server.Documents.ETL.Providers.AI;
using Raven.Server.Logging;
using Raven.Server.ServerWide.Context;
using Sparrow.Json;
using Sparrow.Json.Parsing;
using Sparrow.Server.Json.Sync;
using Sparrow.Logging;
using Tests.Infrastructure;
using Voron;
using Xunit;

namespace SlowTests.Server.Documents.AI
{
    public class RetryClassificationTests : RavenTestBase
    {
        public RetryClassificationTests(ITestOutputHelper output) : base(output)
        {
        }

        // ---- B: the OpenAI 429 explicit-signal gate --------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_429_WithOnlyResetHeaders_StaysTooManyTokens_NotRateLimit()
        {
            await Assert.ThrowsAsync<TooManyTokensException>(() => RunOpenAi(TooManyRequests(h =>
            {
                h.TryAddWithoutValidation("x-ratelimit-reset-tokens", "30s");
                h.TryAddWithoutValidation("x-ratelimit-reset-requests", "10s");
            })));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_429_WithNoSignalAtAll_IsTooManyTokens()
        {
            await Assert.ThrowsAsync<TooManyTokensException>(() => RunOpenAi(TooManyRequests(_ => { })));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_429_WithExplicitSignal_FoldsResetHeaders_MaxWins()
        {
            var ex = await Assert.ThrowsAsync<RateLimitException>(() => RunOpenAi(TooManyRequests(h =>
            {
                h.TryAddWithoutValidation("retry-after-ms", "1000");
                h.TryAddWithoutValidation("x-ratelimit-reset-tokens", "45s");
                h.TryAddWithoutValidation("x-ratelimit-reset-requests", "10s");
            })));

            Assert.Equal(TimeSpan.FromSeconds(45), ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_429_WithRetryAfterSeconds_IsRetryable()
        {
            var ex = await Assert.ThrowsAsync<RateLimitException>(() => RunOpenAi(TooManyRequests(h =>
                h.TryAddWithoutValidation("Retry-After", "7"))));

            Assert.Equal(TimeSpan.FromSeconds(7), ex.RetryAfter);
        }

        // ---- A: permanent failures are classified before headers are parsed --------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_429_WithUnparseableRetryAfter_IsStillRetryable_WithZeroDelay()
        {
            var ex = await Assert.ThrowsAsync<RateLimitException>(() => RunOpenAi(TooManyRequests(h =>
                h.TryAddWithoutValidation("Retry-After", "1;window=60"))));

            Assert.Equal(TimeSpan.Zero, ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_429_WithSignal_MalformedResetHeader_IsSkipped_NotThrown()
        {
            var ex = await Assert.ThrowsAsync<RateLimitException>(() => RunOpenAi(TooManyRequests(h =>
            {
                h.TryAddWithoutValidation("retry-after-ms", "2000");
                h.TryAddWithoutValidation("x-ratelimit-reset-tokens", "not-a-duration");
            })));

            Assert.Equal(TimeSpan.FromSeconds(2), ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_Permanent400_WithRateLimitHeaders_HasNullRetryAfter()
        {
            var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() => RunOpenAi(
                Error(HttpStatusCode.BadRequest, @"{""error"":{""message"":""bad request""}}", h =>
                {
                    h.TryAddWithoutValidation("x-ratelimit-reset-tokens", "60s");
                    h.TryAddWithoutValidation("Retry-After", "30");
                })));

            Assert.Equal(HttpStatusCode.BadRequest, ex.StatusCode);
            Assert.Null(ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_Permanent400_WithMalformedResetHeader_KeepsTheApiError_NotFormatException()
        {
            var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() => RunOpenAi(
                Error(HttpStatusCode.BadRequest, @"{""error"":{""message"":""bad request""}}", h =>
                    h.TryAddWithoutValidation("x-ratelimit-reset-tokens", "not-a-duration"))));

            Assert.Equal(HttpStatusCode.BadRequest, ex.StatusCode);
            Assert.Null(ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_Permanent500_HasNullRetryAfter()
        {
            var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() => RunOpenAi(
                Error(HttpStatusCode.InternalServerError, @"{""error"":{""message"":""boom""}}", _ => { })));

            Assert.Null(ex.RetryAfter);
        }

        // ---- A: Anthropic ----------------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_Permanent400_HasNullRetryAfter_EvenWithARetryAfterHeader()
        {
            var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() => RunAnthropic(
                Error(HttpStatusCode.BadRequest, @"{""type"":""error"",""error"":{""type"":""invalid_request_error"",""message"":""bad""}}",
                    h => h.TryAddWithoutValidation("Retry-After", "30"))));

            Assert.Equal(HttpStatusCode.BadRequest, ex.StatusCode);
            Assert.Null(ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_Permanent401_HasNullRetryAfter()
        {
            var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() => RunAnthropic(
                Error(HttpStatusCode.Unauthorized, @"{""type"":""error"",""error"":{""type"":""authentication_error"",""message"":""bad key""}}", _ => { })));

            Assert.Null(ex.RetryAfter);
        }

        // ---- deterministic non-429 overflow must never become a rate limit ----------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_ContextOverflow400_WithRetryAfter_IsStillTooManyTokens()
        {
            var ex = await Assert.ThrowsAsync<TooManyTokensException>(() => RunOpenAi(
                Error(HttpStatusCode.BadRequest,
                    @"{""error"":{""message"":""This model's maximum context length is 128000 tokens"",""type"":""invalid_request_error"",""code"":""context_length_exceeded""}}",
                    h => h.TryAddWithoutValidation("Retry-After", "30"))));

            Assert.Null(ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Google_TokenOverflow400_WithRetryAfter_IsStillTooManyTokens()
        {
            var ex = await Assert.ThrowsAsync<TooManyTokensException>(() => RunGoogle(
                Error(HttpStatusCode.BadRequest,
                    @"{""error"":{""code"":400,""message"":""The input token count exceeds the maximum allowed"",""status"":""INVALID_ARGUMENT""}}",
                    h => h.TryAddWithoutValidation("Retry-After", "30"))));

            Assert.Null(ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_PromptTooLong400_WithRetryAfter_IsStillTooManyTokens()
        {
            var ex = await Assert.ThrowsAsync<TooManyTokensException>(() => RunAnthropic(
                Error(HttpStatusCode.BadRequest,
                    @"{""type"":""error"",""error"":{""type"":""invalid_request_error"",""message"":""prompt is too long: 250000 tokens > 200000 maximum""}}",
                    h => h.TryAddWithoutValidation("Retry-After", "30"))));

            Assert.Null(ex.RetryAfter);
        }

        // ---- ...while a 429 token rate limit with a signal stays retryable ----------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_TokenRateLimit429_WithSignal_IsRetryable()
        {
            const string body = @"{""error"":{""message"":""TPM exceeded"",""type"":""tokens""}}";
            AssertClassifiesAsTooManyTokens(OpenAi(), HttpStatusCode.TooManyRequests, body);

            var ex = await Assert.ThrowsAsync<RateLimitException>(() => RunOpenAi(
                Error(HttpStatusCode.TooManyRequests, body,
                    h => h.TryAddWithoutValidation("retry-after-ms", "1500"))));

            Assert.Equal(TimeSpan.FromMilliseconds(1500), ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Azure_TokenRateLimit429_WithSignal_IsRetryable()
        {
            const string body = @"{""error"":{""code"":""rate_limit_exceeded"",""message"":""Requests have exceeded token rate limit of your current tier""}}";
            AssertClassifiesAsTooManyTokens(Azure(), HttpStatusCode.TooManyRequests, body);

            var ex = await Assert.ThrowsAsync<RateLimitException>(() => RunAzure(
                Error(HttpStatusCode.TooManyRequests, body,
                    h => h.TryAddWithoutValidation("Retry-After", "5"))));

            Assert.Equal(TimeSpan.FromSeconds(5), ex.RetryAfter);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Google_QuotaRateLimit429_WithBodyRetryInfo_IsRetryable()
        {
            const string body = @"{""error"":{""code"":429,""message"":""quota exceeded"",""status"":""RESOURCE_EXHAUSTED"",""details"":[{""@type"":""type.googleapis.com/google.rpc.QuotaFailure"",""violations"":[{""quotaId"":""GenerateRequestsPerMinutePerProjectPerModel""}]},{""@type"":""type.googleapis.com/google.rpc.RetryInfo"",""retryDelay"":""7s""}]}}";
            AssertClassifiesAsTooManyTokens(Google(), HttpStatusCode.TooManyRequests, body);

            var ex = await Assert.ThrowsAsync<RateLimitException>(() => RunGoogle(
                Error(HttpStatusCode.TooManyRequests, body, _ => { })));

            Assert.Equal(TimeSpan.FromSeconds(7), ex.RetryAfter);
        }

        // ---- harness --------------------------------------------------------------------------------------------

        private static HttpResponseMessage TooManyRequests(Action<System.Net.Http.Headers.HttpResponseHeaders> addHeaders) =>
            Error((HttpStatusCode)429, @"{""error"":{""message"":""rate limited"",""code"":""rate_limit_exceeded""}}", addHeaders);

        private static HttpResponseMessage Error(HttpStatusCode status, string body, Action<System.Net.Http.Headers.HttpResponseHeaders> addHeaders)
        {
            var response = new HttpResponseMessage(status)
            {
                Content = new StringContent(body, System.Text.Encoding.UTF8, "application/json")
            };
            addHeaders(response.Headers);
            return response;
        }

        private static OpenAiChatCompletionClientSettings OpenAi() =>
            new(new OpenAiSettings("sk-test", "https://api.openai.com/v1/", "gpt-4o"));

        private static GoogleChatCompletionClientSettings Google() =>
            new(new GoogleSettings { ApiKey = "k", Model = "m" });

        private static AzureOpenAiChatCompletionClientSettings Azure() =>
            new(new AzureOpenAiSettings { ApiKey = "k", Model = "m", Endpoint = "https://example.openai.azure.com/" });

        private static Task RunOpenAi(HttpResponseMessage response) => Run(OpenAi(), response);

        private static Task RunGoogle(HttpResponseMessage response) => Run(Google(), response);

        private static Task RunAzure(HttpResponseMessage response) => Run(Azure(), response);

        private static void AssertClassifiesAsTooManyTokens(AbstractChatCompletionClientSettings settings, HttpStatusCode status, string body)
        {
            using (var contextPool = new JsonContextPool())
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            using (var response = new HttpResponseMessage(status))
            using (var content = ctx.Sync.ReadForMemory(body, "err"))
            {
                Assert.Equal(ErrorType.TooManyTokens, settings.ParseError(content, response).ErrorType);
            }
        }

        private static Task RunAnthropic(HttpResponseMessage response) =>
            Run(new AnthropicChatCompletionClientSettings(new AnthropicSettings("sk-ant-test", "claude-opus-4-8", "https://api.anthropic.com/v1/")), response);

        private static async Task Run(AbstractChatCompletionClientSettings settings, HttpResponseMessage response)
        {
            using var storageEnv = new StorageEnvironment(StorageEnvironmentOptions.CreateMemoryOnlyForTests());
            using var contextPool = new TransactionContextPool(RavenLogManager.Instance.CreateNullLogger(), storageEnv);
            using var client = new MockClient(contextPool, settings, response);
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                await client.CompleteAsync(ctx,
                    new AiChatRequest
                    {
                        Messages = [ctx.ReadObject(new DynamicJsonValue { ["role"] = "user", ["content"] = "hi" }, "msg")],
                        Schema = ChatCompletionClient.EmptySchema
                    },
                    new AiUsage(), trace: null, CancellationToken.None);
            }
        }

        private sealed class MockClient : ChatCompletionClient
        {
            private readonly HttpResponseMessage _response;

            internal MockClient(IMemoryContextPool contextPool, AbstractChatCompletionClientSettings settings, HttpResponseMessage response)
                : base(contextPool, settings, ConventionsToUse)
            {
                _response = response;
            }

            protected override Task<HttpResponseMessage> SendRequestAsync(HttpRequestMessage request, CancellationToken token) =>
                Task.FromResult(_response);

            protected override Task<HttpResponseMessage> SendStreamingRequestAsync(HttpRequestMessage request, CancellationToken token) =>
                Task.FromResult(_response);
        }
    }
}
