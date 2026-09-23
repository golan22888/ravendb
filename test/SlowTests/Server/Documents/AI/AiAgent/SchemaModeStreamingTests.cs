using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FastTests;
using Raven.Client.Documents.Conventions;
using Raven.Client.Documents.Operations.AI;
using Raven.Server.Documents.AI;
using Raven.Server.Documents.AI.Settings;
using Raven.Server.Documents.ETL.Providers.AI;
using Raven.Server.Logging;
using Raven.Server.ServerWide.Context;
using Sparrow.Json;
using Sparrow.Json.Parsing;
using Sparrow.Logging;
using Tests.Infrastructure;
using Voron;
using Xunit;

namespace SlowTests.Server.Documents.AI.AiAgent
{
    public class SchemaModeStreamingTests : RavenTestBase
    {
        public SchemaModeStreamingTests(ITestOutputHelper output) : base(output)
        {
        }

        private const string AnswerPath = "Answer";

        // ---- OpenAI wire ----------------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_Streaming_WithSchema_ParsesAnswerAndStreamsOnlyThatProperty()
        {
            var sse = OpenAiSse(@"{""Answer"":""hello there""}");

            await WithOpenAi(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = ChatCompletionClient.EmptySchema },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; },
                    new AiUsage(), trace: null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, response.Type);

                var obj = Assert.IsAssignableFrom<BlittableJsonReaderObject>(response.Result);
                Assert.True(obj.TryGet(AnswerPath, out string answer));
                Assert.Equal("hello there", answer);
                Assert.Equal("hello there", Encoding.UTF8.GetString(streamed.ToArray()));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_Streaming_WithoutSchema_ReturnsPlainTextAndStreamsItVerbatim()
        {
            var sse = OpenAiSse("plain prose, {not} json");

            await WithOpenAi(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; },
                    new AiUsage(), trace: null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, response.Type);
                Assert.Equal("plain prose, {not} json", Assert.IsType<string>(response.Result));
                Assert.Equal("plain prose, {not} json", Encoding.UTF8.GetString(streamed.ToArray()));
            });
        }

        // ---- Anthropic wire -------------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_Streaming_WithSchema_ParsesStructuredJson()
        {
            var sse = AnthropicSse(textBlocks: [@"{""Answer"":""structured""}"]);

            await WithAnthropic(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = ChatCompletionClient.EmptySchema },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; },
                    new AiUsage(), trace: null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, response.Type);
                var obj = Assert.IsAssignableFrom<BlittableJsonReaderObject>(response.Result);
                Assert.True(obj.TryGet(AnswerPath, out string answer));
                Assert.Equal("structured", answer);
                Assert.Equal("structured", Encoding.UTF8.GetString(streamed.ToArray()));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_Streaming_WithoutSchema_StreamsPlainTextWithoutTheJsonGate()
        {
            var sse = AnthropicSse(textBlocks: ["just talking, no braces here"]);

            await WithAnthropic(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; },
                    new AiUsage(), trace: null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, response.Type);
                Assert.Equal("just talking, no braces here", Assert.IsType<string>(response.Result));
                Assert.Equal("just talking, no braces here", Encoding.UTF8.GetString(streamed.ToArray()));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_Streaming_WithoutSchema_PlainTextThenToolUse_YieldsToolResponse()
        {
            var sse = AnthropicSse(textBlocks: ["let me check that"], toolUse: ("toolu_1", "get_weather", @"{""city"":""Oslo""}"));

            await WithAnthropic(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                    new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = null },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; },
                    new AiUsage(), trace: null, CancellationToken.None);

                Assert.Equal(AiResponseType.Tool, response.Type);
                Assert.Single(response.ToolCalls);
                Assert.Equal("get_weather", response.ToolCalls[0].Name);

                Assert.Equal("let me check that", Encoding.UTF8.GetString(streamed.ToArray()));

                Assert.True(response.Message.TryGet(AnthropicChatCompletionClientSettings.RawContentSidecarProperty,
                    out BlittableJsonReaderArray raw));
                Assert.NotNull(raw);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_ThinkingDeltas_NeverReachTheCallback_InEitherMode()
        {
            foreach (var schema in new[] { ChatCompletionClient.EmptySchema, null })
            {
                var structured = schema != null;
                var answer = structured ? @"{""Answer"":""visible""}" : "visible";
                var sse = AnthropicSse(textBlocks: [answer], thinking: "secret reasoning the user must not see");

                await WithAnthropic(_ => Sse(sse), async (client, ctx) =>
                {
                    using var streamed = new MemoryStream();
                    var response = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                        new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = schema },
                        m => { streamed.Write(m.Span); return Task.CompletedTask; },
                        new AiUsage(), trace: null, CancellationToken.None);

                    var streamedText = Encoding.UTF8.GetString(streamed.ToArray());
                    Assert.DoesNotContain("secret reasoning", streamedText);
                    Assert.Equal("visible", streamedText);

                    if (structured)
                    {
                        var obj = Assert.IsAssignableFrom<BlittableJsonReaderObject>(response.Result);
                        Assert.True(obj.TryGet(AnswerPath, out string parsed));
                        Assert.Equal("visible", parsed);
                    }
                    else
                    {
                        Assert.Equal("visible", Assert.IsType<string>(response.Result));
                        Assert.DoesNotContain("secret reasoning", (string)response.Result);
                    }
                });
            }
        }

        // ---- one authoritative schema source --------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task RequestSchema_IsTheSingleAuthority_ForWireAndParse()
        {
            foreach (var structured in new[] { true, false })
            {
                var schema = structured ? ChatCompletionClient.EmptySchema : null;

                string openAiBody = null;
                await WithOpenAi(body => { openAiBody = body; return Sse(OpenAiSse(structured ? @"{""Answer"":""x""}" : "x")); },
                    async (client, ctx) =>
                    {
                        using var sink = new MemoryStream();
                        var r = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                            new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = schema },
                            m => { sink.Write(m.Span); return Task.CompletedTask; },
                            new AiUsage(), trace: null, CancellationToken.None);

                        Assert.Equal(structured, openAiBody.Contains("response_format"));
                        if (structured)
                            Assert.IsAssignableFrom<BlittableJsonReaderObject>(r.Result);
                        else
                            Assert.IsType<string>(r.Result);
                    });

                string anthropicBody = null;
                await WithAnthropic(body => { anthropicBody = body; return Sse(AnthropicSse(textBlocks: [structured ? @"{""Answer"":""x""}" : "x"])); },
                    async (client, ctx) =>
                    {
                        using var sink = new MemoryStream();
                        var r = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                            new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = schema },
                            m => { sink.Write(m.Span); return Task.CompletedTask; },
                            new AiUsage(), trace: null, CancellationToken.None);

                        Assert.Equal(structured, anthropicBody.Contains("output_config"));
                        if (structured)
                            Assert.IsAssignableFrom<BlittableJsonReaderObject>(r.Result);
                        else
                            Assert.IsType<string>(r.Result);
                    });
            }
        }

        // ---- streamed vs non-streamed convergence ---------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Anthropic_StreamedAndNonStreamed_Converge_InBothSchemaModes()
        {
            foreach (var schema in new[] { ChatCompletionClient.EmptySchema, null })
            {
                var structured = schema != null;
                var text = structured ? @"{""Answer"":""converged""}" : "converged";

                string streamedAnswer = null;
                string nonStreamedAnswer = null;

                await WithAnthropic(_ => Sse(AnthropicSse(textBlocks: [text])), async (client, ctx) =>
                {
                    using var sink = new MemoryStream();
                    var r = await client.StreamingCompleteAsync(ctx, client.Pool, AnswerPath,
                        new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = schema },
                        m => { sink.Write(m.Span); return Task.CompletedTask; },
                        new AiUsage(), trace: null, CancellationToken.None);
                    streamedAnswer = ExtractAnswer(r.Result, structured);
                });

                await WithAnthropic(_ => Ok(AnthropicNonStreamed(text)), async (client, ctx) =>
                {
                    var r = await client.CompleteAsync(ctx,
                        new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = schema },
                        new AiUsage(), trace: null, CancellationToken.None);
                    nonStreamedAnswer = ExtractAnswer(r.Result, structured);
                });

                Assert.Equal("converged", streamedAnswer);
                Assert.Equal(streamedAnswer, nonStreamedAnswer);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAi_NonStreamed_WithoutSchema_ReturnsPlainTextRatherThanParsingIt()
        {
            await WithOpenAi(_ => Ok(OpenAiNonStreamed("prose, not json")), async (client, ctx) =>
            {
                var r = await client.CompleteAsync(ctx,
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    new AiUsage(), trace: null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, r.Type);
                Assert.Equal("prose, not json", Assert.IsType<string>(r.Result));
            });
        }

        private static string ExtractAnswer(object result, bool structured)
        {
            if (structured == false)
                return Assert.IsType<string>(result);

            var obj = Assert.IsAssignableFrom<BlittableJsonReaderObject>(result);
            Assert.True(obj.TryGet(AnswerPath, out string answer));
            return answer;
        }

        // ---- wire fixtures --------------------------------------------------------------------------------------

        private static string OpenAiSse(string content)
        {
            var chunk = new DynamicJsonValue
            {
                ["choices"] = new DynamicJsonArray
                {
                    new DynamicJsonValue { ["delta"] = new DynamicJsonValue { ["content"] = content } }
                }
            };

            var final = new DynamicJsonValue
            {
                ["choices"] = new DynamicJsonArray { new DynamicJsonValue { ["delta"] = new DynamicJsonValue(), ["finish_reason"] = "stop" } },
                ["usage"] = new DynamicJsonValue { ["prompt_tokens"] = 3, ["completion_tokens"] = 2, ["total_tokens"] = 5 }
            };

            return $"data: {Json(chunk)}\n\ndata: {Json(final)}\n\ndata: [DONE]\n\n";
        }

        private static string OpenAiNonStreamed(string content) => Json(new DynamicJsonValue
        {
            ["choices"] = new DynamicJsonArray
            {
                new DynamicJsonValue
                {
                    ["message"] = new DynamicJsonValue { ["role"] = "assistant", ["content"] = content },
                    ["finish_reason"] = "stop"
                }
            },
            ["usage"] = new DynamicJsonValue { ["prompt_tokens"] = 3, ["completion_tokens"] = 2, ["total_tokens"] = 5 }
        });

        private static string AnthropicSse(string[] textBlocks, string thinking = null, (string Id, string Name, string Input)? toolUse = null)
        {
            var sb = new StringBuilder();
            var index = 0;

            sb.Append(Event("message_start", new DynamicJsonValue
            {
                ["type"] = "message_start",
                ["message"] = new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["usage"] = new DynamicJsonValue { ["input_tokens"] = 3, ["output_tokens"] = 2 }
                }
            }));

            if (thinking != null)
            {
                sb.Append(StartBlock(index, new DynamicJsonValue { ["type"] = "thinking", ["thinking"] = string.Empty }));
                sb.Append(Delta(index, new DynamicJsonValue { ["type"] = "thinking_delta", ["thinking"] = thinking }));
                sb.Append(Delta(index, new DynamicJsonValue { ["type"] = "signature_delta", ["signature"] = "sig-abc" }));
                sb.Append(StopBlock(index++));
            }

            foreach (var text in textBlocks)
            {
                sb.Append(StartBlock(index, new DynamicJsonValue { ["type"] = "text", ["text"] = string.Empty }));
                sb.Append(Delta(index, new DynamicJsonValue { ["type"] = "text_delta", ["text"] = text }));
                sb.Append(StopBlock(index++));
            }

            if (toolUse.HasValue)
            {
                var (id, name, input) = toolUse.Value;
                sb.Append(StartBlock(index, new DynamicJsonValue { ["type"] = "tool_use", ["id"] = id, ["name"] = name, ["input"] = new DynamicJsonValue() }));
                sb.Append(Delta(index, new DynamicJsonValue { ["type"] = "input_json_delta", ["partial_json"] = input }));
                sb.Append(StopBlock(index++));
            }

            sb.Append(Event("message_delta", new DynamicJsonValue
            {
                ["type"] = "message_delta",
                ["delta"] = new DynamicJsonValue { ["stop_reason"] = toolUse.HasValue ? "tool_use" : "end_turn" },
                ["usage"] = new DynamicJsonValue { ["output_tokens"] = 2 }
            }));
            sb.Append(Event("message_stop", new DynamicJsonValue { ["type"] = "message_stop" }));

            return sb.ToString();

            static string StartBlock(int i, DynamicJsonValue block) => Event("content_block_start", new DynamicJsonValue
            {
                ["type"] = "content_block_start", ["index"] = i, ["content_block"] = block
            });

            static string Delta(int i, DynamicJsonValue delta) => Event("content_block_delta", new DynamicJsonValue
            {
                ["type"] = "content_block_delta", ["index"] = i, ["delta"] = delta
            });

            static string StopBlock(int i) => Event("content_block_stop", new DynamicJsonValue
            {
                ["type"] = "content_block_stop", ["index"] = i
            });
        }

        private static string AnthropicNonStreamed(string text) => Json(new DynamicJsonValue
        {
            ["role"] = "assistant",
            ["stop_reason"] = "end_turn",
            ["content"] = new DynamicJsonArray { new DynamicJsonValue { ["type"] = "text", ["text"] = text } },
            ["usage"] = new DynamicJsonValue { ["input_tokens"] = 3, ["output_tokens"] = 2 }
        });

        private static string Event(string name, DynamicJsonValue payload) => $"event: {name}\ndata: {Json(payload)}\n\n";

        private static string Json(DynamicJsonValue djv)
        {
            using var ctx = JsonOperationContext.ShortTermSingleUse();
            return ctx.ReadObject(djv, "fixture").ToString();
        }

        // ---- harness --------------------------------------------------------------------------------------------

        private static BlittableJsonReaderObject Msg(JsonOperationContext ctx, string role, string content) =>
            ctx.ReadObject(new DynamicJsonValue { ["role"] = role, ["content"] = content }, "msg");

        private static HttpResponseMessage Ok(string json) =>
            new(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") };

        private static HttpResponseMessage Sse(string sse) =>
            new(HttpStatusCode.OK) { Content = new StringContent(sse, Encoding.UTF8, "text/event-stream") };

        private static Task WithAnthropic(Func<string, HttpResponseMessage> respond, Func<MockClient, JsonOperationContext, Task> body) =>
            With(new AnthropicChatCompletionClientSettings(new AnthropicSettings("sk-ant-test", "claude-opus-4-8", "https://api.anthropic.com/v1/")), respond, body);

        private static Task WithOpenAi(Func<string, HttpResponseMessage> respond, Func<MockClient, JsonOperationContext, Task> body) =>
            With(new OpenAiChatCompletionClientSettings(new OpenAiSettings("sk-test", "https://api.openai.com/v1/", "gpt-4o")), respond, body);

        private static async Task With(AbstractChatCompletionClientSettings settings, Func<string, HttpResponseMessage> respond,
            Func<MockClient, JsonOperationContext, Task> body)
        {
            using var storageEnv = new StorageEnvironment(StorageEnvironmentOptions.CreateMemoryOnlyForTests());
            using var contextPool = new TransactionContextPool(RavenLogManager.Instance.CreateNullLogger(), storageEnv);
            using var client = new MockClient(contextPool, settings, respond);
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
                await body(client, ctx);
        }

        private sealed class MockClient : ChatCompletionClient
        {
            private readonly Func<string, HttpResponseMessage> _respond;

            public IMemoryContextPool Pool { get; }

            internal MockClient(IMemoryContextPool contextPool, AbstractChatCompletionClientSettings settings, Func<string, HttpResponseMessage> respond)
                : base(contextPool, settings, ConventionsToUse)
            {
                _respond = respond;
                Pool = contextPool;
            }

            protected override Task<HttpResponseMessage> SendRequestAsync(HttpRequestMessage request, CancellationToken token) => Capture(request);

            protected override Task<HttpResponseMessage> SendStreamingRequestAsync(HttpRequestMessage request, CancellationToken token) => Capture(request);

            private async Task<HttpResponseMessage> Capture(HttpRequestMessage request)
            {
                var body = request.Content == null ? null : await request.Content.ReadAsStringAsync();
                return _respond(body);
            }
        }
    }
}
