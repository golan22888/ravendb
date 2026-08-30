using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FastTests;
using Newtonsoft.Json.Linq;
using Raven.Client.Documents.Conventions;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Documents.Operations.AI.Agents;
using Raven.Server.Documents.AI;
using Raven.Server.Documents.AI.Settings;
using Raven.Server.Documents.Handlers.AI.Agents;
using Sparrow.Json;
using Sparrow.Json.Parsing;
using Tests.Infrastructure;
using Xunit;

namespace SlowTests.Server.Documents.AI
{
    public class ProviderSeamFixesTests(ITestOutputHelper output) : RavenTestBase(output)
    {
        // ---- RavenDB bookkeeping must never be replayed to the model ------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAiWire_StripsRavenBookkeeping_AndKeepsEveryRealMessageField()
        {
            using var contextPool = NewContextPool();
            using var client = NewClient(OpenAi());

            string payload;
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            using (var stream = new MemoryStream())
            {
                var message = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = "the answer",
                    ["tool_calls"] = new DynamicJsonArray
                    {
                        new DynamicJsonValue
                        {
                            ["id"] = "call_1",
                            ["type"] = "function",
                            ["function"] = new DynamicJsonValue { ["name"] = "get_weather", ["arguments"] = "{}" }
                        }
                    },
                    ["refusal"] = null,
                    [ConversationDocument.DateProperty] = "2026-07-27T00:00:00.0000000Z",
                    [ConversationDocument.UsageProperty] = new DynamicJsonValue { ["PromptTokens"] = 10 },
                    [ConversationDocument.OutputSchemaProperty] = "none",
                    [ConversationDocument.SummaryProperty] = true,
                    [ChatCompletionClient.Constants.ResponseFields.SubConversationId] = "conversations/sub/1",
                    [ChatCompletionClient.Constants.ResponseFields.ToolCallId] = "call_1",
                    [AnthropicChatCompletionClientSettings.RawContentSidecarProperty] = new DynamicJsonArray()
                }, "assistant/msg");

                await using (var writer = new AsyncBlittableJsonTextWriter(ctx, stream))
                {
                    client.Settings.WritePayload(writer, ctx, new ChatCompletionPayload
                    {
                        Messages = [message],
                        UseTools = true,
                        Schema = ChatCompletionClient.EmptySchema
                    });
                    await writer.FlushAsync();
                }

                payload = Encoding.UTF8.GetString(stream.ToArray());
            }

            var sent = (JObject)((JArray)JObject.Parse(payload)["messages"])[0];

            Assert.Null(sent[ConversationDocument.DateProperty]);
            Assert.Null(sent[ConversationDocument.UsageProperty]);
            Assert.Null(sent[ConversationDocument.OutputSchemaProperty]);
            Assert.Null(sent[ConversationDocument.SummaryProperty]);
            Assert.Null(sent[ChatCompletionClient.Constants.ResponseFields.SubConversationId]);
            Assert.Null(sent[AnthropicChatCompletionClientSettings.RawContentSidecarProperty]);

            Assert.Equal("assistant", (string)sent["role"]);
            Assert.Equal("the answer", (string)sent["content"]);
            Assert.True(sent.ContainsKey("refusal"));
            Assert.Equal("get_weather", (string)((JArray)sent["tool_calls"])[0]["function"]["name"]);
            Assert.Equal("call_1", (string)sent[ChatCompletionClient.Constants.ResponseFields.ToolCallId]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public void EveryOpenAiFamilyProvider_SharesTheOneFilteringWriter()
        {
            foreach (var settings in OpenAiFamily())
                Assert.Equal(typeof(AbstractChatCompletionClientSettings), WriterOwner(settings));

            Assert.Equal(typeof(AnthropicChatCompletionClientSettings), WriterOwner(Anthropic()));

            static Type WriterOwner(AbstractChatCompletionClientSettings settings) =>
                settings.GetType().GetMethod(nameof(AbstractChatCompletionClientSettings.WritePayload)).DeclaringType;
        }

        // ---- provider-shaped tools are materialized once per conversation call, not per model iteration -------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(false)]
        [InlineData(true)]
        public async Task PreparedTools_AreMaterializedOnce_ForTheWholeConversationCall(bool anthropic)
        {
            using var contextPool = NewContextPool();
            using var client = NewClient(anthropic ? Anthropic() : OpenAi(), _ => Ok(anthropic ? AnthropicReply : OpenAiReply));
            var testing = client.ForTestingPurposesOnly(); // must be armed before the run - the counter only moves when it is

            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var descriptors = Descriptors();
                var prepared = client.PrepareTools(ctx, descriptors);

                for (var iteration = 0; iteration < 4; iteration++)
                {
                    await client.CompleteAsync(ctx, new AiChatRequest
                    {
                        Messages = [UserMessage(ctx, "hi")],
                        Tools = descriptors,
                        PreparedTools = prepared,
                        UseTools = true,
                        Schema = ChatCompletionClient.EmptySchema
                    }, new AiUsage(), null, CancellationToken.None);
                }

                Assert.Equal(4, client.RequestCount);
                Assert.Equal(1, testing.ToolPreparationCount);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task WithoutPreparedTools_TheClientStillWorks_ButMaterializesPerRequest()
        {
            using var contextPool = NewContextPool();
            using var client = NewClient(OpenAi(), _ => Ok(OpenAiReply));
            var testing = client.ForTestingPurposesOnly();

            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                for (var iteration = 0; iteration < 4; iteration++)
                {
                    await client.CompleteAsync(ctx, new AiChatRequest
                    {
                        Messages = [UserMessage(ctx, "hi")],
                        Tools = Descriptors(),
                        UseTools = true,
                        Schema = ChatCompletionClient.EmptySchema
                    }, new AiUsage(), null, CancellationToken.None);
                }

                Assert.Equal(4, testing.ToolPreparationCount);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task PreparedTools_KeepTheExactOutboundToolShapes()
        {
            using var contextPool = NewContextPool();

            using (var openAi = NewClient(OpenAi(), _ => Ok(OpenAiReply)))
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                await CompleteWithTools(openAi, ctx, useTools: true);

                var tool = (JObject)((JArray)JObject.Parse(openAi.LastRequestBody)["tools"])[0];
                Assert.Equal("function", (string)tool["type"]);
                Assert.Equal("get_weather", (string)tool["function"]["name"]);
                Assert.Equal("weather by city", (string)tool["function"]["description"]);
                Assert.NotNull(tool["function"]["parameters"]);
                Assert.True((bool)tool["strict"]);
                Assert.Null(JObject.Parse(openAi.LastRequestBody)["tool_choice"]);

                await CompleteWithTools(openAi, ctx, useTools: false);
                Assert.Equal("none", (string)JObject.Parse(openAi.LastRequestBody)["tool_choice"]);
            }

            using (var anthropic = NewClient(Anthropic(), _ => Ok(AnthropicReply)))
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                await CompleteWithTools(anthropic, ctx, useTools: true);

                var tool = (JObject)((JArray)JObject.Parse(anthropic.LastRequestBody)["tools"])[0];
                Assert.Equal("get_weather", (string)tool["name"]);
                Assert.Equal("weather by city", (string)tool["description"]);
                Assert.NotNull(tool["input_schema"]);
                Assert.True((bool)tool["strict"]);
                Assert.Null(JObject.Parse(anthropic.LastRequestBody)["tool_choice"]);

                await CompleteWithTools(anthropic, ctx, useTools: false);
                Assert.Equal("none", (string)JObject.Parse(anthropic.LastRequestBody)["tool_choice"]["type"]);
            }

            static Task CompleteWithTools(MockClient client, JsonOperationContext ctx, bool useTools)
            {
                var descriptors = Descriptors();
                return client.CompleteAsync(ctx, new AiChatRequest
                {
                    Messages = [UserMessage(ctx, "hi")],
                    Tools = descriptors,
                    PreparedTools = client.PrepareTools(ctx, descriptors),
                    UseTools = useTools,
                    Schema = ChatCompletionClient.EmptySchema
                }, new AiUsage(), null, CancellationToken.None);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task OpenAiFamily_MalformedResponse_CarriesTheRequestId_ThroughTheAdapter()
        {
            using var contextPool = NewContextPool();
            using var client = NewClient(OpenAi(), _ =>
            {
                var r = Ok("{}"); // no choices -> malformed response
                r.Headers.TryAddWithoutValidation("X-Request-ID", "oai_req_9");
                return r;
            }, contextPool);

            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var ex = await Assert.ThrowsAsync<UnexpectedResponseException>(() =>
                    client.CompleteAsync(ctx,
                        new AiChatRequest { Messages = [UserMessage(ctx, "hi")], Schema = ChatCompletionClient.EmptySchema },
                        new AiUsage(), null, CancellationToken.None));

                Assert.Equal("oai_req_9", ex.RequestId);
            }
        }

        // ---- the image-input capability probe ------------------------------------------------------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData("A tiny red square on a white background.")] // ordinary prose
        [InlineData("{\"Answer\":\"a red square\"}")]            // structured-looking text
        public async Task ImageProbe_ReportsSupported_ForAnySuccessfulReplyShape(string answer)
        {
            using var contextPool = NewContextPool();
            using var client = NewClient(OpenAi(), _ => Ok(OpenAiReplyWith(answer)), contextPool);

            Assert.True(await client.TestAcceptsImageInputAsync(CancellationToken.None));

            Assert.Null(JObject.Parse(client.LastRequestBody)["response_format"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ImageProbe_LeavesNormalRequests_DrivenBySchemaAlone()
        {
            using var contextPool = NewContextPool();
            using var client = NewClient(OpenAi(), _ => Ok(OpenAiReplyWith("{\"Answer\":\"yes\"}")), contextPool);

            Assert.True(await client.TestAcceptsImageInputAsync(CancellationToken.None));

            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var structured = await client.CompleteAsync(ctx,
                    new AiChatRequest { Messages = [UserMessage(ctx, "hi")], Schema = ChatCompletionClient.EmptySchema },
                    new AiUsage(), null, CancellationToken.None);
                Assert.NotNull(JObject.Parse(client.LastRequestBody)["response_format"]);
                Assert.IsAssignableFrom<BlittableJsonReaderObject>(structured.Result);

                var plain = await client.CompleteAsync(ctx,
                    new AiChatRequest { Messages = [UserMessage(ctx, "hi")], Schema = null },
                    new AiUsage(), null, CancellationToken.None);
                Assert.Null(JObject.Parse(client.LastRequestBody)["response_format"]);
                Assert.IsType<string>(plain.Result);
            }
        }

        // ---- helpers ---------------------------------------------------------------------------------------------

        private const string AnthropicTextReply =
            "{\"role\":\"assistant\",\"stop_reason\":\"end_turn\"," +
            "\"content\":[{\"type\":\"text\",\"text\":\"{\\\"Answer\\\":\\\"ok\\\"}\"}]," +
            "\"usage\":{\"input_tokens\":3,\"output_tokens\":2}}";

        private const string OpenAiReply =
            """
            {"choices":[{"index":0,"message":{"role":"assistant","content":"{\"Answer\":\"yes\"}"},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8}}
            """;

        private const string AnthropicReply =
            """
            {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"{\"Answer\":\"yes\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}
            """;

        // ---- an Anthropic assistant turn is never emitted with content: [] ---------------------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        public async Task AnthropicAssistantTurn_WithNoUsableContent_IsSkipped(string content)
        {
            var messages = await AnthropicMessages(ctx =>
            [
                UserMessage(ctx, "hi"),
                ctx.ReadObject(new DynamicJsonValue { ["role"] = "assistant", ["content"] = content }, "assistant/msg"),
                UserMessage(ctx, "still there?")
            ]);

            Assert.Equal(2, messages.Count);
            Assert.All(messages, m => Assert.Equal("user", (string)m["role"]));
            AssertNoEmptyContentArray(messages);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicAssistantTurn_WithOnlyToolCalls_EmitsToolUseBlocks()
        {
            var messages = await AnthropicMessages(ctx =>
            [
                UserMessage(ctx, "weather?"),
                ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = null,
                    ["tool_calls"] = new DynamicJsonArray
                    {
                        new DynamicJsonValue
                        {
                            ["id"] = "toolu_1",
                            ["type"] = "function",
                            ["function"] = new DynamicJsonValue { ["name"] = "get_weather", ["arguments"] = "{\"city\":\"Oslo\"}" }
                        }
                    }
                }, "assistant/msg")
            ]);

            var assistant = Assert.Single(messages, m => (string)m["role"] == "assistant");
            var block = Assert.Single((JArray)assistant["content"]);
            Assert.Equal("tool_use", (string)block["type"]);
            Assert.Equal("toolu_1", (string)block["id"]);
            Assert.Equal("get_weather", (string)block["name"]);
            Assert.Equal("Oslo", (string)block["input"]["city"]);
            AssertNoEmptyContentArray(messages);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicAssistantTurn_WithSidecar_IsReplayedVerbatim()
        {
            var messages = await AnthropicMessages(ctx =>
            [
                UserMessage(ctx, "weather?"),
                ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = null,
                    [AnthropicChatCompletionClientSettings.RawContentSidecarProperty] = new DynamicJsonArray
                    {
                        new DynamicJsonValue { ["type"] = "thinking", ["thinking"] = "let me check", ["signature"] = "sig-abc" },
                        new DynamicJsonValue { ["type"] = "tool_use", ["id"] = "toolu_1", ["name"] = "get_weather", ["input"] = new DynamicJsonValue { ["city"] = "Oslo" } }
                    }
                }, "assistant/msg")
            ]);

            var blocks = (JArray)Assert.Single(messages, m => (string)m["role"] == "assistant")["content"];
            Assert.Equal(2, blocks.Count);
            Assert.Equal("thinking", (string)blocks[0]["type"]);
            Assert.Equal("sig-abc", (string)blocks[0]["signature"]);   // signature must survive or Anthropic 400s
            Assert.Equal("tool_use", (string)blocks[1]["type"]);
            AssertNoEmptyContentArray(messages);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicAssistantTurn_SidecarWithOnlyAnEmptyTextBlock_IsSkipped()
        {
            var messages = await AnthropicMessages(ctx =>
            [
                UserMessage(ctx, "hi"),
                ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = null,
                    [AnthropicChatCompletionClientSettings.RawContentSidecarProperty] = new DynamicJsonArray
                    {
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "" }
                    }
                }, "assistant/msg")
            ]);

            Assert.Equal("user", (string)Assert.Single(messages)["role"]);
            AssertNoEmptyContentArray(messages);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicAssistantTurn_SidecarWithEmptyThinkingPlusSignature_IsPreserved()
        {
            var messages = await AnthropicMessages(ctx =>
            [
                UserMessage(ctx, "weather?"),
                ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = null,
                    [AnthropicChatCompletionClientSettings.RawContentSidecarProperty] = new DynamicJsonArray
                    {
                        new DynamicJsonValue { ["type"] = "thinking", ["thinking"] = "", ["signature"] = "sig-xyz" },
                        new DynamicJsonValue { ["type"] = "redacted_thinking", ["data"] = "enc-blob" }
                    }
                }, "assistant/msg")
            ]);

            var blocks = (JArray)Assert.Single(messages, m => (string)m["role"] == "assistant")["content"];
            Assert.Equal(2, blocks.Count);
            Assert.Equal("thinking", (string)blocks[0]["type"]);
            Assert.Equal("", (string)blocks[0]["thinking"]);
            Assert.Equal("sig-xyz", (string)blocks[0]["signature"]);
            Assert.Equal("redacted_thinking", (string)blocks[1]["type"]);
            AssertNoEmptyContentArray(messages);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicMixedConversation_SkipsOnlyTheEmptyAssistantTurn()
        {
            var messages = await AnthropicMessages(ctx =>
            [
                UserMessage(ctx, "first"),
                ctx.ReadObject(new DynamicJsonValue { ["role"] = "assistant", ["content"] = "a real answer" }, "a1"),
                ctx.ReadObject(new DynamicJsonValue { ["role"] = "assistant", ["content"] = "  " }, "a2"),
                UserMessage(ctx, "second")
            ]);

            Assert.Equal(3, messages.Count);
            Assert.Equal("user", (string)messages[0]["role"]);
            Assert.Equal("assistant", (string)messages[1]["role"]);
            Assert.Equal("user", (string)messages[2]["role"]);
            AssertNoEmptyContentArray(messages);
        }

        // ---- PreparedAiTools is bound to the context it was prepared in ------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task PreparedTools_UsedWithTheSameContext_Succeeds()
        {
            using var pool = NewContextPool();
            using var client = NewClient(OpenAi(), contextPool: pool);

            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var prepared = client.PrepareTools(ctx, Descriptors());
                await client.CompleteAsync(ctx, new AiChatRequest
                {
                    Messages = [UserMessage(ctx, "hi")],
                    Tools = Descriptors(),
                    PreparedTools = prepared,
                    UseTools = true,
                    Schema = ChatCompletionClient.EmptySchema
                }, new AiUsage(), trace: null, CancellationToken.None);
            }

            Assert.Equal(1, client.RequestCount);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task PreparedTools_UsedWithADifferentContext_ThrowsBeforeTransport()
        {
            using var pool = NewContextPool();
            using var client = NewClient(OpenAi(), contextPool: pool);

            using (pool.AllocateOperationContext(out JsonOperationContext ctx1))
            using (pool.AllocateOperationContext(out JsonOperationContext ctx2))
            {
                var prepared = client.PrepareTools(ctx1, Descriptors());

                var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                    client.CompleteAsync(ctx2, new AiChatRequest
                    {
                        Messages = [UserMessage(ctx2, "hi")],
                        Tools = Descriptors(),
                        PreparedTools = prepared,
                        UseTools = true,
                        Schema = ChatCompletionClient.EmptySchema
                    }, new AiUsage(), trace: null, CancellationToken.None));

                Assert.Contains("same JsonOperationContext", ex.Message);
            }

            Assert.Equal(0, client.RequestCount);
        }

        // ---- helpers for the Anthropic turn tests ----------------------------------------------------------------

        private static async Task<List<JObject>> AnthropicMessages(Func<JsonOperationContext, List<BlittableJsonReaderObject>> build)
        {
            using var pool = NewContextPool();
            using var client = NewClient(Anthropic(), contextPool: pool);

            string payload;
            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            using (var stream = new MemoryStream())
            {
                await using (var writer = new AsyncBlittableJsonTextWriter(ctx, stream))
                {
                    client.Settings.WritePayload(writer, ctx, new ChatCompletionPayload
                    {
                        Messages = build(ctx),
                        Schema = ChatCompletionClient.EmptySchema
                    });
                    await writer.FlushAsync();
                }

                payload = Encoding.UTF8.GetString(stream.ToArray());
            }

            return ((JArray)JObject.Parse(payload)["messages"]).Cast<JObject>().ToList();
        }

        private static void AssertNoEmptyContentArray(List<JObject> messages)
        {
            foreach (var m in messages)
                if (m["content"] is JArray blocks)
                    Assert.NotEmpty(blocks); // content: [] is rejected by Anthropic
        }

        // ---- Anthropic tools are always strict, with every object closed --------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_RootMissingAdditionalProperties_IsClosedAndStrict()
        {
            var tool = await AnthropicTool(@"{""type"":""object"",""properties"":{""city"":{""type"":""string""}},""required"":[""city""]}");

            Assert.True((bool)tool["strict"]);
            Assert.False((bool)tool["input_schema"]["additionalProperties"]);
            Assert.Equal("string", (string)tool["input_schema"]["properties"]["city"]["type"]);
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(@"{""type"":""object"",""additionalProperties"":false,""properties"":{""city"":{""type"":""string""}}}")] // already closed
        [InlineData(@"{""type"":""object"",""properties"":{}}")]                                                            // object with no properties
        [InlineData(@"{""type"":[""object"",""null""],""properties"":{""city"":{""type"":""string""}}}")]                   // nullable object
        public async Task AnthropicTool_ValidObjectRoot_IsClosedAndStrict(string parametersSchema)
        {
            var tool = await AnthropicTool(parametersSchema);

            Assert.True((bool)tool["strict"]);
            Assert.False((bool)tool["input_schema"]["additionalProperties"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_NestedObject_IsClosedRecursively()
        {
            var tool = await AnthropicTool(@"{
                ""type"":""object"",
                ""properties"":{
                    ""address"":{ ""type"":""object"", ""properties"":{ ""city"":{""type"":""string""} } }
                }
            }");

            var schema = tool["input_schema"];
            Assert.False((bool)schema["additionalProperties"]);
            Assert.False((bool)schema["properties"]["address"]["additionalProperties"]);
            Assert.Equal("string", (string)schema["properties"]["address"]["properties"]["city"]["type"]);
            Assert.True((bool)tool["strict"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_ArrayItemsObject_IsClosed()
        {
            var tool = await AnthropicTool(@"{
                ""type"":""object"",
                ""properties"":{
                    ""stops"":{ ""type"":""array"", ""items"":{ ""type"":""object"", ""properties"":{ ""name"":{""type"":""string""} } } }
                }
            }");

            var items = tool["input_schema"]["properties"]["stops"]["items"];
            Assert.False((bool)items["additionalProperties"]);
            Assert.True((bool)tool["strict"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_CompositionKeywordsAndDefs_AreClosedRecursively()
        {
            var tool = await AnthropicTool(@"{
                ""type"":""object"",
                ""properties"":{
                    ""a"":{ ""anyOf"":[ { ""type"":""object"", ""properties"":{ ""x"":{""type"":""string""} } } ] },
                    ""b"":{ ""oneOf"":[ { ""type"":""object"", ""properties"":{ ""y"":{""type"":""string""} } } ] },
                    ""c"":{ ""allOf"":[ { ""type"":""object"", ""properties"":{ ""z"":{""type"":""string""} } } ] }
                },
                ""$defs"":{ ""Inner"":{ ""type"":""object"", ""properties"":{ ""q"":{""type"":""string""} } } },
                ""definitions"":{ ""Legacy"":{ ""type"":""object"", ""properties"":{ ""r"":{""type"":""string""} } } }
            }");

            var schema = tool["input_schema"];
            Assert.False((bool)schema["additionalProperties"]);
            Assert.False((bool)schema["properties"]["a"]["anyOf"][0]["additionalProperties"]);
            Assert.False((bool)schema["properties"]["b"]["oneOf"][0]["additionalProperties"]);
            Assert.False((bool)schema["properties"]["c"]["allOf"][0]["additionalProperties"]);
            Assert.False((bool)schema["$defs"]["Inner"]["additionalProperties"]);
            Assert.False((bool)schema["definitions"]["Legacy"]["additionalProperties"]);
            Assert.True((bool)tool["strict"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_ExplicitlyDynamicSchema_IsRejectedLocally_NotDowngraded()
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                AnthropicTool(@"{""type"":""object"",""additionalProperties"":true,""properties"":{""city"":{""type"":""string""}}}"));

            Assert.Contains("allows additional properties", ex.Message);
            Assert.Contains("get_weather", ex.Message);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_AdditionalPropertiesAsSchema_IsRejectedLocally()
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                AnthropicTool(@"{""type"":""object"",""additionalProperties"":{""type"":""string""},""properties"":{}}"));

            Assert.Contains("allows additional properties", ex.Message);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_NestedDynamicSchema_IsRejectedLocally()
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                AnthropicTool(@"{
                    ""type"":""object"",
                    ""properties"":{ ""nested"":{ ""type"":""object"", ""additionalProperties"":true } }
                }"));

            Assert.Contains("allows additional properties", ex.Message);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTools_EveryToolIsStrictAndClosed()
        {
            using var pool = NewContextPool();
            using var client = NewClient(Anthropic(), contextPool: pool);

            string payload;
            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            using (var stream = new MemoryStream())
            {
                var tools = client.PrepareTools(ctx,
                [
                    new AiToolDescriptor("first", "d1", @"{""type"":""object"",""properties"":{""a"":{""type"":""object"",""properties"":{}}}}"),
                    new AiToolDescriptor("second", "d2", @"{""type"":""object"",""properties"":{""b"":{""type"":""string""}}}")
                ]);

                await using (var writer = new AsyncBlittableJsonTextWriter(ctx, stream))
                {
                    client.Settings.WritePayload(writer, ctx, new ChatCompletionPayload
                    {
                        Messages = [UserMessage(ctx, "hi")],
                        Tools = tools.Tools,
                        UseTools = true,
                        Schema = ChatCompletionClient.EmptySchema
                    });
                    await writer.FlushAsync();
                }

                payload = Encoding.UTF8.GetString(stream.ToArray());
            }

            var emitted = (JArray)JObject.Parse(payload)["tools"];
            Assert.Equal(2, emitted.Count);
            foreach (var tool in emitted)
            {
                Assert.True((bool)tool["strict"]);
                AssertEveryObjectClosed(tool["input_schema"]);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task CanonicalSchema_AndOtherProviders_AreUnaffected()
        {
            const string canonical = @"{""type"":""object"",""properties"":{""city"":{""type"":""string""}},""required"":[""city""]}";
            var descriptor = new AiToolDescriptor("get_weather", "d", canonical);

            _ = await AnthropicTool(canonical);
            Assert.Equal(canonical, descriptor.ParametersSchema); // not mutated

            foreach (var settings in OpenAiFamily())
            {
                using var pool = NewContextPool();
                using (pool.AllocateOperationContext(out JsonOperationContext ctx))
                {
                    var tool = settings.BuildTool(ctx, descriptor.Name, descriptor.Description, descriptor.ParametersSchema);
                    var json = JObject.Parse(ctx.ReadObject(tool, "tool").ToString());

                    var parameters = json["function"]["parameters"];
                    Assert.Null(parameters["additionalProperties"]);
                    Assert.Equal("string", (string)parameters["properties"]["city"]["type"]);
                }
            }
        }

        private static void AssertEveryObjectClosed(JToken schema)
        {
            if (schema is JObject obj)
            {
                if ((string)obj["type"] == "object" || obj["properties"] != null)
                    Assert.False((bool?)obj["additionalProperties"] ?? true, "an object schema was left open");

                foreach (var name in new[] { "properties", "$defs", "definitions" })
                    if (obj[name] is JObject map)
                        foreach (var child in map.Properties())
                            AssertEveryObjectClosed(child.Value);

                if (obj["items"] != null)
                    AssertEveryObjectClosed(obj["items"]);

                foreach (var name in new[] { "anyOf", "oneOf", "allOf" })
                    if (obj[name] is JArray list)
                        foreach (var child in list)
                            AssertEveryObjectClosed(child);
            }
            else if (schema is JArray array)
            {
                foreach (var child in array)
                    AssertEveryObjectClosed(child);
            }
        }

        private static async Task<JObject> AnthropicTool(string parametersSchema)
        {
            using var pool = NewContextPool();
            using var client = NewClient(Anthropic(), contextPool: pool);

            string payload;
            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            using (var stream = new MemoryStream())
            {
                var tools = client.PrepareTools(ctx, [new AiToolDescriptor("get_weather", "weather by city", parametersSchema)]);

                await using (var writer = new AsyncBlittableJsonTextWriter(ctx, stream))
                {
                    client.Settings.WritePayload(writer, ctx, new ChatCompletionPayload
                    {
                        Messages = [UserMessage(ctx, "hi")],
                        Tools = tools.Tools,
                        UseTools = true,
                        Schema = ChatCompletionClient.EmptySchema
                    });
                    await writer.FlushAsync();
                }

                payload = Encoding.UTF8.GetString(stream.ToArray());
            }

            return (JObject)((JArray)JObject.Parse(payload)["tools"])[0];
        }

        // ---- a parameterless tool still needs a real closed object schema -------------------------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(null)]
        [InlineData("")]
        [InlineData("   ")]
        [InlineData("{}")]
        public async Task AnthropicTool_WithoutParameters_EmitsClosedEmptyObject(string parametersSchema)
        {
            var tool = await AnthropicTool(parametersSchema);
            var schema = tool["input_schema"];

            Assert.Equal("object", (string)schema["type"]);
            Assert.NotNull(schema["properties"]);
            Assert.Empty((JObject)schema["properties"]);
            Assert.False((bool)schema["additionalProperties"]);
            Assert.True((bool)tool["strict"]);
        }

        // ---- the root must describe an object ----------------------------------------------------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(@"{""type"":""string""}")]
        [InlineData(@"{""type"":""array"",""items"":{""type"":""string""}}")]
        [InlineData(@"{""type"":""integer""}")]
        [InlineData(@"{""allOf"":[{""type"":""object"",""properties"":{}}]}")]  // object-ness not provable locally
        public async Task AnthropicTool_NonObjectRoot_IsRejectedLocally(string parametersSchema)
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => AnthropicTool(parametersSchema));

            Assert.Contains("must describe", ex.Message);
            Assert.Contains("object at its root", ex.Message);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_ObjectImpliedByPropertiesOnly_GetsAnExplicitType()
        {
            var tool = await AnthropicTool(@"{""properties"":{""city"":{""type"":""string""}}}");

            Assert.Equal("object", (string)tool["input_schema"]["type"]);
            Assert.False((bool)tool["input_schema"]["additionalProperties"]);
        }

        // ---- positive model-capability policy ----------------------------------------------------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData("claude-sonnet-4-5", true)]
        [InlineData("claude-sonnet-4-5-20250929", true)]   // dated ID resolves through its alias prefix
        [InlineData("claude-haiku-4-5", true)]
        [InlineData("claude-opus-4-5", true)]
        [InlineData("claude-opus-5", true)]
        [InlineData("claude-sonnet-5", true)]
        [InlineData("claude-fable-5", true)]
        [InlineData("claude-opus-4-1", false)]            // pre-4.5
        [InlineData("claude-opus-4-0", false)]
        [InlineData("claude-sonnet-4-0", false)]
        [InlineData("claude-3-5-sonnet-20241022", false)] // Claude 3 family
        [InlineData("claude-3-opus-20240229", false)]
        [InlineData("some-future-model", false)]          // unknown - refused, never assumed capable
        [InlineData("", false)]
        public void StrictToolSupport_IsResolvedFromAPositiveList(string model, bool supported)
        {
            using var pool = NewContextPool();
            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var settings = new AnthropicChatCompletionClientSettings(
                    new AnthropicSettings("sk-ant-test", model, "https://api.anthropic.com/v1/"));

                if (supported)
                {
                    var tool = settings.BuildTool(ctx, "get_weather", "d", @"{""type"":""object"",""properties"":{}}");
                    Assert.NotNull(tool);
                    return;
                }

                var ex = Assert.Throws<InvalidOperationException>(() =>
                    settings.BuildTool(ctx, "get_weather", "d", @"{""type"":""object"",""properties"":{}}"));

                Assert.Contains("cannot be used with tools", ex.Message);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public void AnthropicTool_OnUnsupportedModel_FailsLocally()
        {
            using var pool = NewContextPool();
            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                foreach (var model in new[] { "claude-opus-4-1", "claude-3-5-sonnet-20241022", "totally-unknown" })
                {
                    var settings = new AnthropicChatCompletionClientSettings(
                        new AnthropicSettings("sk-ant-test", model, "https://api.anthropic.com/v1/"));

                    var ex = Assert.Throws<InvalidOperationException>(() =>
                        settings.BuildTool(ctx, "get_weather", "d", @"{""type"":""object"",""properties"":{}}"));

                    Assert.Contains("cannot be used with tools", ex.Message);
                    Assert.Contains("Claude 4.5 and later", ex.Message);
                }
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task UnsupportedModel_WithoutTools_PlainRuns_AndASchemaIsRefusedLocally()
        {
            var settings = new AnthropicChatCompletionClientSettings(
                new AnthropicSettings("sk-ant-test", "claude-opus-4-1", "https://api.anthropic.com/v1/"));

            using var pool = NewContextPool();
            using var client = NewClient(settings, _ => Ok(AnthropicTextReply), contextPool: pool);

            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var response = await client.CompleteAsync(ctx,
                    new AiChatRequest { Messages = [UserMessage(ctx, "hi")], Schema = null },
                    new AiUsage(), trace: null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, response.Type);
                Assert.Equal(1, client.RequestCount);

                var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => client.CompleteAsync(ctx,
                    new AiChatRequest { Messages = [UserMessage(ctx, "hi")], Schema = ChatCompletionClient.EmptySchema },
                    new AiUsage(), trace: null, CancellationToken.None));

                Assert.Contains("structured output", ex.Message);
                Assert.Contains("Claude 4.5 and later", ex.Message);
                Assert.Equal(1, client.RequestCount); // the refused request never reached the transport
            }
        }

        // ---- what RavenDB validates, and what it leaves to Anthropic ------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task AnthropicTool_InvalidAdditionalPropertiesValue_ThrowsRatherThanBeingRewritten()
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                AnthropicTool(@"{""type"":""object"",""additionalProperties"":""nope"",""properties"":{}}"));

            Assert.Contains("invalid `additionalProperties`", ex.Message);
            Assert.Contains("get_weather", ex.Message);
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(@"{""type"":""object"",""properties"":{""n"":{""type"":""integer"",""minimum"":1,""maximum"":10}}}")]
        [InlineData(@"{""type"":""object"",""properties"":{""s"":{""type"":""string"",""minLength"":2,""maxLength"":9}}}")]
        [InlineData(@"{""type"":""object"",""properties"":{""a"":{""type"":""array"",""uniqueItems"":true,""maxItems"":3}}}")]
        [InlineData(@"{""type"":""object"",""properties"":{""x"":{""$ref"":""https://example.com/s.json""}}}")]
        public async Task AnthropicTool_ConstraintsAnthropicOwns_ArePassedThroughUntouched(string parametersSchema)
        {
            var tool = await AnthropicTool(parametersSchema);

            Assert.True((bool)tool["strict"]);
            Assert.False((bool)tool["input_schema"]["additionalProperties"]);

            var emitted = tool["input_schema"].ToString();
            foreach (var keyword in new[] { "minimum", "maximum", "minLength", "maxLength", "uniqueItems", "maxItems", "$ref" })
                if (parametersSchema.Contains(keyword))
                    Assert.Contains(keyword, emitted);
        }

        // ---- composed nodes must not lose an explicit additionalProperties -----------------------------------------

        private static string ComposedNodeSchema(string additionalProperties) => $@"{{
            ""type"":""object"",
            ""properties"":{{
                ""x"":{{
                    ""anyOf"":[ {{ ""type"":""object"", ""properties"":{{ ""a"":{{""type"":""string""}} }} }} ]
                    {additionalProperties}
                }}
            }}
        }}";

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ComposedNode_ExplicitFalse_IsPreserved()
        {
            var tool = await AnthropicTool(ComposedNodeSchema(@",""additionalProperties"":false"));
            var composed = tool["input_schema"]["properties"]["x"];

            Assert.False((bool)composed["additionalProperties"]);              // must not be dropped
            Assert.False((bool)composed["anyOf"][0]["additionalProperties"]);  // the provable branch is still closed
            Assert.True((bool)tool["strict"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ComposedNode_ExplicitTrue_IsRejectedLocally()
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                AnthropicTool(ComposedNodeSchema(@",""additionalProperties"":true")));

            Assert.Contains("allows additional properties", ex.Message);
            Assert.Contains("get_weather", ex.Message);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ComposedNode_SchemaValuedAdditionalProperties_IsRejectedLocally()
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                AnthropicTool(ComposedNodeSchema(@",""additionalProperties"":{""type"":""string""}")));

            Assert.Contains("allows additional properties", ex.Message);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ComposedNode_MalformedAdditionalProperties_IsRejectedLocally()
        {
            var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                AnthropicTool(ComposedNodeSchema(@",""additionalProperties"":""nope""")));

            Assert.Contains("invalid `additionalProperties`", ex.Message);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ComposedNode_WithoutAdditionalProperties_IsNotInvented()
        {
            var tool = await AnthropicTool(ComposedNodeSchema(""));
            var composed = tool["input_schema"]["properties"]["x"];

            Assert.Null(composed["additionalProperties"]);
            Assert.False((bool)composed["anyOf"][0]["additionalProperties"]);
            Assert.True((bool)tool["strict"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task DirectObjectNodes_BehaviourUnchangedByTheComposedFix()
        {
            // absent -> closed; explicit false -> preserved; both on a provable nested object
            var absent = await AnthropicTool(@"{""type"":""object"",""properties"":{""o"":{""type"":""object"",""properties"":{}}}}");
            Assert.False((bool)absent["input_schema"]["properties"]["o"]["additionalProperties"]);

            var explicitFalse = await AnthropicTool(
                @"{""type"":""object"",""properties"":{""o"":{""type"":""object"",""additionalProperties"":false,""properties"":{}}}}");
            Assert.False((bool)explicitFalse["input_schema"]["properties"]["o"]["additionalProperties"]);

            Assert.True((bool)absent["strict"]);
            Assert.True((bool)explicitFalse["strict"]);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ComposedNode_CanonicalSchemaIsNotMutated()
        {
            var canonical = ComposedNodeSchema(@",""additionalProperties"":false");
            var descriptor = new AiToolDescriptor("get_weather", "d", canonical);

            _ = await AnthropicTool(descriptor.ParametersSchema);

            Assert.Equal(canonical, descriptor.ParametersSchema);
        }

        // ---- optional properties stay optional under strict --------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task StrictSchema_WithOptionalProperty_PassesThroughUnchanged()
        {
            // Anthropic strict tools allow optional properties - RavenDB must not invent a required-completeness rule.
            var tool = await AnthropicTool(@"{
                ""type"":""object"",
                ""properties"":{ ""city"":{""type"":""string""}, ""units"":{""type"":""string""} },
                ""required"":[""city""],
                ""additionalProperties"":false
            }");

            var schema = tool["input_schema"];
            Assert.True((bool)tool["strict"]);
            Assert.False((bool)schema["additionalProperties"]);

            // `required` is passed through verbatim - not widened to include `units`
            var required = (JArray)schema["required"];
            Assert.Single(required);
            Assert.Equal("city", (string)required[0]);
            Assert.NotNull(schema["properties"]["units"]);
        }

        // ---- the real root cause: the sub-agent schema generator ----------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task RealSubAgentSchema_IsNormalizedForAnthropic_AndUnchangedForOthers()
        {
            // The real generator emits type/properties/required and no additionalProperties - the schema behind the live 400.
            string canonical;
            using (var pool = NewContextPool())
            using (pool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                canonical = ConversationHandler.GetSchemaForSubAgentTool(ctx, new Dictionary<string, ConversationHandler.ParameterDefinition>
                {
                    ["userPrompt"] = new("what the sub-agent should do", AiAgentParameterValueType.String),
                    ["count"] = new("how many", AiAgentParameterValueType.Number)
                });
            }

            // 1) the provider-neutral schema is what we think it is, and is open
            var canonicalJson = JObject.Parse(canonical);
            Assert.Equal("object", (string)canonicalJson["type"]);
            Assert.Null(canonicalJson["additionalProperties"]);

            // 2) the Anthropic copy is closed and strict
            var tool = await AnthropicTool(canonical);
            Assert.True((bool)tool["strict"]);
            Assert.Equal("object", (string)tool["input_schema"]["type"]);
            Assert.False((bool)tool["input_schema"]["additionalProperties"]);

            // 3) the canonical string was not mutated by that
            Assert.Null(JObject.Parse(canonical)["additionalProperties"]);

            // 4) the OpenAI family still emits the original schema, unclosed
            foreach (var settings in OpenAiFamily())
            {
                using var pool = NewContextPool();
                using (pool.AllocateOperationContext(out JsonOperationContext ctx))
                {
                    var openAiTool = settings.BuildTool(ctx, "sub", "d", canonical);
                    var json = JObject.Parse(ctx.ReadObject(openAiTool, "tool").ToString());
                    Assert.Null(json["function"]["parameters"]["additionalProperties"]);
                }
            }
        }

        private static string OpenAiReplyWith(string answer) =>
            "{\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" +
            Newtonsoft.Json.JsonConvert.ToString(answer) +
            "},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":3,\"total_tokens\":8}}";

        // A TransactionContextPool would need a StorageEnvironment that its Dispose does not own, so every call leaked
        // one. The client only ever allocates a JsonOperationContext, so the plain pool is enough - same as
        // BaseAiConnectorForTesting does.
        private static JsonContextPool NewContextPool() => new();

        private static MockClient NewClient(AbstractChatCompletionClientSettings settings, Func<string, HttpResponseMessage> respond = null, IMemoryContextPool contextPool = null) =>
            new(contextPool ?? NewContextPool(), settings, respond ?? (_ => Ok(OpenAiReply)));

        private static AbstractChatCompletionClientSettings OpenAi() =>
            new OpenAiChatCompletionClientSettings(new OpenAiSettings { ApiKey = "sk-test", Model = "gpt-test", Endpoint = "https://api.openai.com/v1/" });

        private static AbstractChatCompletionClientSettings Anthropic() =>
            new AnthropicChatCompletionClientSettings(new AnthropicSettings("sk-ant-test", "claude-opus-4-8", "https://api.anthropic.com/v1/"));

        private static IEnumerable<AbstractChatCompletionClientSettings> OpenAiFamily()
        {
            yield return OpenAi();
            yield return new AzureOpenAiChatCompletionClientSettings(new AzureOpenAiSettings { ApiKey = "k", Model = "m", Endpoint = "https://example.openai.azure.com/" });
            yield return new OllamaChatCompletionClientSettings(new OllamaSettings { Uri = "http://localhost:11434", Model = "m" });
            yield return new GoogleChatCompletionClientSettings(new GoogleSettings { ApiKey = "k", Model = "m" });
        }

        private static List<AiToolDescriptor> Descriptors() =>
        [
            new("get_weather", "weather by city", ChatCompletionClient.GetSchemaForTool(schema: null, sampleObject: "{\"city\":\"Paris\"}")),
            new("get_time", "time by city", ChatCompletionClient.GetSchemaForTool(schema: null, sampleObject: "{\"city\":\"Paris\"}"))
        ];

        private static BlittableJsonReaderObject UserMessage(JsonOperationContext ctx, string content) =>
            ctx.ReadObject(new DynamicJsonValue { ["role"] = "user", ["content"] = content }, "msg");

        private static HttpResponseMessage Ok(string json) =>
            new(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") };

        private sealed class MockClient : ChatCompletionClient
        {
            private readonly Func<string, HttpResponseMessage> _respond;

            public string LastRequestBody;
            public int RequestCount;

            internal MockClient(IMemoryContextPool contextPool, AbstractChatCompletionClientSettings settings, Func<string, HttpResponseMessage> respond)
                : base(contextPool, settings, ConventionsToUse)
            {
                _respond = respond;
            }

            protected override Task<HttpResponseMessage> SendRequestAsync(HttpRequestMessage request, CancellationToken token) => Capture(request, token);

            protected override Task<HttpResponseMessage> SendStreamingRequestAsync(HttpRequestMessage request, CancellationToken token) => Capture(request, token);

            private async Task<HttpResponseMessage> Capture(HttpRequestMessage request, CancellationToken token)
            {
                RequestCount++;
                LastRequestBody = request.Content != null ? await request.Content.ReadAsStringAsync(token) : null;
                return _respond(LastRequestBody);
            }
        }
    }
}
