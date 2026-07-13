using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FastTests;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using Raven.Client.Documents;
using Raven.Client.Documents.AI;
using Raven.Client.Documents.Conventions;
using Raven.Client.Documents.Linq;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Documents.Operations.AI.Agents;
using Raven.Client.Documents.Operations.ConnectionStrings;
using Raven.Client.Exceptions;
using Raven.Server.Documents.AI;
using Raven.Server.Documents.AI.Settings;
using Raven.Server.Documents.ETL.Providers.AI;
using Raven.Server.Documents.Handlers.AI.Agents;
using Raven.Server.Json;
using Raven.Server.Logging;
using Raven.Server.ServerWide.Context;
using Sparrow.Json;
using Sparrow.Json.Parsing;
using Sparrow.Logging;
using Sparrow.Server.Json.Sync;
using Tests.Infrastructure;
using Voron;
using Xunit;

namespace SlowTests.Server.Documents.AI.AiAgent
{
    public class AnthropicChatCompletionClientTests : RavenTestBase
    {
        public AnthropicChatCompletionClientTests(ITestOutputHelper output) : base(output)
        {
        }

        private const string TextResponse =
            """
            {"id":"msg_1","type":"message","role":"assistant","content":[{"type":"text","text":"{\"Answer\":\"yes\"}"}],"stop_reason":"end_turn","usage":{"input_tokens":10,"output_tokens":5,"cache_read_input_tokens":2}}
            """;

        private const string ToolUseWithThinkingResponse =
            """
            {"id":"msg_2","type":"message","role":"assistant","content":[{"type":"thinking","thinking":"let me think","signature":"sig123"},{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Paris"}}],"stop_reason":"tool_use","usage":{"input_tokens":20,"output_tokens":8}}
            """;

        // ---- provider routing (the one split point both Agents and GenAI ETL go through) ------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public void Factory_RoutesAnthropicConnection_ToNativeClient_ElseToOpenAiFamily()
        {
            using var storageEnv = new StorageEnvironment(StorageEnvironmentOptions.CreateMemoryOnlyForTests());
            using var contextPool = new TransactionContextPool(RavenLogManager.Instance.CreateNullLogger(), storageEnv);

            var anthropic = new AiConnectionString
            {
                Name = "claude",
                ModelType = AiModelType.Chat,
                AnthropicSettings = new AnthropicSettings("sk-ant-test", "claude-opus-4-8", "https://api.anthropic.com/v1/")
            };
            using (var client = ChatCompletionClient.CreateChatCompletionClient(contextPool, anthropic))
                Assert.IsType<AnthropicChatCompletionClientSettings>(client.Settings);

            var ollama = new AiConnectionString
            {
                Name = "ollama",
                ModelType = AiModelType.Chat,
                OllamaSettings = new OllamaSettings { Uri = "http://localhost:11434", Model = "x" }
            };
            using (var client = ChatCompletionClient.CreateChatCompletionClient(contextPool, ollama))
            {
                Assert.IsType<ChatCompletionClient>(client);
                Assert.IsNotType<AnthropicChatCompletionClientSettings>(client.Settings);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public void TestConnection_DeserializesAnthropicSettings_AndResolvesTheProvider()
        {
            using var storageEnv = new StorageEnvironment(StorageEnvironmentOptions.CreateMemoryOnlyForTests());
            using var contextPool = new TransactionContextPool(RavenLogManager.Instance.CreateNullLogger(), storageEnv);
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var json = ctx.ReadObject(new DynamicJsonValue
                {
                    ["ApiKey"] = "sk-ant-test",
                    ["Model"] = "claude-sonnet-4-5",
                    ["Endpoint"] = "https://api.anthropic.com/v1/",
                    ["MaxOutputTokens"] = 4096
                }, "anthropic/settings");

                var settings = JsonDeserializationServer.AnthropicSettings(json);
                Assert.Equal("sk-ant-test", settings.ApiKey);
                Assert.Equal("claude-sonnet-4-5", settings.Model);
                Assert.Equal(4096, settings.MaxOutputTokens);

                var connectionString = new AiConnectionString { Name = "claude", ModelType = AiModelType.Chat, AnthropicSettings = settings };
                Assert.Equal(AiConnectorType.Anthropic, connectionString.GetActiveProvider());
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public void StudioModelList_DeserializesAnthropicSettings_AndBuildsTheProvider()
        {
            using var storageEnv = new StorageEnvironment(StorageEnvironmentOptions.CreateMemoryOnlyForTests());
            using var contextPool = new TransactionContextPool(RavenLogManager.Instance.CreateNullLogger(), storageEnv);
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
            {
                var json = ctx.ReadObject(new DynamicJsonValue
                {
                    ["ConnectorType"] = nameof(AiConnectorType.Anthropic),
                    ["AnthropicSettings"] = new DynamicJsonValue
                    {
                        ["ApiKey"] = "sk-ant-test",
                        ["Model"] = "claude-opus-5",
                        ["Endpoint"] = "https://api.anthropic.com/v1/"
                    }
                }, "ai/models/request");

                var request = JsonDeserializationServer.AiModelsRequest(json);
                Assert.Equal(AiConnectorType.Anthropic, request.ConnectorType);
                Assert.NotNull(request.AnthropicSettings);
                Assert.Equal("claude-opus-5", request.AnthropicSettings.Model);

                var settings = new AnthropicChatCompletionClientSettings(request.AnthropicSettings);
                Assert.Equal("models", settings.GetRelativeModelsUri());
                Assert.Equal(new Uri("https://api.anthropic.com/v1/"), settings.GetBaseEndpointUri());
            }
        }

        // ---- request translation ---------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_HoistsSystem_TranslatesUser_UnwrapsSchema_SetsMaxTokens()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "system", "You are helpful."), Msg(ctx, "user", "Hi")],
                    Schema = ChatCompletionClient.GetSchemaFromSampleObject("{\"Answer\":\"the answer\"}")
                };

                var usage = new AiUsage();
                var response = await client.CompleteAsync(ctx, request, usage, trace: null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Equal("You are helpful.", (string)body["system"]);
                Assert.Equal("claude-opus-4-8", (string)body["model"]);
                Assert.Equal(8192, (int)body["max_tokens"]);
                Assert.Equal("user", (string)body["messages"][0]["role"]);
                Assert.Equal("text", (string)body["messages"][0]["content"][0]["type"]);
                Assert.Equal("Hi", (string)body["messages"][0]["content"][0]["text"]);
                Assert.Equal("json_schema", (string)body["output_config"]["format"]["type"]);
                Assert.Equal("object", (string)body["output_config"]["format"]["schema"]["type"]); // unwrapped inner schema

                Assert.Equal(AiResponseType.Result, response.Type);
                Assert.True(((BlittableJsonReaderObject)response.Result).TryGet("Answer", out string answer));
                Assert.Equal("yes", answer);
                Assert.Equal(12, usage.PromptTokens);   // input(10) + cache_read(2)
                Assert.Equal(5, usage.CompletionTokens);
                Assert.Equal(2, usage.CachedTokens);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_UsesXApiKeyAndAnthropicVersionHeaders()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, new AiChatRequest { Messages = [Msg(ctx, "user", "Hi")], Schema = EmptySchema() }, new AiUsage(), null, CancellationToken.None);

                Assert.Equal("sk-ant-test", client.ApiKeyHeader);
                Assert.Equal("2023-06-01", client.VersionHeader);
                Assert.Null(client.AuthorizationHeader); // native uses x-api-key, never Bearer
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_TranslatesToolsWithStrict_AndToolChoiceNoneWhenNotUsed()
        {
            const string toolSchema = "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"],\"additionalProperties\":false}";
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", "Hi")],
                    Tools = [new AiToolDescriptor("get_weather", "Get the weather", toolSchema)],
                    UseTools = false,
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Equal("get_weather", (string)body["tools"][0]["name"]);
                Assert.Equal("Get the weather", (string)body["tools"][0]["description"]);
                Assert.Equal(true, (bool)body["tools"][0]["strict"]);
                Assert.Equal("object", (string)body["tools"][0]["input_schema"]["type"]);
                Assert.Equal("none", (string)body["tool_choice"]["type"]); // UseTools == false
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_GroupsConsecutiveToolResultsIntoOneUserTurn()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var assistant = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = null,
                    ["tool_calls"] = new DynamicJsonArray
                    {
                        ToolCall("call_1", "get_weather", "{}"),
                        ToolCall("call_2", "get_time", "{}")
                    }
                }, "assistant");

                var request = new AiChatRequest
                {
                    Messages =
                    [
                        Msg(ctx, "user", "Hi"),
                        assistant,
                        Msg(ctx, "tool", "sunny", toolCallId: "call_1"),
                        Msg(ctx, "tool", "noon", toolCallId: "call_2")
                    ],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                var toolResultTurn = messages.Last(m => (string)m["role"] == "user" &&
                                                        m["content"] is JArray c && c.Count > 0 && (string)c[0]["type"] == "tool_result");
                var blocks = (JArray)toolResultTurn["content"];
                Assert.Equal(2, blocks.Count);
                Assert.Equal("call_1", (string)blocks[0]["tool_use_id"]);
                Assert.Equal("sunny", (string)blocks[0]["content"]);
                Assert.Equal("call_2", (string)blocks[1]["tool_use_id"]);
                Assert.Equal("noon", (string)blocks[1]["content"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_ReplaysPriorStructuredAssistantAnswer_AsText_NotEmpty()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var priorAnswer = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = new DynamicJsonValue { ["Answer"] = "yes" }
                }, "assistant");

                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", "first?"), priorAnswer, Msg(ctx, "user", "and now?")],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                var assistantContent = (JArray)messages.First(m => (string)m["role"] == "assistant")["content"];
                Assert.NotEmpty(assistantContent);
                Assert.Equal("text", (string)assistantContent[0]["type"]);
                Assert.Contains("Answer", (string)assistantContent[0]["text"]); // the structured answer, stringified
                Assert.Contains("yes", (string)assistantContent[0]["text"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_MultiPartUserContent_BecomesSeparateTextBlocks_NotRawJson()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var multiPart = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "user",
                    ["content"] = new DynamicJsonArray
                    {
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "part one" },
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "part two" }
                    }
                }, "user");

                await client.CompleteAsync(ctx, new AiChatRequest { Messages = [multiPart], Schema = EmptySchema() }, new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                var content = (JArray)messages.First(m => (string)m["role"] == "user")["content"];
                Assert.Equal(2, content.Count);
                Assert.Equal("part one", (string)content[0]["text"]);
                Assert.Equal("part two", (string)content[1]["text"]);
            });
        }

        // ---- extended thinking (request side) --------------------------------------------------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(AiReasoningLevel.Low, "low")]
        [InlineData(AiReasoningLevel.Medium, "medium")]
        [InlineData(AiReasoningLevel.High, "high")]
        public async Task Request_UsesAdaptiveThinkingAndEffort_OnCurrentModels(AiReasoningLevel mode, string expectedEffort)
        {
            var settings = new AnthropicSettings("sk-ant-test", "claude-opus-5", "https://api.anthropic.com/v1/", maxOutputTokens: 8192, reasoning: mode);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);

                var thinking = body["thinking"];
                Assert.NotNull(thinking);
                Assert.Equal("adaptive", (string)thinking["type"]);
                Assert.Null(thinking["budget_tokens"]);

                Assert.Equal(expectedEffort, (string)body["output_config"]["effort"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_MergesEffortAndFormat_IntoOneOutputConfig()
        {
            var settings = new AnthropicSettings("sk-ant-test", "claude-opus-5", "https://api.anthropic.com/v1/", maxOutputTokens: 8192, reasoning: AiReasoningLevel.High);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);

                var outputConfig = (JObject)JObject.Parse(client.LastRequestBody)["output_config"];
                Assert.Equal("high", (string)outputConfig["effort"]);
                Assert.Equal("json_schema", (string)outputConfig["format"]["type"]);
            });
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData("claude-sonnet-4-5", AiReasoningLevel.Low, 2048)]
        [InlineData("claude-sonnet-4-5", AiReasoningLevel.Medium, 4096)]
        [InlineData("claude-sonnet-4-5", AiReasoningLevel.High, 6144)]
        [InlineData("claude-haiku-4-5", AiReasoningLevel.Medium, 4096)]
        [InlineData("claude-opus-4-20250514", AiReasoningLevel.Medium, 4096)]
        [InlineData("claude-sonnet-4-20250514", AiReasoningLevel.Medium, 4096)]
        [InlineData("claude-3-7-sonnet-20250219", AiReasoningLevel.Medium, 4096)]
        [InlineData("claude-3-7-sonnet-latest", AiReasoningLevel.Medium, 4096)]
        public async Task Request_UsesBudgetTokens_OnLegacyModels(string model, AiReasoningLevel mode, int expectedBudget)
        {
            var settings = new AnthropicSettings("sk-ant-test", model, "https://api.anthropic.com/v1/", maxOutputTokens: 8192, reasoning: mode);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                var thinking = body["thinking"];
                Assert.Equal("enabled", (string)thinking["type"]);
                Assert.Equal(expectedBudget, (int)thinking["budget_tokens"]);
                Assert.True((int)thinking["budget_tokens"] < (int)body["max_tokens"]);

                Assert.Null(body["output_config"]?["effort"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_BudgetTokens_NeverDropBelowTheMinimum()
        {
            var settings = new AnthropicSettings("sk-ant-test", "claude-sonnet-4-5", "https://api.anthropic.com/v1/", maxOutputTokens: 2048, reasoning: AiReasoningLevel.Low);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);

                Assert.Equal(1024, (int)JObject.Parse(client.LastRequestBody)["thinking"]["budget_tokens"]);
            });
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData("claude-opus-5")]
        [InlineData("claude-sonnet-4-5")]
        [InlineData("some-unreleased-model")]
        public async Task Request_OmitsThinking_ByDefault(string model)
        {
            var settings = new AnthropicSettings("sk-ant-test", model, "https://api.anthropic.com/v1/", maxOutputTokens: 8192);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Null(body["thinking"]);
                Assert.Null(body["output_config"]?["effort"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_EmitsBudgetTokensAndEffort_OnOpus45()
        {
            var settings = new AnthropicSettings("sk-ant-test", "claude-opus-4-5", "https://api.anthropic.com/v1/", maxOutputTokens: 8192, reasoning: AiReasoningLevel.Medium);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);

                Assert.Equal("enabled", (string)body["thinking"]["type"]);
                Assert.Equal(4096, (int)body["thinking"]["budget_tokens"]);
                Assert.Equal("medium", (string)body["output_config"]["effort"]);

                Assert.Equal("json_schema", (string)body["output_config"]["format"]["type"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public void Reasoning_IsNotValidatedInTheClient()
        {
            var effortModel = new List<string>();
            new AnthropicSettings("k", "claude-opus-5", maxOutputTokens: 512, reasoning: AiReasoningLevel.High).ValidateFields(effortModel);
            Assert.Empty(effortModel);

            var budgetModel = new List<string>();
            new AnthropicSettings("k", "claude-sonnet-4-5", maxOutputTokens: 8192, reasoning: AiReasoningLevel.High).ValidateFields(budgetModel);
            Assert.Empty(budgetModel);

            var defaultLevel = new List<string>();
            new AnthropicSettings("k", "claude-sonnet-4-5", maxOutputTokens: 512).ValidateFields(defaultLevel);
            Assert.Empty(defaultLevel);
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_EffortModel_WithSmallMaxOutputTokens_IsNotFalselyRejected()
        {
            var settings = new AnthropicSettings("sk-ant-test", "claude-opus-5", "https://api.anthropic.com/v1/", maxOutputTokens: 512, reasoning: AiReasoningLevel.High);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Equal("adaptive", (string)body["thinking"]["type"]);
                Assert.Equal("high", (string)body["output_config"]["effort"]);
                Assert.Null(body["thinking"]["budget_tokens"]);

                Assert.Equal(512, (int)body["max_tokens"]);
            });
        }

        // ---- empty canonical content (must never become an empty text block) --------------------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData("")]
        [InlineData("   ")]
        public async Task Request_EmptyUserContent_SkipsTheTurn_InsteadOfSendingAnEmptyTextBlock(string content)
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", content), Msg(ctx, "user", "real question")],
                    Schema = EmptySchema()
                }, new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                Assert.Single(messages);
                Assert.Equal("real question", (string)messages[0]["content"][0]["text"]);
                Assert.DoesNotContain(AllTextBlocks(messages), t => t.Length == 0);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_AllEmptyMultipartContent_SkipsTheTurn()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var empties = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "user",
                    ["content"] = new DynamicJsonArray
                    {
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "" },
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "" }
                    }
                }, "msg");

                await client.CompleteAsync(ctx, new AiChatRequest { Messages = [empties, Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                    new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                Assert.Single(messages);
                Assert.DoesNotContain(AllTextBlocks(messages), t => t.Length == 0);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_MixedEmptyAndValidParts_KeepsOnlyTheValidOnes()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var mixed = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "user",
                    ["content"] = new DynamicJsonArray
                    {
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "" },
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "keep me" },
                        new DynamicJsonValue { ["type"] = "text", ["text"] = "" }
                    }
                }, "msg");

                await client.CompleteAsync(ctx, new AiChatRequest { Messages = [mixed], Schema = EmptySchema() },
                    new AiUsage(), null, CancellationToken.None);

                var blocks = (JArray)((JArray)JObject.Parse(client.LastRequestBody)["messages"])[0]["content"];
                Assert.Single(blocks);
                Assert.Equal("keep me", (string)blocks[0]["text"]);
            });
        }

        // ---- preflight: deterministic configuration failures never reach the transport ----------------------------

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(false)]
        [InlineData(true)]
        public async Task Preflight_ThinkingOnUnknownModel_ThrowsBeforeAnyTransportCall(bool streaming)
        {
            var settings = new AnthropicSettings("sk-ant-test", "some-unreleased-model", "https://api.anthropic.com/v1/", maxOutputTokens: 8192, reasoning: AiReasoningLevel.High);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => RunAsync(client, ctx, Simple(ctx), streaming));

                Assert.Contains("some-unreleased-model", ex.Message);
                Assert.Contains("claude-opus-5", ex.Message);
                Assert.Equal(0, client.RequestCount);
                Assert.Null(client.LastRequestBody);
            });
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(false)]
        [InlineData(true)]
        public async Task Preflight_BudgetWithNoRoomUnderMaxTokens_ThrowsBeforeAnyTransportCall(bool streaming)
        {
            var settings = new AnthropicSettings("sk-ant-test", "claude-sonnet-4-5", "https://api.anthropic.com/v1/", maxOutputTokens: 512, reasoning: AiReasoningLevel.High);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => RunAsync(client, ctx, Simple(ctx), streaming));

                Assert.Contains("claude-sonnet-4-5", ex.Message);
                Assert.Contains("1024", ex.Message);
                Assert.Equal(0, client.RequestCount);
                Assert.Null(client.LastRequestBody);
            });
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(false)]
        [InlineData(true)]
        public async Task Preflight_EveryMessageEmpty_ThrowsBeforeAnyTransportCall(bool streaming)
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest { Messages = [Msg(ctx, "user", ""), Msg(ctx, "user", "   ")], Schema = EmptySchema() };

                var ex = await Assert.ThrowsAsync<InvalidOperationException>(() => RunAsync(client, ctx, request, streaming));

                Assert.Contains("every message was empty", ex.Message);
                Assert.Equal(0, client.RequestCount);
                Assert.Null(client.LastRequestBody);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Preflight_AssistantFirstConversation_FailsLocally_NamingBothRemedies()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var assistant = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["tool_calls"] = new DynamicJsonArray { ToolCall("call_1", "InitialQuery", "{}") }
                }, "a1");

                var request = new AiChatRequest
                {
                    Messages =
                    [
                        Msg(ctx, "system", "You are helpful."),
                        assistant,
                        Msg(ctx, "tool", "{\"rows\":[]}", toolCallId: "call_1"),
                        Msg(ctx, "user", "What do you see?")
                    ],
                    Schema = EmptySchema()
                };

                var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                    client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None));

                Assert.Contains("first conversation turn", ex.Message);
                Assert.Contains("SendToModel", ex.Message);
                Assert.Contains("AddToInitialContext", ex.Message);
                Assert.Equal(0, client.RequestCount); // fails locally, the provider is never contacted
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task PostSummarization_LeadingSummary_IsHoistedIntoSystem()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var summary = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = "Summary: the user asked about fruit and was told apples are red.",
                    [ConversationDocument.SummaryProperty] = true
                }, "summary-msg");

                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "system", "You answer briefly."), summary, Msg(ctx, "user", "Name a vegetable.")],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                var system = (string)body["system"];
                Assert.Contains("You answer briefly.", system);
                Assert.Contains("apples are red", system);

                var messages = (JArray)body["messages"];
                Assert.Single(messages);
                Assert.Equal("user", (string)messages[0]["role"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task MidConversationSummary_StaysAnAssistantTurn()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var summary = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = "Summary: earlier talk.",
                    [ConversationDocument.SummaryProperty] = true
                }, "summary-msg");

                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", "hi"), summary, Msg(ctx, "user", "and now?")],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Null(body["system"]);

                var messages = (JArray)body["messages"];
                Assert.Equal(3, messages.Count);
                Assert.Equal("assistant", (string)messages[1]["role"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Preflight_PlainAssistantFirstConversation_FailsLocally()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "system", "s"), Msg(ctx, "assistant", "an earlier answer"), Msg(ctx, "user", "hi")],
                    Schema = EmptySchema()
                };

                var ex = await Assert.ThrowsAsync<InvalidOperationException>(() =>
                    client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None));

                Assert.Contains("first conversation turn", ex.Message);
                Assert.Equal(0, client.RequestCount);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Preflight_UnknownModel_WithDefaultReasoning_StaysValid()
        {
            var settings = new AnthropicSettings("sk-ant-test", "some-unreleased-model", "https://api.anthropic.com/v1/", maxOutputTokens: 8192);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    new AiUsage(), null, CancellationToken.None);

                Assert.Equal(1, client.RequestCount);
                Assert.Null(JObject.Parse(client.LastRequestBody)["thinking"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Preflight_AdaptiveModel_IsNotSubjectToTheBudgetTokenConstraint()
        {
            var settings = new AnthropicSettings("sk-ant-test", "claude-opus-4-8", "https://api.anthropic.com/v1/", maxOutputTokens: 512, reasoning: AiReasoningLevel.High);
            await WithClient(settings, _ => Ok(TextResponse), async (client, ctx) =>
            {
                await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Equal("adaptive", (string)body["thinking"]["type"]);
                Assert.Null(body["thinking"]["budget_tokens"]);
                Assert.Equal("high", (string)body["output_config"]["effort"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Preflight_NormalizesTheConversationOnce_AndTheWriterReusesIt()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "system", "be brief"), Msg(ctx, "user", "hi"), Msg(ctx, "user", "   ")],
                    Schema = EmptySchema()
                };

                Assert.Null(request.ProviderPrepared);
                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);
                Assert.NotNull(request.ProviderPrepared);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Equal("be brief", (string)body["system"]);
                var messages = (JArray)body["messages"];
                Assert.Single(messages);
                Assert.Equal("hi", (string)((JArray)messages[0]["content"])[0]["text"]);
            });
        }

        // ---- response parsing + thinking sidecar ----------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_ParsesToolUse_AndPreservesRawContentSidecar()
        {
            await WithClient(_ => Ok(ToolUseWithThinkingResponse), async (client, ctx) =>
            {
                var response = await client.CompleteAsync(ctx, new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = EmptySchema() },
                    new AiUsage(), null, CancellationToken.None);

                Assert.Equal(AiResponseType.Tool, response.Type);
                Assert.Single(response.ToolCalls);
                Assert.Equal("toolu_1", response.ToolCalls[0].Id);
                Assert.Equal("get_weather", response.ToolCalls[0].Name);
                Assert.Contains("Paris", response.ToolCalls[0].Arguments);

                var message = JObject.Parse(response.Message.ToString());
                var sidecar = (JArray)message[AnthropicChatCompletionClientSettings.RawContentSidecarProperty];
                Assert.Equal("thinking", (string)sidecar[0]["type"]);
                Assert.Equal("sig123", (string)sidecar[0]["signature"]);
                Assert.Equal("tool_use", (string)sidecar[1]["type"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_EchoesThinkingSidecarVerbatim_NotRebuilt()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var assistant = ctx.ReadObject(new DynamicJsonValue
                {
                    ["role"] = "assistant",
                    ["content"] = null,
                    ["tool_calls"] = new DynamicJsonArray { ToolCall("toolu_9", "f", "{}") },
                    [AnthropicChatCompletionClientSettings.RawContentSidecarProperty] = new DynamicJsonArray
                    {
                        new DynamicJsonValue { ["type"] = "thinking", ["thinking"] = "deep thought", ["signature"] = "sigABC" },
                        new DynamicJsonValue { ["type"] = "tool_use", ["id"] = "toolu_9", ["name"] = "f", ["input"] = new DynamicJsonValue() }
                    }
                }, "assistant");

                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", "go"), assistant, Msg(ctx, "tool", "done", toolCallId: "toolu_9")],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                var assistantTurn = messages.First(m => (string)m["role"] == "assistant");
                var content = (JArray)assistantTurn["content"];

                Assert.Equal("thinking", (string)content[0]["type"]);
                Assert.Equal("deep thought", (string)content[0]["thinking"]);
                Assert.Equal("sigABC", (string)content[0]["signature"]);
                Assert.Equal("tool_use", (string)content[1]["type"]);
                Assert.Equal("toolu_9", (string)content[1]["id"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ToolUse_ThinkingSidecar_SurvivesPersistence_AndEchoesVerbatimOnNextRequest()
        {
            await WithClient(
                body => body.Contains("tool_result") ? Ok(TextResponse) : Ok(ToolUseWithThinkingResponse),
                async (client, ctx) =>
                {
                    var first = await client.CompleteAsync(ctx,
                        new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = EmptySchema() },
                        new AiUsage(), null, CancellationToken.None);
                    Assert.Equal(AiResponseType.Tool, first.Type);

                    var persistedAssistant = ctx.Sync.ReadForMemory(first.Message.ToString(), "persisted/assistant");
                    Assert.True(persistedAssistant.TryGet(AnthropicChatCompletionClientSettings.RawContentSidecarProperty, out BlittableJsonReaderArray _));

                    await client.CompleteAsync(ctx, new AiChatRequest
                    {
                        Messages = [Msg(ctx, "user", "weather?"), persistedAssistant, Msg(ctx, "tool", "sunny, 20C", toolCallId: "toolu_1")],
                        Schema = EmptySchema()
                    }, new AiUsage(), null, CancellationToken.None);

                    var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];

                    var assistantContent = (JArray)messages.First(m => (string)m["role"] == "assistant")["content"];
                    Assert.Equal("thinking", (string)assistantContent[0]["type"]);
                    Assert.Equal("let me think", (string)assistantContent[0]["thinking"]);
                    Assert.Equal("sig123", (string)assistantContent[0]["signature"]);
                    Assert.Equal("tool_use", (string)assistantContent[1]["type"]);
                    Assert.Equal("toolu_1", (string)assistantContent[1]["id"]);

                    var toolResult = (JArray)messages.Last(m => (string)m["role"] == "user" &&
                                                                m["content"] is JArray c && c.Count > 0 && (string)c[0]["type"] == "tool_result")["content"];
                    Assert.Equal("toolu_1", (string)toolResult[0]["tool_use_id"]);
                    Assert.Equal("sunny, 20C", (string)toolResult[0]["content"]);
                });
        }

        // ---- error mapping ---------------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_429_MapsToRateLimit_HonoringRetryAfter()
        {
            await WithClient(_ => Error(HttpStatusCode.TooManyRequests, "rate_limit_error", "slow down", ("retry-after", "30")), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<RateLimitException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));
                Assert.Equal(TimeSpan.FromSeconds(30), ex.RetryAfter);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_429_RetryAfterAsHttpDate_IsAlsoHonored()
        {
            var date = DateTimeOffset.UtcNow.AddSeconds(45).ToString("R", CultureInfo.InvariantCulture);

            await WithClient(_ => Error(HttpStatusCode.TooManyRequests, "rate_limit_error", "slow down", ("retry-after", date)), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<RateLimitException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));

                Assert.InRange(ex.RetryAfter, TimeSpan.FromSeconds(30), TimeSpan.FromSeconds(60));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_429_WithoutRetryAfterHeader_StaysRetryableWithNoDelay()
        {
            await WithClient(_ => Error(HttpStatusCode.TooManyRequests, "rate_limit_error", "slow down"), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<RateLimitException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));

                Assert.Equal(TimeSpan.Zero, ex.RetryAfter);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_529_MapsToOverloaded_HonoringRetryAfter_NotRateLimit()
        {
            await WithClient(_ => Error((HttpStatusCode)529, "overloaded_error", "overloaded", ("retry-after", "10")), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));
                Assert.IsNotType<RateLimitException>(ex);
                Assert.Equal((HttpStatusCode)529, ex.StatusCode);
                Assert.Equal(TimeSpan.FromSeconds(10), ex.RetryAfter.Value);
                Assert.StartsWith("Status Code: 529, Message: ", ex.Message);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_529_WithoutRetryAfterHeader_LeavesRetryAfterNull_ForBaseBackoff()
        {
            await WithClient(_ => Error((HttpStatusCode)529, "overloaded_error", "overloaded"), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));
                Assert.IsNotType<RateLimitException>(ex);
                Assert.Equal((HttpStatusCode)529, ex.StatusCode);
                Assert.Null(ex.RetryAfter);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_MaxTokensStopReason_MapsToTooManyTokens()
        {
            const string truncated = """{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"{\"Ans"}],"stop_reason":"max_tokens","usage":{"input_tokens":1,"output_tokens":1}}""";
            await WithClient(_ => Ok(truncated), async (client, ctx) =>
                await Assert.ThrowsAsync<TooManyTokensException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_ContextWindowExceededStopReason_MapsToTooManyTokens()
        {
            const string truncated = """{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"{\"Ans"}],"stop_reason":"model_context_window_exceeded","usage":{"input_tokens":1,"output_tokens":1}}""";
            await WithClient(_ => Ok(truncated), async (client, ctx) =>
                await Assert.ThrowsAsync<TooManyTokensException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_PromptTooLong400_MapsToTooManyTokens()
        {
            await WithClient(_ => Error(HttpStatusCode.BadRequest, "invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum"), async (client, ctx) =>
                await Assert.ThrowsAsync<TooManyTokensException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_UsageLimit400_StaysUnsuccessful_NotTooManyTokens()
        {
            await WithClient(_ => Error(HttpStatusCode.BadRequest, "invalid_request_error", "You have reached your specified API usage limits. You will regain access on 2026-09-01 at 00:00 UTC."), async (client, ctx) =>
                await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_NoContent_CarriesTheAnthropicRequestId()
        {
            const string noContent = """{"id":"m","type":"message","role":"assistant","stop_reason":"end_turn","usage":{"input_tokens":1,"output_tokens":1}}""";
            await WithClient(_ =>
            {
                var r = Ok(noContent);
                r.Headers.TryAddWithoutValidation("request-id", "req_test_123");
                return r;
            }, async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<UnexpectedResponseException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));

                Assert.Equal("req_test_123", ex.RequestId);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_RefusalStopReason_MapsToRefusedToAnswer()
        {
            const string refusal = """{"id":"m","type":"message","role":"assistant","content":[{"type":"text","text":"I can't help with that"}],"stop_reason":"refusal","usage":{"input_tokens":1,"output_tokens":1}}""";
            await WithClient(_ => Ok(refusal), async (client, ctx) =>
                await Assert.ThrowsAsync<RefusedToAnswerException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Error_400_SchemaRejected_SurfacesCleanly_WithoutSilentRetry()
        {
            await WithClient(_ => Error(HttpStatusCode.BadRequest, "invalid_request_error", "schema is too complex"), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));
                Assert.Equal(HttpStatusCode.BadRequest, ex.StatusCode);
                Assert.Contains("schema is too complex", ex.Message);
                Assert.Equal(1, client.RequestCount); // no silent downgrade/retry
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task SimulateFailureHook_IsHonored_ForFailureInjectionParityWithOpenAi()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                client.ForTestingPurposesOnly().SimulateFailureAsync = _ => throw new RefusedToAnswerException("injected");
                await Assert.ThrowsAsync<RefusedToAnswerException>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));
                Assert.Null(client.LastRequestBody);
            });
        }

        // ---- streaming -------------------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_TextResult_StreamsAnswerAndParsesResult()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":7}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"Answer\":\"hi\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var usage = new AiUsage();
                var response = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, usage, null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, response.Type);
                Assert.True(((BlittableJsonReaderObject)response.Result).TryGet("Answer", out string answer));
                Assert.Equal("hi", answer);
                Assert.Equal(7, usage.PromptTokens);
                Assert.Equal(4, usage.CompletionTokens);
                Assert.Contains("hi", Encoding.UTF8.GetString(streamed.ToArray())); // the answer property was streamed
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_ToolUse_WithThinking_BuildsToolResponseAndSidecar()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":9}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sigZ"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("content_block_start", """{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_5","name":"get_weather","input":{}}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"Paris\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":1}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = EmptySchema() },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None);

                Assert.Equal(AiResponseType.Tool, response.Type);
                Assert.Single(response.ToolCalls);
                Assert.Equal("get_weather", response.ToolCalls[0].Name);
                Assert.Contains("Paris", response.ToolCalls[0].Arguments);

                Assert.Empty(streamed.ToArray());
                var sidecar = (JArray)JObject.Parse(response.Message.ToString())[AnthropicChatCompletionClientSettings.RawContentSidecarProperty];
                Assert.Equal("thinking", (string)sidecar[0]["type"]);
                Assert.Equal("let me", (string)sidecar[0]["thinking"]);
                Assert.Equal("sigZ", (string)sidecar[0]["signature"]);
                Assert.Equal("tool_use", (string)sidecar[1]["type"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_ProseAlongsideToolUse_DoesNotReachTheAnswerParser()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":5}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me look "}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"that up."}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("content_block_start", """{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_9","name":"get_weather","input":{}}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":1}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = EmptySchema() },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None);

                Assert.Equal(AiResponseType.Tool, response.Type);
                Assert.Single(response.ToolCalls);
                Assert.Empty(streamed.ToArray()); // prose was never streamed as the answer

                var sidecar = (JArray)JObject.Parse(response.Message.ToString())[AnthropicChatCompletionClientSettings.RawContentSidecarProperty];
                Assert.Equal("text", (string)sidecar[0]["type"]);
                Assert.Equal("Let me look that up.", (string)sidecar[0]["text"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_ProseEndingTheTurn_FailsAsInvalidStructuredResponse()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":5}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me look that up."}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":8}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var ex = await Assert.ThrowsAnyAsync<Exception>(() => client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = EmptySchema() },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None));

                Assert.Contains("Let me look that up.", ex.Message);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_JsonAnswerWithLeadingWhitespace_StillStreams()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":5}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"  \n"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"Answer\":\"hi\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None);

                Assert.Equal(AiResponseType.Result, response.Type);
                Assert.True(((BlittableJsonReaderObject)response.Result).TryGet("Answer", out string answer));
                Assert.Equal("hi", answer);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_Sidecar_DropsEmptyTextBlocks_ButKeepsEmptyThinkingBlocks()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":9}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("content_block_start", """{"type":"content_block_start","index":1,"content_block":{"type":"thinking","thinking":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":1,"delta":{"type":"signature_delta","signature":"sigQ"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":1}""") +
                Event("content_block_start", """{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_7","name":"get_weather","input":{}}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Rome\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":2}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = EmptySchema() },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None);

                var sidecar = (JArray)JObject.Parse(response.Message.ToString())[AnthropicChatCompletionClientSettings.RawContentSidecarProperty];

                Assert.DoesNotContain(sidecar, b => (string)b["type"] == "text");

                var thinking = Assert.Single(sidecar, b => (string)b["type"] == "thinking");
                Assert.Equal(string.Empty, (string)thinking["thinking"]);
                Assert.Equal("sigQ", (string)thinking["signature"]);

                Assert.Contains(sidecar, b => (string)b["type"] == "tool_use");
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_FragmentedInputJsonDelta_ReassemblesToolArguments()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_frag","name":"get_weather","input":{}}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"ci"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"ty\":\"Par"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"is\",\"un"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"it\":\"c\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":6}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var response = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "weather?")], Schema = EmptySchema() },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None);

                Assert.Equal(AiResponseType.Tool, response.Type);
                Assert.Single(response.ToolCalls);
                var args = JObject.Parse(response.ToolCalls[0].Arguments); // parses => valid JSON, fully reassembled
                Assert.Equal("Paris", (string)args["city"]);
                Assert.Equal("c", (string)args["unit"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_MaxTokensStopReason_ThrowsTooManyTokens()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"Ans"}}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":9}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
                await Assert.ThrowsAsync<TooManyTokensException>(() =>
                    client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                        new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                        _ => Task.CompletedTask, new AiUsage(), null, CancellationToken.None)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_ContextWindowExceededStopReason_ThrowsTooManyTokens()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"Ans"}}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"model_context_window_exceeded"},"usage":{"output_tokens":9}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
                await Assert.ThrowsAsync<TooManyTokensException>(() =>
                    client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                        new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                        _ => Task.CompletedTask, new AiUsage(), null, CancellationToken.None)));
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_ErrorEvent_ThrowsUnsuccessfulMidStream()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("error", """{"type":"error","error":{"type":"overloaded_error","message":"the server is overloaded"}}""");

            await WithClient(_ => SseWithHeaders(sse, ("retry-after", "7")), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAsync<UnsuccessfulAiRequestException>(() =>
                    client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                        new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                        _ => Task.CompletedTask, new AiUsage(), null, CancellationToken.None));
                Assert.Contains("the server is overloaded", ex.Message);
                Assert.NotEqual(HttpStatusCode.OK, ex.StatusCode);
                Assert.Equal((HttpStatusCode)529, ex.StatusCode);
                Assert.Equal(TimeSpan.FromSeconds(7), ex.RetryAfter);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_AggregatesUsageAcrossStartAndDeltaEvents()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":20,"cache_creation_input_tokens":5}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"{\"Answer\":\"ok\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":50}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                var usage = new AiUsage();
                await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                    _ => Task.CompletedTask, usage, null, CancellationToken.None);

                Assert.Equal(125, usage.PromptTokens);      // 100 + 20 + 5
                Assert.Equal(20, usage.CachedTokens);        // cache_read
                Assert.Equal(50, usage.CompletionTokens);    // output
                Assert.Equal(175, usage.TotalTokens);        // prompt + completion
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_EmptyTextDelta_IsIgnored_NotStreamed()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var r = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None);

                Assert.Equal("Hello world", Assert.IsType<string>(r.Result));
                Assert.Equal("Hello world", Encoding.UTF8.GetString(streamed.ToArray()));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_TruncatedStream_FailsInsteadOfReturningAPartialAnswer()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"The answer is"}}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                using var streamed = new MemoryStream();
                var ex = await Assert.ThrowsAsync<UnexpectedResponseException>(() => client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None));

                Assert.Contains("message_stop", ex.Message);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_TruncatedStream_MidToolUse_DoesNotExecuteWithEmptyArguments()
        {
            var sse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}""");

            await WithClient(_ => Sse(sse), async (client, ctx) =>
            {
                var ex = await StreamAndCaptureError(client, ctx);
                Assert.IsType<UnexpectedResponseException>(ex);
                Assert.Contains("message_stop", ex.Message);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streamed_And_NonStreamed_ToolUse_PersistTheSameMessageShape()
        {
            var streamedSse =
                Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":20}}}""") +
                Event("content_block_start", """{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"let me think"}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig123"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":0}""") +
                Event("content_block_start", """{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}""") +
                Event("content_block_delta", """{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}""") +
                Event("content_block_stop", """{"type":"content_block_stop","index":1}""") +
                Event("message_delta", """{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":8}}""") +
                Event("message_stop", """{"type":"message_stop"}""");

            JObject nonStreamed = null, streamed = null;

            await WithClient(_ => Ok(ToolUseWithThinkingResponse), async (client, ctx) =>
            {
                var r = await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);
                nonStreamed = JObject.Parse(r.Message.ToString());
            });

            await WithClient(_ => Sse(streamedSse), async (client, ctx) =>
            {
                var r = await client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                    new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                    _ => Task.CompletedTask, new AiUsage(), null, CancellationToken.None);
                streamed = JObject.Parse(r.Message.ToString());
            });

            foreach (var message in new[] { nonStreamed, streamed })
            {
                Assert.Equal("assistant", (string)message["role"]);
                Assert.True(message["content"] == null || message["content"].Type == JTokenType.Null);
                Assert.Equal("toolu_1", (string)message["tool_calls"][0]["id"]);
                Assert.Equal("get_weather", (string)message["tool_calls"][0]["function"]["name"]);
                Assert.Equal("function", (string)message["tool_calls"][0]["type"]);
                Assert.NotNull(message[AnthropicChatCompletionClientSettings.RawContentSidecarProperty]);
            }
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_RateLimitErrorEvent_IsARetryableRateLimit_NotAnOverload()
        {
            await WithClient(_ => SseWithHeaders(ErrorEvent("rate_limit_error", "slow down"), ("retry-after", "11")), async (client, ctx) =>
            {
                var ex = Assert.IsType<RateLimitException>(await StreamAndCaptureError(client, ctx));

                Assert.Contains("rate_limit_error", ex.Message);
                Assert.Contains("slow down", ex.Message);
                Assert.Equal(TimeSpan.FromSeconds(11), ex.RetryAfter);
            });
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData("authentication_error", HttpStatusCode.Unauthorized)]
        [InlineData("permission_error", HttpStatusCode.Forbidden)]
        [InlineData("invalid_request_error", HttpStatusCode.BadRequest)]
        [InlineData("not_found_error", HttpStatusCode.NotFound)]
        public async Task Streaming_PermanentErrorEvent_IsNotRetryable(string errorType, HttpStatusCode expected)
        {
            await WithClient(_ => SseWithHeaders(ErrorEvent(errorType, "nope"), ("retry-after", "30")), async (client, ctx) =>
            {
                var ex = Assert.IsType<UnsuccessfulAiRequestException>(await StreamAndCaptureError(client, ctx));

                Assert.Equal(expected, ex.StatusCode);
                Assert.Contains(errorType, ex.Message);
                Assert.Contains("nope", ex.Message);

                Assert.Null(ex.RetryAfter);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Streaming_UnknownErrorEvent_PreservesTypeAndMessage_AndIsNotAssumedRetryable()
        {
            await WithClient(_ => SseWithHeaders(ErrorEvent("some_future_error", "who knows"), ("retry-after", "30")), async (client, ctx) =>
            {
                var ex = Assert.IsType<UnsuccessfulAiRequestException>(await StreamAndCaptureError(client, ctx));

                Assert.Contains("some_future_error", ex.Message);
                Assert.Contains("who knows", ex.Message);
                Assert.Null(ex.RetryAfter);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_EmptyTextBlock_IsNotEchoedOnTheNextRequest()
        {
            const string toolUseWithEmptyText = """
                {"id":"msg_1","type":"message","role":"assistant","stop_reason":"tool_use",
                 "content":[{"type":"text","text":""},
                            {"type":"thinking","thinking":"","signature":"sigE"},
                            {"type":"tool_use","id":"toolu_1","name":"get_weather","input":{"city":"Oslo"}}],
                 "usage":{"input_tokens":5,"output_tokens":8}}
                """;

            await WithClient(
                body => body.Contains("tool_result") ? Ok(TextResponse) : Ok(toolUseWithEmptyText),
                async (client, ctx) =>
                {
                    var first = await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);
                    Assert.Equal(AiResponseType.Tool, first.Type);

                    var persisted = ctx.Sync.ReadForMemory(first.Message.ToString(), "assistant/turn");

                    await client.CompleteAsync(ctx, new AiChatRequest
                    {
                        Messages = [Msg(ctx, "user", "weather?"), persisted, Msg(ctx, "tool", "12C", toolCallId: "toolu_1")],
                        Schema = EmptySchema()
                    }, new AiUsage(), null, CancellationToken.None);

                    var assistant = ((JArray)JObject.Parse(client.LastRequestBody)["messages"])
                        .First(m => (string)m["role"] == "assistant");
                    var echoed = (JArray)assistant["content"];

                    Assert.DoesNotContain(echoed, b => (string)b["type"] == "text" && ((string)b["text"]).Length == 0);

                    var thinking = Assert.Single(echoed, b => (string)b["type"] == "thinking");
                    Assert.Equal(string.Empty, (string)thinking["thinking"]);
                    Assert.Equal("sigE", (string)thinking["signature"]);
                    Assert.Contains(echoed, b => (string)b["type"] == "tool_use");
                });
        }

        // ---- attachments -----------------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_TranslatesAttachments_IntoImageDocumentAndTextBlocks()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", "look at these")],
                    Attachments =
                    [
                        new AiAttachment("pic.png", "image/png", AiAttachmentSource.FromAttachment, "aW1n"),
                        new AiAttachment("doc.pdf", "application/pdf", AiAttachmentSource.FromAttachment, "cGRm"),
                        new AiAttachment("notes.txt", "text/plain", AiAttachmentSource.FromAttachment, "hello notes")
                    ],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                var blocks = (JArray)messages.Last(m => (string)m["role"] == "user")["content"];

                var image = blocks.First(b => (string)b["type"] == "image");
                Assert.Equal("base64", (string)image["source"]["type"]);
                Assert.Equal("image/png", (string)image["source"]["media_type"]);
                Assert.Equal("aW1n", (string)image["source"]["data"]);

                var doc = blocks.First(b => (string)b["type"] == "document");
                Assert.Equal("application/pdf", (string)doc["source"]["media_type"]);
                Assert.Equal("cGRm", (string)doc["source"]["data"]);

                var text = blocks.First(b => (string)b["type"] == "text");
                Assert.Equal("hello notes", (string)text["text"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_NotFoundAttachment_BecomesTextNote()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", "look")],
                    Attachments = [new AiAttachment("missing.png", "image/png", AiAttachmentSource.NotFound, null)],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                var blocks = (JArray)messages.Last(m => (string)m["role"] == "user")["content"];
                Assert.Contains(blocks, b => (string)b["type"] == "text" && ((string)b["text"]).Contains("could not be loaded"));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_UnknownAttachmentType_Throws()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "user", "look")],
                    Attachments = [new AiAttachment("archive.zip", "application/zip", AiAttachmentSource.FromAttachment, "emlw")],
                    Schema = EmptySchema()
                };

                await Assert.ThrowsAsync<InvalidOperationException>(() =>
                    client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None));
            });
        }

        // ---- extra request / response cases ----------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Request_JoinsMultipleSystemMessages()
        {
            await WithClient(_ => Ok(TextResponse), async (client, ctx) =>
            {
                var request = new AiChatRequest
                {
                    Messages = [Msg(ctx, "system", "First rule."), Msg(ctx, "system", "Second rule."), Msg(ctx, "user", "hi")],
                    Schema = EmptySchema()
                };

                await client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

                var system = (string)JObject.Parse(client.LastRequestBody)["system"];
                Assert.Contains("First rule.", system);
                Assert.Contains("Second rule.", system);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_ProseAlongsideToolUse_IsNotParsedAsTheAnswer()
        {
            const string json = """
                {"id":"msg_1","type":"message","role":"assistant","stop_reason":"tool_use",
                 "content":[{"type":"text","text":"Let me look that up."},
                            {"type":"tool_use","id":"toolu_3","name":"get_weather","input":{"city":"Paris"}}],
                 "usage":{"input_tokens":5,"output_tokens":8}}
                """;

            await WithClient(_ => Ok(json), async (client, ctx) =>
            {
                var response = await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);

                Assert.Equal(AiResponseType.Tool, response.Type);
                Assert.Single(response.ToolCalls);

                var sidecar = (JArray)JObject.Parse(response.Message.ToString())[AnthropicChatCompletionClientSettings.RawContentSidecarProperty];
                Assert.Equal("Let me look that up.", (string)sidecar[0]["text"]);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_ProseEndingTheTurn_FailsAsInvalidStructuredResponse()
        {
            const string json = """
                {"id":"msg_1","type":"message","role":"assistant","stop_reason":"end_turn",
                 "content":[{"type":"text","text":"Let me look that up."}],
                 "usage":{"input_tokens":5,"output_tokens":8}}
                """;

            await WithClient(_ => Ok(json), async (client, ctx) =>
            {
                var ex = await Assert.ThrowsAnyAsync<Exception>(() =>
                    client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None));

                Assert.Contains("Let me look that up.", ex.Message);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_MultipleToolUseBlocks_ProduceMultipleToolCalls()
        {
            const string twoTools =
                """
                {"id":"m","type":"message","role":"assistant","content":[
                  {"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"Paris"}},
                  {"type":"tool_use","id":"t2","name":"get_time","input":{"tz":"CET"}}],
                "stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":3}}
                """;
            await WithClient(_ => Ok(twoTools), async (client, ctx) =>
            {
                var response = await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);
                Assert.Equal(AiResponseType.Tool, response.Type);
                Assert.Equal(2, response.ToolCalls.Count);
                Assert.Equal("get_weather", response.ToolCalls[0].Name);
                Assert.Equal("get_time", response.ToolCalls[1].Name);
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_ThinkingBlockInTextAnswer_IsIgnored_NotInResult()
        {
            const string thinkingThenText =
                """
                {"id":"m","type":"message","role":"assistant","content":[
                  {"type":"thinking","thinking":"the user greeted me, secret reasoning","signature":"s"},
                  {"type":"text","text":"{\"Answer\":\"hello\"}"}],
                "stop_reason":"end_turn","usage":{"input_tokens":5,"output_tokens":3}}
                """;
            await WithClient(_ => Ok(thinkingThenText), async (client, ctx) =>
            {
                var response = await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);
                Assert.Equal(AiResponseType.Result, response.Type);
                Assert.True(((BlittableJsonReaderObject)response.Result).TryGet("Answer", out string answer));
                Assert.Equal("hello", answer);
                Assert.DoesNotContain("secret reasoning", response.Result.ToString()); // thinking never reaches the answer
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task Response_RedactedThinking_IsPreservedInSidecar()
        {
            const string redacted =
                """
                {"id":"m","type":"message","role":"assistant","content":[
                  {"type":"redacted_thinking","data":"ENCRYPTED_BLOB"},
                  {"type":"tool_use","id":"t1","name":"f","input":{}}],
                "stop_reason":"tool_use","usage":{"input_tokens":5,"output_tokens":3}}
                """;
            await WithClient(_ => Ok(redacted), async (client, ctx) =>
            {
                var response = await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);
                Assert.Equal(AiResponseType.Tool, response.Type);
                var sidecar = (JArray)JObject.Parse(response.Message.ToString())[AnthropicChatCompletionClientSettings.RawContentSidecarProperty];
                Assert.Equal("redacted_thinking", (string)sidecar[0]["type"]);
                Assert.Equal("ENCRYPTED_BLOB", (string)sidecar[0]["data"]);
            });
        }

        // ---- connectivity probes ---------------------------------------------------------------------------------

        [RavenFact(RavenTestCategory.Ai)]
        public async Task TestCompleteAsync_EmptyUserPrompt_SendsNonEmptyUserTurn()
        {
            await WithClient(_ => Ok(TextResponse), async (client, _) =>
            {
                await client.TestCompleteAsync("You are helpful.", userPrompt: "", EmptySchema(), CancellationToken.None);

                var body = JObject.Parse(client.LastRequestBody);
                Assert.Equal("You are helpful.", (string)body["system"]);
                var userText = (string)((JArray)((JArray)body["messages"]).First(m => (string)m["role"] == "user")["content"])[0]["text"];
                Assert.False(string.IsNullOrEmpty(userText));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task TestAcceptsImageInputAsync_SendsImageProbe_AndReturnsTrue()
        {
            await WithClient(_ => Ok(TextResponse), async (client, _) =>
            {
                var accepts = await client.TestAcceptsImageInputAsync(CancellationToken.None);
                Assert.True(accepts);

                var messages = (JArray)JObject.Parse(client.LastRequestBody)["messages"];
                var blocks = (JArray)messages.Last(m => (string)m["role"] == "user")["content"];
                Assert.Contains(blocks, b => (string)b["type"] == "image");
            });
        }

        // ---- image-input capability probe --------------------------------------------------------------------------

        private const string PlainTextResponse =
            """
            {"id":"msg_p","type":"message","role":"assistant","content":[{"type":"text","text":"A tiny red square on a white background."}],"stop_reason":"end_turn","usage":{"input_tokens":9,"output_tokens":11}}
            """;

        [RavenTheory(RavenTestCategory.Ai)]
        [InlineData(PlainTextResponse)]  // ordinary prose
        [InlineData(TextResponse)]       // structured-looking text
        public async Task ImageProbe_ReportsSupported_ForAnySuccessfulReplyShape(string response)
        {
            await WithClient(_ => Ok(response), async (client, _) =>
            {
                Assert.True(await client.TestAcceptsImageInputAsync(CancellationToken.None));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ImageProbe_ReportsUnsupported_WhenTheProviderRejectsTheImage()
        {
            await WithClient(_ => Error(HttpStatusCode.BadRequest, "invalid_request_error", "messages.0.content.1.image: unsupported"), async (client, _) =>
            {
                Assert.False(await client.TestAcceptsImageInputAsync(CancellationToken.None));
            });
        }

        [RavenFact(RavenTestCategory.Ai)]
        public async Task ImageProbe_AsksForNoOutputFormat_AndLeavesNormalRequestsStructured()
        {
            await WithClient(body => Ok(body.Contains("output_config") ? TextResponse : PlainTextResponse), async (client, ctx) =>
            {
                Assert.True(await client.TestAcceptsImageInputAsync(CancellationToken.None));
                Assert.Null(JObject.Parse(client.LastRequestBody)["output_config"]?["format"]);

                var structured = await client.CompleteAsync(ctx, Simple(ctx), new AiUsage(), null, CancellationToken.None);
                Assert.NotNull(JObject.Parse(client.LastRequestBody)["output_config"]["format"]);
                Assert.IsAssignableFrom<BlittableJsonReaderObject>(structured.Result);

                var plain = await client.CompleteAsync(ctx, new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = null },
                    new AiUsage(), null, CancellationToken.None);
                Assert.Null(JObject.Parse(client.LastRequestBody)["output_config"]);
                Assert.IsType<string>(plain.Result);
            });
        }

        // ---- helpers ---------------------------------------------------------------------------------------------

        private static Task RunAsync(MockAnthropicClient client, JsonOperationContext ctx, AiChatRequest request, bool streaming)
        {
            if (streaming == false)
                return client.CompleteAsync(ctx, request, new AiUsage(), null, CancellationToken.None);

            return client.StreamingCompleteAsync(ctx, GetPool(client), "Answer", request,
                _ => Task.CompletedTask, new AiUsage(), null, CancellationToken.None);
        }

        private static async Task WithClient(Func<string, HttpResponseMessage> respond, Func<MockAnthropicClient, JsonOperationContext, Task> body)
        {
            using var storageEnv = new StorageEnvironment(StorageEnvironmentOptions.CreateMemoryOnlyForTests());
            using var contextPool = new TransactionContextPool(RavenLogManager.Instance.CreateNullLogger(), storageEnv);
            var settings = new AnthropicSettings("sk-ant-test", "claude-opus-4-8", "https://api.anthropic.com/v1/");
            using var client = new MockAnthropicClient(contextPool, settings, ChatCompletionClient.ConventionsToUse, respond);
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
                await body(client, ctx);
        }

        private static async Task WithClient(AnthropicSettings settings, Func<string, HttpResponseMessage> respond, Func<MockAnthropicClient, JsonOperationContext, Task> body)
        {
            using var storageEnv = new StorageEnvironment(StorageEnvironmentOptions.CreateMemoryOnlyForTests());
            using var contextPool = new TransactionContextPool(RavenLogManager.Instance.CreateNullLogger(), storageEnv);
            using var client = new MockAnthropicClient(contextPool, settings, ChatCompletionClient.ConventionsToUse, respond);
            using (contextPool.AllocateOperationContext(out JsonOperationContext ctx))
                await body(client, ctx);
        }

        private static IMemoryContextPool GetPool(MockAnthropicClient client) => client.Pool;

        private static string EmptySchema() => ChatCompletionClient.EmptySchema;

        private static AiChatRequest Simple(JsonOperationContext ctx) => new() { Messages = [Msg(ctx, "user", "hi")], Schema = ChatCompletionClient.EmptySchema };

        private static BlittableJsonReaderObject Msg(JsonOperationContext ctx, string role, string content, string toolCallId = null)
        {
            var djv = new DynamicJsonValue { ["role"] = role, ["content"] = content };
            if (toolCallId != null)
                djv["tool_call_id"] = toolCallId;
            return ctx.ReadObject(djv, "msg");
        }

        private static DynamicJsonValue ToolCall(string id, string name, string arguments) => new()
        {
            ["id"] = id,
            ["type"] = "function",
            ["function"] = new DynamicJsonValue { ["name"] = name, ["arguments"] = arguments }
        };

        private static HttpResponseMessage Ok(string json) => new(HttpStatusCode.OK) { Content = new StringContent(json, Encoding.UTF8, "application/json") };

        private static HttpResponseMessage Sse(string sse) => new(HttpStatusCode.OK) { Content = new StringContent(sse, Encoding.UTF8, "text/event-stream") };

        private static HttpResponseMessage SseWithHeaders(string sse, params (string Name, string Value)[] headers)
        {
            var response = Sse(sse);
            foreach (var (name, value) in headers)
                response.Headers.TryAddWithoutValidation(name, value);
            return response;
        }

        private static HttpResponseMessage Error(HttpStatusCode code, string type, string message, (string Name, string Value)? header = null)
        {
            var response = new HttpResponseMessage(code)
            {
                Content = new StringContent($"{{\"type\":\"error\",\"error\":{{\"type\":\"{type}\",\"message\":\"{message}\"}}}}", Encoding.UTF8, "application/json")
            };
            if (header != null)
                response.Headers.TryAddWithoutValidation(header.Value.Name, header.Value.Value);
            return response;
        }

        private static string Event(string name, string data) => $"event: {name}\ndata: {data}\n\n";

        private static IEnumerable<string> AllTextBlocks(JArray messages) =>
            messages.SelectMany(m => ((JArray)m["content"]).Where(b => (string)b["type"] == "text").Select(b => (string)b["text"]));

        private static string ErrorEvent(string type, string message) =>
            Event("message_start", """{"type":"message_start","message":{"usage":{"input_tokens":3}}}""") +
            Event("error", $"{{\"type\":\"error\",\"error\":{{\"type\":\"{type}\",\"message\":\"{message}\"}}}}");

        private static async Task<Exception> StreamAndCaptureError(MockAnthropicClient client, JsonOperationContext ctx)
        {
            using var streamed = new MemoryStream();
            return await Assert.ThrowsAnyAsync<Exception>(() => client.StreamingCompleteAsync(ctx, GetPool(client), "Answer",
                new AiChatRequest { Messages = [Msg(ctx, "user", "hi")], Schema = EmptySchema() },
                m => { streamed.Write(m.Span); return Task.CompletedTask; }, new AiUsage(), null, CancellationToken.None));
        }

        private sealed class MockAnthropicClient : ChatCompletionClient
        {
            private readonly Func<string, HttpResponseMessage> _respond;

            public string LastRequestBody;
            public string ApiKeyHeader;
            public string VersionHeader;
            public string AuthorizationHeader;
            public int RequestCount;
            public IMemoryContextPool Pool { get; }

            internal MockAnthropicClient(IMemoryContextPool contextPool, AnthropicSettings settings, DocumentConventions conventions, Func<string, HttpResponseMessage> respond)
                : base(contextPool, new AnthropicChatCompletionClientSettings(settings), conventions)
            {
                _respond = respond;
                Pool = contextPool;
            }

            protected override Task<HttpResponseMessage> SendRequestAsync(HttpRequestMessage request, CancellationToken token) => Capture(request, token);

            protected override Task<HttpResponseMessage> SendStreamingRequestAsync(HttpRequestMessage request, CancellationToken token) => Capture(request, token);

            private async Task<HttpResponseMessage> Capture(HttpRequestMessage request, CancellationToken token)
            {
                RequestCount++;
                LastRequestBody = request.Content != null ? await request.Content.ReadAsStringAsync(token) : null;
                ApiKeyHeader = request.Headers.TryGetValues("x-api-key", out var k) ? k.FirstOrDefault() : null;
                VersionHeader = request.Headers.TryGetValues("anthropic-version", out var v) ? v.FirstOrDefault() : null;
                AuthorizationHeader = request.Headers.Authorization?.ToString();
                return _respond(LastRequestBody);
            }
        }
    }

    public class AiAgentAnthropicLive : RavenTestBase
    {
        public AiAgentAnthropicLive(ITestOutputHelper output) : base(output)
        {
        }

        private class ShopAnswer
        {
            public string Answer = "the answer to the user's question";
            public List<string> ProductNames = ["exact names of the products you mentioned"];
        }

        private record Product(string Name, string Category);

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Claude_Agent_CallsQueryTool_AndAnswersFromRetrievedData(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            await store.Maintenance.SendAsync(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            using (var session = store.OpenAsyncSession())
            {
                await session.StoreAsync(new Product("Zyxwv Gizmo 3000", "Gadgets"));
                await session.StoreAsync(new Product("Ordinary Stapler", "Office"));
                await session.SaveChangesAsync();

                await session.Query<Product>().Where(p => p.Category == "Gadgets").ToListAsync();
            }
            Indexes.WaitForIndexing(store);

            var agent = new AiAgentConfiguration("shop-assistant", config.ConnectionStringName,
                "You are a shop assistant. You do not know the catalog from memory. Whenever the user asks about products, " +
                "you MUST call the FindProducts tool to look them up, then answer using only the returned data, including exact product names.");
            agent.Identifier = "shop-assistant";
            agent.Queries =
            [
                new AiAgentToolQuery
                {
                    Name = "FindProducts",
                    Description = "Find products in a given category.",
                    Query = "from Products where Category = $category",
                    ParametersSampleObject = "{\"category\": \"the product category to look up\"}"
                }
            ];

            var createResult = await store.AI.CreateAgentAsync(agent, new ShopAnswer());
            var chat = store.AI.Conversation(createResult.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("What gadgets do you sell? Give me their exact names.");

            var r = await chat.RunAsync<ShopAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.Done, r.Status);
            Assert.NotNull(r.Answer);

            var haystack = (r.Answer.Answer + " " + string.Join(" ", r.Answer.ProductNames ?? [])).ToLower();
            Assert.Contains("zyxwv", haystack);
        }
    }

    public class GenAiAnthropicLive(ITestOutputHelper output) : RavenTestBase(output)
    {
        private record BlogComment(string Id, string Text, string Author);

        private record BlogPost(string Title, List<BlogComment> Comments);

        [RavenTheory(RavenTestCategory.Etl | RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Claude_ClassifiesSpam_AndUpdateScriptAppliesStructuredOutput(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            config.Prompt = "You are a spam classifier for blog comments. Decide whether the given comment is spam.";
            config.Collection = "BlogPosts";
            config.Identifier = "claude-spam-check";
            config.SampleObject = JsonConvert.SerializeObject(new { Blocked = true, Reason = "Short reason for the decision" });
            config.UpdateScript = @"
    const idx = this.Comments.findIndex(c => c.Id == $input.Id);
    if (idx >= 0 && $output.Blocked)
        this.Comments.splice(idx, 1);
    ";
            config.GenAiTransformation = new GenAiTransformation
            {
                Script = "for (const c of this.Comments) ai.genContext({ Id: c.Id, Text: c.Text });"
            };

            var etl = Etl.WaitForEtlToComplete(store);

            store.Maintenance.Send(new AddGenAiOperation(config));

            const string docId = "posts/1";
            using (var session = store.OpenSession())
            {
                session.Store(new BlogPost("Understanding RavenDB indexing", new List<BlogComment>
                {
                    new("spam", "FREE CRYPTO AIRDROP!!! Claim your $$$ now at scamcoin.fake — limited time, act quick!", "bot"),
                    new("legit", "Great write-up, this finally made map/reduce indexes click for me. Thanks!", "alex"),
                }), docId);
                session.SaveChanges();
            }

            Assert.True(await etl.WaitAsync(TimeSpan.FromSeconds(90)), "GenAI ETL did not finish in time");

            using (var session = store.OpenSession())
            {
                var post = session.Load<BlogPost>(docId);
                Assert.NotNull(post);

                Assert.DoesNotContain(post.Comments, c => c.Id == "spam");
                Assert.Contains(post.Comments, c => c.Id == "legit");
            }
        }
    }

    public class AnthropicClientApiTests : RavenTestBase
    {
        public AnthropicClientApiTests(ITestOutputHelper output) : base(output)
        {
        }

        private class AgentAnswer
        {
            public string Answer = "the answer to the user's question";
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_MultiTurn_ResumesSameConversation(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var agent = new AiAgentConfiguration("assistant", config.ConnectionStringName, "You are a helpful assistant. Keep answers short.") { Identifier = "assistant" };
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("My favourite colour is teal. Remember it.");
            var first = await chat.RunAsync<AgentAnswer>(CancellationToken.None);
            Assert.Equal(AiConversationResult.Done, first.Status);

            chat.SetUserPrompt("What is my favourite colour?");
            var second = await chat.RunAsync<AgentAnswer>(CancellationToken.None);
            Assert.Equal(AiConversationResult.Done, second.Status);
            Assert.Contains("teal", second.Answer.Answer.ToLowerInvariant());
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_AddUserPrompt_MultiPart(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var agent = new AiAgentConfiguration("assistant", config.ConnectionStringName, "You answer briefly.") { Identifier = "assistant" };
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("I will ask two things.");
            chat.AddUserPrompt(new[] { "First, what is 2+2?", "Second, name a primary colour." });
            var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.Done, r.Status);
            Assert.False(string.IsNullOrEmpty(r.Answer.Answer));
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_Attachment_Image(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var agent = new AiAgentConfiguration("vision", config.ConnectionStringName, "You describe images you are given.") { Identifier = "vision" };
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("What fruit is in this image? Answer with one word.");
            await using (var img = GetImg("banana.png"))
            {
                chat.AddAttachment("banana.png", img, "image/png");
                var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);
                Assert.Equal(AiConversationResult.Done, r.Status);
                Assert.Contains("banana", r.Answer.Answer.ToLowerInvariant());
            }
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_CopyAttachmentFrom_Document(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            using (var session = store.OpenSession())
            {
                session.Store(new { }, "docs/1");
                session.SaveChanges();
            }
            await using (var img = GetImg("heart.png"))
                await store.Operations.SendAsync(new Raven.Client.Documents.Operations.Attachments.PutAttachmentOperation("docs/1", "heart.png", img, "image/png"));

            var agent = new AiAgentConfiguration("vision", config.ConnectionStringName, "You describe images you are given.") { Identifier = "vision" };
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("What shape is in this image? Answer with one word.");
            chat.CopyAttachmentFrom("docs/1", "heart.png");
            var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.Done, r.Status);
            Assert.False(string.IsNullOrEmpty(r.Answer.Answer));
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_ActionTool_ManualLoop(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var agent = new AiAgentConfiguration("assistant", config.ConnectionStringName,
                "You help the user. When asked about the weather you MUST call the GetWeather tool.") { Identifier = "assistant" };
            agent.Actions = [new AiAgentToolAction("GetWeather", "Get the current weather for a city") { ParametersSampleObject = "{\"city\":\"the city\"}" }];
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());

            chat.OnUnhandledAction += _ => Task.CompletedTask;

            chat.SetUserPrompt("What is the weather in Paris? Use your tool.");
            var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.ActionRequired, r.Status);

            foreach (var action in chat.RequiredActions())
                chat.AddActionResponse(action.ToolId, "It is 22C and sunny.");

            r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.Done, r.Status);
            Assert.False(string.IsNullOrEmpty(r.Answer.Answer));
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_QueryTool_WithBoundParameter(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            using (var session = store.OpenSession())
            {
                session.Store(new Product("Zyxwv Gizmo 3000", "Gadgets", "companies/1-A"));
                session.SaveChanges();
                session.Query<Product>().Where(p => p.Company == "companies/1-A").ToList();
            }
            Indexes.WaitForIndexing(store);

            var agent = new AiAgentConfiguration("shop", config.ConnectionStringName,
                "You are a shop assistant. Use FindProducts to look up the company's products, then answer with their exact names.") { Identifier = "shop" };
            agent.Parameters.Add(new AiAgentParameter("company", "the current company id"));
            agent.Queries = [new AiAgentToolQuery { Name = "FindProducts", Description = "Find products for the current company", Query = "from Products where Company = $company", ParametersSampleObject = "{}" }];
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions().AddParameter("company", "companies/1-A"));
            chat.SetUserPrompt("What products do we sell? Give exact names.");
            var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.Done, r.Status);
            Assert.Contains("zyxwv", r.Answer.Answer.ToLowerInvariant()); // only knowable via the parameter-bound query
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_ExtendedThinking(Options options, GenAiConfiguration config)
        {
            config.Connection.AnthropicSettings.Reasoning = AiReasoningLevel.High; // enable extended thinking on the connection
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var agent = new AiAgentConfiguration("thinker", config.ConnectionStringName, "You reason carefully, then answer briefly.") { Identifier = "thinker" };
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("If a bat and ball cost 1.10 and the bat costs 1.00 more than the ball, how much is the ball?");
            var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.Done, r.Status);
            Assert.False(string.IsNullOrEmpty(r.Answer.Answer));
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_GetConversationMessages_DoesNotLeakSidecar(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            using (var session = store.OpenSession())
            {
                session.Store(new Product("Zyxwv Gizmo 3000", "Gadgets", "companies/1-A"));
                session.SaveChanges();
                session.Query<Product>().Where(p => p.Company == "companies/1-A").ToList();
            }
            Indexes.WaitForIndexing(store);

            var agent = new AiAgentConfiguration("shop", config.ConnectionStringName,
                "You are a shop assistant. Use FindProducts before answering.") { Identifier = "shop" };
            agent.Parameters.Add(new AiAgentParameter("company"));
            agent.Queries = [new AiAgentToolQuery { Name = "FindProducts", Description = "Find products for the current company", Query = "from Products where Company = $company", ParametersSampleObject = "{}" }];
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions().AddParameter("company", "companies/1-A"));
            chat.SetUserPrompt("What products do we sell?");
            var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);
            Assert.Equal(AiConversationResult.Done, r.Status);

            var messages = await store.AI.GetConversationMessagesAsync(chat.Id);
            Assert.NotNull(messages);
            Assert.DoesNotContain("anthropic-content", System.Text.Json.JsonSerializer.Serialize(messages.Messages));
            Assert.Contains(messages.Messages, m => m.Role == AiMessageRole.Assistant);
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_SubAgent(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var sub = new AiAgentConfiguration("joke-agent", config.ConnectionStringName, "You tell a one-line joke about the given topic.") { Identifier = "joke-agent" };
            var subId = (await store.AI.CreateAgentAsync(sub, new AgentAnswer())).Identifier;

            var parent = new AiAgentConfiguration("host", config.ConnectionStringName, "You are a host. When the user wants a joke, ask the joke sub-agent.") { Identifier = "host" };
            parent.SubAgents = [new AiAgentToolSubAgent { Identifier = subId, Description = "Ask it to tell a joke about a topic." }];
            var created = await store.AI.CreateAgentAsync(parent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("Tell me a joke about databases.");
            var r = await chat.RunAsync<AgentAnswer>(CancellationToken.None);

            Assert.Equal(AiConversationResult.Done, r.Status);
            Assert.False(string.IsNullOrEmpty(r.Answer.Answer));
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.Anthropic, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task Agent_Summarization(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);
            store.Maintenance.Send(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var agent = new AiAgentConfiguration("assistant", config.ConnectionStringName, "You answer briefly.") { Identifier = "assistant" };
            agent.ChatTrimming = new AiAgentChatTrimmingConfiguration { Tokens = new AiAgentSummarizationByTokens { MaxTokensBeforeSummarization = 0 } }; // summarize every turn
            var created = await store.AI.CreateAgentAsync(agent, new AgentAnswer());

            var chat = store.AI.Conversation(created.Identifier, "chats/", new AiConversationCreationOptions());
            chat.SetUserPrompt("Name a fruit.");
            var first = await chat.RunAsync<AgentAnswer>(CancellationToken.None);
            Assert.Equal(AiConversationResult.Done, first.Status);

            chat.SetUserPrompt("Name a vegetable."); // forces a summarization completion call through Claude
            var second = await chat.RunAsync<AgentAnswer>(CancellationToken.None);
            Assert.Equal(AiConversationResult.Done, second.Status);
        }

        private record Product(string Name, string Category, string Company);

        private static Stream GetImg(string name)
        {
            var resourceName = "SlowTests.Data.RavenDB_24648." + name;
            var stream = typeof(AnthropicClientApiTests).Assembly.GetManifestResourceStream(resourceName);
            if (stream == null)
                throw new FileNotFoundException($"Embedded resource not found: {resourceName}");
            return stream;
        }
    }
}
