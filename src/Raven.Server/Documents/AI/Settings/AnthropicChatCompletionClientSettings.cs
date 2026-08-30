using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Exceptions;
using Raven.Server.Documents.ETL.Providers.AI;
using Raven.Server.Documents.Handlers.AI.Agents;
using Sparrow;
using Sparrow.Json;
using Sparrow.Json.Parsing;
using Sparrow.Server.Json.Sync;

namespace Raven.Server.Documents.AI.Settings;

internal sealed partial class AnthropicChatCompletionClientSettings : AbstractChatCompletionClientSettings
{
    public const string RawContentSidecarProperty = "@anthropic-content";

    private const string AnthropicVersion = "2023-06-01";

    private readonly AnthropicSettings _settings;

    public AnthropicChatCompletionClientSettings(AnthropicSettings settings)
        : base(settings)
    {
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
    }

    private int MaxOutputTokens => _settings.MaxOutputTokens ?? AnthropicSettings.DefaultMaxOutputTokens;
    private const int MinThinkingBudgetTokens = 1024;
    private enum AnthropicReasoningPolicy
    {
        AdaptiveWithEffort,               // thinking:{type:"adaptive"} + output_config.effort
        EnabledWithBudgetTokens,          // thinking:{type:"enabled",budget_tokens:N}
        EnabledWithBudgetTokensAndEffort  // both of the above
    }

    private static readonly (string Prefix, AnthropicReasoningPolicy Policy)[] ReasoningPolicyByModel =
    [
        ("claude-fable-", AnthropicReasoningPolicy.AdaptiveWithEffort),
        ("claude-mythos-", AnthropicReasoningPolicy.AdaptiveWithEffort),
        ("claude-opus-5", AnthropicReasoningPolicy.AdaptiveWithEffort),
        ("claude-opus-4-8", AnthropicReasoningPolicy.AdaptiveWithEffort),
        ("claude-opus-4-7", AnthropicReasoningPolicy.AdaptiveWithEffort),
        ("claude-opus-4-6", AnthropicReasoningPolicy.AdaptiveWithEffort),
        ("claude-sonnet-5", AnthropicReasoningPolicy.AdaptiveWithEffort),
        ("claude-sonnet-4-6", AnthropicReasoningPolicy.AdaptiveWithEffort),

        ("claude-opus-4-5", AnthropicReasoningPolicy.EnabledWithBudgetTokensAndEffort),

        ("claude-opus-4-1", AnthropicReasoningPolicy.EnabledWithBudgetTokens),
        ("claude-opus-4-0", AnthropicReasoningPolicy.EnabledWithBudgetTokens),
        ("claude-sonnet-4-5", AnthropicReasoningPolicy.EnabledWithBudgetTokens),
        ("claude-sonnet-4-0", AnthropicReasoningPolicy.EnabledWithBudgetTokens),
        ("claude-haiku-4-5", AnthropicReasoningPolicy.EnabledWithBudgetTokens),

        ("claude-opus-4-20250514", AnthropicReasoningPolicy.EnabledWithBudgetTokens),
        ("claude-sonnet-4-20250514", AnthropicReasoningPolicy.EnabledWithBudgetTokens),
        ("claude-3-7-sonnet-20250219", AnthropicReasoningPolicy.EnabledWithBudgetTokens),
        ("claude-3-7-sonnet-latest", AnthropicReasoningPolicy.EnabledWithBudgetTokens)
    ];

    public override bool EnablePromptCaching => false; // cache_control is not emitted yet (deferred)

    public override string GetRelativeCompletionUri() => "messages";

    public override void HandleCompletionRequestPayload(AsyncBlittableJsonTextWriter writer)
    {
        // Not used: WritePayload is overridden and emits the whole Anthropic body.
    }

    // ---- authentication --------------------------------------------------------------------------------------

    public override void AddAuthentication(HttpRequestMessage request)
    {
        request.Headers.TryAddWithoutValidation(Wire.HeaderApiKey, ApiKey);
        request.Headers.TryAddWithoutValidation(Wire.HeaderAnthropicVersion, AnthropicVersion);
    }

    // ---- request building ------------------------------------------------------------------------------------

    public override DynamicJsonValue BuildTool(JsonOperationContext ctx, string name, string description, string parametersSchema)
    {
        AssertModelSupportsStrictTools(name);

        var inputSchema = ParseJsonObject(ctx, parametersSchema);

        return new DynamicJsonValue
        {
            [Wire.Name] = name,
            [Wire.Description] = description,
            [Wire.InputSchema] = BuildStrictToolRootSchema(name, inputSchema),
            [Wire.Strict] = true
        };
    }

    private static readonly string[] ModelFamiliesWithStructuredOutputSupport =
    [
        "claude-fable-",
        "claude-mythos-",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-haiku-4-5"
    ];

    private static bool SupportsStructuredOutput(string model) =>
        string.IsNullOrEmpty(model) == false &&
        ModelFamiliesWithStructuredOutputSupport.Any(prefix => model.StartsWith(prefix, StringComparison.OrdinalIgnoreCase));

    private void AssertModelSupportsStrictTools(string toolName)
    {
        if (SupportsStructuredOutput(Model))
            return;

        throw new InvalidOperationException(
            $"Model '{Model}' cannot be used with tools, so tool '{toolName}' cannot be sent. RavenDB requires every " +
            "Anthropic tool to be strict (a tool must accept only the fields its input schema declares), and strict tool " +
            "use is generally available on Claude 4.5 and later models. Either use a Claude 4.5-or-later model, or remove " +
            "the tools from this agent configuration - conversations without tools are unaffected.");
    }

    private void AssertModelSupportsStructuredOutput()
    {
        if (SupportsStructuredOutput(Model))
            return;

        throw new InvalidOperationException(
            $"Model '{Model}' cannot be used with structured output (an answer JSON schema), and this request carries " +
            "one. Structured outputs are generally available on Claude 4.5 and later models. Either use a Claude " +
            "4.5-or-later model, or configure the conversation without an output schema - plain-text conversations " +
            "are unaffected.");
    }

    public override void ValidateRequest(JsonOperationContext ctx, AiChatRequest request)
    {
        // Body/output_config discarded: only the throwing behaviour matters here, the writer rebuilds both.
        DynamicJsonValue discardedOutputConfig = null;
        AppendReasoning(new DynamicJsonValue(), ref discardedOutputConfig);

        if (request.Schema != null)
            AssertModelSupportsStructuredOutput();

        var turns = NormalizeTurns(ctx, request.Messages, request.Attachments);
        AssertFirstTurnIsUser(turns);
        request.ProviderPrepared = turns;
    }

    private static void AssertFirstTurnIsUser(NormalizedTurns turns)
    {
        if (turns.Messages.Items[0] is not DynamicJsonValue first)
            return;

        var role = first.Properties.FirstOrDefault(p => p.Name == Wire.Role).Value as string;
        if (role != Wire.RoleAssistant)
            return;

        throw new InvalidOperationException(
            "Anthropic's Messages API requires the first conversation turn to use the user role, but this conversation " +
            "starts with an assistant turn. The most common cause is an agent with AddToInitialContext queries and no " +
            "model-visible user input before their tool calls. Either add a model-visible (SendToModel) parameter to " +
            "the agent - its parameters message becomes the leading user turn - or disable AddToInitialContext for " +
            "this agent when using Anthropic.");
    }

    public override void WritePayload(AsyncBlittableJsonTextWriter writer, JsonOperationContext ctx, ChatCompletionPayload payload)
    {
        var body = new DynamicJsonValue
        {
            [Wire.Model] = Model,
            [Wire.MaxTokens] = MaxOutputTokens
        };

        if (payload.Streaming)
            body[Wire.Stream] = true;

        DynamicJsonValue outputConfig = null;

        AppendReasoning(body, ref outputConfig);

        var turns = payload.ProviderPrepared as NormalizedTurns ?? NormalizeTurns(ctx, payload.Messages, payload.Attachments);

        if (turns.System != null)
            body[Wire.System] = turns.System;

        body[Wire.Messages] = turns.Messages;

        if (payload.Tools?.Count > 0)
        {
            var tools = new DynamicJsonArray();
            foreach (var tool in payload.Tools)
                tools.Add(tool);
            body[Wire.Tools] = tools;

            if (payload.UseTools == false)
                body[Wire.ToolChoice] = new DynamicJsonValue { [Wire.Type] = Wire.ToolChoiceNone };
        }

        var format = BuildOutputFormat(ctx, payload.Schema);
        if (format != null)
            (outputConfig ??= new DynamicJsonValue())[Wire.Format] = format;

        if (outputConfig != null)
            body[Wire.OutputConfig] = outputConfig;

        ctx.Write(writer, body);
    }

    private sealed class NormalizedTurns
    {
        public DynamicJsonArray Messages;
        public string System;
    }

    private static NormalizedTurns NormalizeTurns(JsonOperationContext ctx, IEnumerable<BlittableJsonReaderObject> payloadMessages, List<AiAttachment> attachments)
    {
        var systemText = new StringBuilder();
        var messages = new DynamicJsonArray();
        DynamicJsonArray pendingToolResults = null;

        void FlushToolResults()
        {
            if (pendingToolResults == null)
                return;

            messages.Add(new DynamicJsonValue { [Wire.Role] = Wire.RoleUser, [Wire.Content] = pendingToolResults });
            pendingToolResults = null;
        }

        foreach (var message in payloadMessages ?? Enumerable.Empty<BlittableJsonReaderObject>())
        {
            if (message.TryGet(Wire.Role, out string role) == false)
                continue;

            if (role == Wire.RoleInternal)
                continue;

            if (role == Wire.RoleSystem)
            {
                if (message.TryGet(Wire.Content, out string sysContent) && string.IsNullOrEmpty(sysContent) == false)
                {
                    if (systemText.Length > 0)
                        systemText.Append("\n\n");
                    systemText.Append(sysContent);
                }
                continue;
            }

            if (role == Wire.RoleTool)
            {
                pendingToolResults ??= new DynamicJsonArray();
                message.TryGet(Wire.ToolCallId, out string toolUseId);
                message.TryGet(Wire.Content, out object toolContent);
                pendingToolResults.Add(new DynamicJsonValue
                {
                    [Wire.Type] = Wire.TypeToolResult,
                    [Wire.ToolUseId] = toolUseId,
                    [Wire.Content] = toolContent?.ToString() ?? string.Empty
                });
                continue;
            }

            FlushToolResults();

            if (role == Wire.RoleAssistant)
            {
                if (messages.Count == 0 &&
                    message.TryGet(ConversationDocument.SummaryProperty, out bool isSummary) && isSummary &&
                    message.TryGet(Wire.Content, out string summaryText) && string.IsNullOrEmpty(summaryText) == false)
                {
                    if (systemText.Length > 0)
                        systemText.Append("\n\n");
                    systemText.Append(summaryText);
                    continue;
                }

                if (TryBuildAssistantTurn(ctx, message, out var assistantTurn))
                    messages.Add(assistantTurn);
                continue;
            }

            message.TryGet(Wire.Content, out object userContent);
            var userBlocks = new DynamicJsonArray();
            AppendContentText(userBlocks, userContent);

            if (userBlocks.Count == 0)
                continue;

            messages.Add(new DynamicJsonValue { [Wire.Role] = Wire.RoleUser, [Wire.Content] = userBlocks });
        }

        FlushToolResults();

        AppendAttachments(messages, attachments);

        if (messages.Count == 0)
            throw new InvalidOperationException(
                "Cannot build an Anthropic request: every message was empty after normalization, leaving no content to send. " +
                "A turn whose content is an empty string (or whose parts are all empty) produces no content block, and Anthropic " +
                "rejects an empty text block, so such turns are dropped rather than padded.");

        return new NormalizedTurns
        {
            Messages = messages,
            System = systemText.Length > 0 ? systemText.ToString() : null
        };
    }

    private void AppendReasoning(DynamicJsonValue body, ref DynamicJsonValue outputConfig)
    {
        // Default returns before the model is resolved: emitting nothing is valid on every model, known or not.
        if (_settings.Reasoning == AiReasoningLevel.Default)
            return;

        var policy = ResolveReasoningPolicy();

        // Adaptive thinking, and the explicit budget, are mutually exclusive - but effort is orthogonal to both.
        body[Wire.Thinking] = policy == AnthropicReasoningPolicy.AdaptiveWithEffort
            ? new DynamicJsonValue { [Wire.Type] = Wire.ThinkingAdaptive }
            : new DynamicJsonValue { [Wire.Type] = Wire.ThinkingEnabled, [Wire.BudgetTokens] = ResolveThinkingBudgetTokens() };

        if (policy != AnthropicReasoningPolicy.EnabledWithBudgetTokens)
            (outputConfig ??= new DynamicJsonValue())[Wire.Effort] = EffortLevel(_settings.Reasoning);
    }

    private AnthropicReasoningPolicy ResolveReasoningPolicy()
    {
        // Longest prefix wins, so a general entry can never shadow a more specific one.
        var bestLength = -1;
        var best = default(AnthropicReasoningPolicy);

        if (Model != null)
        {
            foreach (var (prefix, policy) in ReasoningPolicyByModel)
            {
                if (prefix.Length > bestLength && Model.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
                {
                    bestLength = prefix.Length;
                    best = policy;
                }
            }
        }

        if (bestLength >= 0)
            return best;

        throw new InvalidOperationException(
            $"Cannot apply a reasoning level to model '{Model}': its reasoning capabilities are not known. The wire forms " +
            $"(adaptive thinking, and an explicit token budget) are rejected with a 400 by the models that do not support " +
            $"them, so neither is assumed. Set `{nameof(AnthropicSettings.Reasoning)}` to " +
            $"`{nameof(AiReasoningLevel.Default)}`, or use a model starting with one of: " +
            $"{string.Join(", ", ReasoningPolicyByModel.Select(m => m.Prefix))}.");
    }

    private static string EffortLevel(AiReasoningLevel reasoning) => reasoning switch
    {
        AiReasoningLevel.Low => Wire.EffortLow,
        AiReasoningLevel.Medium => Wire.EffortMedium,
        AiReasoningLevel.High => Wire.EffortHigh,
        _ => throw new ArgumentOutOfRangeException(nameof(reasoning), reasoning, "Not an explicit reasoning level.")
    };

    private int ResolveThinkingBudgetTokens()
    {
        var maxTokens = MaxOutputTokens;
        var budget = _settings.Reasoning switch
        {
            AiReasoningLevel.Low => maxTokens / 4,
            AiReasoningLevel.Medium => maxTokens / 2,
            AiReasoningLevel.High => maxTokens * 3 / 4,
            _ => throw new ArgumentOutOfRangeException(nameof(_settings.Reasoning), _settings.Reasoning, "Not an explicit reasoning level.")
        };

        if (budget < MinThinkingBudgetTokens)
            budget = MinThinkingBudgetTokens;

        if (budget >= maxTokens)
            throw new InvalidOperationException(
                $"Model '{Model}' takes an explicit thinking budget, which must be at least {MinThinkingBudgetTokens} " +
                $"and less than `{nameof(AnthropicSettings.MaxOutputTokens)}` ({maxTokens}). Raise " +
                $"`{nameof(AnthropicSettings.MaxOutputTokens)}` above {MinThinkingBudgetTokens}, or set " +
                $"`{nameof(AnthropicSettings.Reasoning)}` to `{nameof(AiReasoningLevel.Default)}`.");

        return budget;
    }

    private static bool TryBuildAssistantTurn(JsonOperationContext ctx, BlittableJsonReaderObject message, out DynamicJsonValue turn)
    {
        turn = null;

        if (message.TryGet(RawContentSidecarProperty, out BlittableJsonReaderArray rawContent) && rawContent is { Length: > 0 })
        {
            var sidecar = NormalizeSidecar(rawContent);
            if (sidecar.Count > 0)
            {
                turn = new DynamicJsonValue { [Wire.Role] = Wire.RoleAssistant, [Wire.Content] = sidecar };
                return true;
            }
        }

        var content = new DynamicJsonArray();

        message.TryGet(Wire.Content, out object textContent);
        AppendContentText(content, textContent);

        if (message.TryGet(Wire.ToolCalls, out BlittableJsonReaderArray toolCalls) && toolCalls != null)
        {
            foreach (BlittableJsonReaderObject call in toolCalls)
            {
                if (call.TryGet(Wire.Id, out string id) == false ||
                    call.TryGet(Wire.Function, out BlittableJsonReaderObject function) == false ||
                    function.TryGet(Wire.Name, out string name) == false)
                    continue;

                function.TryGet(Wire.Arguments, out string arguments);
                content.Add(new DynamicJsonValue
                {
                    [Wire.Type] = Wire.TypeToolUse,
                    [Wire.Id] = id,
                    [Wire.Name] = name,
                    [Wire.Input] = ParseJsonObject(ctx, arguments)
                });
            }
        }

        if (content.Count == 0)
            return false;

        turn = new DynamicJsonValue { [Wire.Role] = Wire.RoleAssistant, [Wire.Content] = content };
        return true;
    }

    private static void AppendAttachments(DynamicJsonArray messages, List<AiAttachment> attachments)
    {
        if (attachments == null || attachments.Count == 0)
            return;

        var content = new DynamicJsonArray();
        foreach (var attachment in attachments)
        {
            if (attachment.Source == AiAttachmentSource.NotFound)
            {
                content.Add(TextBlock($"File '{attachment.Name}' (of type '{attachment.Type}') could not be loaded: attachment not found"));
                continue;
            }

            content.Add(GetAttachmentBlock(attachment));
        }

        messages.Add(new DynamicJsonValue { [Wire.Role] = Wire.RoleUser, [Wire.Content] = content });
    }

    private static DynamicJsonValue GetAttachmentBlock(AiAttachment attachment)
    {
        switch (attachment.Type)
        {
            case Wire.MediaTypeTextPlain:
                return TextBlock(attachment.Data);
            case Wire.MediaTypeApplicationPdf:
                return new DynamicJsonValue
                {
                    [Wire.Type] = Wire.TypeDocument,
                    [Wire.Source] = Base64Source(Wire.MediaTypeApplicationPdf, attachment.Data)
                };
            case Wire.MediaTypeImageJpeg:
            case Wire.MediaTypeImagePng:
            case Wire.MediaTypeImageGif:
            case Wire.MediaTypeImageWebp:
                return new DynamicJsonValue
                {
                    [Wire.Type] = Wire.TypeImage,
                    [Wire.Source] = Base64Source(attachment.Type, attachment.Data)
                };
            default:
                throw new InvalidOperationException($"Attachment '{attachment.Name}' has unknown type: {attachment.Type}");
        }
    }

    private static DynamicJsonValue BuildOutputFormat(JsonOperationContext ctx, string schema)
    {
        if (string.IsNullOrWhiteSpace(schema))
            return null;

        var wrapper = ParseJsonObject(ctx, schema);
        object innerSchema = wrapper != null && wrapper.TryGetMember(Wire.Schema, out var s) ? s : wrapper;

        return new DynamicJsonValue
        {
            [Wire.Type] = Wire.TypeJsonSchema,
            [Wire.Schema] = innerSchema
        };
    }

    // ---- response parsing (non-streaming) --------------------------------------------------------------------

    public override AiResponse ParseResponse(JsonOperationContext ctx, HttpResponseMessage response, BlittableJsonReaderObject content, AiUsage usage, bool structuredOutput)
    {
        UpdateUsage(content, usage);

        if (content.TryGet(Wire.StopReason, out string stopReason) && stopReason is Wire.StopReasonMaxTokens or Wire.StopReasonModelContextWindowExceeded)
            throw new TooManyTokensException($"The model stopped because it ran out of room (stop_reason='{stopReason}'). Response: {content}") { RequestId = GetRequestId(response.Headers) };

        if (content.TryGet(Wire.Content, out BlittableJsonReaderArray contentArray) == false || contentArray == null)
            throw UnexpectedResponseException.Create("No content in response", response, content, GetRequestId(response.Headers));

        var text = new StringBuilder();
        List<AiToolCall> toolCalls = null;
        var hasToolUse = false;

        foreach (BlittableJsonReaderObject block in contentArray)
        {
            if (block.TryGet(Wire.Type, out string type) == false)
                continue;

            switch (type)
            {
                case Wire.TypeText:
                    if (block.TryGet(Wire.Text, out string blockText))
                        text.Append(blockText);
                    break;

                case Wire.TypeToolUse:
                    hasToolUse = true;
                    block.TryGet(Wire.Id, out string id);
                    block.TryGet(Wire.Name, out string name);
                    block.TryGet(Wire.Input, out BlittableJsonReaderObject input);
                    (toolCalls ??= new List<AiToolCall>()).Add(new AiToolCall(id, name, input?.ToString() ?? "{}"));
                    break;

                // thinking / redacted_thinking are intentionally NOT surfaced into answer content.
            }
        }

        if (stopReason == Wire.StopReasonRefusal)
            RefusedToAnswerException.Throw(text.ToString(), content.ToString(), stopReason, GetRequestId(response.Headers));

        if (hasToolUse)
        {
            var message = new DynamicJsonValue
            {
                [Wire.Role] = Wire.RoleAssistant,
                [Wire.Content] = null,
                [Wire.ToolCalls] = ToWireToolCalls(toolCalls),
                // Preserved raw content, echoed back unmodified on the next request (see NormalizeSidecar).
                [RawContentSidecarProperty] = NormalizeSidecar(contentArray)
            };

            return new AiResponse(AiResponseType.Tool)
            {
                ToolCalls = toolCalls,
                Message = ctx.ReadObject(message, "anthropic/tool-message")
            };
        }

        if (text.Length == 0)
            throw UnexpectedResponseException.Create("No text content in response", response, content, GetRequestId(response.Headers));

        // Unstructured (no schema - RavenDB-24824): the text IS the answer, returned as a string without parsing.
        object result = structuredOutput
            ? ctx.Sync.ReadForMemory(text.ToString(), "ai/output")
            : text.ToString();

        return new AiResponse(AiResponseType.Result)
        {
            Result = result,
            Message = ctx.ReadObject(new DynamicJsonValue
            {
                [Wire.Role] = Wire.RoleAssistant,
                [Wire.Content] = result
            }, "anthropic/result-message")
        };
    }

    // ---- response parsing (streaming) ------------------------------------------------------------------------

    public override StreamEventResult ProcessStreamEvent(JsonOperationContext ctx, BlittableJsonReaderObject sseEvent, ChatStreamState state, AiUsage usage)
    {
        var blocks = (Dictionary<int, StreamingBlock>)(state.ProviderState ??= new Dictionary<int, StreamingBlock>());

        if (sseEvent.TryGet(Wire.Type, out string eventType) == false)
            return default;

        switch (eventType)
        {
            case Wire.EventMessageStart:
                if (sseEvent.TryGet(Wire.Message, out BlittableJsonReaderObject startMessage) && startMessage != null &&
                    startMessage.TryGet(Wire.Usage, out BlittableJsonReaderObject startUsage) && startUsage != null)
                {
                    startUsage.TryGet(Wire.InputTokens, out long inputTokens);
                    startUsage.TryGet(Wire.CacheReadInputTokens, out long cacheRead);
                    startUsage.TryGet(Wire.CacheCreationInputTokens, out long cacheCreation);
                    usage.Add(inputTokens + cacheRead + cacheCreation, 0, cacheRead);
                }
                return default;

            case Wire.EventContentBlockStart:
                if (sseEvent.TryGet(Wire.Index, out int startIndex) &&
                    sseEvent.TryGet(Wire.ContentBlock, out BlittableJsonReaderObject cb) && cb != null)
                {
                    cb.TryGet(Wire.Type, out string blockType);
                    var block = new StreamingBlock { Type = blockType };
                    cb.TryGet(Wire.Id, out block.Id);
                    cb.TryGet(Wire.Name, out block.Name);
                    cb.TryGet(Wire.Data, out block.RedactedData);
                    if (cb.TryGet(Wire.Text, out string seedText) && string.IsNullOrEmpty(seedText) == false)
                        block.Text.Append(seedText);
                    blocks[startIndex] = block;
                }
                return default;

            case Wire.EventContentBlockDelta:
                if (sseEvent.TryGet(Wire.Index, out int deltaIndex) &&
                    sseEvent.TryGet(Wire.Delta, out BlittableJsonReaderObject delta) && delta != null)
                {
                    delta.TryGet(Wire.Type, out string deltaType);
                    blocks.TryGetValue(deltaIndex, out var block);

                    switch (deltaType)
                    {
                        case Wire.DeltaText:
                            delta.TryGet(Wire.Text, out LazyStringValue textChunk);
                            if (textChunk == null || textChunk.Length == 0 || block == null)
                                return default;

                            block.Text.Append(textChunk.ToString());

                            if (state.StructuredOutput == false)
                                return new StreamEventResult(textChunk, stop: false);

                            if (block.IsAnswerJson == null)
                            {
                                var first = FirstNonWhitespace(block.Text);
                                if (first != '\0')
                                    block.IsAnswerJson = first == '{';
                            }

                            return block.IsAnswerJson == true ? new StreamEventResult(textChunk, stop: false) : default;

                        case Wire.DeltaInputJson:
                            delta.TryGet(Wire.PartialJson, out string partial);
                            block?.Json.Append(partial);
                            return default;

                        case Wire.DeltaThinking:
                            delta.TryGet(Wire.Thinking, out string thinkingChunk);
                            block?.Thinking.Append(thinkingChunk);
                            return default;

                        case Wire.DeltaSignature:
                            delta.TryGet(Wire.Signature, out string signature);
                            if (block != null)
                                block.Signature = signature;
                            return default;
                    }
                }
                return default;

            case Wire.EventMessageDelta:
                if (sseEvent.TryGet(Wire.Delta, out BlittableJsonReaderObject messageDelta) && messageDelta != null)
                {
                    messageDelta.TryGet(Wire.StopReason, out string stopReason);
                    state.StopReason = stopReason;
                }
                if (sseEvent.TryGet(Wire.Usage, out BlittableJsonReaderObject deltaUsage) && deltaUsage != null)
                {
                    deltaUsage.TryGet(Wire.OutputTokens, out long outputTokens);
                    usage.Add(0, outputTokens, 0);
                }
                return default;

            case Wire.EventMessageStop:
                return new StreamEventResult(null, stop: true);

            case Wire.EventError:
                throw BuildMidStreamError(sseEvent, state);

            default:
                // "ping" and any unknown event types are intentionally ignored (forward-compatible).
                return default;
        }
    }

    public override AiResponse BuildStreamedResponse(JsonOperationContext streamingCtx, ChatStreamState state, HttpResponseMessage response)
    {
        if (state.SawStop == false)
            throw UnexpectedResponseException.Create("The stream ended before the provider signaled completion (message_stop); the response is incomplete",
                response, string.Empty, GetRequestId(response.Headers));

        if (state.StopReason is Wire.StopReasonMaxTokens or Wire.StopReasonModelContextWindowExceeded)
            throw new TooManyTokensException($"The model stopped because it ran out of room (stop_reason='{state.StopReason}').") { RequestId = GetRequestId(response.Headers) };

        var blocks = state.ProviderState as Dictionary<int, StreamingBlock> ?? new Dictionary<int, StreamingBlock>();
        var ordered = blocks.OrderBy(kv => kv.Key).Select(kv => kv.Value).ToList();
        var hasToolUse = ordered.Any(b => b.Type == Wire.TypeToolUse);

        if (state.StopReason == Wire.StopReasonRefusal)
        {
            var refusalText = string.Concat(ordered.Where(b => b.Type == Wire.TypeText).Select(b => b.Text.ToString()));
            RefusedToAnswerException.Throw(refusalText, response.ToString(), state.StopReason, GetRequestId(response.Headers));
        }

        if (hasToolUse == false)
        {
            if (state.StructuredOutput == false)
            {
                var plainText = string.Concat(ordered.Where(b => b.Type == Wire.TypeText).Select(b => b.Text.ToString()));
                return new AiResponse(AiResponseType.Result)
                {
                    Result = plainText,
                    Message = streamingCtx.ReadObject(new DynamicJsonValue
                    {
                        [Wire.Role] = Wire.RoleAssistant,
                        [Wire.Content] = plainText
                    }, "anthropic/streamed/text")
                };
            }

            var resultMessage = state.FinalResult;
            if (resultMessage == null)
            {
                var text = string.Concat(ordered.Where(b => b.Type == Wire.TypeText).Select(b => b.Text.ToString()));
                if (string.IsNullOrEmpty(text))
                    throw UnexpectedResponseException.Create("No content in streamed response", response, string.Empty, GetRequestId(response.Headers));

                resultMessage = streamingCtx.Sync.ReadForMemory(text, "ai/output");
            }

            return new AiResponse(AiResponseType.Result)
            {
                Result = resultMessage,
                Message = streamingCtx.ReadObject(new DynamicJsonValue
                {
                    [Wire.Role] = Wire.RoleAssistant,
                    [Wire.Content] = resultMessage
                }, "anthropic/streamed/result")
            };
        }

        // Reconstruct the raw content[] verbatim (incl. thinking + signatures) for the echo-back sidecar.
        var toolCalls = new List<AiToolCall>();
        var rawContent = new DynamicJsonArray();
        foreach (var block in ordered)
        {
            switch (block.Type)
            {
                case Wire.TypeText:
                    var blockText = block.Text.ToString();
                    if (KeepInSidecar(Wire.TypeText, blockText))
                        rawContent.Add(TextBlock(blockText));
                    break;
                case Wire.TypeThinking:
                    rawContent.Add(new DynamicJsonValue { [Wire.Type] = Wire.TypeThinking, [Wire.Thinking] = block.Thinking.ToString(), [Wire.Signature] = block.Signature });
                    break;
                case Wire.TypeRedactedThinking:
                    rawContent.Add(new DynamicJsonValue { [Wire.Type] = Wire.TypeRedactedThinking, [Wire.Data] = block.RedactedData });
                    break;
                case Wire.TypeToolUse:
                    var arguments = block.Json.Length > 0 ? block.Json.ToString() : "{}";
                    rawContent.Add(new DynamicJsonValue
                    {
                        [Wire.Type] = Wire.TypeToolUse,
                        [Wire.Id] = block.Id,
                        [Wire.Name] = block.Name,
                        [Wire.Input] = ParseJsonObject(streamingCtx, arguments)
                    });
                    toolCalls.Add(new AiToolCall(block.Id, block.Name, arguments));
                    break;
            }
        }

        return new AiResponse(AiResponseType.Tool)
        {
            ToolCalls = toolCalls,
            Message = streamingCtx.ReadObject(new DynamicJsonValue
            {
                [Wire.Role] = Wire.RoleAssistant,
                [Wire.Content] = null,
                [Wire.ToolCalls] = ToWireToolCalls(toolCalls),
                [RawContentSidecarProperty] = rawContent
            }, "anthropic/streamed/tool")
        };
    }

    private sealed class StreamingBlock
    {
        public string Type;
        public string Id;
        public string Name;
        public string RedactedData;
        public string Signature;

        // Whether this text block carries the structured answer. Null until the first non-whitespace character.
        public bool? IsAnswerJson;

        public readonly StringBuilder Text = new();
        public readonly StringBuilder Json = new();
        public readonly StringBuilder Thinking = new();
    }

    private static char FirstNonWhitespace(StringBuilder text)
    {
        for (var i = 0; i < text.Length; i++)
        {
            if (char.IsWhiteSpace(text[i]) == false)
                return text[i];
        }

        return '\0';
    }

    private static DynamicJsonArray ToWireToolCalls(List<AiToolCall> toolCalls)
    {
        var array = new DynamicJsonArray();
        foreach (var call in toolCalls)
        {
            array.Add(new DynamicJsonValue
            {
                [Wire.Id] = call.Id,
                [Wire.Type] = Wire.TypeFunction,
                [Wire.Function] = new DynamicJsonValue { [Wire.Name] = call.Name, [Wire.Arguments] = call.Arguments }
            });
        }

        return array;
    }

    private static void UpdateUsage(BlittableJsonReaderObject content, AiUsage usage)
    {
        if (content.TryGet(Wire.Usage, out BlittableJsonReaderObject usageJson) == false || usageJson == null)
            return;

        usageJson.TryGet(Wire.InputTokens, out long inputTokens);
        usageJson.TryGet(Wire.OutputTokens, out long outputTokens);
        usageJson.TryGet(Wire.CacheReadInputTokens, out long cacheRead);
        usageJson.TryGet(Wire.CacheCreationInputTokens, out long cacheCreation);

        // Prompt tokens = fresh input + cache-read + cache-creation; cached = cache-read.
        usage.Add(inputTokens + cacheRead + cacheCreation, outputTokens, cacheRead);
    }

    // ---- errors ----------------------------------------------------------------------------------------------

    public override AiError ParseError(BlittableJsonReaderObject content, HttpResponseMessage response)
    {
        var (type, message) = ExtractError(content);

        var errorType = (int)response.StatusCode == 429 ? ErrorType.TooManyRequests
            : IsInputOverflow(response, type, message) ? ErrorType.TooManyTokens
            : ErrorType.Unknown;

        return new AiError
        {
            ErrorType = errorType,
            Message = message
        };
    }

    private static bool IsInputOverflow(HttpResponseMessage response, string errorType, string message) =>
        (int)response.StatusCode == 400 &&
        errorType == Wire.ErrorInvalidRequest &&
        message?.StartsWith("prompt is too long", StringComparison.OrdinalIgnoreCase) == true;

    public override string GetRefusal(BlittableJsonReaderObject choice0, BlittableJsonReaderObject message)
    {
        return null;
    }

    public override TimeSpan? GetRetryAfter(HttpResponseMessage response, AiError error)
    {
        var retryAfter = TryGetRetryAfterHeader(response.Headers, out var fromHeader) ? fromHeader : error.RetryAfter;

        if (retryAfter == null && (int)response.StatusCode == 429)
            return TimeSpan.Zero;

        return retryAfter;
    }

    public override bool CanCarryRetryDelayOnNonRateLimit(HttpResponseMessage response) => (int)response.StatusCode == 529;

    public override string GetRequestId(HttpResponseHeaders headers)
    {
        if (headers.TryGetValues(Wire.HeaderRequestId, out var values))
            return values.FirstOrDefault() ?? string.Empty;

        return string.Empty;
    }

    private static (string Type, string Message) ExtractError(BlittableJsonReaderObject content)
    {
        if (content != null && content.TryGet(Wire.Error, out BlittableJsonReaderObject error) && error != null)
        {
            error.TryGet(Wire.Type, out string type);
            error.TryGet(Wire.Message, out string message);
            return (type, message);
        }

        return (null, null);
    }

    private Exception BuildMidStreamError(BlittableJsonReaderObject sseEvent, ChatStreamState state)
    {
        var (errorType, errorMessage) = ExtractError(sseEvent);
        var requestId = GetRequestId(state.Response.Headers);
        var message = $"The model returned an error mid-stream (type: '{errorType ?? "unknown"}'): {errorMessage ?? sseEvent.ToString()}";

        if (errorType == Wire.ErrorRateLimit)
        {
            // Same contract as the non-streaming 429 path: always retryable, floored to zero when no delay is given.
            return new RateLimitException(message)
            {
                RequestId = requestId,
                RetryAfter = GetRetryAfter(state.Response, new AiError { ErrorType = ErrorType.TooManyRequests }) ?? TimeSpan.Zero
            };
        }

        var (status, transient) = errorType switch
        {
            Wire.ErrorOverloaded => ((HttpStatusCode)529, true),
            Wire.ErrorApi => (HttpStatusCode.InternalServerError, true),
            Wire.ErrorInvalidRequest => (HttpStatusCode.BadRequest, false),
            Wire.ErrorAuthentication => (HttpStatusCode.Unauthorized, false),
            Wire.ErrorPermission => (HttpStatusCode.Forbidden, false),
            Wire.ErrorNotFound => (HttpStatusCode.NotFound, false),
            Wire.ErrorRequestTooLarge => (HttpStatusCode.RequestEntityTooLarge, false),
            _ => (HttpStatusCode.InternalServerError, false)
        };

        return new UnsuccessfulAiRequestException(message, status)
        {
            RequestId = requestId,
            RetryAfter = transient ? GetRetryAfter(state.Response, new AiError { ErrorType = ErrorType.Unknown }) : null
        };
    }

    // ---- helpers ---------------------------------------------------------------------------------------------

    private static DynamicJsonValue TextBlock(string text) => new() { [Wire.Type] = Wire.TypeText, [Wire.Text] = text };

    private static bool KeepInSidecar(string blockType, string text) =>
        blockType != Wire.TypeText || string.IsNullOrEmpty(text) == false;

    private static DynamicJsonArray NormalizeSidecar(BlittableJsonReaderArray contentArray)
    {
        var normalized = new DynamicJsonArray();
        foreach (BlittableJsonReaderObject block in contentArray)
        {
            block.TryGet(Wire.Type, out string blockType);

            string text = null;
            if (blockType == Wire.TypeText)
                block.TryGet(Wire.Text, out text);

            if (KeepInSidecar(blockType, text))
                normalized.Add(block);
        }

        return normalized;
    }

    private static void AppendContentText(DynamicJsonArray target, object content)
    {
        switch (content)
        {
            case null:
                return;

            case BlittableJsonReaderArray parts:
                foreach (var part in parts)
                {
                    if (part is BlittableJsonReaderObject partObj && partObj.TryGet(Wire.Text, out string partText))
                    {
                        if (string.IsNullOrWhiteSpace(partText) == false)
                            target.Add(TextBlock(partText));
                    }
                    else if (part != null)
                    {
                        var s = part.ToString();
                        if (string.IsNullOrWhiteSpace(s) == false)
                            target.Add(TextBlock(s));
                    }
                }
                return;

            case BlittableJsonReaderObject obj:
                target.Add(TextBlock(obj.ToString()));
                return;

            default:
                var text = content.ToString();
                if (string.IsNullOrWhiteSpace(text) == false)
                    target.Add(TextBlock(text));
                return;
        }
    }

    private static DynamicJsonValue Base64Source(string mediaType, string data) => new()
    {
        [Wire.Type] = Wire.SourceBase64,
        [Wire.MediaType] = mediaType,
        [Wire.Data] = data
    };

    private static BlittableJsonReaderObject ParseJsonObject(JsonOperationContext ctx, string json)
    {
        if (string.IsNullOrWhiteSpace(json))
            return ctx.ReadObject(new DynamicJsonValue(), "empty/json");

        return ctx.Sync.ReadForMemory(json, "json");
    }

    private static class Wire
    {
        // request/response field names (Anthropic Messages API + RavenDB canonical shared shape)
        public const string Model = "model";
        public const string MaxTokens = "max_tokens";
        public const string System = "system";
        public const string Messages = "messages";
        public const string Role = "role";
        public const string Content = "content";
        public const string Type = "type";
        public const string Text = "text";
        public const string Source = "source";
        public const string SourceBase64 = "base64";
        public const string MediaType = "media_type";
        public const string Data = "data";
        public const string Tools = "tools";
        public const string ToolChoice = "tool_choice";
        public const string ToolChoiceNone = "none";
        public const string Name = "name";
        public const string Description = "description";
        public const string InputSchema = "input_schema";
        public const string AdditionalProperties = "additionalProperties";
        // JSON-Schema keywords used when closing a tool input schema for strict mode.
        public const string SchemaType = "type";
        public const string SchemaTypeObject = "object";
        public const string SchemaProperties = "properties";
        public const string SchemaItems = "items";
        public const string SchemaAnyOf = "anyOf";
        public const string SchemaOneOf = "oneOf";
        public const string SchemaAllOf = "allOf";
        public const string SchemaDefs = "$defs";
        public const string SchemaDefinitions = "definitions";
        public const string Strict = "strict";
        public const string Input = "input";
        public const string OutputConfig = "output_config";
        public const string Format = "format";
        public const string Effort = "effort";
        public const string EffortLow = "low";
        public const string EffortMedium = "medium";
        public const string EffortHigh = "high";
        public const string Schema = "schema";
        public const string TypeJsonSchema = "json_schema";
        public const string Usage = "usage";
        public const string InputTokens = "input_tokens";
        public const string OutputTokens = "output_tokens";
        public const string CacheReadInputTokens = "cache_read_input_tokens";
        public const string CacheCreationInputTokens = "cache_creation_input_tokens";
        public const string StopReason = "stop_reason";
        public const string StopReasonMaxTokens = "max_tokens";
        public const string StopReasonModelContextWindowExceeded = "model_context_window_exceeded";
        public const string StopReasonRefusal = "refusal";
        public const string Error = "error";
        public const string Message = "message";

        // Anthropic error object `type` values (same set on an HTTP error body and on an SSE error event).
        public const string ErrorInvalidRequest = "invalid_request_error";
        public const string ErrorAuthentication = "authentication_error";
        public const string ErrorPermission = "permission_error";
        public const string ErrorNotFound = "not_found_error";
        public const string ErrorRequestTooLarge = "request_too_large";
        public const string ErrorRateLimit = "rate_limit_error";
        public const string ErrorApi = "api_error";
        public const string ErrorOverloaded = "overloaded_error";

        // content-block types
        public const string TypeText = "text";
        public const string TypeToolUse = "tool_use";
        public const string TypeToolResult = "tool_result";
        public const string TypeImage = "image";
        public const string TypeDocument = "document";
        public const string TypeThinking = "thinking";
        public const string TypeRedactedThinking = "redacted_thinking";
        public const string ToolUseId = "tool_use_id";

        // streaming (SSE): event types, delta types, and their fields
        public const string Stream = "stream";
        public const string Index = "index";
        public const string ContentBlock = "content_block";
        public const string Delta = "delta";
        public const string PartialJson = "partial_json";
        public const string Thinking = "thinking";
        public const string ThinkingAdaptive = "adaptive";
        public const string ThinkingEnabled = "enabled"; // legacy (pre-4.6) models only
        public const string BudgetTokens = "budget_tokens";
        public const string Signature = "signature";
        public const string EventMessageStart = "message_start";
        public const string EventContentBlockStart = "content_block_start";
        public const string EventContentBlockDelta = "content_block_delta";
        public const string EventMessageDelta = "message_delta";
        public const string EventMessageStop = "message_stop";
        public const string EventError = "error";
        public const string DeltaText = "text_delta";
        public const string DeltaInputJson = "input_json_delta";
        public const string DeltaThinking = "thinking_delta";
        public const string DeltaSignature = "signature_delta";

        // canonical stored shape (OpenAI-ish) field names + roles
        public const string ToolCalls = "tool_calls";
        public const string ToolCallId = "tool_call_id";
        public const string Function = "function";
        public const string TypeFunction = "function";
        public const string Id = "id";
        public const string Arguments = "arguments";
        public const string RoleSystem = "system";
        public const string RoleUser = "user";
        public const string RoleAssistant = "assistant";
        public const string RoleTool = "tool";
        public const string RoleInternal = "internal";

        // media types
        public const string MediaTypeTextPlain = "text/plain";
        public const string MediaTypeApplicationPdf = "application/pdf";
        public const string MediaTypeImageJpeg = "image/jpeg";
        public const string MediaTypeImagePng = "image/png";
        public const string MediaTypeImageGif = "image/gif";
        public const string MediaTypeImageWebp = "image/webp";

        // headers
        public const string HeaderApiKey = "x-api-key";
        public const string HeaderAnthropicVersion = "anthropic-version";
        public const string HeaderRequestId = "request-id";
    }
}
