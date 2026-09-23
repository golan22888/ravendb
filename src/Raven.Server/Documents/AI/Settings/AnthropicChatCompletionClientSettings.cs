using System;
using System.Collections.Generic;
using System.Globalization;
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
using Canonical = Raven.Server.Documents.AI.ChatCompletionClient.Constants;

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

    public override string GetRelativeCompletionUri() => "messages";

    // ---- authentication --------------------------------------------------------------------------------------

    public override void AddAuthentication(HttpRequestMessage request)
    {
        request.Headers.TryAddWithoutValidation(Wire.HeaderApiKey, ApiKey);
        request.Headers.TryAddWithoutValidation(Wire.HeaderAnthropicVersion, AnthropicVersion);
    }

    // ---- request building ------------------------------------------------------------------------------------

    public override DynamicJsonValue BuildTool(JsonOperationContext ctx, string name, string description, string parametersSchema)
    {
        var inputSchema = ParseJsonObject(ctx, parametersSchema);

        return new DynamicJsonValue
        {
            [Wire.Name] = name,
            [Wire.Description] = description,
            [Wire.InputSchema] = BuildStrictToolRootSchema(inputSchema),
            [Wire.Strict] = true
        };
    }

    public override void WritePayload(AsyncBlittableJsonTextWriter writer, JsonOperationContext ctx, AiChatRequest request, bool streaming)
    {
        var body = new DynamicJsonValue
        {
            [Wire.Model] = Model,
            [Wire.MaxTokens] = MaxOutputTokens
        };

        if (streaming)
            body[Wire.Stream] = true;

        DynamicJsonValue outputConfig = null;

        AppendReasoning(body, ref outputConfig);

        var turns = NormalizeTurns(ctx, request.Messages, request.Attachments);

        if (turns.System != null)
            body[Wire.System] = turns.System;

        body[Wire.Messages] = turns.Messages;

        if (request.PreparedTools?.Count > 0)
        {
            var tools = new DynamicJsonArray();
            foreach (var tool in request.PreparedTools)
                tools.Add(tool);
            body[Wire.Tools] = tools;

            if (request.UseTools == false)
                body[Wire.ToolChoice] = new DynamicJsonValue { [Wire.Type] = Wire.ToolChoiceNone };
        }

        var format = BuildOutputFormat(ctx, request.Schema);
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
            if (message.TryGet(Canonical.ResponseFields.Role, out string role) == false)
                continue;

            if (role == Canonical.RequestFields.RoleInternalValue)
                continue;

            if (role == Canonical.RequestFields.RoleSystemValue)
            {
                if (message.TryGet(Canonical.ResponseFields.Content, out string sysContent) && string.IsNullOrEmpty(sysContent) == false)
                {
                    if (systemText.Length > 0)
                        systemText.Append("\n\n");
                    systemText.Append(sysContent);
                }
                continue;
            }

            if (role == Canonical.RequestFields.RoleToolValue)
            {
                pendingToolResults ??= new DynamicJsonArray();
                message.TryGet(Canonical.ResponseFields.ToolCallId, out string toolUseId);
                message.TryGet(Canonical.ResponseFields.Content, out object toolContent);
                pendingToolResults.Add(new DynamicJsonValue
                {
                    [Wire.Type] = Wire.TypeToolResult,
                    [Wire.ToolUseId] = toolUseId,
                    [Wire.Content] = toolContent?.ToString() ?? string.Empty
                });
                continue;
            }

            FlushToolResults();

            if (role == Canonical.RequestFields.RoleAssistantValue)
            {
                if (TryBuildAssistantTurn(ctx, message, out var assistantTurn))
                    messages.Add(assistantTurn);
                continue;
            }

            message.TryGet(Canonical.ResponseFields.Content, out object userContent);
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
        if (string.IsNullOrWhiteSpace(_settings.Reasoning))
            return;

        body[Wire.Thinking] = new DynamicJsonValue { [Wire.Type] = Wire.ThinkingAdaptive };
        (outputConfig ??= new DynamicJsonValue())[Wire.Effort] = EffortLevel();
    }

    // Known levels are emitted in their canonical form; anything else goes out as supplied and the API validates it.
    private string EffortLevel()
    {
        var reasoning = _settings.Reasoning.Trim();

        foreach (var level in new[] { Wire.EffortLow, Wire.EffortMedium, Wire.EffortHigh, Wire.EffortXHigh, Wire.EffortMax })
        {
            if (string.Equals(reasoning, level, StringComparison.OrdinalIgnoreCase))
                return level;
        }

        return reasoning;
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

        message.TryGet(Canonical.ResponseFields.Content, out object textContent);
        AppendContentText(content, textContent);

        if (message.TryGet(Canonical.ResponseFields.ToolCalls, out BlittableJsonReaderArray toolCalls) && toolCalls != null)
        {
            foreach (BlittableJsonReaderObject call in toolCalls)
            {
                if (call.TryGet(Canonical.ResponseFields.Id, out string id) == false ||
                    call.TryGet(Canonical.ResponseFields.Function, out BlittableJsonReaderObject function) == false ||
                    function.TryGet(Canonical.ResponseFields.Name, out string name) == false)
                    continue;

                function.TryGet(Canonical.ResponseFields.Arguments, out string arguments);
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
            case Canonical.AttachmentsRequestFields.MediaTypeTextPlain:
                return TextBlock(attachment.Data);
            case Canonical.AttachmentsRequestFields.MediaTypeApplicationPdf:
                return new DynamicJsonValue
                {
                    [Wire.Type] = Wire.TypeDocument,
                    [Wire.Source] = Base64Source(Canonical.AttachmentsRequestFields.MediaTypeApplicationPdf, attachment.Data)
                };
            case Canonical.AttachmentsRequestFields.MediaTypeImageJpeg:
            case Canonical.AttachmentsRequestFields.MediaTypeImagePng:
            case Canonical.AttachmentsRequestFields.MediaTypeImageGif:
            case Canonical.AttachmentsRequestFields.MediaTypeImageWebp:
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
                [Canonical.ResponseFields.Role] = Canonical.RequestFields.RoleAssistantValue,
                [Canonical.ResponseFields.Content] = null,
                [Canonical.ResponseFields.ToolCalls] = ToCanonicalToolCalls(toolCalls),
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
                [Canonical.ResponseFields.Role] = Canonical.RequestFields.RoleAssistantValue,
                [Canonical.ResponseFields.Content] = result
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
                    usage.Update(promptTokens: inputTokens + cacheRead + cacheCreation, completionTokens: 0, cachedTokens: cacheRead);
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
                    usage.Update(promptTokens: 0, completionTokens: outputTokens, cachedTokens: 0);
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
            RefusedToAnswerException.Throw(refusalText, "[streaming]", state.StopReason, GetRequestId(response.Headers));
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
                        [Canonical.ResponseFields.Role] = Canonical.RequestFields.RoleAssistantValue,
                        [Canonical.ResponseFields.Content] = plainText
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
                    [Canonical.ResponseFields.Role] = Canonical.RequestFields.RoleAssistantValue,
                    [Canonical.ResponseFields.Content] = resultMessage
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
                [Canonical.ResponseFields.Role] = Canonical.RequestFields.RoleAssistantValue,
                [Canonical.ResponseFields.Content] = null,
                [Canonical.ResponseFields.ToolCalls] = ToCanonicalToolCalls(toolCalls),
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

    private static DynamicJsonArray ToCanonicalToolCalls(List<AiToolCall> toolCalls)
    {
        var array = new DynamicJsonArray();
        foreach (var call in toolCalls)
        {
            array.Add(new DynamicJsonValue
            {
                [Canonical.ResponseFields.Id] = call.Id,
                [Canonical.ResponseFields.Type] = Canonical.ResponseFields.Function,
                [Canonical.ResponseFields.Function] = new DynamicJsonValue { [Canonical.ResponseFields.Name] = call.Name, [Canonical.ResponseFields.Arguments] = call.Arguments }
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
        usage.Update(promptTokens: inputTokens + cacheRead + cacheCreation, completionTokens: outputTokens, cachedTokens: cacheRead);
    }

    // ---- errors ----------------------------------------------------------------------------------------------

    public override AiError ParseError(BlittableJsonReaderObject content, HttpResponseMessage response)
    {
        var (type, message) = ExtractError(content);

        // A 429 without a usable retry-after has no delay to honour - the monthly spend cap answers that way and keeps
        // failing until it resets - so it backs off like an exhausted quota instead of being retried at once.
        var errorType = (int)response.StatusCode == 429
            ? (TryParseRetryAfterHeaders(response.Headers, out _) ? ErrorType.TooManyRequests : ErrorType.InsufficientQuota)
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

    public override TimeSpan? GetRetryAfter(HttpResponseMessage response, AiError error)
    {
        // "prompt is too long" is a deterministic overflow; a stray Retry-After must not make it retryable.
        if (error.ErrorType == ErrorType.TooManyTokens && response.StatusCode != HttpStatusCode.TooManyRequests)
            return null;

        return TryParseRetryAfterHeaders(response.Headers, out var fromHeader) ? fromHeader : error.RetryAfter;
    }

    // Parses the standard retry-after-ms / Retry-After headers only; GetRetryAfter is the policy on top of it.
    private static bool TryParseRetryAfterHeaders(HttpResponseHeaders headers, out TimeSpan retryAfter)
    {
        if (headers.TryGetValues(ChatCompletionClient.Constants.Headers.RetryAfterMs, out var msValues) &&
            double.TryParse(msValues.FirstOrDefault(), NumberStyles.Float, CultureInfo.InvariantCulture, out var milliseconds) &&
            double.IsFinite(milliseconds))
        {
            retryAfter = FromNonNegative(TimeSpan.FromMilliseconds(milliseconds));
            return true;
        }

        var standard = headers.RetryAfter;
        if (standard?.Delta is { } delta)
        {
            retryAfter = FromNonNegative(delta);
            return true;
        }

        if (standard?.Date is { } date)
        {
            retryAfter = FromNonNegative(date - DateTimeOffset.UtcNow);
            return true;
        }

        if (headers.TryGetValues(ChatCompletionClient.Constants.Headers.RetryAfter, out var rawValues))
        {
            var raw = rawValues.FirstOrDefault();

            if (double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out var seconds) && double.IsFinite(seconds))
            {
                retryAfter = FromNonNegative(TimeSpan.FromSeconds(seconds));
                return true;
            }

            if (DateTimeOffset.TryParse(raw, CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal | DateTimeStyles.AdjustToUniversal, out var httpDate))
            {
                retryAfter = FromNonNegative(httpDate - DateTimeOffset.UtcNow);
                return true;
            }
        }

        retryAfter = default;
        return false;

        static TimeSpan FromNonNegative(TimeSpan value) => value < TimeSpan.Zero ? TimeSpan.Zero : value;
    }

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
            // Same contract as the non-streaming 429: honour a delay when there is one, otherwise back off like an
            // exhausted quota instead of retrying at once.
            var retryAfter = GetRetryAfter(state.Response, new AiError { ErrorType = ErrorType.TooManyRequests });
            if (retryAfter == null)
                return new InsufficientQuotaException(message) { RequestId = requestId };

            return new RateLimitException(message)
            {
                RequestId = requestId,
                RetryAfter = retryAfter.Value
            };
        }

        var status = errorType switch
        {
            Wire.ErrorOverloaded => (HttpStatusCode)529,
            Wire.ErrorApi => HttpStatusCode.InternalServerError,
            Wire.ErrorInvalidRequest => HttpStatusCode.BadRequest,
            Wire.ErrorAuthentication => HttpStatusCode.Unauthorized,
            Wire.ErrorPermission => HttpStatusCode.Forbidden,
            Wire.ErrorNotFound => HttpStatusCode.NotFound,
            Wire.ErrorRequestTooLarge => HttpStatusCode.RequestEntityTooLarge,
            _ => HttpStatusCode.InternalServerError
        };

        return new UnsuccessfulAiRequestException(message, status)
        {
            RequestId = requestId
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
                    if (part is BlittableJsonReaderObject partObj && partObj.TryGet(Canonical.AttachmentsRequestFields.TypeText, out string partText))
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
        // request/response field names (Anthropic Messages API)
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
        // JSON-Schema keywords used when closing a tool input schema for strict mode.
        public const string Strict = "strict";
        public const string Input = "input";
        public const string OutputConfig = "output_config";
        public const string Format = "format";
        public const string Effort = "effort";
        public const string EffortLow = "low";
        public const string EffortMedium = "medium";
        public const string EffortHigh = "high";
        public const string EffortXHigh = "xhigh";
        public const string EffortMax = "max";
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

        // message roles and tool-use ids
        public const string RoleUser = "user";
        public const string RoleAssistant = "assistant";
        public const string Id = "id";

        // headers
        public const string HeaderApiKey = "x-api-key";
        public const string HeaderAnthropicVersion = "anthropic-version";
        public const string HeaderRequestId = "request-id";
    }
}
