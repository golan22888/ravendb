using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Exceptions;
using Raven.Server.Documents.ETL.Providers.AI;
using Raven.Server.Documents.Handlers.AI.Agents;
using Raven.Server.Json;
using Sparrow;
using Sparrow.Json;
using Sparrow.Json.Parsing;
using Sparrow.Server.Json.Sync;

namespace Raven.Server.Documents.AI.Settings;

internal abstract partial class AbstractChatCompletionClientSettings
{
    public virtual void AddAuthentication(HttpRequestMessage request)
    {
        request.Headers.Authorization = string.IsNullOrEmpty(ApiKey)
            ? null
            : new AuthenticationHeaderValue(ChatCompletionClient.Constants.RequestFields.AuthorizationApiKeyProperty, ApiKey);
    }

    public virtual void ValidateRequest(JsonOperationContext ctx, AiChatRequest request)
    {
        // The OpenAI family has nothing to validate up front.
    }

    public virtual DynamicJsonValue BuildTool(JsonOperationContext ctx, string name, string description, string parametersSchema)
    {
        var tool = new DynamicJsonValue
        {
            [ChatCompletionClient.Constants.JsonSchemaFields.Type] = "function",
            [ChatCompletionClient.Constants.ResponseFields.Function] = new DynamicJsonValue
            {
                [ChatCompletionClient.Constants.ResponseFields.Name] = name,
                [ChatCompletionClient.Constants.JsonSchemaFields.Description] = description,
                ["parameters"] = ctx.Sync.ReadForMemory(parametersSchema, "params/schema")
            }
        };

        if (SupportStrictTools)
            tool[ChatCompletionClient.Constants.JsonSchemaFields.Strict] = true;

        return tool;
    }

    public virtual void WritePayload(AsyncBlittableJsonTextWriter writer, JsonOperationContext ctx, ChatCompletionPayload payload)
    {
        writer.WriteStartObject();

        writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.Model);
        writer.WriteString(Model);
        writer.WriteComma();

        List<LazyStringValue> filterProperties =
        [
            ctx.GetLazyString(ConversationDocument.DateProperty),
            ctx.GetLazyString(ConversationDocument.UsageProperty),
            ctx.GetLazyString(ConversationDocument.OutputSchemaProperty),
            ctx.GetLazyString(ConversationDocument.SummaryProperty),
            ctx.GetLazyString(ChatCompletionClient.Constants.ResponseFields.SubConversationId),
            ctx.GetLazyString(AnthropicChatCompletionClientSettings.RawContentSidecarProperty)
        ];

        writer.WriteArray(ctx, ChatCompletionClient.Constants.RequestFields.Messages, BuildMessages(ctx, payload.Messages, payload.Attachments), (w, context, message) =>
        {
            w.WriteStartObject();
            w.WriteObjectWithFilter(message, filterProperties.Contains);
            w.WriteEndObject();
        });

        if (payload.Tools?.Count > 0)
        {
            writer.WriteComma();
            writer.WriteArray(ChatCompletionClient.Constants.RequestFields.Tools, payload.Tools);

            if (payload.UseTools is false)
            {
                writer.WriteComma();
                writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.ToolChoice);
                writer.WriteString("none");
            }
        }

        if (payload.Schema != null)
        {
            writer.WriteComma();
            writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.ResponseFormat);
            writer.WriteStartObject();
            writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.Type);
            writer.WriteString(ChatCompletionClient.Constants.RequestFields.JsonSchema);
            writer.WriteComma();
            writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.JsonSchema);
            writer.WriteObject(GetStructuredOutputSchemaAsBlittable(ctx, payload.Schema));
            writer.WriteEndObject();
        }

        if (payload.Streaming)
        {
            writer.WriteComma();
            writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.Stream);
            writer.WriteBool(true);
            writer.WriteComma();
            writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.StreamOptions);
            writer.WriteStartObject();
            writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.IncludeUsage);
            writer.WriteBool(true);
            writer.WriteEndObject();
        }

        if (payload.PromptCacheKey != null && EnablePromptCaching)
        {
            writer.WriteComma();
            writer.WritePropertyName(ChatCompletionClient.Constants.RequestFields.PromptCacheKey);
            writer.WriteString(payload.PromptCacheKey);
        }

        HandleCompletionRequestPayload(writer);

        writer.WriteEndObject();
    }

    private static BlittableJsonReaderObject GetStructuredOutputSchemaAsBlittable(JsonOperationContext ctx, string schema)
    {
        using (var stream = RecyclableMemoryStreamFactory.GetRecyclableStream(Encoding.UTF8.GetBytes(schema)))
        {
            return ctx.Sync.ReadForMemory(stream, "json");
        }
    }

    protected virtual IEnumerable<BlittableJsonReaderObject> BuildMessages(JsonOperationContext context, IEnumerable<BlittableJsonReaderObject> messages, List<AiAttachment> attachments)
    {
        foreach (var message in messages)
        {
            if (message.TryGet(ChatCompletionClient.Constants.RequestFields.Content, out object content))
            {
                if (content is BlittableJsonReaderObject blittableJson)
                {
                    var msg = message.CloneOnTheSameContext();
                    var modifications = msg.Modifications ??= new DynamicJsonValue(msg);
                    modifications[ChatCompletionClient.Constants.RequestFields.Content] = blittableJson.ToString();
                    yield return msg.CloneOnTheSameContext();
                    continue;
                }
            }

            yield return message;
        }

        if (attachments is not null && attachments.Count > 0)
        {
            var content = new DynamicJsonArray();
            var message = new DynamicJsonValue
            {
                [ChatCompletionClient.Constants.RequestFields.Role] = ChatCompletionClient.Constants.RequestFields.RoleUserValue,
                [ChatCompletionClient.Constants.RequestFields.Content] = content
            };

            foreach (var attachment in attachments)
            {
                if (attachment.Source == AiAttachmentSource.NotFound)
                {
                    content.Add(new DynamicJsonValue
                    {
                        [ChatCompletionClient.Constants.AttachmentsRequestFields.Type] = ChatCompletionClient.Constants.AttachmentsRequestFields.TypeText,
                        [ChatCompletionClient.Constants.AttachmentsRequestFields.TypeText] = $"File '{attachment.Name}' (of type '{attachment.Type}') could not be loaded: attachment not found"
                    });
                    continue;
                }

                content.Add(GetAiAttachmentJson(attachment));
            }

            yield return context.ReadObject(message, "write-ai-attachments");
        }
    }

    public virtual AiResponse ParseResponse(JsonOperationContext ctx, HttpResponseMessage response, BlittableJsonReaderObject content, AiUsage usage, bool structuredOutput)
    {
        if (content.TryGet(ChatCompletionClient.Constants.ResponseFields.Choices, out BlittableJsonReaderArray choices) == false || choices.Length == 0)
            throw UnexpectedResponseException.Create(message: "No choices in response", response, content, GetRequestId(response.Headers));

        var choice0 = (BlittableJsonReaderObject)choices[0];

        if (choice0.TryGet(ChatCompletionClient.Constants.ResponseFields.Message, out BlittableJsonReaderObject message) == false)
            throw UnexpectedResponseException.Create(message: "No message property in choice", response, content, GetRequestId(response.Headers));

        if (content.TryGet(ChatCompletionClient.Constants.ResponseFields.Usage, out BlittableJsonReaderObject usageJson) == false)
            throw UnexpectedResponseException.Create(message: "No usage in response content", response, content, GetRequestId(response.Headers));
        usage.UpdateFrom(usageJson);

        if (message.TryGet(ChatCompletionClient.Constants.ResponseFields.ToolCalls, out BlittableJsonReaderArray calls) && calls.Length > 0)
        {
            var toolCalls = new List<AiToolCall>();
            foreach (BlittableJsonReaderObject call in calls)
            {
                if (call.TryGet(ChatCompletionClient.Constants.ResponseFields.Id, out string callId) is false ||
                    call.TryGet(ChatCompletionClient.Constants.ResponseFields.Function, out BlittableJsonReaderObject function) is false ||
                    function.TryGet(ChatCompletionClient.Constants.ResponseFields.Name, out string name) is false ||
                    function.TryGet(ChatCompletionClient.Constants.ResponseFields.Arguments, out string args) is false)
                    throw UnexpectedResponseException.Create(message: "Invalid function call: " + call, response, content, GetRequestId(response.Headers));
                toolCalls.Add(new AiToolCall(callId, name, args));
            }

            return new AiResponse(AiResponseType.Tool) { ToolCalls = toolCalls, Message = message };
        }

        if (TryGetDeltaContent(message, out var contentStr) == false)
        {
            choice0.TryGet(ChatCompletionClient.Constants.ResponseFields.FinishReason, out string finishReason);
            var refusal = GetRefusal(choice0, message);
            if (string.IsNullOrEmpty(refusal))
                throw UnexpectedResponseException.Create(message: "No response content", response, content, GetRequestId(response.Headers));

            RefusedToAnswerException.Throw(refusal, content.ToString(), finishReason, GetRequestId(response.Headers));
        }

        object result = structuredOutput
            ? ctx.Sync.ReadForMemory(contentStr, "ai/output")
            : contentStr.ToString();

        message.Modifications ??= new DynamicJsonValue(message);
        message.Modifications[ChatCompletionClient.Constants.ResponseFields.Content] = result;

        return new AiResponse(AiResponseType.Result) { Result = result, Message = message };
    }

    public virtual StreamEventResult ProcessStreamEvent(JsonOperationContext ctx, BlittableJsonReaderObject sseEvent, ChatStreamState state, AiUsage usage)
    {
        if (sseEvent.TryGet(ChatCompletionClient.Constants.ResponseFields.Usage, out BlittableJsonReaderObject streamedUsage) && streamedUsage is not null)
            usage.UpdateFrom(streamedUsage);

        if (sseEvent.TryGet(ChatCompletionClient.Constants.ResponseFields.Choices, out BlittableJsonReaderArray choices) is false || choices.Length == 0)
            return default;

        var choice = (BlittableJsonReaderObject)choices[0];
        if (choice.TryGet(ChatCompletionClient.Constants.ResponseFields.Delta, out BlittableJsonReaderObject delta) == false)
            return default;

        LazyStringValue textDelta = null;
        if (TryGetDeltaContent(delta, out var content))
        {
            state.ToolCalls.AddAndReset();
            textDelta = content;
        }

        if (delta.TryGet(ChatCompletionClient.Constants.ResponseFields.ToolCalls, out BlittableJsonReaderArray toolCalls))
        {
            foreach (BlittableJsonReaderObject toolCallChunk in toolCalls)
                state.ToolCalls.Merge(toolCallChunk);
        }

        return new StreamEventResult(textDelta, stop: false);
    }

    public virtual AiResponse BuildStreamedResponse(JsonOperationContext streamingCtx, ChatStreamState state, HttpResponseMessage response)
    {
        state.ToolCalls.AddAndReset();

        if (state.ToolCalls.TryGetToolCallsForMessage(out var allToolCalls))
        {
            return new AiResponse(AiResponseType.Tool)
            {
                Message = streamingCtx.ReadObject(new DynamicJsonValue
                {
                    [ChatCompletionClient.Constants.ResponseFields.Role] = ChatCompletionClient.Constants.RequestFields.RoleAssistantValue,
                    [ChatCompletionClient.Constants.ResponseFields.Content] = null,
                    [ChatCompletionClient.Constants.ResponseFields.ToolCalls] = allToolCalls
                }, "persisted/streamed/toolcalls"),
                ToolCalls = state.ToolCalls.GetAllToolCalls(),
            };
        }

        // Unstructured: the answer is the accumulated plain text, returned as a string rather than a parsed object.
        if (state.StructuredOutput == false)
        {
            var fullText = state.RawText?.ToString() ?? string.Empty;
            return new AiResponse(AiResponseType.Result)
            {
                Message = streamingCtx.ReadObject(new DynamicJsonValue
                {
                    [ChatCompletionClient.Constants.ResponseFields.Role] = ChatCompletionClient.Constants.RequestFields.RoleAssistantValue,
                    [ChatCompletionClient.Constants.ResponseFields.Content] = fullText,
                }, "persisted/streamed/message"),
                Result = fullText,
            };
        }

        return new AiResponse(AiResponseType.Result)
        {
            Message = streamingCtx.ReadObject(new DynamicJsonValue
            {
                [ChatCompletionClient.Constants.ResponseFields.Role] = ChatCompletionClient.Constants.RequestFields.RoleAssistantValue,
                [ChatCompletionClient.Constants.ResponseFields.Content] = state.FinalResult,
            }, "persisted/streamed/message"),
            Result = state.FinalResult,
        };
    }

    private static bool TryGetDeltaContent(BlittableJsonReaderObject delta, out LazyStringValue content)
    {
        // Try content, then reasoning_content, then reasoning (for LM Studio and other reasoning-model compatibility).
        if (delta.TryGet(ChatCompletionClient.Constants.ResponseFields.Content, out content) && content?.Length > 0)
            return true;

        if (delta.TryGet(ChatCompletionClient.Constants.ResponseFields.ReasoningContent, out content) && content?.Length > 0)
            return true;

        if (delta.TryGet(ChatCompletionClient.Constants.ResponseFields.Reasoning, out content) && content?.Length > 0)
            return true;

        content = null;
        return false;
    }

    public virtual TimeSpan? GetRetryAfter(HttpResponseMessage response, AiError error)
    {
        var headers = response.Headers;

        var parsed = TryGetRetryAfterHeader(headers, out var fromHeader);
        if (parsed == false &&
            headers.Contains(ChatCompletionClient.Constants.Headers.RetryAfterMs) == false &&
            headers.Contains(ChatCompletionClient.Constants.Headers.RetryAfter) == false &&
            error.RetryAfter is null)
            return null;

        var retryAfter = parsed ? fromHeader : error.RetryAfter ?? TimeSpan.Zero;

        if (headers.TryGetValues(ChatCompletionClient.Constants.Headers.XRateLimitResetTokens, out var resetTokensValues) &&
            ChatCompletionClient.TryParseResetTime(resetTokensValues.FirstOrDefault(), out var retryAfterForTokens) &&
            retryAfterForTokens > retryAfter)
            retryAfter = retryAfterForTokens;

        if (headers.TryGetValues(ChatCompletionClient.Constants.Headers.XRateLimitResetRequests, out var resetRequestsValues) &&
            ChatCompletionClient.TryParseResetTime(resetRequestsValues.FirstOrDefault(), out var retryAfterForReqs) &&
            retryAfterForReqs > retryAfter)
            retryAfter = retryAfterForReqs;

        return retryAfter;
    }

    public virtual bool CanCarryRetryDelayOnNonRateLimit(HttpResponseMessage response) => false;

    protected static bool TryGetRetryAfterHeader(HttpResponseHeaders headers, out TimeSpan retryAfter)
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

    public virtual string GetRequestId(HttpResponseHeaders headers)
    {
        if (headers.TryGetValues(ChatCompletionClient.Constants.Headers.XRequestId, out IEnumerable<string> values))
            return values.FirstOrDefault() ?? string.Empty;

        // Azure API Management uses a different header name.
        if (headers.TryGetValues("apim-request-id", out values))
            return values.FirstOrDefault() ?? string.Empty;

        return string.Empty;
    }
}

internal sealed class ChatCompletionPayload
{
    public IEnumerable<BlittableJsonReaderObject> Messages;
    public List<AiAttachment> Attachments;
    public List<BlittableJsonReaderObject> Tools;
    public bool UseTools;
    public bool Streaming;
    public string Schema;
    public string PromptCacheKey;
    public object ProviderPrepared;
}

internal sealed class ChatStreamState
{
    public IToolCallState ToolCalls;
    public BlittableJsonReaderObject FinalResult;   // set by the client when the answer JSON parser completes
    public string StopReason;
    public object ProviderState;                    // provider-private (Anthropic: block map by content-block index)
    public HttpResponseMessage Response;

    public bool StructuredOutput;

    public StringBuilder RawText;                   // accumulates the streamed text when StructuredOutput is false

    public bool SawStop;
}

internal readonly struct StreamEventResult(LazyStringValue textDelta, bool stop)
{
    public readonly LazyStringValue TextDelta = textDelta;
    public readonly bool Stop = stop;               // true when the provider signals end-of-stream (e.g. message_stop)
}
