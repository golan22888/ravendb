using System;
using System.IO;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;
using Raven.Client.Documents.Operations.AI;
using Sparrow.Json;
using Sparrow.Json.Parsing;

namespace Raven.Server.Documents.AI.Settings;

internal abstract class AbstractChatCompletionClientSettings
{
    private readonly IAiSettings _settings;
    
    public string ApiKey => _settings.ApiKey;

    public string Model => _settings.Model;

    public Uri GetBaseEndpointUri() => _settings.GetBaseEndpointUri();

    protected AbstractChatCompletionClientSettings(IAiSettings settings)
    {
        _settings = settings;
    }

    public virtual void AddHeaders(HttpRequestMessage request)
    {
    }

    public abstract string GetRelativeCompletionUri();

    public virtual string GetRelativeModelsUri() => "models";
    
    internal static bool TryGetParameters(AiConnectionString connectionString, out AbstractChatCompletionClientSettings settings)
    {
        settings = null;

        switch (connectionString.ModelType)
        {
            case AiModelType.Chat:
                break;
            default:
                throw new InvalidOperationException(
                    $"Invalid provider settings for '{connectionString.Name}' with model type '{connectionString.ModelType}'. " +
                    $"Supported providers for '{nameof(connectionString.ModelType.Chat)}' model type are '{nameof(AiConnectorType.OpenAi)}', '{nameof(AiConnectorType.Ollama)}', '{nameof(AiConnectorType.AzureOpenAi)}', '{nameof(AiConnectorType.Google)}' and '{nameof(AiConnectorType.Anthropic)}'");
        }

        var provider = connectionString.GetActiveProvider();
        switch (provider)
        {
            case AiConnectorType.OpenAi:
                settings = new OpenAiChatCompletionClientSettings(connectionString.OpenAiSettings);
                return true;
            case AiConnectorType.AzureOpenAi:
                settings = new AzureOpenAiChatCompletionClientSettings(connectionString.AzureOpenAiSettings);
                return true;
            case AiConnectorType.Ollama:
                settings = new OllamaChatCompletionClientSettings(connectionString.OllamaSettings);
                return true;
            case AiConnectorType.Google:
                settings = new GoogleChatCompletionClientSettings(connectionString.GoogleSettings);
                return true;
            case AiConnectorType.Anthropic:
                settings = new AnthropicChatCompletionClientSettings(connectionString.AnthropicSettings);
                return true;
        }

        return false;
    }
    
    internal virtual IToolCallState CreateToolCallState()
    {
        return new ToolCallState();
    }

    public abstract AiError ParseError(BlittableJsonReaderObject content, HttpResponseMessage response);

    public abstract void AddAuthentication(HttpRequestMessage request);

    public abstract DynamicJsonValue BuildTool(JsonOperationContext ctx, string name, string description, string parametersSchema);

    // 'request' is resolved by the client: internal messages filtered, tools in this provider's shape.
    public abstract void WritePayload(AsyncBlittableJsonTextWriter writer, JsonOperationContext ctx, AiChatRequest request, bool streaming);

    public abstract AiResponse ParseResponse(JsonOperationContext ctx, HttpResponseMessage response, BlittableJsonReaderObject content, AiUsage usage, bool structuredOutput);

    public abstract StreamEventResult ProcessStreamEvent(JsonOperationContext ctx, BlittableJsonReaderObject sseEvent, ChatStreamState state, AiUsage usage);

    public abstract AiResponse BuildStreamedResponse(JsonOperationContext streamingCtx, ChatStreamState state, HttpResponseMessage response);

    public abstract TimeSpan? GetRetryAfter(HttpResponseMessage response, AiError error);

    public abstract string GetRequestId(HttpResponseHeaders headers);

    public virtual ValueTask<BlittableJsonReaderObject> TryGetResponseContentAsync(JsonOperationContext context, Stream stream)
    {
        return context.ReadForMemoryAsync(stream, "response/object");
    }
}

public class AiError
{
    public string Message { get; set; }
    public ErrorType ErrorType { get; set; }
    public TimeSpan? RetryAfter { get; set; } = null;
}

public enum ErrorType
{
    Unknown,
    InsufficientQuota,
    TooManyTokens,
    TooManyRequests,
    Other429,
    RefusedToAnswer,
}
