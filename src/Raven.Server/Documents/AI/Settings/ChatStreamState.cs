using System.Net.Http;
using System.Text;
using Sparrow.Json;

namespace Raven.Server.Documents.AI.Settings;

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

    // OpenAI-family streamed-result accounting (RavenDB-25401 / 25681). Anthropic keeps its own equivalents:
    // thinking and text are distinct block types there, and refusal/truncation arrive as a stop_reason.
    // Refusal text is held apart from the answer: never streamed as content, never parsed as JSON (RavenDB-26185).
    public StringBuilder RefusalText;
    public bool SawContent;
    public StringBuilder ReasoningFallback;         // reasoning is only ever a fallback, never streamed as the answer
    public string PendingChunk;                     // promoted reasoning the client still has to stream
    public SseStreamingJsonParser Parser;           // owned by the client; read here for IsInvalid
}
