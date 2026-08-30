using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Documents.Operations.AI.Agents;
using Raven.Server.Documents.AI;
using Raven.Server.Documents.ETL.Providers.AI;
using Sparrow.Json;

namespace Raven.Server.Documents.Handlers.AI.Agents;

internal class Talker(ConversationHandler handler, JsonOperationContext context, AiAgentConfiguration configuration, string schema, ConversationDocument document, string firstStreamPropertyPath, Func<Memory<byte>, Task> streaming) : IDisposable
{
    private List<AiToolDescriptor> _tools;
    private PreparedAiTools _preparedTools;

    public AiUsage AiUsage;
    public ChatCompletionClient Client;
    public ConversationDocument Document => document;

    public void Init()
    {
        document.EnsureInitialized();

        Client = handler.CreateClient();
        _tools = handler.BuildToolDescriptors(context, configuration);

        // Prepare once per conversation call - per-iteration preparation leaks schema blittables into the
        // conversation-scoped context.
        _preparedTools = Client.PrepareTools(context, _tools);
    }

    public AiChatRequest CreateRequest(List<AiAttachment> attachments)
    {
        AiUsage = new();
        return new AiChatRequest
        {
            Messages = document.Messages,
            Attachments = attachments,
            Tools = _tools,
            PreparedTools = _preparedTools,
            UseTools = document.RemainingToolIterations-- > 0,
            Schema = schema,
            PromptCacheKey = document.Id
        };
    }

    public async Task<AiResponse> RunAsync(IMemoryContextPool contextPool, AiChatRequest request, AiDebugTrace trace, CancellationToken token)
    {
        if (streaming is null)
        {
            return await Client.CompleteAsync(
                context,
                request,
                AiUsage,
                trace,
                token
            );
        }

        return await Client.StreamingCompleteAsync(
            context,
            contextPool,
            firstStreamPropertyPath,
            request,
            streaming,
            AiUsage,
            trace,
            token
        );
    }

    public void Dispose()
    {
        Client?.Dispose();
    }
}
