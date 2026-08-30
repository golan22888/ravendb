using System;
using System.Collections.Generic;
using Raven.Server.Documents.ETL.Providers.AI;
using Sparrow.Json;

namespace Raven.Server.Documents.AI;

// A provider-independent chat-completion request; the selected adapter translates it into its own wire format.
public sealed class AiChatRequest
{
    public IEnumerable<BlittableJsonReaderObject> Messages;

    public List<AiAttachment> Attachments;

    public IReadOnlyList<AiToolDescriptor> Tools;

    // Provider-shaped tools, prepared once per conversation call; the client falls back to Tools when null.
    public PreparedAiTools PreparedTools;

    public bool UseTools;

    public string Schema;

    public string PromptCacheKey;

    // Whatever ValidateRequest produced, reused by the same provider's WritePayload. Opaque to the client.
    internal object ProviderPrepared;
}

// Bound to the context the tools were prepared in - reading blittables through another context does not fail cleanly.
public sealed class PreparedAiTools
{
    private readonly JsonOperationContext _ownerContext;

    internal readonly List<BlittableJsonReaderObject> Tools;

    internal PreparedAiTools(JsonOperationContext ownerContext, List<BlittableJsonReaderObject> tools)
    {
        _ownerContext = ownerContext ?? throw new ArgumentNullException(nameof(ownerContext));
        Tools = tools;
    }

    internal void AssertContext(JsonOperationContext context)
    {
        if (ReferenceEquals(_ownerContext, context) == false)
            throw new InvalidOperationException(
                "Prepared AI tools must be used with the same JsonOperationContext in which they were prepared.");
    }
}

public sealed class AiToolDescriptor
{
    public string Name;
    public string Description;
    public string ParametersSchema;

    public AiToolDescriptor(string name, string description, string parametersSchema)
    {
        Name = name;
        Description = description;
        ParametersSchema = parametersSchema;
    }
}
