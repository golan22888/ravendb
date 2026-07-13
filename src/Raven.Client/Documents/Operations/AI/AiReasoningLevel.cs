namespace Raven.Client.Documents.Operations.AI;

/// <summary>How much reasoning a chat model should do before answering - an intent, not a wire format; each provider translates it.</summary>
public enum AiReasoningLevel
{
    /// <summary>Send no reasoning configuration and keep the model's own default.</summary>
    Default,

    Low,

    Medium,

    High
}
