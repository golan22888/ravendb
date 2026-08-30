using Raven.Client.Documents.Operations.AI;

namespace Tests.Infrastructure.ConnectionString.AI;

public class GenAiAnthropicConnectorForTesting : AbstractGenAiConnectorForTesting<GenAiAnthropicConnectorForTesting>
{
    private const string Model = "claude-haiku-4-5";

    public GenAiAnthropicConnectorForTesting()
    {
        RequiredEnvironmentVariables = [RavenTestHelper.EnvironmentVariables.AiIntegrationAnthropicApiKeyEnvName];
    }

    public override AiConnectorType AiConnectorType { get; init; } = AiConnectorType.Anthropic;

    protected override AiConnectionString CreateAiConnectionStringImpl() => new AiConnectionString
    {
        ModelType = AiModelType.Chat,
        AnthropicSettings = new AnthropicSettings(RavenTestHelper.EnvironmentVariables.AiIntegrationAnthropicApiKey, Model)
    };
}
