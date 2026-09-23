using Sparrow.Json;
using Sparrow.Json.Parsing;
using JsonSchema = Raven.Server.Documents.AI.ChatCompletionClient.Constants.JsonSchemaFields;

namespace Raven.Server.Documents.AI.Settings;

// Anthropic's strict mode requires `additionalProperties: false` on every object in a tool schema, and RavenDB's own
// sub-agent schema generator does not emit it on the root. Only the root is repaired: the schemas RavenDB generates are either
// flat (sub-agent parameters are primitives) or already closed at every level (the sample-object generator), and a
// hand-written schema belongs to the agent author - if it declares something Anthropic rejects, Anthropic says so.
internal sealed partial class AnthropicChatCompletionClientSettings
{
    // What RavenDB generates for a sub-agent tool (GetSchemaForSubAgentTool), e.g. one string parameter:
    //   {"type":"object","properties":{"customerId":{"description":"The customer to look up","type":"string"}},"required":["customerId"]}
    // goes out to Anthropic as the same object with "additionalProperties": false appended at the root.
    // A schema that already states additionalProperties is sent as-is; a tool with no parameters gets
    //   {"type":"object","properties":{},"additionalProperties":false}
    private static object BuildStrictToolRootSchema(BlittableJsonReaderObject schema)
    {
        if (schema == null || schema.Count == 0)
            return new DynamicJsonValue
            {
                [JsonSchema.Type] = JsonSchema.TypeObject,
                [JsonSchema.Properties] = new DynamicJsonValue(),
                [JsonSchema.AdditionalProperties] = false
            };

        if (schema.TryGetMember(JsonSchema.AdditionalProperties, out _))
            return schema;

        var root = new DynamicJsonValue();
        foreach (var property in schema.GetPropertyNames())
        {
            if (schema.TryGetMember(property, out var value))
                root[property] = value;
        }

        root[JsonSchema.AdditionalProperties] = false;
        return root;
    }
}
