using System;
using System.Linq;
using Sparrow.Json;
using Sparrow.Json.Parsing;

namespace Raven.Server.Documents.AI.Settings;

// Closes a canonical tool input schema for Anthropic strict mode: every object node must declare
// `additionalProperties: false`, or the API rejects the whole request with a 400.
internal sealed partial class AnthropicChatCompletionClientSettings
{
    private static DynamicJsonValue BuildStrictToolRootSchema(string toolName, BlittableJsonReaderObject schema)
    {
        if (schema == null || schema.Count == 0)
            return new DynamicJsonValue
            {
                [Wire.SchemaType] = Wire.SchemaTypeObject,
                [Wire.SchemaProperties] = new DynamicJsonValue(),
                [Wire.AdditionalProperties] = false
            };

        if (IsObjectSchema(schema) == false)
            throw new InvalidOperationException(
                $"Tool '{toolName}' cannot be used as an Anthropic strict tool because a tool input schema must describe " +
                "an object at its root. Declare the root as `{\"type\": \"object\", \"properties\": { ... }}`.");

        var root = BuildStrictToolSchema(toolName, schema);

        if (schema.TryGetMember(Wire.SchemaType, out _) == false)
            root[Wire.SchemaType] = Wire.SchemaTypeObject;

        if (schema.TryGetMember(Wire.SchemaProperties, out _) == false)
            root[Wire.SchemaProperties] = new DynamicJsonValue();

        return root;
    }

    private static readonly string[] SubSchemaMaps = [Wire.SchemaProperties, Wire.SchemaDefs, Wire.SchemaDefinitions];

    private static readonly string[] SubSchemaLists = [Wire.SchemaAnyOf, Wire.SchemaOneOf, Wire.SchemaAllOf];

    // Builds a new tree - the canonical ParametersSchema is shared with other providers and must not be mutated.
    private static DynamicJsonValue BuildStrictToolSchema(string toolName, BlittableJsonReaderObject schema)
    {
        if (schema == null)
            return new DynamicJsonValue
            {
                [Wire.SchemaType] = Wire.SchemaTypeObject,
                [Wire.AdditionalProperties] = false
            };

        var result = new DynamicJsonValue();

        foreach (var property in schema.GetPropertyNames())
        {
            if (schema.TryGetMember(property, out var value) == false)
                continue;

            if (property == Wire.AdditionalProperties)
                continue; // re-emitted below, after validation

            // Follow only real sub-schema keywords: a `default`/`const`/`examples` object is data, not a schema.
            if (SubSchemaMaps.Contains(property) && value is BlittableJsonReaderObject map)
            {
                var clonedMap = new DynamicJsonValue();
                foreach (var entry in map.GetPropertyNames())
                {
                    if (map.TryGetMember(entry, out var entryValue) == false)
                        continue;

                    clonedMap[entry] = entryValue is BlittableJsonReaderObject entrySchema
                        ? BuildStrictToolSchema(toolName, entrySchema)
                        : entryValue;
                }

                result[property] = clonedMap;
                continue;
            }

            if (SubSchemaLists.Contains(property) && value is BlittableJsonReaderArray list)
            {
                result[property] = CloneSubSchemaList(toolName, list);
                continue;
            }

            if (property == Wire.SchemaItems)
            {
                result[property] = value switch
                {
                    BlittableJsonReaderObject itemSchema => BuildStrictToolSchema(toolName, itemSchema),
                    BlittableJsonReaderArray itemList => CloneSubSchemaList(toolName, itemList),
                    _ => value
                };
                continue;
            }

            result[property] = value;
        }

        // An explicit declaration is honoured on any node; only the "absent" case needs the object-ness proof.
        if (schema.TryGetMember(Wire.AdditionalProperties, out _))
            result[Wire.AdditionalProperties] = ResolveAdditionalProperties(toolName, schema);
        else if (IsObjectSchema(schema))
            result[Wire.AdditionalProperties] = false; // provably an object, so close it

        return result;
    }

    private static DynamicJsonArray CloneSubSchemaList(string toolName, BlittableJsonReaderArray list)
    {
        var cloned = new DynamicJsonArray();
        foreach (var item in list)
            cloned.Add(item is BlittableJsonReaderObject itemSchema ? BuildStrictToolSchema(toolName, itemSchema) : item);

        return cloned;
    }

    // Anything open-ended is refused - closing it silently would change what the schema says it accepts.
    private static bool ResolveAdditionalProperties(string toolName, BlittableJsonReaderObject schema)
    {
        if (schema.TryGetMember(Wire.AdditionalProperties, out var declared) == false || declared == null)
            return false; // absent - close it

        switch (declared)
        {
            case bool allowed when allowed == false:
                return false; // already closed

            case bool:            // explicitly true
            case BlittableJsonReaderObject:  // a sub-schema: arbitrary property NAMES are intentionally allowed
                throw new InvalidOperationException(
                    $"Tool '{toolName}' cannot be used as an Anthropic strict tool because its input schema allows " +
                    "additional properties. RavenDB Anthropic tools require a closed schema in which every object " +
                    $"declares `{Wire.AdditionalProperties}: false`.");

            default:
                throw new InvalidOperationException(
                    $"Tool '{toolName}' has an invalid `{Wire.AdditionalProperties}` value of type " +
                    $"'{declared.GetType().Name}' in its input schema. It must be omitted, `false`, `true`, or a schema " +
                    "object; RavenDB Anthropic tools additionally require it to be omitted or `false`.");
        }
    }

    // `properties` without `type` still means object; a nullable object declares its type as a list.
    private static bool IsObjectSchema(BlittableJsonReaderObject schema)
    {
        if (schema.TryGetMember(Wire.SchemaType, out var type))
        {
            switch (type)
            {
                case LazyStringValue single:
                    return single == Wire.SchemaTypeObject;

                case BlittableJsonReaderArray many:
                    return many.Any(t => t is LazyStringValue s && s == Wire.SchemaTypeObject);
            }
        }

        return schema.TryGetMember(Wire.SchemaProperties, out var properties) && properties is BlittableJsonReaderObject;
    }
}
