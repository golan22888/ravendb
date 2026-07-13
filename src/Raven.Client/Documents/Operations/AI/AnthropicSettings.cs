using System;
using System.Collections.Generic;
using Sparrow.Json.Parsing;

namespace Raven.Client.Documents.Operations.AI;

/// <summary>Settings for Anthropic's native Claude API (the Messages API). Chat model type only.</summary>
public sealed class AnthropicSettings : AbstractAiSettings, IAiSettings
{
    /// <summary>The default Anthropic API endpoint.</summary>
    public const string DefaultEndpoint = "https://api.anthropic.com/v1/";

    /// <summary>The default maximum number of output tokens to request when the user does not specify one.</summary>
    public const int DefaultMaxOutputTokens = 8192;

    public AnthropicSettings()
    {
        // deserialization
    }

    public AnthropicSettings(string apiKey, string model, string endpoint = null, int? maxOutputTokens = null, AiReasoningLevel reasoning = AiReasoningLevel.Default)
    {
        ApiKey = apiKey;
        Model = model;
        Endpoint = endpoint;
        MaxOutputTokens = maxOutputTokens;
        Reasoning = reasoning;
    }

    /// <summary>The Anthropic API key (sent as the <c>x-api-key</c> header).</summary>
    public string ApiKey { get; set; }

    /// <summary>The Claude model id.</summary>
    public string Model { get; set; }

    /// <summary>The API endpoint. Optional - defaults to <see cref="DefaultEndpoint"/>.</summary>
    public string Endpoint { get; set; }

    /// <summary>The output-token cap per request (required by the Messages API); when null, <see cref="DefaultMaxOutputTokens"/> is applied.</summary>
    public int? MaxOutputTokens { get; set; }

    /// <summary>How much reasoning the model should do; <see cref="AiReasoningLevel.Default"/> sends no reasoning configuration at all.</summary>
    public AiReasoningLevel Reasoning { get; set; } = AiReasoningLevel.Default;

    public Uri GetBaseEndpointUri()
    {
        var endpoint = string.IsNullOrWhiteSpace(Endpoint) ? DefaultEndpoint : Endpoint;
        if (endpoint.EndsWith("/") == false)
            endpoint += "/";

        return new Uri(endpoint);
    }

    public override void ValidateFields(List<string> errors)
    {
        if (string.IsNullOrWhiteSpace(ApiKey))
            errors.Add($"Value of `{nameof(ApiKey)}` field cannot be empty.");

        if (string.IsNullOrWhiteSpace(Model))
            errors.Add($"Value of `{nameof(Model)}` field cannot be empty.");

        if (MaxOutputTokens is <= 0)
            errors.Add($"Value of `{nameof(MaxOutputTokens)}` field must be positive.");

        // Reasoning is validated server-side, once the model's wire form is known.
    }

    public override AiSettingsCompareDifferences Compare(AbstractAiSettings other)
    {
        if (other is not AnthropicSettings anthropicSettings)
            return AiSettingsCompareDifferences.All;

        var differences = AiSettingsCompareDifferences.None;

        if (ApiKey != anthropicSettings.ApiKey)
            differences |= AiSettingsCompareDifferences.AuthenticationSettings;

        if (Model != anthropicSettings.Model)
            differences |= AiSettingsCompareDifferences.ModelArchitecture;

        if (Endpoint != anthropicSettings.Endpoint)
            differences |= AiSettingsCompareDifferences.EndpointConfiguration;

        if (MaxOutputTokens != anthropicSettings.MaxOutputTokens)
            differences |= AiSettingsCompareDifferences.EndpointConfiguration;

        if (Reasoning != anthropicSettings.Reasoning)
            differences |= AiSettingsCompareDifferences.EndpointConfiguration;

        return differences;
    }

    public override DynamicJsonValue ToJson()
    {
        var json = base.ToJson();
        json[nameof(Model)] = Model;
        json[nameof(ApiKey)] = ApiKey;

        if (string.IsNullOrWhiteSpace(Endpoint) == false)
            json[nameof(Endpoint)] = Endpoint;

        if (MaxOutputTokens.HasValue)
            json[nameof(MaxOutputTokens)] = MaxOutputTokens.Value;

        if (Reasoning != AiReasoningLevel.Default)
            json[nameof(Reasoning)] = Reasoning.ToString();

        return json;
    }
}
