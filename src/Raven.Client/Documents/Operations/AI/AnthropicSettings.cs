using System;
using System.Collections.Generic;
using Sparrow.Json.Parsing;

namespace Raven.Client.Documents.Operations.AI;

/// <summary>Settings for Anthropic's native Claude API (the Messages API). Chat model type only.</summary>
public sealed class AnthropicSettings : AbstractAiSettings, IAiSettings
{
    /// <summary>The default Anthropic API endpoint.</summary>
    internal const string DefaultEndpoint = "https://api.anthropic.com/v1/";

    /// <summary>The default maximum number of output tokens to request when the user does not specify one.</summary>
    internal const int DefaultMaxOutputTokens = 8192;

    public AnthropicSettings()
    {
        // deserialization
    }

    public AnthropicSettings(string apiKey, string model, string endpoint = null, int? maxOutputTokens = null, string reasoning = null)
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

    /// <summary>
    /// How much reasoning the model should do before answering. When not set, no reasoning configuration is sent
    /// and the model applies its own default.
    ///
    /// The levels are <c>low</c>, <c>medium</c>, <c>high</c>, <c>xhigh</c> and <c>max</c>; not every model supports
    /// all of them, and a model without effort support rejects the request. A known level is sent in lowercase, any
    /// other value as supplied apart from trimming, so a new provider value can be used without waiting for a RavenDB
    /// release.
    /// </summary> 
    public string Reasoning { get; set; }

    // The documented base URL has no version segment; the Messages API is served under v1/.
    private static readonly Uri AnthropicBaseUri = new Uri("https://api.anthropic.com/");

    public Uri GetBaseEndpointUri()
    {
        var endpoint = string.IsNullOrWhiteSpace(Endpoint) ? DefaultEndpoint : Endpoint;
        if (endpoint.EndsWith("/") == false)
            endpoint += "/";

        var uri = new Uri(endpoint);
        return uri.Equals(AnthropicBaseUri) ? new Uri(uri, "v1/") : uri;
    }

    public override void ValidateFields(List<string> errors)
    {
        if (string.IsNullOrWhiteSpace(ApiKey))
            errors.Add($"Value of `{nameof(ApiKey)}` field cannot be empty.");

        if (string.IsNullOrWhiteSpace(Model))
            errors.Add($"Value of `{nameof(Model)}` field cannot be empty.");

        if (MaxOutputTokens is <= 0)
            errors.Add($"Value of `{nameof(MaxOutputTokens)}` field must be positive.");

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

        if (string.IsNullOrWhiteSpace(Reasoning) == false)
            json[nameof(Reasoning)] = Reasoning;

        return json;
    }
}
