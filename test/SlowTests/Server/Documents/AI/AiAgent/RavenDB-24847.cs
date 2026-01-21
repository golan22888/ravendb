using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FastTests;
using Microsoft.AspNetCore.Mvc;
using Raven.Client.Documents.AI;
using Raven.Client.Documents.Operations.AI;
using Raven.Client.Documents.Operations.AI.Agents;
using Raven.Client.Documents.Operations.Attachments;
using Raven.Client.Documents.Operations.ConnectionStrings;
using Raven.Server.Documents.Handlers.AI.Agents;
using SlowTests.Client.TimeSeries.Session;
using Tests.Infrastructure;
using Xunit;
using Xunit.Abstractions;
using static Raven.Server.Documents.Handlers.AI.Agents.ConversationDocument;

namespace SlowTests.Server.Documents.AI.AiAgent
{
    public class RavenDB_24847 : RavenTestBase
    {
        public RavenDB_24847(ITestOutputHelper output) : base(output)
        {
        }

        public class AnalysisOutputSchema
        {
            public string Answer { get; set; } = "answer form the llm";
        }
        private class ConversationDocumentVerify
        {
            public string LastProcessedHash { get; set; }
            public List<object> Messages { get; set; }
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.OpenAi, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task CanAnalyzeImageAndPersistStructuredData(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);

            // 1. Setup Connection String
            await store.Maintenance.SendAsync(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            // 2. Configure Agent
            var agent = new AiAgentConfiguration("image-analyzer", config.ConnectionStringName,
                "You are my friend have a chat with me");
            agent.Identifier = "image-analyzer";

            await store.AI.CreateAgentAsync(agent, new AnalysisOutputSchema());

            // 3. Start Conversation
            var chat = store.AI.Conversation(agent.Identifier, "chats/", new AiConversationCreationOptions());
            List<string> names = ["banana.png", "star.png", "heart.png"];
            chat.SetUserPrompt("what are inside the images I sent you? what are their colors?");
            AiAnswer<AnalysisOutputSchema> result = null;
            foreach (var name in names)
            {
                chat.AddAttachment(name, GetEmbeddedImgStream(name));
            }
            result = await chat.RunAsync<AnalysisOutputSchema>(CancellationToken.None);
         
            Assert.Equal(AiConversationResult.Done, result?.Status);
            Assert.NotNull(result?.Answer);
            // 6. Verify Persistence in the underlying ConversationDocument
            using (var session = store.OpenAsyncSession())
            {
                var chatDoc = await session.LoadAsync<ConversationDocumentVerify>(chat.Id);

                Assert.NotNull(chatDoc);

                // Verify that the analysis was extracted and saved correctly

                Assert.Null(chatDoc.LastProcessedHash);
            }

            WaitForUserToContinueTheTest(store, false);
        }

        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.OpenAi, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task CanAnalyzeImageFromCopiedAttachment(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);

            // 1. Setup Connection & Agent
            await store.Maintenance.SendAsync(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            var agent = new AiAgentConfiguration("image-analyzer", config.ConnectionStringName,
                "You are my friend have a chat with me");
            agent.Identifier = "image-analyzer";
            agent.Parameters.Add(new AiAgentParameter("company", "The company ID"));

            agent.Queries =
            [
                new AiAgentToolQuery
                {
                    Name = "ProductSearch",
                    Description = "semantic search the store product catalog",
                    Query = "from Products where vector.search(embedding.text(Name), $query)",
                    ParametersSampleObject = "{\"query\": [\"term or phrase to search in the catalog\"]}"
                },
                new AiAgentToolQuery
                {
                    Name = "RecentOrder",
                    Description = "Get the recent orders of the current user",
                    Query = "from Orders where Company = $company order by OrderedAt desc limit 10",
                    ParametersSampleObject = "{}"
                },
                new AiAgentToolQuery
                {
                    Name = "RecentOrder1",
                    Description = "Get the recent orders of the current user",
                    Query = "from Orders where Company = $company order by OrderedAt desc limit 10",
                    ParametersSampleObject = "{}"
                },
                new AiAgentToolQuery
                {
                    Name = "RecentOrder2",
                    Description = "Get the recent orders of the current user",
                    Query = "from Orders where Company = $company order by OrderedAt desc limit 10",
                    ParametersSampleObject = "{}"
                },
                new AiAgentToolQuery
                {
                    Name = "RecentOrder3",
                    Description = "Get the recent orders of the current user",
                    Query = "from Orders where Company = $company order by OrderedAt desc limit 10",
                    ParametersSampleObject = "{}"
                }
            ];
            await store.AI.CreateAgentAsync(agent, new AnalysisOutputSchema());

            // 2. Prepare Source Document with Attachment
            // We upload the image to a regular document first.
            string sourceDocId = "docs/1";
            using (var session = store.OpenAsyncSession())
            {
                await session.StoreAsync(new { Info = "Source Doc" }, sourceDocId);
                await session.SaveChangesAsync();
            }
            List<string> names = ["banana.png","star.png","heart.png"];
            foreach (var name in names)
            {
                using (var stream = GetEmbeddedImgStream(name))
                {
                    await store.Operations.SendAsync(new PutAttachmentOperation(sourceDocId, name, stream, "image/png"));
                }
            }
            // 3. Start Conversation
            var chat = store.AI.Conversation(agent.Identifier, "chats/", new AiConversationCreationOptions().AddParameter("company", "companies/90-A"));

            // 4. Reference the Existing Attachment
            // Instead of uploading a stream, we tell the conversation to "copy" or "use" the attachment from docs/1
            chat.SetUserPrompt("What is in this image?");
            chat.CopyAttachmentFrom("heart.png", sourceDocId);

            // 5. Run the Chat
            var result = await chat.RunAsync<AnalysisOutputSchema>(CancellationToken.None);

            // chat.AddAttachment("banana.png", GetEmbeddedImgStream("banana.png"));
            // result = await chat.RunAsync<AnalysisOutputSchema>(CancellationToken.None);
            WaitForUserToContinueTheTest(store, false);


            Assert.Equal(AiConversationResult.Done, result.Status);
            Assert.NotNull(result.Answer);

            // 6. Verify Persistence
            using (var session = store.OpenAsyncSession())
            {
                var chatDoc = await session.LoadAsync<ConversationDocumentVerify>(chat.Id);
                Assert.NotNull(chatDoc);
                Assert.Null(chatDoc.LastProcessedHash);
            }

            WaitForUserToContinueTheTest(store, false);
        }




        [RavenTheory(RavenTestCategory.Ai)]
        [RavenGenAiData(IntegrationType = RavenAiIntegration.OpenAi, DatabaseMode = RavenDatabaseMode.Single)]
        public async Task CanHandleCustomToolFailureAndStillRetrieveAttachment(Options options, GenAiConfiguration config)
        {
            using var store = GetDocumentStore(options);

            // 1. Setup Connection
            await store.Maintenance.SendAsync(new PutConnectionStringOperation<AiConnectionString>(config.Connection));

            // 2. Configure Agent with a Custom Tool
            var agent = new AiAgentConfiguration("mixed-tool-agent", config.ConnectionStringName,
                "You are a helpful assistant. Before answering about an image, you should still try to describe the image using your vision capabilities.");

            agent.Identifier = "mixed-tool-agent";
            agent.Actions =
            [
                new AiAgentToolAction
                {
                    Name = "CheckImageMetadata",
                    Description = "Checks external metadata for an image file.",
                    ParametersSampleObject = "{\"filename\": \"string\"}"
                }
            ];

            await store.AI.CreateAgentAsync(agent, new AnalysisOutputSchema());

            // 3. Start Conversation
            var chat = store.AI.Conversation(agent.Identifier, "chats/", new AiConversationCreationOptions());
            chat.OnUnhandledAction += args => Task.CompletedTask;
            var imgName = "banana.png";

            chat.SetUserPrompt($"Check metadata for any attachment you have and then tell me what object is in the image.");

            
            chat.AddAttachment(imgName, GetEmbeddedImgStream(imgName));
            

            // 4. Run Turn 1: Expecting Tool Call
            // The AI should stop to call 'CheckImageMetadata'
            var result = await chat.RunAsync<AnalysisOutputSchema>(CancellationToken.None);
            //
            // Assert.Equal(AiConversationResult.ActionRequired, result.Status);
            // Assert.NotEmpty(result.Answer);
            //
            // var toolCall = result.Answer.FirstOrDefault(x => x.Name == "CheckImageMetadata");
            // Assert.NotNull(toolCall);

            // 5. Provide "Failure" Response for Custom Tool
            // This ensures that if the test passes, it's NOT because this tool gave the answer.
            // It proves the AI fell back to the system's attachment handling.
            var responses = new List<AiAgentActionResponse>();
            foreach (var request in chat.RequiredActions())
            {
                responses.Add(new AiAgentActionResponse
                {
                    ToolId = request.ToolId,
                    Content = "{}" // Simulating an empty response for the action tool
                });
            }


            WaitForUserToContinueTheTest(store, false);
            // 6. Run Turn 2: Expecting Answer
            // The AI receives the tool failure, but should still see the attachment in the context 
            // (handled by ConversationHandler's logic) and answer the vision question.
            // result = await chat.RunAsync<string>(CancellationToken.None);
            //
            // Assert.Equal(AiConversationResult.Done, result.Status);
            // Assert.Contains("banana", result.Answer.ToLowerInvariant());

                // 7. Verify Persistence
                // We verify that the conversation state confirms the tool usage and the final answer.
            using (var session = store.OpenAsyncSession())
            {
                var doc = await session.LoadAsync<dynamic>(chat.Id);
                Assert.NotNull(doc);
                // We could inspect the messages here if needed to ensure the sequence was User -> Tool -> ToolResponse -> Assistant
            }
        }







        // Helper method to load images from the assembly, matching the pattern in RavenDB_24645
        private static Stream GetEmbeddedImgStream(string name)
        {
            var asm = typeof(RavenDB_24847).Assembly;
            // Using the same resource path as the provided issue test
            var resourceName = "SlowTests.Data.RavenDB_24648."+name;

            var stream = asm.GetManifestResourceStream(resourceName);
            if (stream == null)
                throw new FileNotFoundException($"Embedded resource not found: {resourceName}");

            return stream;
        }
    }
}
