﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading.Tasks;
using Raven.Client.Documents.Attachments;
using Raven.Client.Documents.Conventions;
using Raven.Client.Documents.Replication.Messages;
using Raven.Client.Extensions;
using Raven.Client.Http;
using Raven.Client.Json;
using Raven.Client.Util;
using Sparrow.Json;

namespace Raven.Client.Documents.Operations
{
    public class GetAttachmentOperation : IOperation<AttachmentResult>
    {
        private readonly string _documentId;
        private readonly string _name;
        private readonly AttachmentType _type;
        private readonly ChangeVectorEntry[] _changeVector;

        public GetAttachmentOperation(string documentId, string name, AttachmentType type, ChangeVectorEntry[] changeVector)
        {
            _documentId = documentId;
            _name = name;
            _type = type;
            _changeVector = changeVector;
        }

        public RavenCommand<AttachmentResult> GetCommand(IDocumentStore store, JsonOperationContext context, HttpCache cache)
        {
            return new GetAttachmentCommand(context, _documentId, _name, _type, _changeVector);
        }

        private class GetAttachmentCommand : RavenCommand<AttachmentResult>
        {
            private readonly JsonOperationContext _context;
            private readonly string _documentId;
            private readonly string _name;
            private readonly AttachmentType _type;
            private readonly ChangeVectorEntry[] _changeVector;

            public GetAttachmentCommand(JsonOperationContext context, string documentId, string name, AttachmentType type, ChangeVectorEntry[] changeVector)
            {
                if (string.IsNullOrWhiteSpace(documentId))
                    throw new ArgumentNullException(nameof(documentId));
                if (string.IsNullOrWhiteSpace(name))
                    throw new ArgumentNullException(nameof(name));

                if (type != AttachmentType.Document && changeVector == null)
                    throw new ArgumentNullException(nameof(changeVector), $"Change Vector cannot be null for attachment type {type}");

                _context = context;
                _documentId = documentId;
                _name = name;
                _type = type;
                _changeVector = changeVector;

                ResponseType = RavenCommandResponseType.Empty;
            }

            public override HttpRequestMessage CreateRequest(ServerNode node, out string url)
            {
                url = $"{node.Url}/databases/{node.Database}/attachments?id={Uri.EscapeDataString(_documentId)}&name={Uri.EscapeDataString(_name)}";
                var request = new HttpRequestMessage
                {
                    Method = HttpMethods.Get,
                };

                if (_type != AttachmentType.Document)
                {
                    request.Method = HttpMethod.Post;

                    request.Content = new BlittableJsonContent(stream =>
                    {
                        using (var writer = new BlittableJsonTextWriter(_context, stream))
                        {
                            writer.WriteStartObject();

                            writer.WritePropertyName("Type");
                            writer.WriteString(_type.ToString());
                            writer.WriteComma();

                            writer.WritePropertyName("ChangeVector");
                            writer.WriteStartArray();
                            bool first = true;
                            foreach (var vectorEntry in _changeVector)
                            {
                                if (first == false)
                                    writer.WriteComma();
                                first = false;

                                writer.WriteStartObject();

                                writer.WritePropertyName(nameof(vectorEntry.DbId));
                                writer.WriteString(vectorEntry.DbId.ToString());
                                writer.WriteComma();

                                writer.WritePropertyName(nameof(vectorEntry.Etag));
                                writer.WriteInteger(vectorEntry.Etag);

                                writer.WriteEndObject();
                            }
                            writer.WriteEndArray();

                            writer.WriteEndObject();
                        }
                    });
                }

                return request;
            }

            public override async Task<ResponseDisposeHandling> ProcessResponse(JsonOperationContext context, HttpCache cache, HttpResponseMessage response, string url)
            {
                var contentType = response.Content.Headers.TryGetValues("Content-Type", out IEnumerable<string> contentTypeVale) ? contentTypeVale.First() : null;
                var etag = response.GetRequiredEtagHeader();
                var hash = response.Headers.TryGetValues("Attachment-Hash", out IEnumerable<string> hashVal) ? hashVal.First() : null;
                long size = 0;
                if (response.Headers.TryGetValues("Attachment-Size", out IEnumerable<string> sizeVal))
                    long.TryParse(sizeVal.First(), out size);

                var attachmentDetails = new AttachmentDetails
                {
                    ContentType = contentType,
                    Name = _name,
                    Hash = hash,
                    Size = size,
                    Etag = etag,
                    DocumentId = _documentId,
                };

                var stream = new AttachmentStream(response, await response.Content.ReadAsStreamAsync().ConfigureAwait(false));

                Result = new AttachmentResult
                {
                    Stream = stream,
                    Details = attachmentDetails,
                };

                return ResponseDisposeHandling.Manually;
            }

            public override bool IsReadRequest => true;
        }
    }
}