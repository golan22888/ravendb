﻿using System;
using System.Net.Http;
using Raven.Client.Documents.Conventions;
using Raven.Client.Http;
using Raven.Client.Json.Serialization;
using Raven.Client.Util;
using Sparrow.Json;

namespace Raven.Client.Documents.Operations.ConnectionStrings
{
    public sealed class RemoveConnectionStringOperation<T> : IMaintenanceOperation<RemoveConnectionStringResult> where T : ConnectionString
    {
        private readonly T _connectionString;

        public RemoveConnectionStringOperation(T connectionString)
        {
            _connectionString = connectionString;
        }

        public RavenCommand<RemoveConnectionStringResult> GetCommand(DocumentConventions conventions, JsonOperationContext context)
        {
            return new RemoveConnectionStringCommand(_connectionString);
        }

        private sealed class RemoveConnectionStringCommand : RavenCommand<RemoveConnectionStringResult>, IRaftCommand
        {
            private readonly T _connectionString;

            public RemoveConnectionStringCommand(T connectionString)
            {
                _connectionString = connectionString;
            }

            public override bool IsReadRequest => false;

            public override HttpRequestMessage CreateRequest(JsonOperationContext ctx, ServerNode node, out string url)
            {
                url = $"{node.Url}/databases/{node.Database}/admin/connection-strings?connectionString={Uri.EscapeDataString(_connectionString.Name)}&type={_connectionString.Type}";

                var request = new HttpRequestMessage
                    {
                        Method = HttpMethod.Delete
                    };

                return request;
            }

            public override void SetResponse(JsonOperationContext context, BlittableJsonReaderObject response, bool fromCache)
            {
                if (response == null)
                    ThrowInvalidResponse();

                Result = JsonDeserializationClient.RemoveConnectionStringResult(response);
            }

            public string RaftUniqueRequestId { get; } = RaftIdGenerator.NewId();
        }
    }

    public sealed class RemoveConnectionStringResult
    {
        public long RaftCommandIndex { get; set; }
    }
}
