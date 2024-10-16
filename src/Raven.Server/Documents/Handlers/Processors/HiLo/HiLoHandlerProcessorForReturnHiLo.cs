﻿using System.Threading.Tasks;
using JetBrains.Annotations;
using Raven.Client.Documents.Identity;
using Raven.Server.Documents.TransactionMerger.Commands;
using Raven.Server.ServerWide.Context;
using Sparrow.Json;
using Sparrow.Json.Parsing;

namespace Raven.Server.Documents.Handlers.Processors.HiLo;

internal sealed class HiLoHandlerProcessorForReturnHiLo : AbstractHiLoHandlerProcessorForReturnHiLo<DatabaseRequestHandler, DocumentsOperationContext>
{
    public HiLoHandlerProcessorForReturnHiLo([NotNull] DatabaseRequestHandler requestHandler) : base(requestHandler)
    {
    }

    public override async ValueTask ExecuteAsync()
    {
        var tag = GetTag();
        var end = RequestHandler.GetLongQueryString("end");
        var last = RequestHandler.GetLongQueryString("last");

        var cmd = new MergedHiLoReturnCommand
        {
            Database = RequestHandler.Database,
            Key = tag,
            End = end,
            Last = last
        };

        await RequestHandler.Database.TxMerger.Enqueue(cmd);
    }

    internal sealed class MergedHiLoReturnCommand : DocumentMergedTransactionCommand
    {
        public string Key;
        public DocumentDatabase Database;
        public long End;
        public long Last;

        protected override long ExecuteCmd(DocumentsOperationContext context)
        {
            var hiLoDocumentId = HiLoHandler.RavenHiloIdPrefix + Key;

            var document = Database.DocumentsStorage.Get(context, hiLoDocumentId);

            if (document == null)
                return 1;

            document.Data.TryGet(nameof(HiloDocument.Max), out long oldMax);
            if (oldMax != End || Last > oldMax)
                return 1;

            document.Data.Modifications = new DynamicJsonValue
            {
                [nameof(HiloDocument.Max)] = Last
            };

            using (var hiloReader = context.ReadObject(document.Data, hiLoDocumentId, BlittableJsonDocumentBuilder.UsageMode.ToDisk))
            {
                Database.DocumentsStorage.Put(context, hiLoDocumentId, null, hiloReader);
            }

            return 1;
        }

        public override IReplayableCommandDto<DocumentsOperationContext, DocumentsTransaction, DocumentMergedTransactionCommand> ToDto(DocumentsOperationContext context)
        {
            return new MergedHiLoReturnCommandDto
            {
                Key = Key,
                End = End,
                Last = Last
            };
        }
    }

    internal sealed class MergedHiLoReturnCommandDto : IReplayableCommandDto<DocumentsOperationContext, DocumentsTransaction, MergedHiLoReturnCommand>
    {
        public string Key;
        public long End;
        public long Last;

        public MergedHiLoReturnCommand ToCommand(DocumentsOperationContext context, DocumentDatabase database)
        {
            return new MergedHiLoReturnCommand
            {
                Key = Key,
                End = End,
                Last = Last,
                Database = database
            };
        }
    }
}
