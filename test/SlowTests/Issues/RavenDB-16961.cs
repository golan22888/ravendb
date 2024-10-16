﻿using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using FastTests;
using FastTests.Utils;
using Raven.Client.Documents.Operations;
using Raven.Client.Documents.Operations.Revisions;
using Raven.Server.Documents;
using Raven.Server.ServerWide;
using Raven.Server.ServerWide.Context;
using Raven.Tests.Core.Utils.Entities;
using Tests.Infrastructure;
using Xunit;
using Xunit.Abstractions;

namespace SlowTests.Issues
{
    public class RavenDB_16961 : ReplicationTestBase
    {
        public RavenDB_16961(ITestOutputHelper output) : base(output)
        {
        }

        [RavenTheory(RavenTestCategory.Revisions)]
        [RavenData(DatabaseMode = RavenDatabaseMode.All)]
        public async Task StripRevisionFlagFromTombstone(Options options)
        {
            using (var store = GetDocumentStore(options))
            {
                await RevisionsHelper.SetupRevisionsAsync(store, configuration: new RevisionsConfiguration()
                {
                    Default = new RevisionsCollectionConfiguration()
                    {
                        Disabled = false
                    }
                });
                var user = new User() { Name = "Toli" };
                using (var session = store.OpenAsyncSession())
                {
                    for (int i = 0; i < 3; i++)
                    {
                        user.Age = i;
                        await session.StoreAsync(user, "users/1");
                        await session.SaveChangesAsync();
                    }
                    session.Delete("users/1");
                    await session.SaveChangesAsync();
                }

                await RevisionsHelper.SetupRevisionsAsync(store, configuration: new RevisionsConfiguration());

                var db = await GetDocumentDatabaseInstanceForAsync(store, options.DatabaseMode, "users/1");
                using (var token = new OperationCancelToken(db.Configuration.Databases.OperationTimeout.AsTimeSpan, db.DatabaseShutdown, CancellationToken.None))
                {
                    await db.DocumentsStorage.RevisionsStorage.EnforceConfigurationAsync(_ => { }, token);
                }

                using (db.DocumentsStorage.ContextPool.AllocateOperationContext(out DocumentsOperationContext ctx))
                using (ctx.OpenReadTransaction())
                {
                    var tombstone = db.DocumentsStorage.GetDocumentOrTombstone(ctx, "users/1");
                    Assert.False(tombstone.Tombstone.Flags.Contain(DocumentFlags.HasRevisions));
                }
            }
        }

        [RavenTheory(RavenTestCategory.Revisions | RavenTestCategory.Replication)]
        [RavenData(DatabaseMode = RavenDatabaseMode.All)]
        public async Task StripRevisionFlagFromTombstoneWithExternalReplication(Options options)
        {
            using (var store1 = GetDocumentStore(new Options(options)
            {
                ModifyDatabaseName = s => $"{s}_FooBar-1"
            }))
            using (var store2 = GetDocumentStore(new Options(options)
            {
                ModifyDatabaseName = s => $"{s}_FooBar-2"
            }))
            {
                await SetupReplicationAsync(store1, store2);
                await RevisionsHelper.SetupRevisionsAsync(store1, configuration: new RevisionsConfiguration()
                {
                    Default = new RevisionsCollectionConfiguration()
                    {
                        Disabled = false
                    }
                });
                var user = new User() { Name = "Toli" };
                using (var session = store1.OpenAsyncSession())
                {
                    for (int i = 0; i < 3; i++)
                    {
                        user.Age = i;
                        await session.StoreAsync(user, "users/1");
                        await session.SaveChangesAsync();
                    }
                    session.Delete("users/1");
                    await session.SaveChangesAsync();
                }

                await EnsureReplicatingAsync(store1, store2);

                var db2 = await GetDocumentDatabaseInstanceForAsync(store2, options.DatabaseMode, "users/1");
                var val2 = await WaitForValueAsync(() =>
                    {
                        using (db2.DocumentsStorage.ContextPool.AllocateOperationContext(out DocumentsOperationContext ctx))
                        using (ctx.OpenReadTransaction())
                        {
                            var rev = db2.DocumentsStorage.RevisionsStorage.GetRevisions(ctx, "users/1", 0, 1);
                            return rev.Count;
                        }
                    }, 4
                );

                Assert.Equal(4, val2);
                await RevisionsHelper.SetupRevisionsAsync(store1, configuration: new RevisionsConfiguration());

                var db1 = await GetDocumentDatabaseInstanceForAsync(store1, options.DatabaseMode, "users/1");
                IOperationResult enforceResult;
                using (var token = new OperationCancelToken(db1.Configuration.Databases.OperationTimeout.AsTimeSpan, db1.DatabaseShutdown, CancellationToken.None))
                    enforceResult = await db1.DocumentsStorage.RevisionsStorage.EnforceConfigurationAsync(_ => { }, token);

                var val = await WaitForValueAsync(() =>
                    {
                        using (db1.DocumentsStorage.ContextPool.AllocateOperationContext(out DocumentsOperationContext ctx))
                        using (ctx.OpenReadTransaction())
                        {
                            var tombstone = db1.DocumentsStorage.GetDocumentOrTombstone(ctx, "users/1");
                            return tombstone.Tombstone.Flags.Contain(DocumentFlags.HasRevisions);
                        }
                    }, false
                );
                Assert.False(val, AddErrorInfo(enforceResult));

                using (var session = store1.OpenAsyncSession())
                {
                    await session.StoreAsync(user, "marker");
                    await session.SaveChangesAsync();
                }

                var res = WaitForDocument(store2, "marker");
                Assert.True(res);

                val2 = await WaitForValueAsync(() =>
                    {
                        using (db2.DocumentsStorage.ContextPool.AllocateOperationContext(out DocumentsOperationContext ctx))
                        using (ctx.OpenReadTransaction())
                        {
                            var rev = db2.DocumentsStorage.RevisionsStorage.GetRevisions(ctx, "users/1", 0, 1);
                            return rev.Count;
                        }
                    }, 0
                );
                Assert.Equal(0, val2);

                val = await WaitForValueAsync(() =>
                {
                    using (db2.DocumentsStorage.ContextPool.AllocateOperationContext(out DocumentsOperationContext ctx))
                    using (ctx.OpenReadTransaction())
                    {
                        var tombstone = db2.DocumentsStorage.GetDocumentOrTombstone(ctx, "users/1");
                        return tombstone.Tombstone != null && tombstone.Tombstone.Flags.Contain(DocumentFlags.HasRevisions);
                    }
                }, false
                );
                Assert.False(val);
            }
        }

        [RavenTheory(RavenTestCategory.Revisions | RavenTestCategory.Replication)]
        [RavenData(DatabaseMode = RavenDatabaseMode.All)]
        public async Task EnforceRevisionConfigurationWithConflict(Options options)
        {
            options = UpdateConflictSolverAndGetMergedOptions(options);
            using (var store1 = GetDocumentStore(new Options(options)
            {
                ModifyDatabaseName = s => $"{s}_foo1",
            }))
            using (var store2 = GetDocumentStore(new Options(options)
            {
                ModifyDatabaseName = s => $"{s}_foo2",
            }))
            {
                await RevisionsHelper.SetupRevisionsAsync(store1, configuration: new RevisionsConfiguration()
                {
                    Default = new RevisionsCollectionConfiguration()
                    {
                        Disabled = false
                    }
                });

                using (var s1 = store1.OpenSession())
                {
                    s1.Store(new User { Name = "test" }, "users/1");
                    s1.SaveChanges();
                }

                using (var s2 = store2.OpenSession())
                {
                    s2.Store(new User { Name = "test3" }, "users/1");
                    s2.SaveChanges();
                }

                await SetupReplicationAsync(store2, store1);
                var conflicts = WaitUntilHasConflict(store1, "users/1");
                await RevisionsHelper.SetupRevisionsAsync(store1, configuration: new RevisionsConfiguration());

                var db = await GetDocumentDatabaseInstanceForAsync(store1, options.DatabaseMode, "users/1");
                using (var token = new OperationCancelToken(db.Configuration.Databases.OperationTimeout.AsTimeSpan, db.DatabaseShutdown, CancellationToken.None))
                    await db.DocumentsStorage.RevisionsStorage.EnforceConfigurationAsync(_ => { }, token);

                await UpdateConflictResolver(store1, resolveToLatest: true);

                WaitForValue(() => store1.Commands().GetConflictsFor("users/1").Length, 0);

                using (var session = store1.OpenAsyncSession())
                {
                    var revision = await session.Advanced.Revisions.GetForAsync<User>("users/1");
                    Assert.Equal(3, revision.Count);
                }
            }
        }

        private static string AddErrorInfo(IOperationResult operationResult)
        {
            var msg = new StringBuilder()
                .AppendLine("tombstone still has `HasRevisions` flag");

            if (operationResult is not EnforceConfigurationResult enforceResult)
                return msg.ToString();

            msg.AppendLine("EnforceConfiguration result :")
                .AppendLine($"\tRemovedRevisions : {enforceResult.RemovedRevisions}")
                .AppendLine($"\tScannedDocuments : {enforceResult.ScannedDocuments}")
                .AppendLine($"\tScannedRevisions : {enforceResult.ScannedRevisions}")
                .AppendLine($"\tMessage : {enforceResult.Message}")
                .AppendLine($"\tWarnings : [{string.Join(',', enforceResult.Warnings.Select(kvp => $"{kvp.Key} : {kvp.Value}"))}]");

            return msg.ToString();
        }
    }
}
