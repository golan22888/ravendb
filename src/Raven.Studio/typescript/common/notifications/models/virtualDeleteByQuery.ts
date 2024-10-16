/// <reference path="../../../../typings/tsd.d.ts" />
import database = require("models/resources/database");
import pluralizeHelpers = require("common/helpers/text/pluralizeHelpers");
import moment = require("moment");
import groupedVirtualNotification from "common/notifications/models/groupedVirtualNotification";

class virtualDeleteByQuery extends groupedVirtualNotification<queryBasedVirtualBulkOperationItem> {
    
    static readonly Id = "virtual$$deleteByQuery";
    
    constructor(db: database) {
        super(db, {
            Id: virtualDeleteByQuery.Id,
            IsPersistent: false,
            Type: "CumulativeDeleteByQuery",
            Database: db.name,
            
            // properties below will be initialized later
            Message: null,
            CreatedAt: null,
            Title: null,
            Severity: null,
        });
        
        this.title("Delete by query");
        this.severity("Success");
    }
    
    merge(dto: Raven.Server.NotificationCenter.Notifications.OperationChanged) {
        this.createdAt(dto.CreatedAt ? moment.utc(dto.CreatedAt) : null);
        
        const existingItemIndex = this.operations().findIndex(x => x.id === dto.Id);
        
        const bulkResult = dto.State.Result as Raven.Client.Documents.Operations.BulkOperationResult;
        
        const query = dto.TaskType === "DeleteByQuery" 
            ? (dto.DetailedDescription as Raven.Client.Documents.Operations.BulkOperationResult.OperationDetails).Query
            : "n/a";
        const indexOrCollection = dto.TaskType === "DeleteByQuery"
            ? dto.Message
            : "dynamic/" + dto.Message;
        
        const item = {
            id: dto.Id,
            date: dto.StartTime,
            duration: moment.utc(dto.EndTime).diff(moment.utc(dto.StartTime)),
            totalItemsProcessed: bulkResult.Total,
            query: query,
            indexOrCollectionUsed: indexOrCollection
        } as queryBasedVirtualBulkOperationItem;
        
        if (existingItemIndex !== -1) {
            this.operations.splice(existingItemIndex, 1, item);
        } else {
            this.operations.unshift(item);
        }
        
        this.message(pluralizeHelpers.pluralize(this.operations().length, "operation", "operations")
            + " has been completed successfully. ");
    }
}

export = virtualDeleteByQuery;
