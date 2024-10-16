using System;

namespace Raven.Client.Exceptions.Documents.Subscriptions
{
    public sealed class SubscriptionInvalidStateException : SubscriptionException
    {
        public SubscriptionInvalidStateException(string message) : base(message)
        {
        }

        public SubscriptionInvalidStateException(string message, Exception inner) : base(message, inner)
        {
        }
    }
}