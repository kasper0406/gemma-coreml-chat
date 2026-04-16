/// Chat data model.

import Foundation

/// Role of a chat message participant.
public enum MessageRole: String, Codable, Sendable {
    case user
    case assistant
    case system
}

/// A single chat message.
public struct ChatMessage: Identifiable, Sendable {
    public let id: UUID
    public let role: MessageRole
    public let content: String
    public let timestamp: Date

    public init(
        id: UUID = UUID(),
        role: MessageRole,
        content: String,
        timestamp: Date = .now
    ) {
        self.id = id
        self.role = role
        self.content = content
        self.timestamp = timestamp
    }
}
