/// Chat data model shared across the app.

import Foundation

/// Role of a chat message participant.
enum MessageRole: String, Codable, Sendable {
    case user
    case assistant
    case system
}

/// A single chat message.
struct ChatMessage: Identifiable, Sendable {
    let id: UUID
    let role: MessageRole
    let content: String
    let timestamp: Date

    init(
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
