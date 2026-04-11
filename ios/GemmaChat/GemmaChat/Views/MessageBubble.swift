/// Chat message bubble view.

import SwiftUI

struct MessageBubble: View {
    let message: ChatMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            VStack(alignment: message.role == .user ? .trailing : .leading, spacing: 4) {
                if message.role == .system {
                    Text(message.content)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .italic()
                } else {
                    Text(message.content)
                        .textSelection(.enabled)
                }
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 10)
            .background(backgroundColor)
            .foregroundStyle(foregroundColor)
            .clipShape(RoundedRectangle(cornerRadius: 16))

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }

    private var backgroundColor: Color {
        switch message.role {
        case .user: .blue
        case .assistant: Color(.systemGray5)
        case .system: Color(.systemGray6)
        }
    }

    private var foregroundColor: Color {
        switch message.role {
        case .user: .white
        default: .primary
        }
    }
}
