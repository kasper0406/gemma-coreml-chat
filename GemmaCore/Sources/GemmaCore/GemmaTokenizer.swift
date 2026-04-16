/// Swift tokenizer wrapper using swift-transformers.
///
/// Loads tokenizer files from a directory or auto-downloads from HuggingFace Hub.
/// Provides encode/decode and Gemma4 chat template formatting.

import Foundation
import Tokenizers

public final class GemmaTokenizer: @unchecked Sendable {
    private let tokenizer: any Tokenizer

    /// Load tokenizer from a local directory containing tokenizer.json + tokenizer_config.json.
    public init(from directory: URL) async throws {
        self.tokenizer = try await AutoTokenizer.from(modelFolder: directory)
    }

    /// Load tokenizer from a HuggingFace model ID (downloads on first use).
    public init(pretrained modelID: String) async throws {
        self.tokenizer = try await AutoTokenizer.from(pretrained: modelID)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token IDs to text.
    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }

    /// Tokenize conversation history using Gemma4's chat template.
    ///
    /// Returns token IDs directly (no intermediate string).
    /// Uses the tokenizer's built-in chat template with `<|turn>` / `<turn|>` markers.
    public func encodeChatPrompt(
        history: [ChatMessage],
        systemPrompt: String? = nil
    ) -> [Int] {
        var messages: [Message] = []
        if let sys = systemPrompt {
            messages.append(["role": "system", "content": sys])
        }
        for msg in history {
            let role = msg.role == .assistant ? "assistant" : "user"
            messages.append(["role": role, "content": msg.content])
        }
        do {
            return try tokenizer.applyChatTemplate(messages: messages)
        } catch {
            // Fallback: manually construct template and encode
            return encode(manualChatTemplate(messages: messages))
        }
    }

    /// Manual fallback template for Gemma4 if applyChatTemplate fails.
    private func manualChatTemplate(messages: [Message]) -> String {
        var result = ""
        for msg in messages {
            let role = (msg["role"] as? String) ?? "user"
            let content = (msg["content"] as? String) ?? ""
            result += "<|turn>\(role)\n\(content)<turn|>\n"
        }
        result += "<|turn>model\n"
        return result
    }
}

public enum GemmaTokenizerError: Error, LocalizedError {
    case missingResources

    public var errorDescription: String? {
        switch self {
        case .missingResources:
            "tokenizer.json or tokenizer_config.json not found"
        }
    }
}
