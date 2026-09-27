/// Swift tokenizer wrapper using swift-transformers.
///
/// Loads tokenizer files from a directory, from inside a .mlpackage, or
/// auto-downloads from HuggingFace Hub.
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

    /// Load tokenizer embedded inside a .mlpackage or .mlmodelc directory.
    ///
    /// Looks for a `Tokenizer/` subdirectory containing `tokenizer.json` and
    /// `tokenizer_config.json` inside the model package.
    /// The export script (`gemma-export`) embeds these automatically.
    public init(fromModelPackage packageURL: URL) async throws {
        let tokDir = packageURL.appendingPathComponent("Tokenizer")
        let tokFile = tokDir.appendingPathComponent("tokenizer.json")
        guard FileManager.default.fileExists(atPath: tokFile.path) else {
            throw GemmaTokenizerError.missingResources
        }
        self.tokenizer = try await AutoTokenizer.from(modelFolder: tokDir)
    }

    /// Encode text to token IDs.
    public func encode(_ text: String) -> [Int] {
        tokenizer.encode(text: text)
    }

    /// Decode token IDs to text.
    public func decode(_ ids: [Int]) -> String {
        tokenizer.decode(tokens: ids, skipSpecialTokens: true)
    }

    /// Whether `id` is a byte-fallback token (`<0xNN>`): one UTF-8 byte of a
    /// character the vocabulary has no token for. Same test as the
    /// tokenizer's `ByteFallback` decoder.
    func isByteToken(_ id: Int) -> Bool {
        guard let piece = tokenizer.convertIdToToken(id) else { return false }
        return piece.count == 6 && piece.hasPrefix("<0x") && piece.hasSuffix(">")
    }

    /// Tokenize a conversation with Gemma4's chat template, dropping its
    /// oldest turns while the prompt is longer than `budget` tokens.
    ///
    /// Turns go whole, from the front, and the kept history always starts at
    /// a user message; the system prompt and the newest user message are
    /// always kept, so the result can still exceed `budget` — the engine
    /// refuses a prompt that does not fit the context at all
    /// (``InferenceError/promptTooLong(tokens:limit:)``). Cutting tokens off
    /// the front instead would drop the template's framing (`<bos>`, the
    /// first `<|turn>user`) and the model would answer nonsense.
    public func encodeChatPrompt(
        history: [ChatMessage],
        systemPrompt: String? = nil,
        budget: Int
    ) -> [Int] {
        var start = history.startIndex
        var ids = encodeChatPrompt(history: history[start...], systemPrompt: systemPrompt)
        while ids.count > budget,
              let next = history[(start + 1)...].firstIndex(where: { $0.role == .user }) {
            start = next
            ids = encodeChatPrompt(history: history[start...], systemPrompt: systemPrompt)
        }
        return ids
    }

    /// Token IDs of `history` in Gemma4's chat template (`<|turn>` /
    /// `<turn|>` markers), no intermediate string.
    private func encodeChatPrompt(
        history: ArraySlice<ChatMessage>,
        systemPrompt: String?
    ) -> [Int] {
        var messages: [Message] = []
        if let sys = systemPrompt {
            messages.append(["role": "system", "content": sys])
        }
        for msg in history {
            // Gemma's chat template hardcodes `<|turn>model\n` for the generation
            // prompt (turn 1) but renders assistant messages as `<|turn>{role}\n`.
            // The template's role mapping (`assistant` → `model`) is not evaluated
            // by swift-transformers' Jinja, so we pre-map here to keep the rendered
            // prefix stable across turns — otherwise KV reuse breaks at the role
            // token (turn 1 sees `model`, turn 2 sees `assistant`).
            let role = msg.role == .assistant ? "model" : "user"
            messages.append(["role": role, "content": msg.content])
        }
        var ids: [Int]
        do {
            ids = try tokenizer.applyChatTemplate(messages: messages)
        } catch {
            // Fallback: manually construct template and encode
            ids = encode(manualChatTemplate(messages: messages))
        }
        // Ensure BOS token is present — swift-transformers may not add it
        if ids.first != GemmaConfig.bosTokenID {
            ids.insert(GemmaConfig.bosTokenID, at: 0)
        }
        return ids
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
