/// `GemmaTokenizer.encodeChatPrompt(history:systemPrompt:budget:)` against the
/// model's own tokenizer.

import Foundation
import XCTest

@testable import GemmaCore

/// The tokenizer of the repository's exported package (`gemma-export` writes
/// it to the repository root); skips the test when there is none.
func exportedTokenizer() async throws -> GemmaTokenizer {
    let root = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()
    let package = root.appendingPathComponent("gemma4-e2b.mlpackage")
    guard FileManager.default.fileExists(atPath: package.appendingPathComponent("Tokenizer").path) else {
        throw XCTSkip("no exported gemma4-e2b.mlpackage at \(root.path)")
    }
    return try await GemmaTokenizer(fromModelPackage: package)
}

final class ChatPromptTests: XCTestCase {
    /// A conversation over its budget loses whole turns from the front and
    /// keeps the template's framing; the newest message always stays.
    func testOldTurnsAreDroppedWholeAndTheFramingKept() async throws {
        let tokenizer = try await exportedTokenizer()
        let long = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 40)
        let history = [
            ChatMessage(role: .user, content: "First question. " + long),
            ChatMessage(role: .assistant, content: "First answer. " + long),
            ChatMessage(role: .user, content: "Second question."),
            ChatMessage(role: .assistant, content: "Second answer."),
            ChatMessage(role: .user, content: "Third question?"),
        ]
        let full = tokenizer.encodeChatPrompt(history: history, budget: .max)
        let recent = tokenizer.encodeChatPrompt(history: Array(history[2...]), budget: .max)
        let newest = tokenizer.encodeChatPrompt(history: Array(history[4...]), budget: .max)
        XCTAssertGreaterThan(full.count, recent.count + 500)

        XCTAssertEqual(tokenizer.encodeChatPrompt(history: history, budget: full.count), full)
        XCTAssertEqual(tokenizer.encodeChatPrompt(history: history, budget: recent.count), recent)
        XCTAssertEqual(tokenizer.encodeChatPrompt(history: history, budget: recent.count - 1), newest)
        XCTAssertEqual(tokenizer.encodeChatPrompt(history: history, budget: 1), newest)
        for ids in [full, recent, newest] {
            XCTAssertEqual(Array(ids.prefix(4)), Array(full.prefix(4)), "<bos><|turn>user\\n")
            XCTAssertEqual(Array(ids.suffix(8)), Array(full.suffix(8)))
        }
    }

    /// With nothing left to drop, a prompt over its budget comes back whole
    /// (the engine then refuses it) instead of trapping.
    func testAPromptThatCannotShrinkComesBackWhole() async throws {
        let tokenizer = try await exportedTokenizer()
        let long = String(repeating: "The quick brown fox jumps over the lazy dog. ", count: 40)

        let empty = tokenizer.encodeChatPrompt(history: [], budget: .max)
        XCTAssertEqual(tokenizer.encodeChatPrompt(history: [], budget: 0), empty)

        let system = tokenizer.encodeChatPrompt(history: [], systemPrompt: long, budget: .max)
        XCTAssertGreaterThan(system.count, 400)
        XCTAssertEqual(tokenizer.encodeChatPrompt(history: [], systemPrompt: long, budget: 10), system)

        let huge = [ChatMessage(role: .user, content: long)]
        let message = tokenizer.encodeChatPrompt(history: huge, budget: .max)
        XCTAssertGreaterThan(message.count, 400)
        XCTAssertEqual(tokenizer.encodeChatPrompt(history: huge, budget: 10), message)
        let withSystem = tokenizer.encodeChatPrompt(history: huge, systemPrompt: long, budget: .max)
        XCTAssertEqual(tokenizer.encodeChatPrompt(history: huge, systemPrompt: long, budget: 10), withSystem)
    }
}
