/// Streamed detokenization equals the decode of the whole reply: the pieces
/// ``TextStream`` returns, plus what ``TextStream/finish()`` flushes, always
/// concatenate to it, and nothing returned is ever taken back.

import Foundation
import XCTest

@testable import GemmaCore

final class TextStreamTests: XCTestCase {
    // MARK: - A tokenizer with swift-transformers' decode semantics

    /// Gemma's decoder (`Replace("▁", " ")`, `ByteFallback`, `Fuse`) followed
    /// by `PreTrainedTokenizer.cleanUp`, over a small vocabulary. `specials`
    /// are dropped, as `decode(skipSpecialTokens: true)` drops them.
    /// `flushesTrailingBytes: false` is swift-transformers' own ByteFallback
    /// (0.1.24), which loses byte tokens at the very end of a decode; `true` is
    /// what `tokenizers` does.
    struct FakeTokenizer {
        static let words = [
            "<special>", "Hello", "▁", ".", "X", "\u{FFFD}", "▁do", "▁n", "'t", "▁'", "s",
            "▁world", ",", "'", "ve", "re", "m", "!", "?", "\n", "▁▁", "n", "▁know", "I", "▁'s",
            "\u{0301}",
        ]
        static let byteBase = words.count
        static let vocabulary = words + (0..<256).map { String(format: "<0x%02X>", $0) }
        static func byte(_ b: UInt8) -> Int { byteBase + Int(b) }
        static func id(_ word: String) -> Int { words.firstIndex(of: word)! }

        let flushesTrailingBytes: Bool

        func isByteToken(_ id: Int) -> Bool { id >= Self.byteBase }

        func decode(_ ids: [Int]) -> String {
            var text = ""
            var bytes: [UInt8] = []
            for id in ids where id != 0 {
                if isByteToken(id) {
                    bytes.append(UInt8(id - Self.byteBase))
                    continue
                }
                if !bytes.isEmpty { text += String(decoding: bytes, as: UTF8.self); bytes.removeAll() }
                text += Self.vocabulary[id].replacingOccurrences(of: "▁", with: " ")
            }
            if flushesTrailingBytes, !bytes.isEmpty { text += String(decoding: bytes, as: UTF8.self) }
            for (pattern, replacement) in TextStream.cleanupRules {
                text = text.replacingOccurrences(of: pattern, with: replacement)
            }
            return text
        }

        func stream() -> TextStream { TextStream(decode: decode, isByteToken: isByteToken) }
    }

    /// Streams `ids`, asserting that nothing returned is later contradicted,
    /// and returns the pieces (the last one from `finish()`).
    @discardableResult
    private func assertStreamsAsFullDecode(
        _ ids: [Int], decode: @escaping ([Int]) -> String, stream: TextStream,
        file: StaticString = #filePath, line: UInt = #line
    ) -> [String] {
        var stream = stream
        let full = decode(ids)
        var pieces: [String] = []
        var soFar = String.UnicodeScalarView()
        for id in ids {
            pieces.append(stream.push(Int32(id)))
            soFar.append(contentsOf: pieces.last!.unicodeScalars)
            XCTAssertTrue(full.unicodeScalars.starts(with: soFar),
                          "returned \(String(soFar).debugDescription), not a prefix of \(full.debugDescription) (ids \(ids))",
                          file: file, line: line)
        }
        pieces.append(stream.finish())
        XCTAssertEqual(pieces.joined(), full, "ids \(ids)", file: file, line: line)
        return pieces
    }

    private func check(_ ids: [Int], flushes: Bool = false) -> [String] {
        let tokenizer = FakeTokenizer(flushesTrailingBytes: flushes)
        return assertStreamsAsFullDecode(ids, decode: tokenizer.decode, stream: tokenizer.stream())
    }

    private typealias T = FakeTokenizer

    // MARK: - Cases

    func testALiteralReplacementCharacterIsReturnedRightAway() {
        let pieces = check([T.id("Hello"), T.id("\u{FFFD}")])
        XCTAssertEqual(pieces, ["Hello", "\u{FFFD}", ""])
    }

    func testCleanupIsHeldUntilTheNextTokenSettlesIt() {
        // "Hello", " ", ".", "X" decode to "Hello.X": cleanup drops the space.
        let pieces = check([T.id("Hello"), T.id("▁"), T.id("."), T.id("X")])
        XCTAssertEqual(pieces, ["Hello", "", "", ".X", ""])
    }

    func testContractionsAreCleanedUpAcrossTokens() {
        check([T.id("I"), T.id("▁do"), T.id("▁n"), T.id("'t"), T.id("▁know")])     // "I don't know"
        check([T.id("I"), T.id("▁'"), T.id("ve"), T.id("▁'"), T.id("▁world")])      // " 've", " ' "
        check([T.id("Hello"), T.id("▁'s"), T.id("▁'"), T.id("re")])
    }

    func testACombiningMarkUndoesACleanup() {
        // " 's" is cleaned up to "'s" until an accent joins the "s".
        let pieces = check([T.id("I"), T.id("▁'s"), T.id("\u{0301}"), T.id("X")])
        XCTAssertEqual(pieces.joined(), "I 's\u{0301}X")
        XCTAssertEqual(pieces[1], "")
    }

    func testHeldTextIsFlushedAtTheEnd() {
        XCTAssertEqual(check([T.id("Hello"), T.id("▁")]), ["Hello", "", " "])
        XCTAssertEqual(check([T.id("Hello"), T.id("▁n")]), ["Hello", "", " n"])
    }

    func testAByteFallbackCharacterIsReturnedWhole() {
        let euro = Array("€".utf8).map(T.byte)   // E2 82 AC
        for flushes in [false, true] {
            let pieces = check([T.id("Hello")] + euro + [T.id("X")], flushes: flushes)
            XCTAssertEqual(pieces.joined(), "Hello€X")
            XCTAssertFalse(pieces.contains { $0.unicodeScalars.contains("\u{FFFD}") })
        }
    }

    func testAnUnfinishedByteSequenceAtTheEnd() {
        let partial = Array("€".utf8).prefix(2).map(T.byte)
        XCTAssertEqual(check([T.id("Hello")] + partial, flushes: true).joined(), "Hello\u{FFFD}")
        XCTAssertEqual(check([T.id("Hello")] + partial, flushes: false).joined(), "Hello")
    }

    func testASpecialTokenInsideAByteSequence() {
        let euro = Array("€".utf8).map(T.byte)
        let ids = [T.id("Hello"), euro[0], 0, euro[1], euro[2], T.id("X")]
        XCTAssertEqual(check(ids, flushes: true).joined(), "Hello€X")
    }

    func testRandomSequences() {
        // Weighted towards what interacts: spaces, punctuation, apostrophes,
        // byte tokens (a few characters' worth, some cut short) and specials.
        var rng = SplitMix64(seed: 7)
        let chars = ["é", "€", "🦜", "日"].map { Array($0.utf8).map(T.byte) }
        for flushes in [false, true] {
            for _ in 0..<3000 {
                var ids: [Int] = []
                for _ in 0..<Int.random(in: 1...30, using: &rng) {
                    switch Int.random(in: 0..<10, using: &rng) {
                    case 0..<6: ids.append(Int.random(in: 0..<T.byteBase, using: &rng))
                    case 6..<8: ids += chars.randomElement(using: &rng)!
                    case 8: ids += chars.randomElement(using: &rng)!.prefix(Int.random(in: 1...3, using: &rng))
                    default: ids.append(T.byte(UInt8.random(in: 0...255, using: &rng)))
                    }
                }
                check(ids, flushes: flushes)
            }
        }
    }

    // MARK: - The model's own tokenizer

    func testTheModelTokenizer() async throws {
        let tokenizer = try await exportedTokenizer()
        func check(_ ids: [Int]) {
            assertStreamsAsFullDecode(ids, decode: tokenizer.decode, stream: TextStream(tokenizer: tokenizer))
        }
        check([9259, 238479])                    // "Hello" + a literal U+FFFD token
        check([9259, 236743, 236761, 236917])    // "Hello", " ", ".", "X" -> "Hello.X"
        let text = """
            Café naïve — 日本語のテキスト、한국어. Emoji: 🦜🧑‍🚀👩🏽‍💻 and e\u{301}. \
            I do n't know , he said ' yes ' . Isn't it ? We 've , they 're ! 𓀀𐎀 ꧁꧂
            """
        let encoded = tokenizer.encode(text).filter { $0 != GemmaConfig.bosTokenID }
        check(encoded)
        // Byte-fallback spellings (ids 238 + byte) mixed into real text.
        let bytes = Array("🦜€é".utf8).map { 238 + Int($0) }
        check(encoded.prefix(10) + bytes + encoded.suffix(10) + bytes.prefix(2) + [236761])
        var rng = SplitMix64(seed: 11)
        for _ in 0..<200 {
            check((0..<40).map { _ in Int.random(in: 0..<262_144, using: &rng) })
        }
    }
}

/// A seedable generator, so a failure reproduces.
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64
    init(seed: UInt64) { state = seed }
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}
