/// Incremental detokenization of a reply as it is generated.

import Foundation

/// The text of a growing token sequence, a piece per token, such that the
/// pieces — with ``finish()``'s — concatenate to exactly the decode of the
/// whole sequence.
///
/// Each ``push(_:)`` decodes a window of the latest tokens and returns what is
/// new in it, compared by Unicode scalars. That is only valid while text
/// already returned can no longer change, so the tail that a later token still
/// could change is held back until it cannot:
///
/// - **An unfinished byte-fallback character.** A character outside the
///   vocabulary is spelled as `<0xNN>` byte tokens; until its last byte
///   arrives it decodes to nothing or to U+FFFD. The trailing U+FFFDs are
///   held while the last token with any text is a byte token — a U+FFFD
///   after any other token is a character of its own and is returned right
///   away.
/// - **Space cleanup.** The tokenizer's decode drops the space before
///   punctuation and contractions (`" ."` → `"."`, `" n't"` → `"n't"`, …; see
///   ``cleanupRules``). Cleanup only deletes spaces, and whether it deletes
///   one depends only on the few characters after it, so a later token can
///   change nothing but the text's last few scalars: a pattern it may still
///   complete (`" n"`, `" '"`, …), or a rewrite whose last character a
///   combining mark may still join, undoing it (`" 's"` cleans up to `"'s"`,
///   but not once U+0301 joins the `s`).
///   See ``unfinished`` and ``rewritten``: at most four scalars are held, and
///   none after most tokens.
///
/// The window restarts after the latest token at which the text ended in no
/// unfinished pattern — no later rewrite reaches back past it — once
/// everything before that point has been returned, so it spans a few tokens
/// (longer only across a run of tokens that each end in one, such as spaces).
///
/// Call ``finish()`` when generation ends, however it ends: it returns
/// whatever was held.
public struct TextStream {
    /// swift-transformers' `PreTrainedTokenizer.cleanUp`, applied in this
    /// order. Gemma's `tokenizer_config.json` leaves
    /// `clean_up_tokenization_spaces` at its default, which turns it on.
    static let cleanupRules = [
        (" .", "."), (" ?", "?"), (" !", "!"), (" ,", ","), (" ' ", "'"),
        (" n't", "n't"), (" 'm", "'m"), (" 's", "'s"), (" 've", "'ve"), (" 're", "'re"),
    ]

    /// How the decoded text can end in a pattern that a later token may
    /// complete, so that cleanup deletes a space in it: the patterns' proper
    /// prefixes, and where `" ' "` feeds another rule — its cleanup `"'"`,
    /// whose second space a `" ."` may still take instead; `" n ' t"`, which
    /// cleans up to `" n't"` (`" n "`, `" n '"`, and `" n'"` again); and a
    /// space before it, which it leaves in front of the `"'"` (`"  ' s"` →
    /// `" 's"` → `"'s"`: `"  "`, `"  '"`, and `" '"` again).
    private static let unfinished: [[Unicode.Scalar]] = [
        " ", " '", "'", " 'v", " 'r", " n", " n'", " n ", " n '", "  ", "  '",
    ].map { Array($0.unicodeScalars) }

    /// How it can end in a finished rewrite that a combining mark joining its
    /// last character would undo: the rules' replacements, and `" '."`,
    /// where `" ."` took the space `" ' "` needed (a mark on the `"."` gives
    /// it back, and `" ' "` deletes it together with the first).
    private static let rewritten: [[Unicode.Scalar]] = [
        ".", "?", "!", ",", "'m", "'s", "'ve", "'re", "n't", " '.", " '?", " '!", " ',",
    ].map { Array($0.unicodeScalars) }

    private let decode: ([Int]) -> String
    private let isByteToken: (Int) -> Bool
    private var tokens: [Int] = []
    /// Start of the decode window.
    private var windowStart = 0
    /// Scalars of the window's text returned so far.
    private var returned = 0
    /// Where the window restarts once everything before it is returned.
    private var nextStart: Int?
    /// Whether the last token with any text was a byte token: a later byte
    /// may still complete its character.
    private var inCharacter = false

    public init(tokenizer: GemmaTokenizer) {
        self.init(decode: tokenizer.decode, isByteToken: tokenizer.isByteToken)
    }

    init(decode: @escaping ([Int]) -> String, isByteToken: @escaping (Int) -> Bool) {
        self.decode = decode
        self.isByteToken = isByteToken
    }

    /// Append `token`; returns the text that is now final (often empty).
    public mutating func push(_ token: Int32) -> String {
        let id = Int(token)
        tokens.append(id)
        // A token that decodes to nothing on its own (a special token, or a
        // byte the decoder has not rendered yet) ends no text.
        var endsText = false
        if isByteToken(id) {
            inCharacter = true
        } else if !decode([id]).isEmpty {
            inCharacter = false
            endsText = true
        }
        let text = decode(Array(tokens[windowStart...])).unicodeScalars
        let piece = take(upTo: text.count - heldBack(text), of: text)

        // A token after which nothing is unfinished ends a character, and no
        // pattern reaches back across it: the text after it decodes the same
        // on its own.
        var length = text.count
        if let start = nextStart {
            length = restartWindow(at: start, length: length)
        }
        if nextStart == nil, endsText, Self.longest(Self.unfinished, endingOf: text) == 0 {
            nextStart = tokens.count
            restartWindow(at: tokens.count, length: length)
        }
        return piece
    }

    /// The text still held back. Call once, when generation ends.
    public mutating func finish() -> String {
        let text = decode(Array(tokens[windowStart...])).unicodeScalars
        return take(upTo: text.count, of: text)
    }

    /// Returns `text` from what was returned up to `end`.
    private mutating func take(upTo end: Int, of text: String.UnicodeScalarView) -> String {
        guard end > returned else { return "" }
        defer { returned = end }
        return String(String.UnicodeScalarView(text.dropFirst(returned).prefix(end - returned)))
    }

    /// Moves the window's start to `start` if everything before it was
    /// returned; returns the window's text length after that.
    @discardableResult
    private mutating func restartWindow(at start: Int, length: Int) -> Int {
        let before = length - decode(Array(tokens[start...])).unicodeScalars.count
        guard returned >= before else { return length }
        windowStart = start
        returned -= before
        nextStart = nil
        return length - before
    }

    /// How many trailing scalars of `text` a later token could still change.
    private func heldBack(_ text: String.UnicodeScalarView) -> Int {
        let bytes = inCharacter ? text.reversed().prefix { $0 == "\u{FFFD}" }.count : 0
        return bytes + Self.longest(Self.unfinished + Self.rewritten, endingOf: text.dropLast(bytes))
    }

    /// The length of the longest of `tails` that `text` ends with (0 if none).
    private static func longest(
        _ tails: [[Unicode.Scalar]], endingOf text: some BidirectionalCollection<Unicode.Scalar>
    ) -> Int {
        tails.filter { text.reversed().starts(with: $0.reversed()) }.map(\.count).max() ?? 0
    }
}
