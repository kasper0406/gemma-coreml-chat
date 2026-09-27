/// Incremental detokenization of a reply as it is generated.

import Foundation

/// The text of a growing token sequence, a piece per token, such that the
/// pieces — with ``finish()``'s — concatenate to exactly the decode of the
/// whole sequence.
///
/// Each ``push(_:)`` decodes a short window (the last settled stretch of
/// tokens, as context, plus everything after it) and returns what is new in
/// it, compared by Unicode scalars, so the work per token stays constant
/// however long the reply gets. That is only valid while text already
/// returned can no longer change, so the tail that a later token still could
/// change is held back until one does not.
///
/// - **An unfinished byte-fallback character.** A character outside the
///   vocabulary is spelled as `<0xNN>` byte tokens; until its last byte
///   arrives it decodes to nothing or to U+FFFD. The trailing U+FFFDs are
///   held while the last token with any text is a byte token — a U+FFFD
///   after any other token is a character of its own and is returned right
///   away.
/// Call ``finish()`` when generation ends, however it ends: it returns
/// whatever was held.
public struct TextStream {
    private let decode: ([Int]) -> String
    private let isByteToken: (Int) -> Bool
    private var tokens: [Int] = []
    /// Start of the decode window.
    private var windowStart = 0
    /// End of the window's settled stretch: its text has all been returned,
    /// and no later token can change it.
    private var settled = 0
    /// Scalars returned past the settled stretch's text.
    private var returnedPastSettled = 0
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
        // byte the decoder has not rendered yet) settles nothing.
        var hasText = false
        if isByteToken(id) {
            inCharacter = true
        } else if !decode([id]).isEmpty {
            inCharacter = false
            hasText = true
        }
        let (piece, held) = emit(holding: true)
        // Settle only when nothing is held and a later byte cannot join the
        // text, so that the next window starts on a character boundary.
        if held == 0, hasText {
            windowStart = settled
            settled = tokens.count
            returnedPastSettled = 0
        }
        return piece
    }

    /// The text still held back. Call once, when generation ends.
    public mutating func finish() -> String {
        emit(holding: false).piece
    }

    private mutating func emit(holding: Bool) -> (piece: String, held: Int) {
        let returned = decode(Array(tokens[windowStart..<settled])).unicodeScalars.count
            + returnedPastSettled
        let text = decode(Array(tokens[windowStart...])).unicodeScalars
        let held = holding ? heldBack(text) : 0
        let end = text.count - held
        guard end > returned else { return ("", held) }
        returnedPastSettled += end - returned
        return (String(String.UnicodeScalarView(text.dropFirst(returned).prefix(end - returned))), held)
    }

    /// How many trailing scalars of `text` a later token could still change.
    private func heldBack(_ text: String.UnicodeScalarView) -> Int {
        inCharacter ? text.reversed().prefix { $0 == "\u{FFFD}" }.count : 0
    }
}
