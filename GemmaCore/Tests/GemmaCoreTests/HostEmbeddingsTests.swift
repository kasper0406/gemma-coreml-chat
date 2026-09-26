/// The Swift embedding reader against the rows the old in-graph lookup
/// produced. The fixture is written by `tests/test_host_embeddings.py`, which
/// also checks it is current.

import CoreML
import Foundation
import XCTest

@testable import GemmaCore

final class HostEmbeddingsTests: XCTestCase {
    private struct Expected: Decodable {
        let tokens: [Int32]
        let token_embed: [[UInt16]]
        let ple_rows: [[UInt16]]
    }

    func testRowsMatchTheGraphLookupBitForBit() throws {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let expected = try JSONDecoder().decode(
            Expected.self, from: Data(contentsOf: fixtures.appendingPathComponent("expected.json"))
        )
        let embeddings = try HostEmbeddings(packageURL: fixtures)

        let n = expected.tokens.count
        let tokenEmbed = try MLMultiArray(
            shape: [1, NSNumber(value: n), NSNumber(value: embeddings.token.cols)], dataType: .float16
        )
        let pleRows = try MLMultiArray(
            shape: [1, NSNumber(value: n), NSNumber(value: embeddings.perLayer.cols)], dataType: .float16
        )
        try embeddings.fill(tokens: expected.tokens, tokenEmbed: tokenEmbed, pleRows: pleRows)

        for (array, rows, name) in [
            (tokenEmbed, expected.token_embed, "token_embed"),
            (pleRows, expected.ple_rows, "ple_rows"),
        ] {
            let got = array.withUnsafeBufferPointer(ofType: Float16.self) { $0.map(\.bitPattern) }
            XCTAssertEqual(got, rows.flatMap { $0 }, name)
        }
    }

    func testOutOfRangeTokenThrows() throws {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let embeddings = try HostEmbeddings(packageURL: fixtures)
        let row = try MLMultiArray(
            shape: [1, 1, NSNumber(value: embeddings.token.cols)], dataType: .float16
        )
        let pleRow = try MLMultiArray(
            shape: [1, 1, NSNumber(value: embeddings.perLayer.cols)], dataType: .float16
        )
        XCTAssertThrowsError(try embeddings.fill(
            tokens: [Int32(embeddings.token.rows)], tokenEmbed: row, pleRows: pleRow
        ))
    }
}
