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

    /// A manifest whose dimensions are malformed, overflow, or disagree with
    /// the files must throw — never trap or read out of bounds.
    func testMalformedManifestThrows() throws {
        let fixtures = try XCTUnwrap(Bundle.module.url(forResource: "Fixtures", withExtension: nil))
        let source = fixtures.appendingPathComponent(HostEmbeddings.directoryName)
        let good = try JSONSerialization.jsonObject(
            with: Data(contentsOf: source.appendingPathComponent("embeddings.json"))
        ) as! [String: [String: Any]]

        let cases: [(String, [String: Any])] = [
            ("rows overflow", ["rows": Int.max]),
            ("rows × cols overflow", ["rows": Int.max / 64 + 1]),
            ("zero rows", ["rows": 0]),
            ("negative rows", ["rows": -8]),
            ("negative cols", ["cols": -64]),
            ("cols not a multiple of the group", ["cols": 48]),
            ("odd cols", ["cols": 63]),
            ("unsupported group size", ["group_size": 16]),
            ("rows disagree with the files", ["rows": 9]),
            ("cols disagree with the files", ["cols": 96]),
        ]
        for (label, override) in cases {
            let package = FileManager.default.temporaryDirectory
                .appendingPathComponent("HostEmbeddingsTests-\(UUID().uuidString)")
            defer { try? FileManager.default.removeItem(at: package) }
            let dir = package.appendingPathComponent(HostEmbeddings.directoryName)
            try FileManager.default.createDirectory(at: package, withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: source, to: dir)
            var manifest = good
            manifest[HostEmbeddings.tokenInputName]!.merge(override) { _, new in new }
            try JSONSerialization.data(withJSONObject: manifest)
                .write(to: dir.appendingPathComponent("embeddings.json"))

            XCTAssertThrowsError(try HostEmbeddings(packageURL: package), label) { error in
                guard case HostEmbeddingsError.malformed = error else {
                    return XCTFail("\(label): unexpected \(error)")
                }
            }
        }
    }
}
