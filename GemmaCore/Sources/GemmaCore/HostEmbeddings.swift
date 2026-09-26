/// Host-side embedding lookups for the exported model.
///
/// The exported functions take embedding *rows*, not token ids: `token_embed`
/// `[1, L, embed_dim]` and `ple_rows` `[1, L, num_layers × per_layer_dim]`,
/// both fp16. The exporter ships the two int4 tables they come from in the
/// package's `Embeddings/` directory (see `gemma_chat/host_embeddings.py` for
/// the format), and this type memory-maps them and dequantizes one row per
/// token — exactly the values the old in-graph `gather` produced:
///
/// - element `c` of row `r` is nibble `c % 2` of byte `r * cols / 2 + c / 2`
///   (even columns low, odd columns high), a 4-bit two's-complement integer `q`;
/// - its value is `fp16(q × scale[r, c / 32])` — exact in fp32, so one rounding;
/// - then `fp16(value × multiplier)` — `fp16(√embed_dim)` for `token_embed`,
///   1 for `ple_rows`.
///
/// The tables are memory-mapped, so they cost page cache, not resident memory.

import CoreML
import Foundation

/// One int4 block-32 table, memory-mapped.
struct EmbeddingTable {
    let rows: Int
    let cols: Int
    /// fp16 factor applied after dequantization.
    let multiplier: Float16
    private let data: Data
    private let scales: Data

    static let groupSize = 32

    init(directory: URL, name: String, rows: Int, cols: Int, groupSize: Int, multiplier: Float16) throws {
        guard groupSize == Self.groupSize, cols % groupSize == 0, rows > 0 else {
            throw HostEmbeddingsError.malformed(
                "\(name): rows=\(rows) cols=\(cols) group_size=\(groupSize); expected cols to be a multiple of \(Self.groupSize)"
            )
        }
        self.rows = rows
        self.cols = cols
        self.multiplier = multiplier
        self.data = try Data(
            contentsOf: directory.appendingPathComponent("\(name).int4"), options: .alwaysMapped
        )
        self.scales = try Data(
            contentsOf: directory.appendingPathComponent("\(name).scales"), options: .alwaysMapped
        )
        guard data.count == rows * cols / 2,
              scales.count == rows * (cols / groupSize) * MemoryLayout<Float16>.size else {
            throw HostEmbeddingsError.malformed(
                "\(name): \(data.count) data bytes / \(scales.count) scale bytes do not match [\(rows), \(cols)]"
            )
        }
        // Start paging the tables in now, behind the model load. A row read
        // from a cold page costs a page fault — ~0.3 ms per token when a
        // prompt's rows all miss — where the old in-graph tables were read in
        // whole at load. Advisory and asynchronous; a failure costs nothing.
        for mapped in [data, scales] {
            mapped.withUnsafeBytes { raw in
                _ = madvise(UnsafeMutableRawPointer(mutating: raw.baseAddress), raw.count, MADV_WILLNEED)
            }
        }
    }

    /// Dequantize row `token` into `out[0 ..< cols]`.
    func writeRow(_ token: Int32, into out: UnsafeMutablePointer<Float16>) throws {
        let row = Int(token)
        guard row >= 0, row < rows else {
            throw HostEmbeddingsError.tokenOutOfRange(token: token, rows: rows)
        }
        let groups = cols / Self.groupSize
        let bytesPerGroup = Self.groupSize / 2
        let mult = Float(multiplier)
        let scaled = multiplier != 1
        data.withUnsafeBytes { (packed: UnsafeRawBufferPointer) in
            scales.withUnsafeBytes { (scaleBytes: UnsafeRawBufferPointer) in
                let q = packed.baseAddress!.assumingMemoryBound(to: UInt8.self)
                    + row * (cols / 2)
                let s = scaleBytes.baseAddress!.assumingMemoryBound(to: Float16.self)
                    + row * groups
                for g in 0..<groups {
                    let scale = Float(s[g])
                    let src = q + g * bytesPerGroup
                    let dst = out + g * Self.groupSize
                    for j in 0..<bytesPerGroup {
                        let byte = src[j]
                        // Sign-extend each nibble: shift it to the top of an
                        // Int8, then arithmetic-shift back down.
                        let lo = Int8(bitPattern: byte << 4) >> 4
                        let hi = Int8(bitPattern: byte) >> 4
                        var v0 = Float16(Float(lo) * scale)
                        var v1 = Float16(Float(hi) * scale)
                        if scaled {
                            v0 = Float16(Float(v0) * mult)
                            v1 = Float16(Float(v1) * mult)
                        }
                        dst[2 * j] = v0
                        dst[2 * j + 1] = v1
                    }
                }
            }
        }
    }
}

/// The two tables the exported functions' embedding inputs are fed from.
struct HostEmbeddings: Sendable {
    /// Feeds the `token_embed` input.
    let token: EmbeddingTable
    /// Feeds the `ple_rows` input.
    let perLayer: EmbeddingTable

    static let directoryName = "Embeddings"
    static let tokenInputName = "token_embed"
    static let perLayerInputName = "ple_rows"

    private struct ManifestEntry: Decodable {
        let rows: Int
        let cols: Int
        let group_size: Int
        let multiplier: Double
    }

    /// Load the tables shipped inside `packageURL` (a `.mlpackage`, or a
    /// `.mlmodelc` someone copied `Embeddings/` into).
    init(packageURL: URL) throws {
        let dir = packageURL.appendingPathComponent(Self.directoryName)
        let manifestURL = dir.appendingPathComponent("embeddings.json")
        guard FileManager.default.fileExists(atPath: manifestURL.path) else {
            throw HostEmbeddingsError.missing(packageURL.lastPathComponent)
        }
        let manifest = try JSONDecoder().decode(
            [String: ManifestEntry].self, from: Data(contentsOf: manifestURL)
        )
        func table(_ name: String) throws -> EmbeddingTable {
            guard let e = manifest[name] else {
                throw HostEmbeddingsError.malformed("embeddings.json has no '\(name)' table")
            }
            return try EmbeddingTable(
                directory: dir, name: name, rows: e.rows, cols: e.cols,
                groupSize: e.group_size, multiplier: Float16(e.multiplier)
            )
        }
        self.token = try table(Self.tokenInputName)
        self.perLayer = try table(Self.perLayerInputName)
    }

    /// Fill `tokenEmbed` `[1, L, cols]` and `pleRows` `[1, L, cols]` with the
    /// rows of `tokens` (`tokens.count == L`).
    func fill(tokens: some Collection<Int32>, tokenEmbed: MLMultiArray, pleRows: MLMultiArray) throws {
        try Self.fill(table: token, tokens: tokens, into: tokenEmbed)
        try Self.fill(table: perLayer, tokens: tokens, into: pleRows)
    }

    private static func fill(
        table: EmbeddingTable, tokens: some Collection<Int32>, into array: MLMultiArray
    ) throws {
        guard array.dataType == .float16, array.count == tokens.count * table.cols else {
            throw KVCacheError.unexpectedBufferLayout(
                "embedding input \(array.shape) (dtype \(array.dataType.rawValue)) cannot hold \(tokens.count) rows of \(table.cols)"
            )
        }
        try PredictionBuffer.requireTightlyPacked(array, what: "embedding input")
        try array.withUnsafeMutableBufferPointer(ofType: Float16.self) { ptr, _ in
            guard let base = ptr.baseAddress else { return }
            for (i, t) in tokens.enumerated() {
                try table.writeRow(t, into: base + i * table.cols)
            }
        }
    }
}

public enum HostEmbeddingsError: Error, LocalizedError {
    /// The package has no `Embeddings/` directory.
    case missing(String)
    case malformed(String)
    case tokenOutOfRange(token: Int32, rows: Int)

    public var errorDescription: String? {
        switch self {
        case .missing(let name):
            "\(name) has no Embeddings/ directory — it predates host-side embedding lookups. Re-run `uv run gemma-export`."
        case .malformed(let detail):
            "Malformed embedding tables: \(detail)"
        case .tokenOutOfRange(let token, let rows):
            "Token id \(token) is outside the \(rows)-row embedding table"
        }
    }
}
