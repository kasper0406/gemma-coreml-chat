/// The per-step inputs the host computes for the layer-chunk functions.
///
/// The exported functions take no position (`gemma_chat/decode_coreml.py`,
/// "Host inputs"): everything a position used to derive in the graph — integer
/// and fp32 arithmetic the Neural Engine cannot run — is computed here for each
/// step of `L` tokens at positions `p0 ..< p0 + L`, and passed in as fp16:
///
/// - `rope_sliding` / `rope_global` `[1, L, 1, 2·half]`: `cos | sin` of each
///   row's RoPE angles. The angle is `Float(p) / timescale[i]` in fp32, with
///   `timescale[i]` the fp32 rounding of `base^(2i / headDim)`; its cosine and
///   sine are taken in fp64 and rounded once to fp16.
/// - `mask_sliding` `[1, 1, L, R]` / `mask_global` `[1, 1, L, N]`: additive,
///   0 where row `q` may attend — a ring slot holding position `p` iff
///   `q − W < p ≤ q`, a global slot `s` iff `s ≤ q` — and ``maskValue``
///   elsewhere.
/// - `write_sliding` `[1, R, L]` / `write_global` `[1, N, L]`: 1 where cache row
///   `slot` takes the step's row `l` — `p % R` in the ring, `p` in a global cache
///   (when it fits) — else 0.
///
/// `decode_coreml.host_inputs` is the reference this matches. A chunk takes the
/// subset its layers need; every buffer here is one the chunks read in place.

import Accelerate
import CoreML
import Foundation

enum HostInputs {
    static let ropeSliding = "rope_sliding"
    static let ropeGlobal = "rope_global"
    static let maskSliding = "mask_sliding"
    static let maskGlobal = "mask_global"
    static let writeSliding = "write_sliding"
    static let writeGlobal = "write_global"
    static let all: Set<String> = [
        ropeSliding, ropeGlobal, maskSliding, maskGlobal, writeSliding, writeGlobal,
    ]

    /// The additive mask of a slot a query may not see (`decode_coreml.MASK_VALUE`).
    static let maskValue = Float16(-10_000)

    /// The dimension of `name` that is the global cache length `N`, if any.
    static func cacheLengthDim(_ name: String) -> Int? {
        switch name {
        case maskGlobal: 3
        case writeGlobal: 1
        default: nil
        }
    }

    /// Per-pair RoPE timescales, `fp32(base^(2i / headDim))` for `i < half`.
    static func timescales(for name: String, half: Int) -> [Float] {
        let (base, headDim) = name == ropeSliding
            ? (GemmaConfig.slidingRopeBase, GemmaConfig.slidingHeadDim)
            : (GemmaConfig.globalRopeBase, GemmaConfig.globalHeadDim)
        return (0..<half).map { Float(pow(base, Double(2 * $0) / Double(headDim))) }
    }

    /// `[1, L, 1, 2·half]`: `cos | sin` of positions `start ..< start + L`.
    static func fillRope(_ array: MLMultiArray, start: Int, timescale: [Float]) throws {
        let rows = array.shape[1].intValue
        let half = timescale.count
        try check(array, count: rows * 2 * half, "RoPE rows")
        var angles = [Double](repeating: 0, count: rows * half)
        for r in 0..<rows {
            let p = Float(start + r)
            for i in 0..<half { angles[r * half + i] = Double(p / timescale[i]) }
        }
        var sines = angles, cosines = angles
        var n = Int32(angles.count)
        vvsincos(&sines, &cosines, angles, &n)
        array.withUnsafeMutableBufferPointer(ofType: Float16.self) { out, _ in
            for r in 0..<rows {
                for i in 0..<half {
                    out[r * 2 * half + i] = Float16(cosines[r * half + i])
                    out[r * 2 * half + half + i] = Float16(sines[r * half + i])
                }
            }
        }
    }

    /// `[1, 1, L, R]`: the ring slots each of positions `start ..< start + L`
    /// sees, from the ring *after* those positions were marked.
    static func fillSlidingMask(_ array: MLMultiArray, start: Int, ring: [Int32]) throws {
        let rows = array.shape[2].intValue
        try check(array, count: rows * ring.count, "sliding mask")
        let window = GemmaConfig.slidingWindow
        array.withUnsafeMutableBufferPointer(ofType: Float16.self) { out, _ in
            for r in 0..<rows {
                let q = start + r
                for (slot, p) in ring.enumerated() {
                    let p = Int(p)
                    out[r * ring.count + slot] = p >= 0 && p <= q && p > q - window ? 0 : maskValue
                }
            }
        }
    }

    /// `[1, 1, L, N]`: global slots `0 ... q` for each row `q`.
    static func fillGlobalMask(_ array: MLMultiArray, start: Int) throws {
        let rows = array.shape[2].intValue, slots = array.shape[3].intValue
        try check(array, count: rows * slots, "global mask")
        array.withUnsafeMutableBufferPointer(ofType: Float16.self) { out, _ in
            for r in 0..<rows {
                let visible = min(max(start + r + 1, 0), slots)
                let row = UnsafeMutableBufferPointer(rebasing: out[(r * slots)..<((r + 1) * slots)])
                UnsafeMutableBufferPointer(rebasing: row[..<visible]).update(repeating: 0)
                UnsafeMutableBufferPointer(rebasing: row[visible...]).update(repeating: maskValue)
            }
        }
    }

    /// `[1, S, L]`: the cache row each of positions `start ..< start + L` is
    /// written to — `p % S` for the ring (`wraps`), `p` for a global cache,
    /// where a position past the end takes no row.
    static func fillWrite(_ array: MLMultiArray, start: Int, wraps: Bool) throws {
        let slots = array.shape[1].intValue, rows = array.shape[2].intValue
        try check(array, count: slots * rows, "write selection")
        array.withUnsafeMutableBufferPointer(ofType: Float16.self) { out, _ in
            out.update(repeating: 0)
            for r in 0..<rows {
                let p = start + r
                let slot = wraps ? p % slots : p
                if slot < slots { out[slot * rows + r] = 1 }
            }
        }
    }

    private static func check(_ array: MLMultiArray, count: Int, _ what: String) throws {
        guard array.dataType == .float16, array.count == count else {
            throw KVCacheError.unexpectedBufferLayout(
                "\(what) buffer \(array.shape) (dtype \(array.dataType.rawValue)) does not hold \(count) fp16 values"
            )
        }
        try PredictionBuffer.requireTightlyPacked(array, what: what)
    }
}
