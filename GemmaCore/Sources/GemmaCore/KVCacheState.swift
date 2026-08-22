/// KV cache state for Gemma4-E2B CoreML inference.
///
/// Every KV cache is a CoreML **state** feature: the sliding-window caches and
/// the global-attention ones (`k_4`/`v_4`, `k_9`/`v_9`, `k_14`/`v_14`) alike.
/// The model reads them with `read_state` and writes them back with
/// `coreml_update_state`, so no cache tensor crosses the prediction boundary —
/// which is what makes decode cost independent of context length. The one
/// exception is `sliding_pos_ring`: CoreML states must be floating point, so
/// the int32 ring stays an ordinary input/output.
///
/// A materialized function bakes its state shapes in, so an `MLState` belongs
/// to exactly one size N: it is created from the `decode_N` / `prefill_N`
/// handle and no other pair will accept it. Growing the context therefore means
/// allocating a state on the *next* size's handle and migrating the contents —
/// see ``CoreMLModel/grownToFit(_:needed:)``.
///
/// This object also owns the per-conversation prediction scratch: the ring
/// double buffer, the reusable token/position input buffers, and the logits
/// output backings. Keeping them here rather than on ``CoreMLModel`` is what
/// lets two caches coexist (iOS runs an eager-prefill cache alongside the one
/// the current generation is decoding into) without writing over each other.
///
/// A conversation reset means a *fresh* `KVCacheState`, never a reused one with
/// a cleared ring: stale K/V left in a sliding slot becomes valid again the
/// moment a re-populated `sliding_pos_ring` points at it.

import CoreML
import CoreVideo
import Foundation

/// Errors raised while allocating, migrating, or feeding the KV cache.
public enum KVCacheError: Error, LocalizedError {
    /// A state can only be made from the function pair it belongs to, and that
    /// pair has not been loaded yet.
    case functionNotLoaded(size: Int)
    /// Could not allocate a prediction buffer of the requested shape.
    case bufferAllocationFailed(shape: [Int], dataType: MLMultiArrayDataType)
    /// An MLMultiArray had a dtype/shape/stride we can't safely memcpy.
    case unexpectedBufferLayout(String)

    public var errorDescription: String? {
        switch self {
        case .functionNotLoaded(let size):
            "No loaded function for cache size \(size) — call ensureLoaded(forGlobalCacheSize:) first"
        case .bufferAllocationFailed(let shape, let dataType):
            "Could not allocate a \(shape) buffer of dtype \(dataType.rawValue)"
        case .unexpectedBufferLayout(let reason):
            "Unexpected MLMultiArray layout: \(reason)"
        }
    }
}

/// The single source of truth for how big a global KV cache may be.
///
/// A materialized model can only run the concrete sizes it was exported with,
/// so **every** place that allocates or grows a cache has to round through this
/// one policy. Rounding independently — "next power of two" — is only
/// accidentally correct for the default contiguous power-of-two export: with
/// `--materialize-sizes 512,2048`, crossing 512 tokens grows the cache to 1024
/// while function resolution picks `decode_2048`, and every turn then fails on
/// a shape mismatch.
///
/// Vend one from ``CoreMLModel/cacheSizePolicy`` rather than constructing it ad
/// hoc: the model wrapper is what knows the sizes that were actually loaded.
public struct KVCacheSizePolicy: Sendable {
    /// Concrete materialized sizes in ascending order.
    public let materializedSizes: [Int]

    /// Largest cache size the loaded model can serve.
    public let maxLen: Int

    public init(materializedSizes: [Int], maxLen: Int) {
        self.materializedSizes = materializedSizes.sorted()
        self.maxLen = maxLen
    }

    /// Smallest runnable cache size that holds `needed` tokens.
    ///
    /// When `needed` exceeds `maxLen` the result is clamped to `maxLen`, so
    /// callers must independently cap how many token positions they feed the
    /// model — a clamped cache cannot hold every requested position.
    public func size(forNeeded needed: Int) -> Int {
        let clamped = min(max(needed, 1), maxLen)
        guard let largest = materializedSizes.last else { return clamped }
        return materializedSizes.first { $0 >= clamped } ?? largest
    }
}

/// Live KV cache for one conversation, bound to one materialized size.
///
/// Predictions mutate it in place (the `MLState` buffers by the model itself,
/// the ring by the double-buffer swap below), so callers hold one instance for
/// as long as the conversation lives rather than threading snapshots around.
public final class KVCacheState: @unchecked Sendable {
    /// Materialized cache length this state is bound to. Only the
    /// `decode_<size>` / `prefill_<size>` pair accepts it.
    public let size: Int

    /// All KV caches, sliding and global, updated in place by every prediction.
    let caches: MLState

    /// `sliding_pos_ring`, double-buffered: `ring` holds the live contents fed
    /// to the next prediction and `ringSpare` is handed to CoreML as that same
    /// prediction's output backing. One buffer cannot be both — CoreML would be
    /// reading the ring while overwriting it.
    private var ringBuffers: [MLMultiArray]
    private var ringIndex = 0

    /// Reusable int32 scalars for the token and position inputs. Allocating
    /// these per decode step showed up as pure overhead once the KV caches
    /// stopped crossing the boundary.
    let tokenScalar: MLMultiArray
    let positionScalar: MLMultiArray

    /// Reusable int32 scalar for the cache-length input `N`, on exports that
    /// still declare one. Always fed ``size``.
    let nScalar: MLMultiArray

    /// Reusable `[1, chunkSize]` int32 buffer for prefill token chunks.
    let chunkTokens: MLMultiArray

    /// Decode logits output backings, alternating so the array returned by step
    /// N survives until step N+1 has been sampled. Allocated on first decode,
    /// so a cache that only ever prefills never pays for them.
    private var decodeLogitsBackings: [MLMultiArray] = []
    private var decodeLogitsIndex = 0

    /// Prefill logits output backing (`[chunkSize, vocab]`), allocated on first
    /// prefill. Single, not double: the caller copies the one row it wants out
    /// before the next chunk runs.
    private var prefillLogitsBacking: MLMultiArray?

    /// Set once CoreML has been seen to ignore an output backing, so the
    /// diagnostic is logged a single time per conversation instead of per step.
    private var warnedAboutIgnoredBacking = false

    init(
        size: Int,
        caches: MLState,
        ringShape: [NSNumber],
        ringDataType: MLMultiArrayDataType,
        chunkSize: Int
    ) throws {
        self.size = size
        self.caches = caches
        // -1 is the "empty slot" sentinel: a zeroed ring would claim position 0
        // is live in every sliding slot.
        self.ringBuffers = [
            try PredictionBuffer.make(shape: ringShape, dataType: ringDataType, fill: -1),
            try PredictionBuffer.make(shape: ringShape, dataType: ringDataType, fill: -1),
        ]
        self.tokenScalar = try MLMultiArray(shape: [1], dataType: .int32)
        self.positionScalar = try MLMultiArray(shape: [1], dataType: .int32)
        self.nScalar = try MLMultiArray(shape: [1], dataType: .int32)
        self.chunkTokens = try MLMultiArray(
            shape: [1, NSNumber(value: chunkSize)], dataType: .int32
        )
    }

    // MARK: - Prediction scratch

    /// The ring contents to feed the next prediction.
    var ring: MLMultiArray { ringBuffers[ringIndex] }

    /// The buffer to hand CoreML as the next prediction's ring output backing.
    var ringSpare: MLMultiArray { ringBuffers[1 - ringIndex] }

    /// Take `produced` — whatever the prediction actually returned for the ring
    /// output — as the new live ring. When CoreML honoured our backing this is
    /// a pointer swap; when it allocated its own buffer we copy into the spare,
    /// because the framework's buffer may be recycled by the next prediction.
    func adoptRing(_ produced: MLMultiArray) throws {
        if produced !== ringSpare {
            noteIgnoredBacking("sliding_pos_ring")
            try PredictionBuffer.copyPrefix(from: produced, to: ringSpare, what: "sliding_pos_ring")
        }
        ringIndex = 1 - ringIndex
    }

    /// Next decode logits backing, alternating between two buffers.
    func nextDecodeLogitsBacking(
        shape: [NSNumber], dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        if decodeLogitsBackings.isEmpty {
            decodeLogitsBackings = [
                try PredictionBuffer.make(shape: shape, dataType: dataType),
                try PredictionBuffer.make(shape: shape, dataType: dataType),
            ]
        }
        decodeLogitsIndex = 1 - decodeLogitsIndex
        return decodeLogitsBackings[decodeLogitsIndex]
    }

    /// The prefill logits backing, allocated on first use.
    func prefillLogits(
        shape: [NSNumber], dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        if let existing = prefillLogitsBacking { return existing }
        let fresh = try PredictionBuffer.make(shape: shape, dataType: dataType)
        prefillLogitsBacking = fresh
        return fresh
    }

    /// Log once per conversation that CoreML declined a preallocated backing —
    /// correctness is unaffected (we copy), but every step pays an allocation.
    func noteIgnoredBacking(_ feature: String) {
        guard !warnedAboutIgnoredBacking else { return }
        warnedAboutIgnoredBacking = true
        Log.info("[CoreML] Output backing for '\(feature)' was not used by the framework — predictions will allocate their own buffers")
    }

    /// Write a scalar into one of the reusable int32 inputs.
    func setScalar(_ value: Int32, in array: MLMultiArray) {
        array.withUnsafeMutableBufferPointer(ofType: Int32.self) { ptr, _ in
            ptr[0] = value
        }
    }

    /// Fill the reusable prefill token buffer. `tokens.count` must match the
    /// model's chunk size, which the engine guarantees by padding.
    func loadChunk(_ tokens: [Int32]) throws {
        guard tokens.count == chunkTokens.count else {
            throw KVCacheError.unexpectedBufferLayout(
                "prefill chunk has \(tokens.count) tokens, model expects \(chunkTokens.count)"
            )
        }
        chunkTokens.withUnsafeMutableBufferPointer(ofType: Int32.self) { ptr, _ in
            for (i, t) in tokens.enumerated() { ptr[i] = t }
        }
    }

    // MARK: - Growth

    /// Copy every cache — and the ring — from `old` into `self`.
    ///
    /// State buffers are row-major `[1, length, …]`, so "the first N rows" is a
    /// byte prefix: one `memcpy` of `min(oldBytes, newBytes)` handles both the
    /// sliding caches (identical shape at every size, so a full copy) and the
    /// global ones (length N content landing in the first N rows of the new
    /// length-2N buffer). Rows past the copied prefix stay as CoreML made them
    /// — zeroed, and masked out until the positions they hold are written.
    func adoptContents(of old: KVCacheState, stateNames: [String]) throws {
        for name in stateNames {
            try old.caches.withMultiArray(for: name) { src in
                try caches.withMultiArray(for: name) { dst in
                    try PredictionBuffer.copyPrefix(from: src, to: dst, what: "state '\(name)'")
                }
            }
        }
        // The ring indexes sliding slots, not context positions: its shape is
        // the same at every size and it has to survive growth intact.
        try PredictionBuffer.copyPrefix(from: old.ring, to: ring, what: "sliding_pos_ring")
    }
}

// MARK: - Prediction buffers

/// Allocation and byte-level copying for the buffers we hand CoreML.
enum PredictionBuffer {
    /// Allocate a buffer suitable for `MLPredictionOptions.outputBackings`.
    ///
    /// fp16 buffers are IOSurface-backed (via `CVPixelBuffer`), so a GPU or ANE
    /// prediction writes its result straight into memory we already own instead
    /// of into a framework surface we then copy out of. Everything else — the
    /// int32 ring, and fp32 logits from an export that hasn't moved to fp16 —
    /// gets a page-aligned allocation, the layout CoreML documents for
    /// user-allocated backings.
    ///
    /// Either way the result is tightly packed: an IOSurface whose row pitch
    /// forced padding is rejected in favour of the aligned allocation, so the
    /// copy helpers below can assume row-major contiguity.
    static func make(
        shape: [NSNumber], dataType: MLMultiArrayDataType, fill: Int32? = nil
    ) throws -> MLMultiArray {
        let dims = shape.map { $0.intValue }
        let array = try makeSurfaceBacked(dims: dims, dataType: dataType)
            ?? makePageAligned(dims: dims, dataType: dataType)
        if let fill {
            array.withUnsafeMutableBufferPointer(ofType: Int32.self) { ptr, _ in
                for i in 0..<ptr.count { ptr[i] = fill }
            }
        } else {
            array.withUnsafeMutableBytes { raw, _ in
                if let base = raw.baseAddress { memset(base, 0, raw.count) }
            }
        }
        return array
    }

    /// fp16 only: `kCVPixelFormatType_OneComponent16Half` is the sole 16-bit
    /// float pixel format `MLMultiArray(pixelBuffer:shape:)` accepts. Returns
    /// nil when the format doesn't apply or the surface came back padded.
    private static func makeSurfaceBacked(
        dims: [Int], dataType: MLMultiArrayDataType
    ) -> MLMultiArray? {
        guard dataType == .float16, let width = dims.last, width > 0 else { return nil }
        let height = dims.dropLast().reduce(1, *)
        guard height > 0 else { return nil }

        var pixelBuffer: CVPixelBuffer?
        let attributes = [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault, width, height,
            kCVPixelFormatType_OneComponent16Half, attributes, &pixelBuffer
        )
        guard status == kCVReturnSuccess, let pixelBuffer else { return nil }

        let array = MLMultiArray(
            pixelBuffer: pixelBuffer, shape: dims.map { NSNumber(value: $0) }
        )
        guard isTightlyPacked(array) else { return nil }
        return array
    }

    /// Page-aligned allocation, which CoreML documents as the fastest layout
    /// for a user-allocated backing.
    private static func makePageAligned(
        dims: [Int], dataType: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let count = dims.reduce(1, *)
        let bytes = count * bytesPerElement(of: dataType)
        let alignment = Int(getpagesize())
        let buffer = UnsafeMutableRawPointer.allocate(byteCount: bytes, alignment: alignment)

        var strides = [Int](repeating: 1, count: dims.count)
        for i in stride(from: dims.count - 2, through: 0, by: -1) {
            strides[i] = strides[i + 1] * dims[i + 1]
        }
        do {
            return try MLMultiArray(
                dataPointer: buffer,
                shape: dims.map { NSNumber(value: $0) },
                dataType: dataType,
                strides: strides.map { NSNumber(value: $0) },
                deallocator: { $0.deallocate() }
            )
        } catch {
            buffer.deallocate()
            throw KVCacheError.bufferAllocationFailed(shape: dims, dataType: dataType)
        }
    }

    /// Copy `min(source, destination)` bytes, front-aligned.
    static func copyPrefix(
        from src: MLMultiArray, to dst: MLMultiArray, what: String
    ) throws {
        guard src.dataType == dst.dataType else {
            throw KVCacheError.unexpectedBufferLayout(
                "\(what): dtype \(src.dataType.rawValue) → \(dst.dataType.rawValue)"
            )
        }
        try requireTightlyPacked(src, what: "\(what) source")
        try requireTightlyPacked(dst, what: "\(what) destination")
        let bytes = min(
            src.count * bytesPerElement(of: src.dataType),
            dst.count * bytesPerElement(of: dst.dataType)
        )
        src.withUnsafeBytes { source in
            dst.withUnsafeMutableBytes { destination, _ in
                guard let s = source.baseAddress, let d = destination.baseAddress else { return }
                memcpy(d, s, bytes)
            }
        }
    }

    /// Copy row `row` of a `[rows, width]` array into a fresh tightly-packed
    /// `[width]` array of the same dtype.
    static func extractRow(
        _ row: Int, from array: MLMultiArray, what: String
    ) throws -> MLMultiArray {
        try requireTightlyPacked(array, what: what)
        let shape = array.shape.map { $0.intValue }
        let width = shape.last ?? array.count
        let rows = array.count / max(width, 1)
        guard row >= 0, row < rows else {
            throw KVCacheError.unexpectedBufferLayout(
                "\(what): row \(row) out of range for shape \(shape)"
            )
        }
        let out = try MLMultiArray(shape: [NSNumber(value: width)], dataType: array.dataType)
        let elementSize = bytesPerElement(of: array.dataType)
        array.withUnsafeBytes { source in
            out.withUnsafeMutableBytes { destination, _ in
                guard let s = source.baseAddress, let d = destination.baseAddress else { return }
                memcpy(d, s.advanced(by: row * width * elementSize), width * elementSize)
            }
        }
        return out
    }

    /// Row-major with no padding, so byte-level copies are valid.
    static func isTightlyPacked(_ array: MLMultiArray) -> Bool {
        let shape = array.shape.map { $0.intValue }
        let strides = array.strides.map { $0.intValue }
        guard shape.count == strides.count else { return false }
        var expected = 1
        for i in stride(from: shape.count - 1, through: 0, by: -1) {
            if strides[i] != expected { return false }
            expected *= shape[i]
        }
        return true
    }

    static func requireTightlyPacked(_ array: MLMultiArray, what: String) throws {
        guard isTightlyPacked(array) else {
            throw KVCacheError.unexpectedBufferLayout(
                "\(what) is not contiguous (shape \(array.shape), strides \(array.strides))"
            )
        }
    }

    /// Bytes per element for the dtypes this package uses.
    static func bytesPerElement(of dtype: MLMultiArrayDataType) -> Int {
        switch dtype {
        case .float16: return 2
        case .float32: return 4
        case .float64: return 8
        case .int32:   return 4
        default:       return 2
        }
    }
}
