/// KV cache state for Gemma4-E2B CoreML inference.
///
/// Every KV cache is a CoreML **state** feature: the sliding-window caches and
/// the global-attention ones (`k_4`/`v_4`, `k_9`/`v_9`, `k_14`/`v_14`) alike.
/// The layer-chunk functions read them with `read_state` and write the ones
/// their layers own back with `coreml_update_state`, so no cache tensor
/// crosses the prediction boundary — which is what makes decode cost
/// independent of context length. Core ML shares states across a package's
/// functions by name, so the one `MLState` here serves every chunk.
///
/// The one piece of cache bookkeeping that is not state is `sliding_pos_ring`
/// (CoreML states must be floating point): which absolute position each
/// sliding-cache slot holds, `-1` when empty. The host owns it — every step
/// records its positions in it before the chunks run (``markRing(start:count:)``)
/// and the chunks only read it to mask their sliding layers. The ring (like
/// the sliding caches) has one prefill chunk more rows than the attention
/// window, so a chunk never overwrites a position its own rows still attend
/// to; the chunks' masks enforce the window itself. The host only needs the
/// ring's length, which it takes from the model.
///
/// A materialized function bakes its state shapes in, so an `MLState` belongs
/// to exactly one size N: it is created from the `state_N` function and only
/// that size's chunks accept it. Growing the context therefore means
/// allocating a state for the *next* size and migrating the contents — see
/// ``CoreMLModel/grownToFit(_:needed:)``.
///
/// This object also owns the per-conversation prediction scratch: the ring,
/// the reusable input buffers and feature providers, and the output backings.
/// Keeping them here rather than on ``CoreMLModel`` is what lets two caches
/// coexist (iOS runs an eager-prefill cache alongside the one the current
/// generation is decoding into) without writing over each other.
///
/// A conversation reset means a *fresh* `KVCacheState`, never a reused one with
/// a cleared ring: stale K/V left in a sliding slot becomes valid again the
/// moment a re-populated `sliding_pos_ring` points at it.

import CoreML
import CoreVideo
import Foundation

/// Errors raised while allocating, migrating, or feeding the KV cache.
public enum KVCacheError: Error, LocalizedError {
    /// A state can only be made from the size it belongs to, and that size's
    /// functions have not been loaded yet.
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
/// the ring by the host), so callers hold one instance for as long as the
/// conversation lives rather than threading snapshots around.
public final class KVCacheState: @unchecked Sendable {
    /// Materialized cache length this state is bound to. Only that size's
    /// functions accept it.
    public let size: Int

    /// All KV caches, sliding and global, updated in place by the chunks.
    let caches: MLState

    /// `sliding_pos_ring`: the absolute position each sliding slot holds.
    let ring: MLMultiArray

    /// Reusable int32 scalar for the `position` input.
    let positionScalar: MLMultiArray

    /// Per-phase input buffers, providers and output backings, built on first
    /// use — a cache that only ever decodes never pays for the prefill ones.
    private var steps: [CoreMLModel.Phase: StepScratch] = [:]

    /// Logits output backings with their prediction options, alternating so
    /// the array returned by step N survives until step N+1 has been sampled.
    private var logits: [(backing: MLMultiArray, options: MLPredictionOptions)] = []
    private var logitsIndex = 0

    /// Set once CoreML has been seen to ignore an output backing, so the
    /// diagnostic is logged a single time per conversation instead of per step.
    private var warnedAboutIgnoredBacking = false

    init(size: Int, caches: MLState, ringShape: [NSNumber]) throws {
        self.size = size
        self.caches = caches
        // -1 is the "empty slot" sentinel: a zeroed ring would claim position 0
        // is live in every sliding slot.
        self.ring = try PredictionBuffer.make(shape: ringShape, dataType: .int32, fill: -1)
        self.positionScalar = try MLMultiArray(shape: [1], dataType: .int32)
    }

    // MARK: - Prediction scratch

    func setPosition(_ position: Int32) {
        positionScalar.withUnsafeMutableBufferPointer(ofType: Int32.self) { ptr, _ in
            ptr[0] = position
        }
    }

    /// Record that positions `start ..< start + count` now occupy their
    /// sliding slots (`p % ring length`), before the chunks that write them
    /// run — the step's own tokens are visible to its sliding attention,
    /// exactly as the graph used to update the ring itself. `start` must not
    /// be negative; ``CoreMLModel`` rejects such a position before it gets here.
    func markRing(start: Int32, count: Int) {
        ring.withUnsafeMutableBufferPointer(ofType: Int32.self) { ptr, _ in
            let length = Int32(ptr.count)
            for i in 0..<Int32(count) {
                let p = start + i
                ptr[Int(p % length)] = p
            }
        }
    }

    /// The reusable buffers and feature providers for `phase`.
    func step(
        _ phase: CoreMLModel.Phase, io: CoreMLModel.PhaseIO, headIO: CoreMLModel.HeadIO
    ) throws -> StepScratch {
        if let existing = steps[phase] { return existing }
        let fresh = try StepScratch(
            phase: phase, io: io, headIO: headIO, position: positionScalar, ring: ring
        )
        steps[phase] = fresh
        return fresh
    }

    /// Next logits backing and the options that hand it to `head`.
    func nextLogits(
        _ headIO: CoreMLModel.HeadIO
    ) throws -> (backing: MLMultiArray, options: MLPredictionOptions) {
        if logits.isEmpty {
            logits = try (0..<2).map { _ in
                let backing = try PredictionBuffer.make(
                    shape: headIO.logitsShape, dataType: headIO.logitsDataType
                )
                let options = MLPredictionOptions()
                options.outputBackings = [CoreMLModel.Feature.logits: backing]
                return (backing, options)
            }
        }
        logitsIndex = 1 - logitsIndex
        return logits[logitsIndex]
    }

    /// Log once per conversation that CoreML declined a preallocated backing —
    /// correctness is unaffected (we copy), but every step pays an allocation.
    func noteIgnoredBacking(_ feature: String) {
        guard !warnedAboutIgnoredBacking else { return }
        warnedAboutIgnoredBacking = true
        Log.info("[CoreML] Output backing for '\(feature)' was not used by the framework — predictions will allocate their own buffers")
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

/// One phase's reusable prediction plumbing: the input buffers the host fills,
/// each chunk's output backing, and feature providers / prediction options
/// built once around them. Chunk `k` reads `hidden[k - 1]` and writes
/// `hidden[k]`; `head` reads `headInput`.
final class StepScratch {
    let tokenEmbed: MLMultiArray
    /// One per chunk: that chunk's columns of the per-layer rows.
    let pleRows: [MLMultiArray]
    /// One per chunk: its `hidden_out` backing, `[1, L, D]`.
    let hidden: [MLMultiArray]
    let chunkInputs: [MLFeatureProvider]
    let chunkOptions: [MLPredictionOptions]
    /// `head`'s input: decode reads the last chunk's output directly; prefill
    /// copies the one row it needs into a `[1, 1, D]` buffer of its own.
    let headInput: MLMultiArray
    let headInputs: MLFeatureProvider

    init(
        phase: CoreMLModel.Phase, io: CoreMLModel.PhaseIO, headIO: CoreMLModel.HeadIO,
        position: MLMultiArray, ring: MLMultiArray
    ) throws {
        typealias F = CoreMLModel.Feature
        tokenEmbed = try PredictionBuffer.make(shape: io.hiddenShape, dataType: .float16)
        pleRows = try io.chunks.map { try PredictionBuffer.make(shape: $0.pleRowsShape, dataType: .float16) }
        hidden = try io.chunks.map { _ in try PredictionBuffer.make(shape: io.hiddenShape, dataType: .float16) }
        var inputs: [MLFeatureProvider] = []
        var options: [MLPredictionOptions] = []
        for (k, chunk) in io.chunks.enumerated() {
            var values: [String: MLMultiArray] = [
                F.tokenEmbed: tokenEmbed, F.pleRows: pleRows[k], F.position: position,
            ]
            if chunk.takesHidden { values[F.hidden] = hidden[k - 1] }
            if chunk.takesRing { values[F.ring] = ring }
            inputs.append(CoreMLInputProvider(values: values))
            let o = MLPredictionOptions()
            o.outputBackings = [F.hiddenOut: hidden[k]]
            options.append(o)
        }
        chunkInputs = inputs
        chunkOptions = options
        headInput = phase == .decode
            ? hidden[hidden.count - 1]
            : try PredictionBuffer.make(shape: headIO.inputShape, dataType: .float16)
        headInputs = CoreMLInputProvider(values: [F.hidden: headInput])
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
    /// int32 ring and the fp32 logits — gets a page-aligned allocation, the
    /// layout CoreML documents for user-allocated backings.
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

    /// Copy row `row` of `[…, rows, width]` `src` into `dst`, which holds
    /// exactly one row of the same dtype.
    static func copyRow(_ row: Int, of src: MLMultiArray, into dst: MLMultiArray) throws {
        try requireTightlyPacked(src, what: "row source")
        try requireTightlyPacked(dst, what: "row destination")
        let width = src.shape.last?.intValue ?? src.count
        guard src.dataType == dst.dataType, dst.count == width,
              row >= 0, row < src.count / max(width, 1) else {
            throw KVCacheError.unexpectedBufferLayout(
                "cannot copy row \(row) of \(src.shape) into \(dst.shape)"
            )
        }
        let bytes = width * bytesPerElement(of: src.dataType)
        src.withUnsafeBytes { source in
            dst.withUnsafeMutableBytes { destination, _ in
                guard let s = source.baseAddress, let d = destination.baseAddress else { return }
                memcpy(d, s.advanced(by: row * bytes), bytes)
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
