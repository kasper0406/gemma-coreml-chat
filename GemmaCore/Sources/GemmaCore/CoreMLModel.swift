/// CoreML model wrapper for the materialized multifunction Gemma4-E2B package.
///
/// The artifact exports concrete-shape function pairs `prefill_<N>` /
/// `decode_<N>`, one per size in the export's `--materialize-sizes` (powers of
/// two by default, but any ascending set is legal, and a `--decode-only` export
/// has no `prefill_<N>` at all). Each function is specialized to one global KV
/// cache length, with no dynamic shape ops, so it runs on every backend
/// including ANE and iPhone. The runtime selects the pair matching the current
/// cache size — see ``KVCacheSizePolicy``, the only place bucketing lives.
///
/// **Every** KV cache is a CoreML state: the sliding-window caches and the
/// global-attention ones alike. A prediction therefore takes the token,
/// the position, and the int32 `sliding_pos_ring` (states must be floating
/// point, so the ring cannot be one), and returns logits plus the updated ring.
/// An artifact that still declares `k_<slot>` / `v_<slot>` as inputs predates
/// that change and is rejected at load — re-run `gemma-export`.
///
/// State buffer shapes are baked into each function, so an `MLState` belongs to
/// exactly one size: ``makeEmptyKVState(size:)`` creates it from that pair's
/// model handle, and ``grownToFit(_:needed:)`` migrates the contents when the
/// conversation outgrows it.

import CoreML
import CryptoKit
import Foundation

public final class CoreMLModel: @unchecked Sendable {
    /// I/O names, shapes, and dtypes of one function, read from its description
    /// once so predictions never re-derive them.
    struct FunctionIO {
        let logitsOutputName: String
        let logitsShape: [NSNumber]
        let logitsDataType: MLMultiArrayDataType
        /// Token input, `[1, chunk]` for prefill and `[1]` for decode.
        let tokenInputName: String
        let tokenLength: Int
        let positionInputName: String
        let ringInputName: String
        let ringOutputName: String
        let ringShape: [NSNumber]
        let ringDataType: MLMultiArrayDataType
        /// Every KV cache, sliding and global, sorted for deterministic
        /// migration order.
        let stateNames: [String]
    }

    let decodeIO: FunctionIO
    let prefillIO: FunctionIO

    /// Tokens per prefill call, read from the prefill function's token input.
    ///
    /// A decode-only artifact has no prefill function and prefills by looping
    /// `decode`, so its chunk is 1: any larger value would only pad the prompt
    /// out to a chunk boundary and spend real decode steps on the padding.
    public let chunkSize: Int

    /// Available materialized sizes, ascending. If the caller passed
    /// `maxContextSize`, this is the filtered list.
    public let materializedSizes: [Int]

    /// Largest sequence length this model can actually handle: the largest
    /// retained size, either everything the manifest declared or the
    /// caller-imposed `maxContextSize` cap. The engine uses this instead of
    /// `GemmaConfig.maxSeqLen` so KV growth never exceeds a size we loaded a
    /// function for.
    public let effectiveMaxSeqLen: Int

    /// True when only decode functions are loaded. `prefill()` falls back to
    /// running `decode()` per token — slower (no chunked prefill kernel), but
    /// halves resident-MLModel count, which is the difference between fitting
    /// and OOM on tight devices like iPhone 12 Pro.
    public let isDecodeOnly: Bool

    /// URL of the compiled .mlmodelc (for lazy function loading).
    private let modelURL: URL
    /// URL the caller originally passed to `load(from:)` (.mlpackage or
    /// .mlmodelc). Part of the warm-cache sentinel's identity, so two models
    /// that merely share a basename don't share a sentinel.
    private let sourceURL: URL
    /// Content fingerprint of the artifact at `sourceURL` (spec + sampled
    /// weights), or nil when it couldn't be computed. Recorded in the warm
    /// sentinel so a re-export invalidates it — see `artifactFingerprint`.
    private let sourceFingerprint: String?
    /// Compute units used for all function loads.
    private let computeUnits: MLComputeUnits

    /// `MLModel` isn't `Sendable`, so we can't use `Task<MLModel, Error>`
    /// directly. Wrap it in an @unchecked-Sendable box: CoreML's own loading
    /// is already thread-safe, and we never mutate the model instance.
    private struct SendableMLModel: @unchecked Sendable {
        let model: MLModel
    }

    /// Per-function state: either fully loaded, or a pending load Task that
    /// concurrent callers can join rather than re-issuing the load.
    ///
    /// Pending loads carry a monotonic `id` so a late failure handler can only
    /// evict *its own* entry. Without it: T1 fails with awaiters A and B, A
    /// evicts, C starts T2, then B's eviction removes T2's entry and D starts
    /// T3 — two concurrent multi-GB loads of the same function.
    private enum LoadState {
        case loaded(MLModel)
        case loading(id: UInt64, task: Task<SendableMLModel, Error>)
    }

    /// Function state keyed by function name (e.g. "decode_512").
    private var functions: [String: LoadState]
    /// Functions that have already run their throwaway specialization
    /// prediction — see ``specialize(name:size:)``. Tracked separately from
    /// `functions` because a bulk preload deliberately loads without
    /// specializing.
    private var specializedFunctions: Set<String> = []
    private var nextLoadID: UInt64 = 0
    private let cacheLock = NSLock()

    private init(
        prefillIO: FunctionIO,
        decodeIO: FunctionIO,
        chunkSize: Int,
        materializedSizes: [Int],
        effectiveMaxSeqLen: Int,
        isDecodeOnly: Bool,
        modelURL: URL,
        sourceURL: URL,
        sourceFingerprint: String?,
        computeUnits: MLComputeUnits,
        initialFunctions: [String: MLModel]
    ) {
        self.prefillIO = prefillIO
        self.decodeIO = decodeIO
        self.chunkSize = chunkSize
        self.materializedSizes = materializedSizes
        self.effectiveMaxSeqLen = effectiveMaxSeqLen
        self.isDecodeOnly = isDecodeOnly
        self.modelURL = modelURL
        self.sourceURL = sourceURL
        self.sourceFingerprint = sourceFingerprint
        self.computeUnits = computeUnits
        self.functions = initialFunctions.mapValues { .loaded($0) }
    }

    // MARK: - KV cache lifecycle

    /// A zeroed cache for `size` (rounded up through ``cacheSizePolicy``),
    /// defaulting to the smallest materialized pair.
    ///
    /// The `MLState` is made from that pair's own model handle: state buffer
    /// shapes are baked into each materialized function, so a state made at one
    /// size is meaningless at another. The pair must already be loaded —
    /// `ensureLoaded(forGlobalCacheSize:)` first for anything but the bootstrap
    /// size, which `load` brings up.
    ///
    /// Make a new one per conversation. Reusing one across a reset would leave
    /// stale K/V that a re-populated `sliding_pos_ring` marks valid again.
    public func makeEmptyKVState(size requested: Int? = nil) throws -> KVCacheState {
        let target = cacheSizePolicy.size(forNeeded: requested ?? materializedSizes[0])
        guard let model = loadedFunction(named: functionName(prefix: "decode", size: target)) else {
            throw KVCacheError.functionNotLoaded(size: target)
        }
        return try KVCacheState(
            size: target,
            caches: model.makeState(),
            ringShape: decodeIO.ringShape,
            ringDataType: decodeIO.ringDataType,
            chunkSize: chunkSize
        )
    }

    /// Return a cache big enough for `needed` tokens, migrating `kv` into a
    /// larger pair's state when it no longer fits.
    ///
    /// Returns `kv` untouched in the common case. When growth is required the
    /// next pair is loaded first (state buffers can only be made from a loaded
    /// handle) and every cache is copied across — see
    /// ``KVCacheState/adoptContents(of:stateNames:)``.
    public func grownToFit(_ kv: KVCacheState, needed: Int) async throws -> KVCacheState {
        let target = cacheSizePolicy.size(forNeeded: needed)
        guard target > kv.size else { return kv }
        try await ensureLoaded(forGlobalCacheSize: target)
        let grown = try makeEmptyKVState(size: target)
        try grown.adoptContents(of: kv, stateNames: decodeIO.stateNames)
        Log.info("[KV] Grew caches \(kv.size) → \(target) (needed \(needed))")
        return grown
    }

    /// Bucketing policy for this model's caches. Hand this to anything that
    /// needs to size a cache, so cache shape and resolved function never
    /// disagree.
    public var cacheSizePolicy: KVCacheSizePolicy {
        KVCacheSizePolicy(materializedSizes: materializedSizes, maxLen: effectiveMaxSeqLen)
    }

    // MARK: - Loading

    /// Load the multifunction model from a .mlpackage or .mlmodelc URL.
    ///
    /// For .mlpackage files, the model is compiled and cached as .mlmodelc
    /// next to the source for fast subsequent loads (E5RT cache reuse).
    /// For .mlmodelc files, loads directly without recompilation.
    ///
    /// - Parameter maxContextSize: Only retain function pairs ≤ this size.
    ///   Loading fewer functions is critical on memory-constrained devices like
    ///   iPhone, where loading all 16 pairs OOMs.
    /// - Parameter decodeOnly: Skip loading prefill functions entirely.
    ///   `prefill()` falls back to per-token `decode()` internally — slower but
    ///   halves resident MLModel count, which is the only way the model fits on
    ///   iPhone 12 Pro / 6 GB devices. Forced on for `--decode-only` artifacts,
    ///   which export no prefill functions to load.
    /// - Parameter backgroundPreload: Kick off a detached load of every
    ///   retained function pair once the bootstrap pair is up. Right for
    ///   interactive apps (later size transitions become instant), wrong for
    ///   benchmarks — multi-GB loads running under a measured window contend
    ///   for CPU/ANE/disk and race the engine's own `ensureLoaded`, so
    ///   `GemmaBench` passes false and pre-loads exactly what it needs.
    public static func load(
        from url: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU,
        maxContextSize: Int? = nil,
        decodeOnly: Bool = false,
        backgroundPreload: Bool = true
    ) async throws -> CoreMLModel {
        let compiledURL: URL
        let fingerprint = artifactFingerprint(of: url)

        if url.pathExtension == "mlpackage" {
            let cachedURL = try defaultCacheURL(for: url)
            compiledURL = try await compileAndCache(
                source: url, cached: cachedURL, fingerprint: fingerprint
            )
        } else {
            // Already compiled (.mlmodelc)
            compiledURL = url
        }

        return try await loadCompiled(
            from: compiledURL,
            sourceURL: url,
            sourceFingerprint: fingerprint,
            computeUnits: computeUnits,
            maxContextSize: maxContextSize,
            decodeOnly: decodeOnly,
            backgroundPreload: backgroundPreload
        )
    }

    /// Pick where to persist the compiled `.mlmodelc`.
    ///
    /// Prefers the directory next to the source (convenient for desktop use
    /// where the source lives in a writable project folder). Falls back to
    /// Application Support when the source parent isn't writable — which is
    /// exactly the iOS case, since the app bundle is read-only. Without this
    /// fallback, `MLModel.compileModel` returns a `/tmp`-rooted bundle that
    /// our move-to-cache step can't land anywhere persistent, so the caller
    /// ends up loading from a path that later fails to mmap.
    private static func defaultCacheURL(for source: URL) throws -> URL {
        let nextTo = source.deletingPathExtension().appendingPathExtension("mlmodelc")
        let parent = nextTo.deletingLastPathComponent()
        if FileManager.default.isWritableFile(atPath: parent.path) {
            return nextTo
        }
        let appSupport = try FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        )
        let dir = appSupport.appendingPathComponent("GemmaCore/compiled", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let base = source.deletingPathExtension().lastPathComponent
        return dir.appendingPathComponent("\(base).mlmodelc")
    }

    /// Compile .mlpackage → .mlmodelc, caching at `cached` path.
    ///
    /// Invalidates via `artifactFingerprint(of:)` stored in a sidecar. Mtime
    /// comparison is unreliable here: swapping in a different `.mlpackage`
    /// build can leave the source older than an existing cache, masking a real
    /// change.
    private static func compileAndCache(
        source: URL, cached: URL, fingerprint: String?
    ) async throws -> URL {
        let sidecar = cached.appendingPathExtension("src-sha256")
        let currentHash = fingerprint

        if FileManager.default.fileExists(atPath: cached.path) {
            let cachedHash = (try? String(contentsOf: sidecar, encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let c = currentHash, let s = cachedHash, c == s {
                Log.info("[CoreML] Using cached compiled model at \(cached.path)")
                return cached
            }
            if currentHash == nil {
                Log.info("[CoreML] WARNING: no fingerprint for \(source.lastPathComponent) — discarding the compile cache and recompiling. The sidecar can never be written, so EVERY launch will pay full compilation until the source becomes readable.")
            } else {
                Log.info("[CoreML] Cache hash \(cachedHash == nil ? "missing" : "mismatch") — recompiling")
            }
            try? FileManager.default.removeItem(at: cached)
            try? FileManager.default.removeItem(at: sidecar)
        }

        Log.info("[CoreML] Compiling \(source.lastPathComponent)...")
        let compiledURL = try await MLModel.compileModel(at: source)
        Log.info("[CoreML] Compiled to \(compiledURL.path)")

        try? FileManager.default.removeItem(at: cached)
        do {
            try FileManager.default.moveItem(at: compiledURL, to: cached)
        } catch {
            // On iOS this previously hit: cached was in the read-only bundle,
            // move silently failed, loading then blew up on mmap from /tmp.
            // `defaultCacheURL` now chooses Application Support on iOS, so
            // this path should stay dry — but log loudly if it fires again.
            Log.info("[CoreML] Failed to move compiled model to \(cached.path): \(error) — using temp at \(compiledURL.path)")
        }
        let finalURL = FileManager.default.fileExists(atPath: cached.path) ? cached : compiledURL
        if finalURL == cached, let hash = currentHash {
            try? hash.write(to: sidecar, atomically: true, encoding: .utf8)
        }
        return finalURL
    }

    /// Bytes sampled from each end of a weight blob. Weight files run to
    /// several GB, so hashing them whole on every launch would cost seconds of
    /// I/O; the byte length plus the first and last megabyte catches a
    /// re-export from a different checkpoint or quantization, which rewrites
    /// the whole blob.
    private static let weightSampleBytes = 1 << 20

    /// Content fingerprint of a model artifact (`.mlpackage` or `.mlmodelc`).
    ///
    /// Covers the structure spec **and** the weight blobs. Hashing only the
    /// spec (as this used to) misses a weights-only re-export: the mlprogram
    /// spec references blobs by `fileName` + `offset` with no content digest,
    /// so re-exporting an identical architecture from a new checkpoint leaves
    /// the spec byte-identical and a stale `.mlmodelc` keeps serving the OLD
    /// weights — silently wrong output with no error anywhere.
    ///
    /// Returns nil (and logs loudly) if nothing could be read: callers must
    /// treat that as "staleness detection unavailable", not as a match.
    static func artifactFingerprint(of url: URL) -> String? {
        var hasher = SHA256()
        var sawSpec = false
        for spec in specFileURLs(for: url) {
            guard let data = try? Data(contentsOf: spec) else { continue }
            hasher.update(data: Data(spec.lastPathComponent.utf8))
            hasher.update(data: data)
            sawSpec = true
        }
        guard sawSpec else {
            Log.info("[CoreML] WARNING: no readable spec file under \(url.path) — compile-cache and warm-cache staleness detection are DISABLED for this model")
            return nil
        }

        let blobs = weightBlobURLs(for: url)
        if blobs.isEmpty {
            Log.info("[CoreML] WARNING: no weight blobs found under \(url.lastPathComponent) — fingerprint covers the spec only, so a weights-only re-export will NOT invalidate the compile cache")
        }
        for blob in blobs {
            guard let sample = weightSample(of: blob) else {
                Log.info("[CoreML] WARNING: could not sample weight blob \(blob.lastPathComponent) — staleness detection DISABLED for this model")
                return nil
            }
            hasher.update(data: Data(blob.lastPathComponent.utf8))
            hasher.update(data: sample)
        }
        return hexString(hasher.finalize())
    }

    /// Files describing the model's structure, hashed in full (tens of MB).
    private static func specFileURLs(for url: URL) -> [URL] {
        if url.pathExtension == "mlpackage" {
            return [url.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")]
        }
        // .mlmodelc: model.mil carries the full function set and shapes.
        return [
            url.appendingPathComponent("model.mil"),
            url.appendingPathComponent("coremldata.bin"),
        ]
    }

    /// Weight blobs, sorted by name so the digest is order-independent.
    private static func weightBlobURLs(for url: URL) -> [URL] {
        let dir = url.pathExtension == "mlpackage"
            ? url.appendingPathComponent("Data/com.apple.CoreML/weights")
            : url.appendingPathComponent("weights")
        let contents = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil
        )) ?? []
        return contents.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Digest of a weight blob's byte length plus its first and last
    /// `weightSampleBytes`. Deliberately not a full hash — see the constant.
    private static func weightSample(of url: URL) -> Data? {
        guard let size = (try? FileManager.default
            .attributesOfItem(atPath: url.path)[.size]) as? NSNumber else { return nil }
        let byteCount = max(size.int64Value, 0)
        guard let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }

        var hasher = SHA256()
        withUnsafeBytes(of: byteCount.littleEndian) { hasher.update(data: Data($0)) }
        let window = Int64(weightSampleBytes)
        do {
            if let head = try handle.read(upToCount: Int(min(window, byteCount))) {
                hasher.update(data: head)
            }
            if byteCount > window {
                try handle.seek(toOffset: UInt64(byteCount - window))
                if let tail = try handle.read(upToCount: weightSampleBytes) {
                    hasher.update(data: tail)
                }
            }
        } catch {
            return nil
        }
        return Data(hasher.finalize())
    }

    private static func hexString<D: Sequence>(_ digest: D) -> String where D.Element == UInt8 {
        digest.map { String(format: "%02x", $0) }.joined()
    }

    /// Load a pre-compiled multifunction .mlmodelc.
    ///
    /// The function set comes from `model.mil` — a text parse, no
    /// `MLModel.load` — rather than from trial loads, so the bootstrap is never
    /// the multi-MLModel memory spike that OOMs an iPhone. Only when the
    /// manifest is unreadable does this fall back to probing.
    ///
    /// Strategy, tuned for memory-constrained devices:
    ///   1. Take the declared function set from `model.mil`.
    ///   2. Load `decode_{smallest}` and `prefill_{smallest}` SERIALLY. Peak
    ///      live MLModel count is 1 during bootstrap.
    ///   3. Optionally background-preload the remaining retained sizes.
    private static func loadCompiled(
        from url: URL,
        sourceURL: URL,
        sourceFingerprint: String?,
        computeUnits: MLComputeUnits,
        maxContextSize: Int?,
        decodeOnly: Bool,
        backgroundPreload: Bool
    ) async throws -> CoreMLModel {
        Log.info("[CoreML] Loading decode\(decodeOnly ? "" : " + prefill") functions from \(url.lastPathComponent)...")

        var effectiveDecodeOnly = decodeOnly
        let sizes: [Int]

        if let declared = enumerateMaterializedFunctions(compiledURL: url) {
            // A `gemma-export --decode-only` artifact has decode_<N> and no
            // prefill_<N>. Insisting on the decode∩prefill intersection there
            // yields no sizes at all.
            if declared.prefillSizes.isEmpty && !effectiveDecodeOnly {
                Log.info("[CoreML] Artifact exports no prefill functions — switching to decode-only mode")
                effectiveDecodeOnly = true
            }
            sizes = declared.usableSizes(decodeOnly: effectiveDecodeOnly)
            guard !sizes.isEmpty else {
                throw CoreMLModelError.noUsableMaterializedFunctions(
                    decodeSizes: declared.decodeSizes,
                    prefillSizes: declared.prefillSizes
                )
            }
            Log.info("[CoreML] Materialized sizes (from manifest): \(sizes)")
        } else {
            Log.info("[CoreML] Manifest enumeration unavailable; falling back to parallel probe")
            sizes = await probeMaterializedSizes(url: url, computeUnits: computeUnits)
            guard !sizes.isEmpty else { throw CoreMLModelError.notMaterialized(url.lastPathComponent) }
        }

        // Restrict retained sizes to `maxContextSize` before any heavy load,
        // so the bootstrap only pulls functions we'll actually keep.
        let retainedSizes: [Int]
        if let cap = maxContextSize {
            let under = sizes.filter { $0 <= cap }
            retainedSizes = under.isEmpty ? [sizes[0]] : under
            if retainedSizes != sizes {
                Log.info("[CoreML] Restricting to sizes \(retainedSizes) (maxContextSize=\(cap))")
            }
        } else {
            retainedSizes = sizes
        }

        let bootSize = retainedSizes[0]
        let decodeName = "decode_\(bootSize)"
        let prefillName = "prefill_\(bootSize)"

        // Serial loads keep peak live-MLModel count at 1 during bootstrap.
        let decodeModel = try await loadFunction(
            url: url, computeUnits: computeUnits, function: decodeName
        )
        let prefillModel: MLModel?
        if effectiveDecodeOnly {
            prefillModel = nil
            Log.info("[CoreML] Loaded \(decodeName) (decode-only; prefill skipped)")
        } else {
            prefillModel = try await loadFunction(
                url: url, computeUnits: computeUnits, function: prefillName
            )
            Log.info("[CoreML] Loaded \(decodeName) + \(prefillName) (serial)")
        }

        let decodeIO = try classifyIO(model: decodeModel, function: decodeName)
        // In decode-only mode, prefill metadata is borrowed from decode: the
        // per-token loop in `decodeOnlyPrefill` runs the decode function, so
        // those are the names and shapes it needs.
        let prefillIO = try prefillModel.map { try classifyIO(model: $0, function: prefillName) }
            ?? decodeIO
        // Decode-only prefills one token at a time, so a chunk larger than 1
        // would only pad the prompt and spend real decode steps on padding.
        let chunkSize = prefillModel == nil ? 1 : prefillIO.tokenLength
        if prefillModel != nil {
            // The engine reads the row of the last *real* token out of the
            // chunk, so a prefill that emits fewer rows than it consumes tokens
            // cannot serve a padded final chunk. Fail here rather than at the
            // first prompt that isn't a chunk multiple.
            let rows = prefillIO.logitsShape.dropLast().map { $0.intValue }.reduce(1, *)
            guard rows == chunkSize else {
                throw CoreMLModelError.unexpectedSignature(
                    function: prefillName,
                    detail: "takes \(chunkSize) tokens but emits \(rows) logits row(s); the runtime needs one row per chunk position"
                )
            }
        }
        logIOSummary(decodeIO: decodeIO, prefillIO: prefillIO, chunkSize: chunkSize)

        var initialFunctions: [String: MLModel] = [decodeName: decodeModel]
        if let p = prefillModel {
            initialFunctions[prefillName] = p
        }

        let instance = CoreMLModel(
            prefillIO: prefillIO,
            decodeIO: decodeIO,
            chunkSize: chunkSize,
            materializedSizes: retainedSizes,
            effectiveMaxSeqLen: retainedSizes[retainedSizes.count - 1],
            isDecodeOnly: effectiveDecodeOnly,
            modelURL: url,
            sourceURL: sourceURL,
            sourceFingerprint: sourceFingerprint,
            computeUnits: computeUnits,
            initialFunctions: initialFunctions
        )
        if backgroundPreload {
            instance.preloadAllSizes()
        }
        return instance
    }

    /// Last-resort size discovery when `model.mil` can't be parsed: try loading
    /// `decode_<N>` for every plausible N and keep the ones that succeed. The
    /// probed models are dropped — this only answers "which sizes exist"; the
    /// bootstrap reloads what it needs.
    private static func probeMaterializedSizes(
        url: URL, computeUnits: MLComputeUnits
    ) async -> [Int] {
        let candidateSizes = (6...16).map { 1 << $0 }  // 64..65536
        return await withTaskGroup(of: Int?.self) { group in
            for size in candidateSizes {
                group.addTask {
                    let config = MLModelConfiguration()
                    config.computeUnits = computeUnits
                    config.functionName = "decode_\(size)"
                    let model = try? await MLModel.load(contentsOf: url, configuration: config)
                    return model == nil ? nil : size
                }
            }
            var found: [Int] = []
            for await size in group {
                if let size { found.append(size) }
            }
            return found.sorted()
        }
    }

    /// Load a single function by name (used at bootstrap).
    private static func loadFunction(
        url: URL, computeUnits: MLComputeUnits, function: String
    ) async throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.functionName = function
        return try await MLModel.load(contentsOf: url, configuration: config)
    }

    /// The `{decode,prefill}_<N>` function sets a compiled artifact declares.
    struct MaterializedFunctions {
        /// Sizes with a `decode_<N>` function, ascending.
        let decodeSizes: [Int]
        /// Sizes with a `prefill_<N>` function, ascending. Empty for artifacts
        /// exported with `gemma-export --decode-only`.
        let prefillSizes: [Int]

        /// Sizes runnable in the requested mode: decode-only needs just a
        /// decode function, otherwise both halves of the pair must exist.
        func usableSizes(decodeOnly: Bool) -> [Int] {
            guard !decodeOnly else { return decodeSizes }
            let prefill = Set(prefillSizes)
            return decodeSizes.filter { prefill.contains($0) }
        }
    }

    /// Scan the compiled `model.mil` manifest for `func decode_N<…>` /
    /// `func prefill_N<…>` declarations.
    ///
    /// Returns nil ONLY when the manifest is missing or unparseable. An empty
    /// `decodeSizes` means "parsed fine, this isn't a materialized artifact" —
    /// callers must not conflate the two, because nil is what licenses the
    /// expensive parallel probe.
    static func enumerateMaterializedFunctions(compiledURL: URL) -> MaterializedFunctions? {
        let milURL = compiledURL.appendingPathComponent("model.mil")
        guard let text = try? String(contentsOf: milURL, encoding: .utf8) else {
            return nil
        }
        guard let re = try? NSRegularExpression(
            pattern: #"\bfunc\s+(decode|prefill)_(\d+)\s*[<(]"#
        ) else { return nil }

        var decodeSizes = Set<Int>()
        var prefillSizes = Set<Int>()
        let full = NSRange(text.startIndex..<text.endIndex, in: text)
        re.enumerateMatches(in: text, range: full) { match, _, _ in
            guard let m = match, m.numberOfRanges >= 3,
                  let kr = Range(m.range(at: 1), in: text),
                  let sr = Range(m.range(at: 2), in: text),
                  let size = Int(text[sr]) else { return }
            if text[kr] == "decode" { decodeSizes.insert(size) }
            else { prefillSizes.insert(size) }
        }
        return MaterializedFunctions(
            decodeSizes: decodeSizes.sorted(),
            prefillSizes: prefillSizes.sorted()
        )
    }

    /// Log I/O classification summary.
    private static func logIOSummary(
        decodeIO: FunctionIO, prefillIO: FunctionIO, chunkSize: Int
    ) {
        Log.info("[CoreML] Decode: logits=\(decodeIO.logitsOutputName)\(decodeIO.logitsShape.map { $0.intValue }) dtype=\(decodeIO.logitsDataType.rawValue), token=\(decodeIO.tokenInputName), pos=\(decodeIO.positionInputName), ring=\(decodeIO.ringInputName)→\(decodeIO.ringOutputName), caches=\(decodeIO.stateNames.count) states")
        Log.info("[CoreML] Prefill: logits=\(prefillIO.logitsOutputName)\(prefillIO.logitsShape.map { $0.intValue }) dtype=\(prefillIO.logitsDataType.rawValue), chunk=\(chunkSize)")
    }

    // MARK: - Function Resolution

    /// Name of the function serving `size`, which callers get from a
    /// ``KVCacheState`` and is therefore always one of `materializedSizes`.
    private func functionName(prefix: String, size: Int) -> String {
        "\(prefix)_\(size)"
    }

    /// The loaded model for `name`, or nil if it hasn't been loaded yet.
    private func loadedFunction(named name: String) -> MLModel? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if case .loaded(let model) = functions[name] { return model }
        return nil
    }

    /// The loaded model for a prediction, or a clear error naming the call the
    /// caller skipped.
    private func function(prefix: String, size: Int) throws -> MLModel {
        let name = functionName(prefix: prefix, size: size)
        guard let model = loadedFunction(named: name) else {
            throw KVCacheError.functionNotLoaded(size: size)
        }
        return model
    }

    /// Pre-load *and specialize* the decode and prefill functions for a given
    /// cache size. Call from an async context before sync `prefill()`/`decode()`
    /// calls; every path that is about to predict at a new size goes through
    /// here, which is what keeps ``specialize(name:size:)`` off the token loop.
    public func ensureLoaded(forGlobalCacheSize cacheSize: Int) async throws {
        let size = cacheSizePolicy.size(forNeeded: cacheSize)
        let decodeName = functionName(prefix: "decode", size: size)
        let prefillName = functionName(prefix: "prefill", size: size)

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { _ = try await self.loadIfNeeded(name: decodeName) }
            if !self.isDecodeOnly {
                group.addTask { _ = try await self.loadIfNeeded(name: prefillName) }
            }
            try await group.waitForAll()
        }

        // Serial, deliberately: a specialization transiently allocates tens of
        // GB (MPSGraph materializes the dequantized embedding tables), so two
        // at once exhausts memory on machines that comfortably run one.
        try specialize(name: decodeName, size: size)
        if !isDecodeOnly {
            try specialize(name: prefillName, size: size)
        }
    }

    /// Run one throwaway prediction on a freshly loaded function, into a
    /// scratch `MLState` that no conversation owns.
    ///
    /// A GPU-backed CoreML function does not finish compiling when it loads:
    /// `MLModel.load` only builds the E5RT plan, and MPSGraph specializes the
    /// executable lazily inside the *first* `predictionFromFeatures:`. For this
    /// model that first call costs ~17 s of single-threaded MLIR work, almost
    /// all of it constant-folding the block-32 int4 embedding tables that feed
    /// `gather` (`LowerDequantizeND` → `foldCastAttribute`, one LLVM `APFloat`
    /// per weight element), with a ~27 GB transient peak. Nothing caches it:
    /// it is redone in every process, for every materialized function.
    ///
    /// So pay it here — at load, or at the moment a conversation grows into a
    /// new size — instead of inside the first token the user is waiting on.
    /// The scratch state matters: predictions mutate KV caches in place, so
    /// warming through the live cache would write a phantom token 0 into it.
    private func specialize(name: String, size: Int) throws {
        cacheLock.lock()
        let alreadyDone = !specializedFunctions.insert(name).inserted
        cacheLock.unlock()
        guard !alreadyDone else { return }

        guard let model = loadedFunction(named: name) else {
            throw KVCacheError.functionNotLoaded(size: size)
        }
        let start = CFAbsoluteTimeGetCurrent()
        try autoreleasepool {
            let scratch = try KVCacheState(
                size: size,
                caches: model.makeState(),
                ringShape: decodeIO.ringShape,
                ringDataType: decodeIO.ringDataType,
                chunkSize: chunkSize
            )
            if name.hasPrefix("decode") {
                _ = try decode(token: 0, position: 0, kvState: scratch)
            } else {
                _ = try prefill(
                    tokens: [Int32](repeating: 0, count: chunkSize),
                    startPosition: 0, logitsRow: 0, kvState: scratch
                )
            }
        }
        Log.info("[CoreML] Specialized '\(name)' in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - start))s")
    }

    /// Result of checking the cache for `name`: an already-loaded model, or a
    /// load Task (either newly started by us or one a concurrent caller had
    /// already kicked off).
    private enum CacheLookup {
        case existing(MLModel)
        case pending(id: UInt64, task: Task<SendableMLModel, Error>)
    }

    /// Atomically look up `name`; if absent, start a new load Task and record
    /// it. All `NSLock` traffic is confined to this sync method so callers in
    /// async contexts never touch the lock directly.
    private func lookupOrStart(name: String) -> CacheLookup {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if let state = functions[name] {
            switch state {
            case .loaded(let m): return .existing(m)
            case .loading(let id, let task): return .pending(id: id, task: task)
            }
        }
        let url = modelURL
        let units = computeUnits
        let task: Task<SendableMLModel, Error> = Task {
            let config = MLModelConfiguration()
            config.computeUnits = units
            config.functionName = name
            let model = try await MLModel.load(contentsOf: url, configuration: config)
            return SendableMLModel(model: model)
        }
        nextLoadID += 1
        let id = nextLoadID
        functions[name] = .loading(id: id, task: task)
        return .pending(id: id, task: task)
    }

    private func markLoaded(name: String, model: MLModel) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        functions[name] = .loaded(model)
    }

    /// Evict the pending entry for `name` — but only if it is still *our*
    /// attempt. Removing whatever happens to be there lets a late awaiter of a
    /// failed load evict a successor task's entry, after which the next caller
    /// starts a second concurrent multi-GB load of the same function.
    private func clearPending(name: String, id: UInt64) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if case .loading(let storedID, _) = functions[name], storedID == id {
            functions.removeValue(forKey: name)
        }
    }

    /// Load a single function by name. Concurrent callers for the same name
    /// share one in-flight Task instead of issuing duplicate loads.
    @discardableResult
    private func loadIfNeeded(name: String) async throws -> MLModel {
        switch lookupOrStart(name: name) {
        case .existing(let model):
            return model
        case .pending(let id, let task):
            do {
                let model = try await task.value.model
                markLoaded(name: name, model: model)
                Log.info("[CoreML] Function '\(name)' loaded.")
                return model
            } catch {
                clearPending(name: name, id: id)
                throw error
            }
        }
    }

    /// Kick off background loads for every materialized function pair in
    /// ascending size order. Non-blocking: later calls to `ensureLoaded` join
    /// in-flight tasks rather than issuing duplicate loads.
    public func preloadAllSizes(concurrency: Int = 2) {
        Task.detached { [self] in
            let start = CFAbsoluteTimeGetCurrent()
            let allOK = await self.drainLoads(
                names: self.allFunctionNames, concurrency: concurrency, progress: nil
            )
            let elapsed = CFAbsoluteTimeGetCurrent() - start
            Log.info("[CoreML] Background preload complete (\(String(format: "%.1f", elapsed))s, ok=\(allOK))")
            if allOK { self.markWarmed() }
        }
    }

    /// Block until every materialized function pair is loaded, reporting
    /// progress as each completes. On first-run installs this is what warms
    /// the ANE / E5RT cache before the first chat turn — otherwise the user
    /// hits multi-minute stalls mid-session. Safe to call even when the bg
    /// preload is running: both join the same in-flight tasks.
    public func warmSynchronously(
        concurrency: Int = 4,
        progress: @Sendable @escaping (_ completed: Int, _ total: Int) -> Void
    ) async {
        let allOK = await drainLoads(
            names: allFunctionNames, concurrency: concurrency, progress: progress
        )
        if allOK { markWarmed() }
    }

    /// Every function this load retains, in ascending size order.
    private var allFunctionNames: [String] {
        materializedSizes.flatMap {
            isDecodeOnly ? ["decode_\($0)"] : ["decode_\($0)", "prefill_\($0)"]
        }
    }

    /// Core worker used by both `preloadAllSizes` and `warmSynchronously`:
    /// walks `names` with a bounded-concurrency task group and returns
    /// whether every load succeeded. Progress is reported in completion
    /// order whenever a load finishes.
    private func drainLoads(
        names: [String],
        concurrency: Int,
        progress: (@Sendable (Int, Int) -> Void)?
    ) async -> Bool {
        let total = names.count
        progress?(0, total)
        var completed = 0
        var allOK = true
        await withTaskGroup(of: Bool.self) { group in
            var iter = names.makeIterator()
            var active = 0
            while active < concurrency, let n = iter.next() {
                group.addTask { await self.preloadOne(name: n) }
                active += 1
            }
            while let ok = await group.next() {
                if !ok { allOK = false }
                completed += 1
                progress?(completed, total)
                if let n = iter.next() {
                    group.addTask { await self.preloadOne(name: n) }
                }
            }
        }
        return allOK
    }

    private func preloadOne(name: String) async -> Bool {
        do { _ = try await loadIfNeeded(name: name); return true }
        catch {
            Log.info("[CoreML] Preload '\(name)' failed: \(error.localizedDescription)")
            return false
        }
    }

    // MARK: - Warm sentinel

    /// Whether the functions *this* load retains have previously been compiled
    /// to the ANE / E5RT cache. When false, the first run will pay
    /// multi-minute compilation on each new function; the app should call
    /// `warmSynchronously(progress:)` before entering the chat.
    ///
    /// Validity is decided by the recorded artifact fingerprint, not by mtime:
    /// re-exporting a model can leave it *older* than the sentinel.
    public var isWarmed: Bool {
        guard let fingerprint = sourceFingerprint else {
            Log.info("[CoreML] Warm sentinel unavailable: could not fingerprint \(sourceURL.lastPathComponent) — assuming cold")
            return false
        }
        guard let sentinel = warmSentinelURL,
              let recorded = try? String(contentsOf: sentinel, encoding: .utf8) else {
            return false
        }
        return recorded.trimmingCharacters(in: .whitespacesAndNewlines) == fingerprint
    }

    private func markWarmed() {
        guard let fingerprint = sourceFingerprint else {
            Log.info("[CoreML] Not recording a warm sentinel: no artifact fingerprint available")
            return
        }
        guard let sentinel = warmSentinelURL else { return }
        let dir = sentinel.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        do {
            try fingerprint.write(to: sentinel, atomically: true, encoding: .utf8)
            Log.info("[CoreML] Marked warm cache: \(sentinel.lastPathComponent)")
        } catch {
            Log.info("[CoreML] Failed to write warm sentinel \(sentinel.path): \(error)")
        }
    }

    /// Sentinel path in Application Support. Lives outside the .mlmodelc so it
    /// survives re-compilation and works on iOS where the bundle (and thus the
    /// .mlmodelc next to it) is read-only.
    ///
    /// The name is keyed by the *warm scope*, not just the file name: the full
    /// source path (two models called `gemma4-e2b.mlpackage` in different
    /// directories are different models), the compute units, and the exact
    /// function set this load retains. That last part matters — a
    /// `maxContextSize`-capped `gemma-bench` run only ever compiles its small
    /// subset, and a filename-keyed sentinel let it tell the CLI and the iOS
    /// app that all 16 pairs were ready, which they then discovered mid-chat
    /// as multi-minute E5RT compile stalls.
    private var warmSentinelURL: URL? {
        guard let appSupport = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        ) else { return nil }
        let dir = appSupport.appendingPathComponent("GemmaCore", isDirectory: true)
        let base = sourceURL.deletingPathExtension().lastPathComponent
            .replacingOccurrences(of: "/", with: "_")
        return dir.appendingPathComponent("warmed-\(base)-\(warmScopeKey).marker")
    }

    /// Digest of everything that makes this load's warm-up distinct.
    private var warmScopeKey: String {
        let scope = [
            sourceURL.standardizedFileURL.path,
            Self.computeUnitsTag(computeUnits),
            "decodeOnly=\(isDecodeOnly)",
            "sizes=\(materializedSizes.map(String.init).joined(separator: ","))",
        ].joined(separator: "|")
        return String(Self.hexString(SHA256.hash(data: Data(scope.utf8))).prefix(16))
    }

    private static func computeUnitsTag(_ cu: MLComputeUnits) -> String {
        switch cu {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndANE"
        case .all: return "all"
        @unknown default: return "unknown"
        }
    }

    // MARK: - Prediction

    /// Run one prefill chunk and return the logits row for `logitsRow`.
    ///
    /// `tokens.count` must equal ``chunkSize``; the engine pads the prompt to a
    /// chunk boundary. Only one row of the `[chunk, vocab]` output is ever
    /// wanted (the last *real* token of the chunk), so the row is copied out
    /// and the big output buffer is reused for the next chunk.
    public func prefill(
        tokens: [Int32],
        startPosition: Int32,
        logitsRow: Int,
        kvState: KVCacheState
    ) throws -> MLMultiArray {
        // This chunk writes cache rows startPosition ..< +count, so every one
        // of them has to fit. The real prefill kernel would fault; the
        // per-token loop would silently scribble past the end.
        guard Int(startPosition) + tokens.count <= kvState.size else {
            throw CoreMLModelError.positionOutOfRange(
                position: Int(startPosition) + tokens.count - 1, cacheSize: kvState.size
            )
        }
        guard logitsRow >= 0, logitsRow < tokens.count else {
            throw KVCacheError.unexpectedBufferLayout(
                "prefill logits row \(logitsRow) outside chunk of \(tokens.count)"
            )
        }
        if isDecodeOnly {
            return try decodeOnlyPrefill(
                tokens: tokens, startPosition: startPosition,
                logitsRow: logitsRow, kvState: kvState
            )
        }

        let model = try function(prefix: "prefill", size: kvState.size)
        try kvState.loadChunk(tokens)
        kvState.setScalar(startPosition, in: kvState.positionScalar)

        let inputs: [String: MLMultiArray] = [
            prefillIO.tokenInputName: kvState.chunkTokens,
            prefillIO.positionInputName: kvState.positionScalar,
            prefillIO.ringInputName: kvState.ring,
        ]

        let logitsBacking = try kvState.prefillLogits(
            shape: prefillIO.logitsShape, dataType: prefillIO.logitsDataType
        )
        let logits = try predict(
            model: model, io: prefillIO, inputs: inputs,
            logitsBacking: logitsBacking, kvState: kvState
        )
        return try PredictionBuffer.extractRow(logitsRow, from: logits, what: "prefill logits")
    }

    /// Per-token prefill via repeated `decode()` calls — the fallback used when
    /// only decode functions are loaded. Slower than a real prefill function
    /// (no fused chunk kernel), but keeps the resident MLModel count at 1
    /// instead of 2, the only way to fit on iPhone 12 Pro / 6 GB.
    ///
    /// The chunk is 1 token in this mode, so in practice this runs one decode
    /// and copies its logits — the loop is here for symmetry with a chunked
    /// caller, not because a decode-only artifact ever gets a wide chunk.
    private func decodeOnlyPrefill(
        tokens: [Int32],
        startPosition: Int32,
        logitsRow: Int,
        kvState: KVCacheState
    ) throws -> MLMultiArray {
        var row: MLMultiArray?
        for (i, token) in tokens.enumerated() {
            // autoreleasepool: without this, Metal-backed prediction temporaries
            // (IOSurface buffers, intermediate MLMultiArrays) accumulate across
            // the inner decodes — small per call, large enough cumulatively to
            // OOM on iPhone 12 Pro the moment the user starts typing.
            try autoreleasepool {
                let logits = try decode(
                    token: token, position: startPosition + Int32(i), kvState: kvState
                )
                if i == logitsRow {
                    // Copy: the decode logits live in a backing that the next
                    // step overwrites.
                    row = try PredictionBuffer.extractRow(0, from: logits, what: "decode logits")
                }
            }
        }
        guard let row else {
            throw KVCacheError.unexpectedBufferLayout("prefill chunk produced no logits row")
        }
        return row
    }

    /// Run one decode step. The returned logits live in a double-buffered
    /// backing: they stay valid across the *next* decode and are overwritten by
    /// the one after, which is exactly the lifetime the sampling loop needs.
    public func decode(
        token: Int32,
        position: Int32,
        kvState: KVCacheState
    ) throws -> MLMultiArray {
        guard Int(position) < kvState.size else {
            throw CoreMLModelError.positionOutOfRange(
                position: Int(position), cacheSize: kvState.size
            )
        }
        let model = try function(prefix: "decode", size: kvState.size)
        kvState.setScalar(token, in: kvState.tokenScalar)
        kvState.setScalar(position, in: kvState.positionScalar)

        let inputs: [String: MLMultiArray] = [
            decodeIO.tokenInputName: kvState.tokenScalar,
            decodeIO.positionInputName: kvState.positionScalar,
            decodeIO.ringInputName: kvState.ring,
        ]

        let logitsBacking = try kvState.nextDecodeLogitsBacking(
            shape: decodeIO.logitsShape, dataType: decodeIO.logitsDataType
        )
        return try predict(
            model: model, io: decodeIO, inputs: inputs,
            logitsBacking: logitsBacking, kvState: kvState
        )
    }

    /// Shared prediction body: hand CoreML our preallocated output buffers, run
    /// the stateful prediction, take the new ring, and return the logits.
    ///
    /// The KV caches never appear here — they are state, mutated in place
    /// inside `kvState.caches`.
    private func predict(
        model: MLModel,
        io: FunctionIO,
        inputs: [String: MLMultiArray],
        logitsBacking: MLMultiArray,
        kvState: KVCacheState
    ) throws -> MLMultiArray {
        let options = MLPredictionOptions()
        options.outputBackings = [
            io.logitsOutputName: logitsBacking,
            io.ringOutputName: kvState.ringSpare,
        ]
        let result = try model.prediction(
            from: CoreMLInputProvider(values: inputs),
            using: kvState.caches,
            options: options
        )
        guard let ring = result.featureValue(for: io.ringOutputName)?.multiArrayValue else {
            throw CoreMLModelError.missingOutput(io.ringOutputName)
        }
        guard let logits = result.featureValue(for: io.logitsOutputName)?.multiArrayValue else {
            throw CoreMLModelError.missingOutput(io.logitsOutputName)
        }
        try kvState.adoptRing(ring)
        if logits !== logitsBacking {
            kvState.noteIgnoredBacking(io.logitsOutputName)
        }
        return logits
    }

    // MARK: - I/O Classification

    /// Read one function's I/O names, shapes, and dtypes, rejecting artifacts
    /// that predate stateful KV caches.
    ///
    /// With every cache declared as state, the signature is small and rigid:
    /// inputs are the token, the position, the int32 `sliding_pos_ring`, and
    /// (on exports that keep it) the cache-length `N`; outputs are the
    /// float logits and the updated ring. Anything named `k_<n>` / `v_<n>` on
    /// the signature means the caches still cross the boundary.
    static func classifyIO(model: MLModel, function: String) throws -> FunctionIO {
        let description = model.modelDescription
        let stateNames = description.stateDescriptionsByName.keys.sorted()
        let inputs = description.inputDescriptionsByName
        let outputs = description.outputDescriptionsByName

        let cacheIO = (Array(inputs.keys) + Array(outputs.keys)).filter(isCacheName).sorted()
        guard cacheIO.isEmpty, !stateNames.isEmpty else {
            throw CoreMLModelError.modelPredatesCacheStates(
                function: function, cacheFeatures: cacheIO
            )
        }

        // Outputs: one float tensor (logits) and one int32 tensor (the ring).
        var logitsName: String?
        var ringOutputName: String?
        for (name, desc) in outputs {
            guard let c = desc.multiArrayConstraint else { continue }
            if c.dataType == .int32 { ringOutputName = name } else { logitsName = name }
        }
        guard let logitsName, let logitsConstraint = outputs[logitsName]?.multiArrayConstraint else {
            throw CoreMLModelError.unexpectedSignature(
                function: function, detail: "no float logits output among \(outputs.keys.sorted())"
            )
        }
        guard let ringOutputName else {
            throw CoreMLModelError.unexpectedSignature(
                function: function, detail: "no int32 ring output among \(outputs.keys.sorted())"
            )
        }

        // The ring input is the same feature one step earlier, so the exporter
        // names it either identically or with the `_out` suffix stripped.
        var ringCandidates = [ringOutputName]
        if ringOutputName.hasSuffix("_out") {
            ringCandidates.append(String(ringOutputName.dropLast("_out".count)))
        }
        let ringInputName = ringCandidates.first { inputs[$0] != nil }
        guard let ringInputName, let ringConstraint = inputs[ringInputName]?.multiArrayConstraint else {
            throw CoreMLModelError.unexpectedSignature(
                function: function,
                detail: "no input matching ring output '\(ringOutputName)' among \(inputs.keys.sorted())"
            )
        }

        // Token and position are all that is left: the caches are state, and
        // `concretize_cache_length` folds each function's own cache length into
        // the graph, so nothing else crosses the boundary.
        let control = inputs.keys.filter { $0 != ringInputName }.sorted()
        guard control.count == 2 else {
            throw CoreMLModelError.unexpectedSignature(
                function: function,
                detail: "expected token + position inputs, got \(control)"
            )
        }
        let (tokenName, positionName) = identifyTokenAndPosition(control, inputs: inputs)
        let tokenLength = inputs[tokenName]?.multiArrayConstraint?.shape
            .map { $0.intValue }.reduce(1, *) ?? 1

        return FunctionIO(
            logitsOutputName: logitsName,
            logitsShape: logitsConstraint.shape,
            logitsDataType: logitsConstraint.dataType,
            tokenInputName: tokenName,
            tokenLength: tokenLength,
            positionInputName: positionName,
            ringInputName: ringInputName,
            ringOutputName: ringOutputName,
            ringShape: ringConstraint.shape,
            ringDataType: ringConstraint.dataType,
            stateNames: stateNames
        )
    }

    /// `k_<n>` / `v_<n>`: a KV cache tensor on the function signature.
    private static func isCacheName(_ name: String) -> Bool {
        guard name.count >= 3 else { return false }
        var chars = Array(name)
        guard chars[0] == "k" || chars[0] == "v", chars[1] == "_" else { return false }
        chars.removeFirst(2)
        // `k_4_out` counts too — it is the same cache leaving the function.
        let digits = chars.prefix { $0.isNumber }
        guard !digits.isEmpty else { return false }
        let rest = String(chars.dropFirst(digits.count))
        return rest.isEmpty || rest == "_out"
    }

    /// Tell the token input from the position input. Prefill's token input is
    /// `[1, chunk]` so element count settles it; decode's are both `[1]`, where
    /// the name does.
    private static func identifyTokenAndPosition(
        _ names: [String], inputs: [String: MLFeatureDescription]
    ) -> (token: String, position: String) {
        func count(_ name: String) -> Int {
            inputs[name]?.multiArrayConstraint?.shape.map { $0.intValue }.reduce(1, *) ?? 1
        }
        let (a, b) = (names[0], names[1])
        if count(a) != count(b) {
            return count(a) > count(b) ? (a, b) : (b, a)
        }
        if a.contains("token") { return (a, b) }
        if b.contains("token") { return (b, a) }
        return (a, b)
    }
}

// MARK: - Input Provider

/// Minimal `MLFeatureProvider` over a name → array dictionary.
final class CoreMLInputProvider: MLFeatureProvider {
    let featureNames: Set<String>
    private let values: [String: MLFeatureValue]

    init(values: [String: MLMultiArray]) {
        self.values = values.mapValues { MLFeatureValue(multiArray: $0) }
        self.featureNames = Set(values.keys)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        values[featureName]
    }
}

// MARK: - Errors

public enum CoreMLModelError: Error, LocalizedError {
    /// The artifact declares no materialized function pairs at all.
    case notMaterialized(String)
    /// The artifact declares materialized functions, but none usable in the
    /// requested mode (e.g. prefill wanted, only `decode_<N>` exported).
    case noUsableMaterializedFunctions(decodeSizes: [Int], prefillSizes: [Int])
    /// A KV cache still crosses the function signature, i.e. the artifact was
    /// exported before the caches became CoreML state.
    case modelPredatesCacheStates(function: String, cacheFeatures: [String])
    /// The function's inputs/outputs are not the shape this runtime expects.
    case unexpectedSignature(function: String, detail: String)
    /// A declared output was missing from a prediction result.
    case missingOutput(String)
    /// A token position doesn't fit the allocated KV cache.
    case positionOutOfRange(position: Int, cacheSize: Int)

    public var errorDescription: String? {
        switch self {
        case .notMaterialized(let name):
            "\(name) declares no `decode_<N>` functions — run `uv run gemma-materialize` on the exported package"
        case .noUsableMaterializedFunctions(let decodeSizes, let prefillSizes):
            "No usable materialized function pairs (decode sizes: \(decodeSizes), prefill sizes: \(prefillSizes))"
        case .modelPredatesCacheStates(let function, let cacheFeatures):
            "Function '\(function)' passes KV caches through its signature (\(cacheFeatures.isEmpty ? "no state features at all" : cacheFeatures.joined(separator: ", "))) — this model predates global-cache states. Re-run `uv run gemma-export`."
        case .unexpectedSignature(let function, let detail):
            "Function '\(function)' has an unexpected signature: \(detail)"
        case .missingOutput(let name):
            "Prediction result is missing output '\(name)'"
        case .positionOutOfRange(let position, let cacheSize):
            "Token position \(position) does not fit a KV cache of size \(cacheSize)"
        }
    }
}
