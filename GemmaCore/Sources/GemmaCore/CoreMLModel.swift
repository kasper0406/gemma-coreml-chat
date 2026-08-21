/// CoreML model wrapper for the multifunction Gemma4-E2B .mlpackage.
///
/// Supports two model layouts:
/// - **Standard**: Two functions named `decode` and `prefill`, with optional
///   RangeDim on global KV cache inputs.
/// - **Materialized**: Concrete-shape functions named `decode_<N>` /
///   `prefill_<N>`, one pair per size in the export's `--materialize-sizes`
///   (powers of two by default, but any ascending set is legal, and a
///   `--decode-only` export has no `prefill_<N>` at all). Each function is
///   specialized to a specific global KV cache size (no dynamic shape ops).
///   The runtime selects the function matching the current cache size — see
///   ``KVCacheSizePolicy``, which is the only place that bucketing lives.
///
/// Materialized models are produced by `gemma-materialize` and work on all
/// backends including ANE and iPhone, whereas standard RangeDim models only
/// work on GPU.
///
/// Every function declares the 12 sliding-window KV caches as CoreML **state**
/// (`k_<slot>` / `v_<slot>`), so only the 3 global caches and the int32
/// `sliding_pos_ring` are passed in and out per call. An artifact without those
/// state features predates the change and is rejected at load — re-run
/// `gemma-export`.

import CoreML
import CryptoKit
import Foundation

public final class CoreMLModel: @unchecked Sendable {
    /// Logits output name for each function.
    public let decodeLogitsName: String
    public let prefillLogitsName: String

    /// Token input name for each function.
    public let decodeTokenInputName: String
    public let prefillTokenInputName: String

    /// Position input name for each function.
    public let decodePositionInputName: String
    public let prefillPositionInputName: String

    /// "N" phantom input name (global cache dim), nil for materialized/fixed-shape models.
    public let decodeNInputName: String?
    public let prefillNInputName: String?

    /// KV cache input/output names, in matched order.
    public let decodeKVInputNames: [String]
    public let decodeKVOutputNames: [String]
    public let prefillKVInputNames: [String]
    public let prefillKVOutputNames: [String]

    /// Global KV input names (caches whose dim-1 varies with context length).
    public let globalKVInputNames: Set<String>

    /// Names of the sliding KV caches, which are CoreML state rather than I/O.
    public let slidingStateNames: Set<String>

    /// Shapes of each prefill KV input, extracted once so the prefill model
    /// can be released without losing the metadata needed to re-seed KV caches.
    public let prefillKVShapes: [String: [NSNumber]]
    /// Dtypes of each prefill KV input (companion to `prefillKVShapes`).
    public let prefillKVDtypes: [String: MLMultiArrayDataType]

    /// Available materialized sizes (sorted ascending), or nil for standard models.
    /// If the caller passed `maxContextSize`, this is the filtered list.
    public let materializedSizes: [Int]?

    /// Largest sequence length this model can actually handle. For standard
    /// (RangeDim) models, `GemmaConfig.maxSeqLen`. For materialized models,
    /// the largest retained size — either all sizes from the manifest or the
    /// caller-imposed `maxContextSize` cap. The engine uses this instead of
    /// `GemmaConfig.maxSeqLen` so KV growth never exceeds a size we actually
    /// loaded a function for.
    public let effectiveMaxSeqLen: Int

    /// True when only decode functions are loaded. `prefill()` calls fall back
    /// to running `decode()` per-token internally — slower (no chunked prefill
    /// kernel), but halves resident-MLModel count, which is the difference
    /// between fitting and OOM on tight devices like iPhone 12 Pro.
    public let isDecodeOnly: Bool

    /// URL of the compiled .mlmodelc (for lazy function loading).
    private let modelURL: URL
    /// URL the caller originally passed to `load(from:)` (.mlpackage or .mlmodelc).
    /// Part of the warm-cache sentinel's identity, so two models that merely
    /// share a basename don't share a sentinel.
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

    /// Function state keyed by function name (e.g. "decode" or "decode_512").
    private var functions: [String: LoadState]
    private var nextLoadID: UInt64 = 0
    private let cacheLock = NSLock()

    /// The function instance `makeState()` allocates from. Any loaded instance
    /// works — states are keyed by name and shared across every function of the
    /// package — so this is just "one we know is loaded" (the bootstrap decode).
    private let stateSourceModel: MLModel

    private init(
        prefillIO: ClassifiedIO,
        prefillKVShapes: [String: [NSNumber]],
        prefillKVDtypes: [String: MLMultiArrayDataType],
        decodeIO: ClassifiedIO,
        globalKVInputNames: Set<String>,
        materializedSizes: [Int]?,
        effectiveMaxSeqLen: Int,
        isDecodeOnly: Bool,
        modelURL: URL,
        sourceURL: URL,
        sourceFingerprint: String?,
        computeUnits: MLComputeUnits,
        stateSourceModel: MLModel,
        initialFunctions: [String: MLModel]
    ) {
        self.prefillLogitsName = prefillIO.logitsOutputName
        self.decodeLogitsName = decodeIO.logitsOutputName
        self.prefillTokenInputName = prefillIO.tokenInputName
        self.decodeTokenInputName = decodeIO.tokenInputName
        self.prefillPositionInputName = prefillIO.positionInputName
        self.decodePositionInputName = decodeIO.positionInputName
        self.prefillNInputName = prefillIO.nInputName
        self.decodeNInputName = decodeIO.nInputName
        self.prefillKVInputNames = prefillIO.kvInputNames
        self.prefillKVOutputNames = prefillIO.kvOutputNames
        self.decodeKVInputNames = decodeIO.kvInputNames
        self.decodeKVOutputNames = decodeIO.kvOutputNames
        self.prefillKVShapes = prefillKVShapes
        self.prefillKVDtypes = prefillKVDtypes
        self.globalKVInputNames = globalKVInputNames
        self.slidingStateNames = decodeIO.stateNames
        self.materializedSizes = materializedSizes
        self.effectiveMaxSeqLen = effectiveMaxSeqLen
        self.isDecodeOnly = isDecodeOnly
        self.modelURL = modelURL
        self.sourceURL = sourceURL
        self.sourceFingerprint = sourceFingerprint
        self.computeUnits = computeUnits
        self.stateSourceModel = stateSourceModel
        self.functions = initialFunctions.mapValues { .loaded($0) }
    }

    /// Allocate a fresh, zero-filled set of sliding KV cache buffers.
    ///
    /// One `MLState` serves every function in the package: `decode_512`,
    /// `prefill_1024`, … all bind the same named state buffers, so the runtime
    /// can switch sizes mid-conversation without touching the sliding caches.
    /// Make a new one per conversation — reusing one across a reset would leave
    /// stale K/V that a re-populated `sliding_pos_ring` marks valid again.
    public func makeState() -> MLState {
        stateSourceModel.makeState()
    }

    /// A zeroed KV state: fresh global caches, fresh `sliding_pos_ring`, and a
    /// fresh `MLState` for the sliding caches.
    public func makeEmptyKVState(initialGlobalSize: Int? = nil) throws -> KVCacheState {
        try KVCacheState.empty(
            kvInputNames: prefillKVInputNames,
            shapes: prefillKVShapes,
            dtypes: prefillKVDtypes,
            globalNames: globalKVInputNames,
            initialGlobalSize: initialGlobalSize,
            slidingCaches: makeState()
        )
    }

    /// Bucketing policy for this model's global KV caches. Hand this to
    /// ``KVCacheState/grownToFit(needed:policy:)`` and anything else that needs
    /// to size a cache, so cache shape and resolved function never disagree.
    public var cacheSizePolicy: KVCacheSizePolicy {
        KVCacheSizePolicy(materializedSizes: materializedSizes, maxLen: effectiveMaxSeqLen)
    }

    /// Load the multifunction model from a .mlpackage or .mlmodelc URL.
    ///
    /// Auto-detects whether the model uses standard (`decode`/`prefill`)
    /// or materialized (`decode_64`/`prefill_64`/…) function names.
    ///
    /// For .mlpackage files, the model is compiled and cached as .mlmodelc
    /// next to the source for fast subsequent loads (E5RT cache reuse).
    /// For .mlmodelc files, loads directly without recompilation.
    ///
    /// - Parameter maxContextSize: For materialized models, only retain function
    ///   pairs ≤ this size. Loading fewer functions is critical on memory-
    ///   constrained devices like iPhone, where loading all 16 pairs OOMs.
    ///   Ignored for standard models.
    /// - Parameter decodeOnly: For materialized models, skip loading prefill
    ///   functions entirely. `prefill()` falls back to per-token `decode()`
    ///   internally — slower but halves resident MLModel count, which is the
    ///   only way the model fits on iPhone 12 Pro / 6 GB devices. Ignored for
    ///   standard models; forced on for `--decode-only` artifacts, which
    ///   export no prefill functions to load.
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
    /// build (e.g. non-materialized ↔ materialized) can leave the source older
    /// than an existing cache, masking a real change.
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
    /// The layout is decided from `model.mil` — a text parse, no `MLModel.load`
    /// — rather than by trial-loading. An artifact that declares `decode_<N>`
    /// functions goes straight down the materialized path and is never probed
    /// with parallel loads, which is exactly the multi-MLModel memory spike the
    /// serial bootstrap exists to avoid.
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

        let declared = enumerateMaterializedFunctions(compiledURL: url)
        if let declared, !declared.decodeSizes.isEmpty {
            return try await loadMaterialized(
                from: url,
                sourceURL: sourceURL,
                sourceFingerprint: sourceFingerprint,
                computeUnits: computeUnits,
                maxContextSize: maxContextSize,
                decodeOnly: decodeOnly,
                backgroundPreload: backgroundPreload,
                declared: declared
            )
        }

        // Not a materialized artifact, or `model.mil` was unreadable. Standard
        // models have no meaningful decode-only mode — there is one `decode`
        // function and one `prefill` function — so `decodeOnly` is ignored
        // rather than making the load impossible.
        if decodeOnly && declared != nil {
            Log.info("[CoreML] decodeOnly requested but \(url.lastPathComponent) declares no materialized functions — loading decode + prefill")
        }
        do {
            return try await loadStandard(
                from: url,
                sourceURL: sourceURL,
                sourceFingerprint: sourceFingerprint,
                computeUnits: computeUnits
            )
        } catch let standardError {
            // Reached when `model.mil` was unreadable (layout genuinely
            // unknown) or when it declared no materialized functions but the
            // standard load still failed — including a `.mil` parse that
            // missed real declarations due to format drift. Probe for
            // materialized functions as a desktop-only safety net; on a truly
            // standard artifact the probe fails fast on unknown function
            // names. If it fails too, surface BOTH errors —
            // the standard one is usually the real cause (corrupt bundle,
            // unsupported compute units) and used to be demoted to a log line
            // and replaced by an unrelated `.modelNotFound`.
            Log.info("[CoreML] Standard function load failed (\(standardError.localizedDescription)), probing for materialized functions...")
            do {
                return try await loadMaterialized(
                    from: url,
                    sourceURL: sourceURL,
                    sourceFingerprint: sourceFingerprint,
                    computeUnits: computeUnits,
                    maxContextSize: maxContextSize,
                    decodeOnly: decodeOnly,
                    backgroundPreload: backgroundPreload,
                    declared: nil
                )
            } catch let materializedError {
                throw CoreMLModelError.loadFailed(
                    standard: standardError, materialized: materializedError
                )
            }
        }
    }

    /// Load a standard two-function model (decode + prefill).
    private static func loadStandard(
        from url: URL,
        sourceURL: URL,
        sourceFingerprint: String?,
        computeUnits: MLComputeUnits
    ) async throws -> CoreMLModel {
        let decodeConfig = MLModelConfiguration()
        decodeConfig.computeUnits = computeUnits
        decodeConfig.functionName = "decode"

        let prefillConfig = MLModelConfiguration()
        prefillConfig.computeUnits = computeUnits
        prefillConfig.functionName = "prefill"

        async let decodeTask = MLModel.load(contentsOf: url, configuration: decodeConfig)
        async let prefillTask = MLModel.load(contentsOf: url, configuration: prefillConfig)

        let decodeModel = try await decodeTask
        let prefillModel = try await prefillTask
        Log.info("[CoreML] Both functions loaded (standard mode).")

        let decodeIO = classifyIO(model: decodeModel)
        let prefillIO = classifyIO(model: prefillModel)
        try requireStatefulKV(decodeIO, function: "decode")
        try requireStatefulKV(prefillIO, function: "prefill")
        let (prefillKVShapes, prefillKVDtypes) = extractKVMetadata(
            model: prefillModel, kvInputNames: prefillIO.kvInputNames
        )
        logIOSummary(decodeIO: decodeIO, prefillIO: prefillIO)

        let globalNames = detectFlexibleGlobalKV(
            model: decodeModel, kvInputNames: decodeIO.kvInputNames
        )
        if !globalNames.isEmpty {
            Log.info("[CoreML] Flexible global KV caches: \(globalNames.sorted())")
        }

        return CoreMLModel(
            prefillIO: prefillIO,
            prefillKVShapes: prefillKVShapes,
            prefillKVDtypes: prefillKVDtypes,
            decodeIO: decodeIO,
            globalKVInputNames: globalNames,
            materializedSizes: nil,
            effectiveMaxSeqLen: GemmaConfig.maxSeqLen,
            isDecodeOnly: false,
            modelURL: url,
            sourceURL: sourceURL,
            sourceFingerprint: sourceFingerprint,
            computeUnits: computeUnits,
            stateSourceModel: decodeModel,
            initialFunctions: ["decode": decodeModel, "prefill": prefillModel]
        )
    }

    /// Reject artifacts exported before the sliding KV caches became state.
    ///
    /// Such a model takes all 30 caches as inputs, so it would load and then
    /// fail per-prediction on missing features — or worse, silently run with
    /// whatever the runtime defaulted them to. Fail at load with the fix.
    private static func requireStatefulKV(
        _ io: ClassifiedIO, function: String
    ) throws {
        guard io.stateNames.isEmpty else { return }
        throw CoreMLModelError.modelPredatesStatefulKVCaches(
            function: function, kvInputCount: io.kvInputNames.count
        )
    }

    /// Load a materialized model with concrete function names.
    ///
    /// Strategy tuned for memory-constrained devices (iPhone OOMs on parallel
    /// loads of 2–3 concurrent `MLModel` instances):
    ///   1. Take the declared function set from `model.mil` (text parse, no
    ///      MLModel.load), supplied by the caller as `declared`.
    ///   2. Classify globals vs. slidings by comparing decode function
    ///      signatures in `model.mil` text — also no MLModel.load. This
    ///      replaces the prior "probe-load decode_{second}" classifier which
    ///      was the memory spike that killed the iPhone path.
    ///   3. Load decode_{smallest} and prefill_{smallest} SERIALLY. Peak
    ///      live MLModel count is 1 during bootstrap.
    ///   4. Optionally background-preload the remaining retained sizes.
    ///
    /// `declared == nil` means the manifest was unreadable, and only then does
    /// this fall back to a parallel probe (desktop-only safety net).
    private static func loadMaterialized(
        from url: URL,
        sourceURL: URL,
        sourceFingerprint: String?,
        computeUnits: MLComputeUnits,
        maxContextSize: Int?,
        decodeOnly: Bool,
        backgroundPreload: Bool,
        declared: MaterializedFunctions?
    ) async throws -> CoreMLModel {
        let sizes: [Int]
        let globalNames: Set<String>
        var effectiveDecodeOnly = decodeOnly

        if let declared {
            // A `gemma-export --decode-only` artifact has decode_<N> and no
            // prefill_<N>. Insisting on the decode∩prefill intersection there
            // yields no sizes at all, which used to drop us into the parallel
            // probe and then fail loading a `prefill_<N>` that never existed.
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
            if sizes.count >= 2 {
                // A failed parse must NOT degrade to "no global caches":
                // globalKVInputNames would be empty, growth a no-op, decode
                // forever pinned to the smallest function, and every
                // conversation silently truncated. Fail loudly instead.
                guard let g = detectGlobalsFromMil(
                    compiledURL: url, sizeA: sizes[0], sizeB: sizes[1]
                ) else {
                    throw CoreMLModelError.globalKVDetectionFailed(
                        "could not parse the decode_\(sizes[0]) / decode_\(sizes[1]) signatures out of model.mil"
                    )
                }
                globalNames = g
                Log.info("[CoreML] Global KV caches (from .mil, \(g.count)): \(g.sorted())")
            } else {
                // Exactly one size: growth is a no-op, so an empty global set
                // is correct rather than a degraded guess.
                globalNames = []
                Log.info("[CoreML] Only one materialized size; no classification needed")
            }
        } else {
            Log.info("[CoreML] Manifest enumeration unavailable; falling back to parallel probe")
            let candidateSizes = (6...16).map { 1 << $0 }  // 64..65536
            let probeResults: [(size: Int, model: SendableMLModel?)] = await withTaskGroup(
                of: (Int, SendableMLModel?).self
            ) { group in
                for s in candidateSizes {
                    group.addTask {
                        let config = MLModelConfiguration()
                        config.computeUnits = computeUnits
                        config.functionName = "decode_\(s)"
                        let model = try? await MLModel.load(contentsOf: url, configuration: config)
                        return (s, model.map { SendableMLModel(model: $0) })
                    }
                }
                var out: [(size: Int, model: SendableMLModel?)] = []
                for await r in group { out.append(r) }
                return out
            }
            let loaded = probeResults
                .compactMap { r in r.model.map { (size: r.size, model: $0.model) } }
                .sorted { $0.size < $1.size }
            guard !loaded.isEmpty else { throw CoreMLModelError.modelNotFound }
            sizes = loaded.map { $0.size }
            // Fallback classifier: use the probed MLModels since we already have them.
            let decodeIO0 = classifyIO(model: loaded[0].model)
            if loaded.count >= 2 {
                let g = detectGlobalsByShape(
                    modelA: loaded[0].model,
                    modelB: loaded[1].model,
                    kvInputNames: decodeIO0.kvInputNames
                )
                guard !g.isEmpty else {
                    throw CoreMLModelError.globalKVDetectionFailed(
                        "no KV input changes dim-1 between decode_\(sizes[0]) and decode_\(sizes[1])"
                    )
                }
                globalNames = g
            } else {
                globalNames = []
            }
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
        let prefillName = "prefill_\(bootSize)"

        // Serial loads keep peak live-MLModel count at 1 during bootstrap.
        let decodeModel = try await Self.loadFunction(
            url: url, computeUnits: computeUnits, function: "decode_\(bootSize)"
        )
        let prefillModel: MLModel?
        if effectiveDecodeOnly {
            prefillModel = nil
            Log.info("[CoreML] Loaded decode_\(bootSize) (decode-only; prefill skipped)")
        } else {
            prefillModel = try await Self.loadFunction(
                url: url, computeUnits: computeUnits, function: prefillName
            )
            Log.info("[CoreML] Loaded decode_\(bootSize) + \(prefillName) (serial)")
        }

        let decodeIO = classifyIO(model: decodeModel)
        // In decode-only mode, prefill metadata is borrowed from decode: the
        // KV cache layout (input names, shapes, dtypes) is identical between
        // the two functions, and the per-token loop in `decodeOnlyPrefill`
        // doesn't use prefill's own token/position input names.
        let prefillIO = prefillModel.map { classifyIO(model: $0) } ?? decodeIO
        try requireStatefulKV(decodeIO, function: "decode_\(bootSize)")
        if prefillModel != nil {
            try requireStatefulKV(prefillIO, function: prefillName)
        }
        let prefillSourceModel = prefillModel ?? decodeModel
        let (prefillKVShapes, prefillKVDtypes) = extractKVMetadata(
            model: prefillSourceModel, kvInputNames: prefillIO.kvInputNames
        )
        logIOSummary(decodeIO: decodeIO, prefillIO: prefillIO)

        var initialFunctions: [String: MLModel] = [
            "decode_\(bootSize)": decodeModel
        ]
        if let p = prefillModel {
            initialFunctions[prefillName] = p
        }

        let effectiveMax = retainedSizes.last ?? sizes.last ?? GemmaConfig.maxSeqLen
        let instance = CoreMLModel(
            prefillIO: prefillIO,
            prefillKVShapes: prefillKVShapes,
            prefillKVDtypes: prefillKVDtypes,
            decodeIO: decodeIO,
            globalKVInputNames: globalNames,
            materializedSizes: retainedSizes,
            effectiveMaxSeqLen: effectiveMax,
            isDecodeOnly: effectiveDecodeOnly,
            modelURL: url,
            sourceURL: sourceURL,
            sourceFingerprint: sourceFingerprint,
            computeUnits: computeUnits,
            stateSourceModel: decodeModel,
            initialFunctions: initialFunctions
        )
        if backgroundPreload {
            instance.preloadAllSizes()
        }
        return instance
    }

    /// Identify global KV inputs by parsing two decode function signatures out
    /// of `model.mil` and comparing each parameter's dim-1. A parameter whose
    /// dim-1 scales with the materialized size is a global cache; one that
    /// stays fixed (= sliding window) is a sliding cache. Pure text parse —
    /// avoids an `MLModel.load` just to read shapes, which is the peak-memory
    /// hotspot on iPhone.
    ///
    /// Returns nil on any parse failure so the caller can decide how to
    /// degrade (fallback probe load, or an empty-globals assumption).
    static func detectGlobalsFromMil(
        compiledURL: URL, sizeA: Int, sizeB: Int
    ) -> Set<String>? {
        let milURL = compiledURL.appendingPathComponent("model.mil")
        guard let text = try? String(contentsOf: milURL, encoding: .utf8) else {
            return nil
        }
        guard let dimsA = parseDecodeDim1(text: text, size: sizeA),
              let dimsB = parseDecodeDim1(text: text, size: sizeB) else {
            return nil
        }
        var globals = Set<String>()
        for (name, a) in dimsA {
            if let b = dimsB[name], a != b {
                globals.insert(name)
            }
        }
        return globals
    }

    /// Return `[param-name: dim-1]` for every tensor parameter of `decode_<size>`
    /// in the supplied `.mil` text, or nil if the function or its signature
    /// can't be found.
    private static func parseDecodeDim1(text: String, size: Int) -> [String: Int]? {
        let sigPattern = #"func\s+decode_\#(size)\b[^(]*\(([^)]*)\)"#
        guard let sigRe = try? NSRegularExpression(pattern: sigPattern) else { return nil }
        let full = NSRange(text.startIndex..<text.endIndex, in: text)
        guard let m = sigRe.firstMatch(in: text, range: full),
              m.numberOfRanges >= 2,
              let r = Range(m.range(at: 1), in: text) else {
            return nil
        }
        let sig = String(text[r])

        // Each tensor param is `tensor<dtype, [d0, d1, ...]> name`. The sliding
        // KV caches are state params, spelled `state<tensor<…>> name`, and the
        // trailing `>>` makes them fall out of this pattern — which is what we
        // want: their length is the sliding window, never the context size.
        let paramPattern = #"tensor<[^,]+,\s*\[([^\]]+)\]>\s*([A-Za-z_][A-Za-z0-9_]*)"#
        guard let paramRe = try? NSRegularExpression(pattern: paramPattern) else { return nil }
        var result: [String: Int] = [:]
        let sigRange = NSRange(sig.startIndex..<sig.endIndex, in: sig)
        paramRe.enumerateMatches(in: sig, range: sigRange) { match, _, _ in
            guard let m = match, m.numberOfRanges >= 3,
                  let shapeRange = Range(m.range(at: 1), in: sig),
                  let nameRange = Range(m.range(at: 2), in: sig) else { return }
            let dims = sig[shapeRange].split(separator: ",").compactMap {
                Int($0.trimmingCharacters(in: .whitespaces))
            }
            guard dims.count >= 2 else { return }
            result[String(sig[nameRange])] = dims[1]
        }
        return result.isEmpty ? nil : result
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

    /// Identify global KV inputs by comparing dim-1 across two sizes.
    /// Globals scale (different dim-1); slidings stay fixed.
    private static func detectGlobalsByShape(
        modelA: MLModel, modelB: MLModel, kvInputNames: [String]
    ) -> Set<String> {
        let descsA = modelA.modelDescription.inputDescriptionsByName
        let descsB = modelB.modelDescription.inputDescriptionsByName
        var globals = Set<String>()
        for name in kvInputNames {
            guard let ca = descsA[name]?.multiArrayConstraint,
                  let cb = descsB[name]?.multiArrayConstraint else { continue }
            let sa = ca.shape.map { $0.intValue }
            let sb = cb.shape.map { $0.intValue }
            guard sa.count >= 2, sb.count >= 2 else { continue }
            if sa[1] != sb[1] { globals.insert(name) }
        }
        return globals
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

    /// Extract KV shape/dtype metadata from a model's input descriptions.
    private static func extractKVMetadata(
        model: MLModel, kvInputNames: [String]
    ) -> ([String: [NSNumber]], [String: MLMultiArrayDataType]) {
        let inputDescs = model.modelDescription.inputDescriptionsByName
        var shapes: [String: [NSNumber]] = [:]
        var dtypes: [String: MLMultiArrayDataType] = [:]
        for name in kvInputNames {
            guard let c = inputDescs[name]?.multiArrayConstraint else { continue }
            shapes[name] = c.shape
            dtypes[name] = c.dataType
        }
        return (shapes, dtypes)
    }

    /// Log I/O classification summary.
    private static func logIOSummary(decodeIO: ClassifiedIO, prefillIO: ClassifiedIO) {
        Log.info("[CoreML] Decode: logits=\(decodeIO.logitsOutputName), token=\(decodeIO.tokenInputName), pos=\(decodeIO.positionInputName), kvIn=\(decodeIO.kvInputNames.count), kvOut=\(decodeIO.kvOutputNames.count), slidingStates=\(decodeIO.stateNames.count)")
        Log.info("[CoreML] Prefill: logits=\(prefillIO.logitsOutputName), token=\(prefillIO.tokenInputName), pos=\(prefillIO.positionInputName), kvIn=\(prefillIO.kvInputNames.count), kvOut=\(prefillIO.kvOutputNames.count), slidingStates=\(prefillIO.stateNames.count)")
        if decodeIO.nInputName != nil {
            Log.info("[CoreML] Dynamic context: N input detected (decode=\(decodeIO.nInputName!), prefill=\(prefillIO.nInputName ?? "none"))")
        }
    }

    // MARK: - Function Resolution

    /// Round a cache size up to the nearest materialized size, clamped to the
    /// largest size this model actually loaded. Returns nil for standard
    /// (non-materialized) models.
    ///
    /// Delegates to ``cacheSizePolicy`` so this and `KVCacheState.grownToFit`
    /// can never disagree. Note the clamp: asking for more than
    /// `effectiveMaxSeqLen` yields `effectiveMaxSeqLen`, which is *smaller*
    /// than requested — callers must cap their own token counts too.
    public func materializedSize(forCacheSize cacheSize: Int) -> Int? {
        guard materializedSizes != nil else { return nil }
        return cacheSizePolicy.size(forNeeded: cacheSize)
    }

    /// Resolve the function name for a given prefix and cache size.
    /// For materialized models, always picks a concrete `{prefix}_{size}` —
    /// even when the caller passes no cache-size hint — because `prefix`
    /// alone (e.g. `"decode"`) isn't an exported function.
    private func functionName(prefix: String, cacheSize: Int?) -> String {
        if let sizes = materializedSizes {
            if let cacheSize, let size = materializedSize(forCacheSize: cacheSize) {
                return "\(prefix)_\(size)"
            }
            return "\(prefix)_\(sizes[0])"
        }
        return prefix
    }

    /// Get a loaded model by function name. Crashes if not loaded.
    private func getFunction(_ name: String) -> MLModel {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        guard case .loaded(let model) = functions[name] else {
            fatalError("[CoreML] Function '\(name)' not loaded. Call ensureLoaded(forGlobalCacheSize:) first.")
        }
        return model
    }

    /// Pre-load decode and prefill functions for a given global cache size.
    ///
    /// For standard models this is a no-op. For materialized models, loads the
    /// function pair matching the given cache size (if not already cached).
    /// Call from an async context before sync `prefill()`/`decode()` calls.
    public func ensureLoaded(forGlobalCacheSize cacheSize: Int) async throws {
        guard materializedSizes != nil else { return }
        let decodeName = functionName(prefix: "decode", cacheSize: cacheSize)
        let prefillName = functionName(prefix: "prefill", cacheSize: cacheSize)

        try await withThrowingTaskGroup(of: Void.self) { group in
            group.addTask { _ = try await self.loadIfNeeded(name: decodeName) }
            if !self.isDecodeOnly {
                group.addTask { _ = try await self.loadIfNeeded(name: prefillName) }
            }
            try await group.waitForAll()
        }
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
        guard let sizes = materializedSizes else { return }
        let names = sizes.flatMap {
            isDecodeOnly ? ["decode_\($0)"] : ["decode_\($0)", "prefill_\($0)"]
        }
        Task.detached { [self] in
            let start = CFAbsoluteTimeGetCurrent()
            let allOK = await self.drainLoads(names: names, concurrency: concurrency, progress: nil)
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
        guard let sizes = materializedSizes else {
            progress(1, 1)
            markWarmed()
            return
        }
        let names = sizes.flatMap {
            isDecodeOnly ? ["decode_\($0)"] : ["decode_\($0)", "prefill_\($0)"]
        }
        let allOK = await drainLoads(names: names, concurrency: concurrency, progress: progress)
        if allOK { markWarmed() }
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
        let sizes = materializedSizes.map {
            $0.map(String.init).joined(separator: ",")
        } ?? "standard"
        let scope = [
            sourceURL.standardizedFileURL.path,
            Self.computeUnitsTag(computeUnits),
            "decodeOnly=\(isDecodeOnly)",
            "sizes=\(sizes)",
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

    /// Run one prefill chunk.
    public func prefill(
        tokens: MLMultiArray,
        seqLen: Int32,
        kvState: KVCacheState,
        globalCacheSize: Int32? = nil
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        if isDecodeOnly {
            return try decodeOnlyPrefill(
                tokens: tokens, seqLen: seqLen,
                kvState: kvState, globalCacheSize: globalCacheSize
            )
        }
        let activeModel = getFunction(
            functionName(prefix: "prefill", cacheSize: globalCacheSize.map { Int($0) })
        )

        var features: [String: MLMultiArray] = [:]
        features[prefillTokenInputName] = tokens
        features[prefillPositionInputName] = MLMultiArray.int32Scalar(seqLen)
        if let nName = prefillNInputName, let nValue = globalCacheSize {
            features[nName] = MLMultiArray.int32Scalar(nValue)
        }

        let provider = try CoreMLInputProvider(
            features: features,
            kvNames: prefillKVInputNames,
            kvState: kvState
        )
        // The sliding caches are updated in place inside `slidingCaches`.
        let result = try activeModel.prediction(from: provider, using: kvState.slidingCaches)
        let logits = result.featureValue(for: prefillLogitsName)!.multiArrayValue!
        let newKV = try KVCacheState.from(
            prediction: result,
            outputNames: prefillKVOutputNames,
            inputNames: prefillKVInputNames,
            globalNames: kvState.globalNames,
            slidingCaches: kvState.slidingCaches
        )
        return (logits, newKV)
    }

    /// Per-token prefill via repeated `decode()` calls — the fallback used
    /// when only decode functions are loaded. Synthesizes a chunk-shaped
    /// `[chunkSize, vocabSize]` logits buffer by stacking each step's
    /// logits[0:vocabSize] row, so the engine's downstream
    /// `extractLogitsAt(position:)` math works unchanged.
    ///
    /// Slower than a real prefill function (no fused chunk kernel), but
    /// keeps the resident MLModel count at 1 instead of 2 — the only way
    /// to fit on iPhone 12 Pro / 6 GB.
    private func decodeOnlyPrefill(
        tokens: MLMultiArray,
        seqLen: Int32,
        kvState: KVCacheState,
        globalCacheSize: Int32?
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        let chunkSize = tokens.count
        let vocabSize = GemmaConfig.vocabSize

        guard tokens.dataType == .int32, tokens.strides.last?.intValue == 1 else {
            throw CoreMLModelError.unexpectedBufferLayout(
                "prefill tokens are \(tokens.dataType) with strides \(tokens.strides); expected contiguous int32"
            )
        }
        // This chunk writes KV rows seqLen ..< seqLen+chunkSize, so every one
        // of them has to fit the global cache. The real prefill kernel would
        // fault; the per-token loop would silently scribble past the end.
        if let gc = globalCacheSize, Int(seqLen) + chunkSize > Int(gc) {
            throw CoreMLModelError.positionOutOfRange(
                position: Int(seqLen) + chunkSize - 1, cacheSize: Int(gc)
            )
        }

        // Copy the token IDs out up front: `dataPointer` is only guaranteed
        // valid inside the accessor, and chunkSize is 8.
        var tokenIDs = [Int32](repeating: 0, count: chunkSize)
        tokens.withUnsafeBufferPointer(ofType: Int32.self) { src in
            for i in 0..<chunkSize { tokenIDs[i] = src[i] }
        }

        let result = try MLMultiArray(
            shape: [NSNumber(value: chunkSize), NSNumber(value: vocabSize)],
            dataType: .float32
        )

        var currentKV = kvState
        for i in 0..<chunkSize {
            // autoreleasepool: without this, Metal-backed prediction temporaries
            // (IOSurface buffers, intermediate MLMultiArrays) accumulate across
            // the 8 inner decodes — small per call, large enough cumulatively to
            // OOM on iPhone 12 Pro the moment the user starts typing.
            currentKV = try autoreleasepool {
                let (stepLogits, newKV) = try decode(
                    token: tokenIDs[i],
                    position: seqLen + Int32(i),
                    kvState: currentKV,
                    globalCacheSize: globalCacheSize
                )
                try Self.copyLogitsRow(
                    from: stepLogits, into: result, row: i, vocabSize: vocabSize
                )
                return newKV
            }
        }
        return (result, currentKV)
    }

    /// Copy one decode step's next-token distribution into row `row` of a
    /// `[chunkSize, vocabSize]` Float32 buffer.
    ///
    /// Decode logits arrive as `[1, vocabSize]` (or `[vocabSize]`), so the
    /// distribution occupies the first `vocabSize` elements. Everything is
    /// validated before any raw-memory access: a dtype, length or stride
    /// mismatch here is an out-of-bounds copy, not a crash.
    private static func copyLogitsRow(
        from stepLogits: MLMultiArray,
        into destination: MLMultiArray,
        row: Int,
        vocabSize: Int
    ) throws {
        guard stepLogits.dataType == .float32 else {
            throw CoreMLModelError.unexpectedBufferLayout(
                "decode logits have dtype \(stepLogits.dataType), expected float32"
            )
        }
        guard stepLogits.count >= vocabSize else {
            throw CoreMLModelError.unexpectedBufferLayout(
                "decode logits hold \(stepLogits.count) elements, expected at least \(vocabSize)"
            )
        }
        guard stepLogits.strides.last?.intValue == 1 else {
            throw CoreMLModelError.unexpectedBufferLayout(
                "decode logits are not contiguous (strides \(stepLogits.strides))"
            )
        }
        stepLogits.withUnsafeBufferPointer(ofType: Float32.self) { src in
            destination.withUnsafeMutableBufferPointer(ofType: Float32.self) { dst, _ in
                guard let srcBase = src.baseAddress, let dstBase = dst.baseAddress else { return }
                dstBase.advanced(by: row * vocabSize).update(from: srcBase, count: vocabSize)
            }
        }
    }

    /// Run one decode step.
    public func decode(
        token: Int32,
        position: Int32,
        kvState: KVCacheState,
        globalCacheSize: Int32? = nil
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        let activeModel = getFunction(
            functionName(prefix: "decode", cacheSize: globalCacheSize.map { Int($0) })
        )

        var features: [String: MLMultiArray] = [:]
        features[decodeTokenInputName] = MLMultiArray.int32Scalar(token)
        features[decodePositionInputName] = MLMultiArray.int32Scalar(position)
        if let nName = decodeNInputName, let nValue = globalCacheSize {
            features[nName] = MLMultiArray.int32Scalar(nValue)
        }

        let provider = try CoreMLInputProvider(
            features: features,
            kvNames: decodeKVInputNames,
            kvState: kvState
        )
        // The sliding caches are updated in place inside `slidingCaches`.
        let result = try activeModel.prediction(from: provider, using: kvState.slidingCaches)
        let logits = result.featureValue(for: decodeLogitsName)!.multiArrayValue!
        let newKV = try KVCacheState.from(
            prediction: result,
            outputNames: decodeKVOutputNames,
            inputNames: decodeKVInputNames,
            globalNames: kvState.globalNames,
            slidingCaches: kvState.slidingCaches
        )
        return (logits, newKV)
    }

    // MARK: - I/O Classification

    /// Classified I/O names for a model function.
    struct ClassifiedIO {
        let logitsOutputName: String
        let tokenInputName: String
        let positionInputName: String
        let nInputName: String?
        /// KV features passed per call: the global caches plus `sliding_pos_ring`.
        let kvInputNames: [String]
        let kvOutputNames: [String]
        /// CoreML state features: the sliding KV caches. Empty only for an
        /// artifact exported before they became state.
        let stateNames: Set<String>
    }

    /// Classify model I/O using name matching with positional fallback.
    ///
    /// The sliding KV caches are state, so they appear in neither
    /// `inputDescriptionsByName` nor `outputDescriptionsByName`; what remains to
    /// pair up is the global caches (`k_<slot>` ↔ `k_<slot>_out`) and
    /// `sliding_pos_ring`.
    static func classifyIO(model: MLModel) -> ClassifiedIO {
        let description = model.modelDescription
        let stateNames = Set(description.stateDescriptionsByName.keys)
        let outputDescs = description.outputDescriptionsByName
        // Defensive: keep state features out of the input classification even
        // if a future CoreML release starts listing them there too.
        let inputDescs = description.inputDescriptionsByName
            .filter { !stateNames.contains($0.key) }

        // Outputs: float32 = logits, everything else = KV
        var logitsName = ""
        var kvOutputs: [String] = []
        for (name, desc) in outputDescs {
            if let c = desc.multiArrayConstraint, c.dataType == .float32 {
                logitsName = name
            } else {
                kvOutputs.append(name)
            }
        }
        kvOutputs.sort { naturalCompare($0, $1) }

        // Try name matching: input name ∈ output names → KV
        let stateOutputSet = Set(kvOutputs)
        var controlInputs: [String] = []
        var kvInputs: [String] = []
        for name in inputDescs.keys {
            if stateOutputSet.contains(name) || stateOutputSet.contains(name + "_out") {
                kvInputs.append(name)
            } else {
                controlInputs.append(name)
            }
        }

        // Fallback: if name matching found no state inputs, use positional split
        if kvInputs.isEmpty && !kvOutputs.isEmpty {
            let allInputs = Array(inputDescs.keys).sorted { naturalCompare($0, $1) }
            let stateCount = kvOutputs.count
            let controlCount = allInputs.count - stateCount
            controlInputs = Array(allInputs.prefix(controlCount))
            kvInputs = Array(allInputs.suffix(stateCount))
        } else {
            controlInputs.sort { naturalCompare($0, $1) }
            kvInputs.sort { naturalCompare($0, $1) }
        }

        assert(!logitsName.isEmpty, "No Float32 output found (logits)")
        assert(kvInputs.count == kvOutputs.count,
               "KV input count (\(kvInputs.count)) != output count (\(kvOutputs.count))")

        let (tokenName, posName, nName) = identifyControlInputs(
            controlInputs, inputDescs: inputDescs
        )

        return ClassifiedIO(
            logitsOutputName: logitsName,
            tokenInputName: tokenName,
            positionInputName: posName,
            nInputName: nName,
            kvInputNames: kvInputs,
            kvOutputNames: kvOutputs,
            stateNames: stateNames
        )
    }

    /// Distinguish token, position, and optional N inputs among control inputs.
    private static func identifyControlInputs(
        _ names: [String],
        inputDescs: [String: MLFeatureDescription]
    ) -> (tokenName: String, positionName: String, nInputName: String?) {
        precondition(names.count == 2 || names.count == 3,
                     "Expected 2 or 3 control inputs, got \(names.count): \(names)")

        var nName: String? = nil
        var remaining = names
        if let idx = names.firstIndex(of: "N") {
            nName = names[idx]
            remaining.remove(at: idx)
        }
        precondition(remaining.count == 2,
                     "After removing N, expected 2 control inputs, got \(remaining.count)")

        // By element count: token input has more elements (prefill: 8 vs 1)
        let count0 = inputDescs[remaining[0]]?.multiArrayConstraint?.shape
            .map { $0.intValue }.reduce(1, *) ?? 1
        let count1 = inputDescs[remaining[1]]?.multiArrayConstraint?.shape
            .map { $0.intValue }.reduce(1, *) ?? 1
        if count0 != count1 {
            return count0 > count1
                ? (remaining[0], remaining[1], nName)
                : (remaining[1], remaining[0], nName)
        }

        // By name: "token" in name → token input
        if remaining[0].contains("token") { return (remaining[0], remaining[1], nName) }
        if remaining[1].contains("token") { return (remaining[1], remaining[0], nName) }

        // Fallback: first naturally-sorted name is token
        let sorted = remaining.sorted { naturalCompare($0, $1) }
        return (sorted[0], sorted[1], nName)
    }

    /// Detect KV inputs with flexible shapes (RangeDim) on dim 1.
    private static func detectFlexibleGlobalKV(
        model: MLModel,
        kvInputNames: [String]
    ) -> Set<String> {
        var flex: Set<String> = []
        for name in kvInputNames {
            guard let desc = model.modelDescription.inputDescriptionsByName[name],
                  let constraint = desc.multiArrayConstraint else { continue }
            if constraint.shapeConstraint.type == .range {
                flex.insert(name)
            }
        }
        return flex
    }

    /// Natural string comparison: numeric segments are compared by value.
    static func naturalCompare(_ a: String, _ b: String) -> Bool {
        let aComponents = splitNumeric(a)
        let bComponents = splitNumeric(b)
        for (ac, bc) in zip(aComponents, bComponents) {
            switch (ac, bc) {
            case let (.text(at), .text(bt)):
                if at != bt { return at < bt }
            case let (.number(an), .number(bn)):
                if an != bn { return an < bn }
            case (.number, .text):
                return true
            case (.text, .number):
                return false
            }
        }
        return aComponents.count < bComponents.count
    }

    private enum NameComponent {
        case text(String)
        case number(Int)
    }

    private static func splitNumeric(_ s: String) -> [NameComponent] {
        var result: [NameComponent] = []
        var current = ""
        var inDigits = false
        for ch in s {
            if ch.isNumber {
                if !inDigits && !current.isEmpty {
                    result.append(.text(current)); current = ""
                }
                inDigits = true
                current.append(ch)
            } else {
                if inDigits && !current.isEmpty {
                    result.append(.number(Int(current)!)); current = ""
                }
                inDigits = false
                current.append(ch)
            }
        }
        if !current.isEmpty {
            result.append(inDigits ? .number(Int(current)!) : .text(current))
        }
        return result
    }
}

// MARK: - Input Provider

/// Custom MLFeatureProvider that combines token/position inputs with KV cache arrays.
final class CoreMLInputProvider: MLFeatureProvider {
    let featureNames: Set<String>
    private var values: [String: MLFeatureValue]

    init(
        features: [String: MLMultiArray],
        kvNames: [String],
        kvState: KVCacheState
    ) throws {
        var values: [String: MLFeatureValue] = [:]
        values.reserveCapacity(features.count + kvNames.count)
        for (name, array) in features {
            values[name] = MLFeatureValue(multiArray: array)
        }
        for name in kvNames {
            guard let array = kvState.arraysByName[name] else {
                throw CoreMLModelError.missingKVInput(name)
            }
            values[name] = MLFeatureValue(multiArray: array)
        }
        self.values = values
        self.featureNames = Set(values.keys)
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        values[featureName]
    }
}

// MARK: - MLMultiArray Helpers

extension MLMultiArray {
    /// Create a single-element Int32 MLMultiArray.
    public static func int32Scalar(_ value: Int32) -> MLMultiArray {
        let array = try! MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: value)
        return array
    }

    /// Create an Int32 array of shape (1, length) from a Swift array.
    public static func int32Row(_ values: [Int32]) -> MLMultiArray {
        let array = try! MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for (i, v) in values.enumerated() { ptr[i] = v }
        return array
    }

    /// Deep-copy an MLMultiArray to a new, independent buffer.
    ///
    /// CoreML reuses output MLMultiArray buffers across predictions.
    /// Without copying, passing prediction N's outputs as prediction N+1's
    /// inputs causes silent data corruption (aliased read/write).
    public func deepCopy() throws -> MLMultiArray {
        let copy = try MLMultiArray(shape: self.shape, dataType: self.dataType)
        memcpy(copy.dataPointer, self.dataPointer,
               self.count * KVCacheState.bytesPerElement(of: self.dataType))
        return copy
    }
}

// MARK: - Errors

public enum CoreMLModelError: Error, LocalizedError {
    case modelNotFound
    case missingKVInput(String)
    /// The artifact declares materialized functions, but none usable in the
    /// requested mode (e.g. prefill wanted, only `decode_<N>` exported).
    case noUsableMaterializedFunctions(decodeSizes: [Int], prefillSizes: [Int])
    /// Could not work out which KV caches are global. Fatal rather than
    /// degraded: an empty global set silently caps every conversation.
    case globalKVDetectionFailed(String)
    /// An MLMultiArray had a dtype/shape/stride we can't safely memcpy.
    case unexpectedBufferLayout(String)
    /// The artifact declares no CoreML state features, i.e. it was exported
    /// before the sliding KV caches moved into state.
    case modelPredatesStatefulKVCaches(function: String, kvInputCount: Int)
    /// A token position doesn't fit the allocated global KV cache.
    case positionOutOfRange(position: Int, cacheSize: Int)
    /// Both load strategies failed; both causes are kept because the standard
    /// one is usually the real diagnosis.
    case loadFailed(standard: Error, materialized: Error)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "Model not found at the specified path"
        case .missingKVInput(let name):
            "KV cache is missing the array for input '\(name)'"
        case .noUsableMaterializedFunctions(let decodeSizes, let prefillSizes):
            "No usable materialized function pairs (decode sizes: \(decodeSizes), prefill sizes: \(prefillSizes))"
        case .globalKVDetectionFailed(let reason):
            "Could not identify the global KV cache inputs: \(reason)"
        case .unexpectedBufferLayout(let reason):
            "Unexpected MLMultiArray layout: \(reason)"
        case .modelPredatesStatefulKVCaches(let function, let kvInputCount):
            "Function '\(function)' declares no CoreML state features (it takes \(kvInputCount) KV inputs) — this model predates stateful KV caches. Re-run `uv run gemma-export`."
        case .positionOutOfRange(let position, let cacheSize):
            "Token position \(position) does not fit a global KV cache of size \(cacheSize)"
        case .loadFailed(let standard, let materialized):
            "Model load failed. Standard (decode/prefill): \(standard.localizedDescription). Materialized (decode_N/prefill_N): \(materialized.localizedDescription)"
        }
    }
}
