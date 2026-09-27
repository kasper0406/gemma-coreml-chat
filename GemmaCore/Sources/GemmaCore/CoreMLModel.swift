/// CoreML model wrapper for the chunked, materialized multifunction Gemma4-E2B
/// package.
///
/// Every step runs as a sequence of **layer-chunk functions** and then one
/// shared logit head, because the Neural Engine takes no function past a size
/// limit (and one too-large function takes the whole package off it). For
/// each size N in the export's `--materialize-sizes` the package declares
///
/// - `decode_c<k>_<N>` / `prefill_c<k>_<N>`, k = 0 ..< ``layerChunkCount``:
///   the hidden state in (`token_embed` itself for chunk 0) and out, the
///   final one normed;
/// - `state_<N>`: declares every KV cache; only used to make the `MLState`;
///
/// plus the size-independent `head` (final hidden `[1, 1, D]` → logits).
/// A `--decode-only` export has no `prefill_*` functions.
///
/// **Every** KV cache is CoreML state, and Core ML shares states across the
/// functions of one package by name: one `MLState`, made from `state_<N>`,
/// serves every chunk of size N (each chunk declares only the caches its
/// layers touch). The inputs besides the state are the embedding rows
/// (`token_embed`, and each chunk's columns of `ple_rows` — looked up on the
/// host, see ``HostEmbeddings``), the position and the int32
/// `sliding_pos_ring`, which the host keeps up to date itself (see
/// ``KVCacheState``). The runtime selects the size through
/// ``KVCacheSizePolicy``, the only place bucketing lives.
///
/// State buffer shapes are baked into each size's functions, so an `MLState`
/// belongs to exactly one size: ``makeEmptyKVState(size:)`` creates it and
/// ``grownToFit(_:needed:)`` migrates the contents when the conversation
/// outgrows it. Artifacts from before the layer chunks, the cache states or
/// the host lookups are rejected at load — re-run `gemma-export`.
///
/// Every loaded function is shared: all conversations of a size run the same
/// chunk functions, and every size runs the same `head`. Core ML's synchronous
/// prediction is not safe to call concurrently on one `MLModel`, so each
/// function is a ``SerialFunction`` that lets one prediction at a time through
/// — two conversations (or a specialization warming a new size while another
/// generates) interleave step by step rather than race.

import CoreML
import CryptoKit
import Foundation

public final class CoreMLModel: @unchecked Sendable {
    /// The two kinds of step: a `chunkSize`-token prompt chunk, or one token.
    enum Phase: String, CaseIterable, Sendable {
        case prefill, decode
    }

    /// Feature names the exporter gives every function (`gemma_chat/export.py`).
    enum Feature {
        static let hidden = "hidden"
        static let hiddenOut = "hidden_out"
        static let tokenEmbed = HostEmbeddings.tokenInputName
        static let pleRows = HostEmbeddings.perLayerInputName
        static let position = "position"
        static let ring = "sliding_pos_ring"
        static let logits = "logits"
    }

    /// Signature of one layer-chunk function — the same at every size.
    struct ChunkIO {
        /// False for chunk 0, whose hidden state is `token_embed` itself.
        let takesHidden: Bool
        /// Whether the chunk has a sliding-window layer to mask.
        let takesRing: Bool
        /// `[1, L, columns]`: this chunk's slice of the per-layer rows.
        let pleRowsShape: [NSNumber]
    }

    /// Signature of one phase's chunk sequence.
    struct PhaseIO {
        let chunks: [ChunkIO]
        /// `[1, L, D]`, the shape of `token_embed` and of every chunk's
        /// hidden state in and out.
        let hiddenShape: [NSNumber]
        /// Tokens per call (L).
        var tokenLength: Int { hiddenShape[1].intValue }
    }

    /// Signature of `head`.
    struct HeadIO {
        let inputShape: [NSNumber]
        let logitsShape: [NSNumber]
        let logitsDataType: MLMultiArrayDataType
    }

    let decodeIO: PhaseIO
    /// The decode signature again in decode-only mode, which prefills by
    /// looping decode.
    let prefillIO: PhaseIO
    let headIO: HeadIO
    /// `sliding_pos_ring`'s shape, `[1, sliding_window]`.
    let ringShape: [NSNumber]
    /// Every KV cache state, as `state_<N>` declares them, sorted for a
    /// deterministic migration order.
    let stateNames: [String]

    /// The tables the embedding-row inputs are looked up in.
    private let embeddings: HostEmbeddings
    /// The logit head, shared by every size and both phases.
    private let head: SerialFunction

    /// Tokens per prefill call, read from the prefill chunks' `token_embed`.
    ///
    /// A decode-only load has no prefill functions and prefills by looping
    /// `decode`, so its chunk is 1: any larger value would only pad the prompt
    /// out to a chunk boundary and spend real decode steps on the padding.
    public let chunkSize: Int

    /// Layer-chunk functions per step (per phase and size).
    public let layerChunkCount: Int

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
    /// halves the resident function count, which is the difference between
    /// fitting and OOM on tight devices like iPhone 12 Pro.
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

    /// Per-function state: either fully loaded, or a pending load Task that
    /// concurrent callers can join rather than re-issuing the load.
    ///
    /// Pending loads carry a monotonic `id` so a late failure handler can only
    /// evict *its own* entry. Without it: T1 fails with awaiters A and B, A
    /// evicts, C starts T2, then B's eviction removes T2's entry and D starts
    /// T3 — two concurrent multi-GB loads of the same function.
    private enum LoadState {
        case loaded(SerialFunction)
        case loading(id: UInt64, task: Task<SerialFunction, Error>)
    }

    /// Function state keyed by function name (e.g. "decode_c0_512").
    private var functions: [String: LoadState]
    /// Sizes that have already run their throwaway specialization step — see
    /// ``specialize(size:)``. Tracked separately from `functions` because a
    /// bulk preload deliberately loads without specializing.
    private var specializedSizes: Set<Int> = []
    private var nextLoadID: UInt64 = 0
    private let cacheLock = NSLock()

    private init(
        prefillIO: PhaseIO,
        decodeIO: PhaseIO,
        headIO: HeadIO,
        ringShape: [NSNumber],
        stateNames: [String],
        embeddings: HostEmbeddings,
        head: SerialFunction,
        chunkSize: Int,
        materializedSizes: [Int],
        isDecodeOnly: Bool,
        modelURL: URL,
        sourceURL: URL,
        sourceFingerprint: String?,
        computeUnits: MLComputeUnits,
        initialFunctions: [String: SerialFunction]
    ) {
        self.prefillIO = prefillIO
        self.decodeIO = decodeIO
        self.headIO = headIO
        self.ringShape = ringShape
        self.stateNames = stateNames
        self.embeddings = embeddings
        self.head = head
        self.chunkSize = chunkSize
        self.layerChunkCount = decodeIO.chunks.count
        self.materializedSizes = materializedSizes
        self.effectiveMaxSeqLen = materializedSizes[materializedSizes.count - 1]
        self.isDecodeOnly = isDecodeOnly
        self.modelURL = modelURL
        self.sourceURL = sourceURL
        self.sourceFingerprint = sourceFingerprint
        self.computeUnits = computeUnits
        self.functions = initialFunctions.mapValues { .loaded($0) }
    }

    // MARK: - Function names

    static let headFunctionName = "head"

    static func chunkFunctionName(_ phase: Phase, chunk: Int, size: Int) -> String {
        "\(phase.rawValue)_c\(chunk)_\(size)"
    }

    static func stateFunctionName(size: Int) -> String {
        "state_\(size)"
    }

    /// Every per-size function this load uses at `size`: the state function
    /// first, then decode's chunks, then prefill's.
    private func functionNames(size: Int) -> [String] {
        let phases: [Phase] = isDecodeOnly ? [.decode] : [.decode, .prefill]
        return [Self.stateFunctionName(size: size)] + phases.flatMap { phase in
            (0..<layerChunkCount).map { Self.chunkFunctionName(phase, chunk: $0, size: size) }
        }
    }

    // MARK: - KV cache lifecycle

    /// A zeroed cache for `size` (rounded up through ``cacheSizePolicy``),
    /// defaulting to the smallest materialized size.
    ///
    /// The `MLState` is made from that size's `state_<N>` function, the one
    /// that declares every cache: state buffer shapes are baked into each
    /// size's functions, so a state made at one size is meaningless at
    /// another. The size must already be loaded — `ensureLoaded(forGlobalCacheSize:)`
    /// first for anything but the bootstrap size, which `load` brings up.
    ///
    /// Make a new one per conversation. Reusing one across a reset would leave
    /// stale K/V that a re-populated `sliding_pos_ring` marks valid again.
    public func makeEmptyKVState(size requested: Int? = nil) throws -> KVCacheState {
        let target = cacheSizePolicy.size(forNeeded: requested ?? materializedSizes[0])
        guard let function = loadedFunction(named: Self.stateFunctionName(size: target)) else {
            throw KVCacheError.functionNotLoaded(size: target)
        }
        return try KVCacheState(size: target, caches: function.makeState(), ringShape: ringShape)
    }

    /// Return a cache big enough for `needed` tokens, migrating `kv` into a
    /// larger size's state when it no longer fits.
    ///
    /// Returns `kv` untouched in the common case. When growth is required the
    /// next size is loaded first (state buffers can only be made from a loaded
    /// handle) and every cache is copied across — see
    /// ``KVCacheState/adoptContents(of:stateNames:)``.
    public func grownToFit(_ kv: KVCacheState, needed: Int) async throws -> KVCacheState {
        let target = cacheSizePolicy.size(forNeeded: needed)
        guard target > kv.size else { return kv }
        try await ensureLoaded(forGlobalCacheSize: target)
        let grown = try makeEmptyKVState(size: target)
        try grown.adoptContents(of: kv, stateNames: stateNames)
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
    /// - Parameter maxContextSize: Only retain sizes ≤ this one.
    ///   Loading fewer functions is critical on memory-constrained devices like
    ///   iPhone, where loading all 16 pairs OOMs.
    /// - Parameter decodeOnly: Skip loading prefill functions entirely.
    ///   `prefill()` falls back to per-token `decode()` internally — slower but
    ///   halves resident MLModel count, which is the only way the model fits on
    ///   iPhone 12 Pro / 6 GB devices. Forced on for `--decode-only` artifacts,
    ///   which export no prefill functions to load.
    /// - Parameter backgroundPreload: Kick off a detached load of every
    ///   retained size's functions once the bootstrap size is up. Right for
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
    /// `MLModel.load` — so the bootstrap never loads more than it keeps.
    ///
    /// Strategy, tuned for memory-constrained devices:
    ///   1. Take the declared function set from `model.mil`.
    ///   2. Load `head`, then the smallest size's state and chunk functions,
    ///      one at a time.
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

        guard let declared = enumerateMaterializedFunctions(compiledURL: url) else {
            throw CoreMLModelError.notMaterialized(url.lastPathComponent)
        }
        guard declared.hasHead, let layerChunkCount = declared.layerChunkCount else {
            throw declared.predatesLayerChunks
                ? CoreMLModelError.modelPredatesLayerChunks(url.lastPathComponent)
                : CoreMLModelError.notMaterialized(url.lastPathComponent)
        }
        // A `gemma-export --decode-only` artifact has no prefill chunks.
        // Insisting on them there yields no sizes at all.
        var effectiveDecodeOnly = decodeOnly
        if declared.prefillSizes.isEmpty && !effectiveDecodeOnly {
            Log.info("[CoreML] Artifact exports no prefill functions — switching to decode-only mode")
            effectiveDecodeOnly = true
        }
        let sizes = declared.usableSizes(decodeOnly: effectiveDecodeOnly)
        guard !sizes.isEmpty else {
            throw CoreMLModelError.noUsableMaterializedFunctions(
                decodeSizes: declared.decodeSizes, prefillSizes: declared.prefillSizes
            )
        }
        Log.info("[CoreML] Materialized sizes (from manifest): \(sizes), \(layerChunkCount) layer chunks")

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

        // Serial loads keep the bootstrap's peak to one function load at a time.
        func load(_ name: String) async throws -> SerialFunction {
            try await loadFunction(url: url, computeUnits: computeUnits, function: name)
        }
        let head = try await load(headFunctionName)
        let stateName = stateFunctionName(size: bootSize)
        var loaded: [String: SerialFunction] = [stateName: try await load(stateName)]
        var chunkModels: [Phase: [SerialFunction]] = [:]
        for phase in effectiveDecodeOnly ? [Phase.decode] : [.decode, .prefill] {
            for k in 0..<layerChunkCount {
                let name = chunkFunctionName(phase, chunk: k, size: bootSize)
                let model = try await load(name)
                loaded[name] = model
                chunkModels[phase, default: []].append(model)
            }
        }
        Log.info("[CoreML] Loaded head + \(loaded.count) functions of size \(bootSize) (serial)")

        let stateNames = loaded[stateName]!.model.modelDescription.stateDescriptionsByName.keys.sorted()
        let headIO = try classifyHead(model: head.model)
        var ringShape: [NSNumber]?
        func classify(_ phase: Phase) throws -> PhaseIO {
            let names = (0..<layerChunkCount).map { chunkFunctionName(phase, chunk: $0, size: bootSize) }
            let io = try classifyPhase(
                models: chunkModels[phase]!.map(\.model), names: names, stateNames: Set(stateNames),
                ringShape: &ringShape
            )
            guard io.hiddenShape[2] == headIO.inputShape[2] else {
                throw CoreMLModelError.unexpectedSignature(
                    function: headFunctionName,
                    detail: "takes a hidden state of \(headIO.inputShape) but \(names.last!) emits \(io.hiddenShape)"
                )
            }
            return io
        }
        let decodeIO = try classify(.decode)
        // In decode-only mode, prefill metadata is borrowed from decode: the
        // per-token loop in `decodeOnlyPrefill` runs the decode chunks.
        let prefillIO = effectiveDecodeOnly ? decodeIO : try classify(.prefill)
        guard let ringShape else {
            throw CoreMLModelError.unexpectedSignature(
                function: chunkFunctionName(.decode, chunk: 0, size: bootSize),
                detail: "no layer chunk takes `\(Feature.ring)`"
            )
        }
        // After the signatures, so an artifact that still takes token ids
        // gets that error rather than a missing-directory one.
        let embeddings = try HostEmbeddings(packageURL: sourceURL)
        for io in [decodeIO, prefillIO] {
            try checkEmbeddingShapes(io, against: embeddings)
        }
        let chunkSize = effectiveDecodeOnly ? 1 : prefillIO.tokenLength
        Log.info("[CoreML] \(layerChunkCount) layer chunks; decode hidden=\(decodeIO.hiddenShape.map(\.intValue)), prefill chunk=\(chunkSize), head logits=\(headIO.logitsShape.map(\.intValue)) dtype=\(headIO.logitsDataType.rawValue), ring=\(ringShape.map(\.intValue)), \(stateNames.count) cache states")

        let instance = CoreMLModel(
            prefillIO: prefillIO,
            decodeIO: decodeIO,
            headIO: headIO,
            ringShape: ringShape,
            stateNames: stateNames,
            embeddings: embeddings,
            head: head,
            chunkSize: chunkSize,
            materializedSizes: retainedSizes,
            isDecodeOnly: effectiveDecodeOnly,
            modelURL: url,
            sourceURL: sourceURL,
            sourceFingerprint: sourceFingerprint,
            computeUnits: computeUnits,
            initialFunctions: loaded
        )
        if backgroundPreload {
            instance.preloadAllSizes()
        }
        return instance
    }

    /// Load a single function by name.
    private static func loadFunction(
        url: URL, computeUnits: MLComputeUnits, function: String
    ) async throws -> SerialFunction {
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.functionName = function
        do {
            return SerialFunction(try await MLModel.load(contentsOf: url, configuration: config))
        } catch {
            throw CoreMLModelError.functionLoadFailed(
                function: function, computeUnits: computeUnitsTag(computeUnits), underlying: error
            )
        }
    }

    /// The per-size function sets a compiled artifact declares.
    struct MaterializedFunctions {
        /// Chunk indices declared per size, per phase.
        let chunks: [Phase: [Int: Set<Int>]]
        /// Sizes with a `state_<N>` function.
        let stateSizes: Set<Int>
        let hasHead: Bool
        /// The artifact declares pre-chunk `decode_<N>` functions.
        let predatesLayerChunks: Bool

        /// Chunks per step: the decode chunk count at the smallest size, or nil
        /// when there are none.
        var layerChunkCount: Int? {
            guard let smallest = chunks[.decode]?.keys.min() else { return nil }
            return chunks[.decode]?[smallest]?.count
        }

        /// Sizes whose chunks are complete in `phase`, ascending.
        func sizes(_ phase: Phase) -> [Int] {
            guard let count = layerChunkCount else { return [] }
            let full = Set(0..<count)
            return (chunks[phase] ?? [:]).filter { $0.value == full }.keys.sorted()
        }

        var decodeSizes: [Int] { sizes(.decode) }
        var prefillSizes: [Int] { sizes(.prefill) }

        /// Sizes runnable in the requested mode: a state function, the decode
        /// chunks and, unless decode-only, the prefill chunks.
        func usableSizes(decodeOnly: Bool) -> [Int] {
            let prefill = Set(prefillSizes)
            return decodeSizes.filter {
                stateSizes.contains($0) && (decodeOnly || prefill.contains($0))
            }
        }
    }

    /// Scan the compiled `model.mil` manifest for `func decode_c<k>_<N>`,
    /// `func prefill_c<k>_<N>`, `func state_<N>` and `func head` declarations.
    ///
    /// Returns nil only when the manifest is missing or unparseable.
    static func enumerateMaterializedFunctions(compiledURL: URL) -> MaterializedFunctions? {
        let milURL = compiledURL.appendingPathComponent("model.mil")
        guard let text = try? String(contentsOf: milURL, encoding: .utf8),
              let re = try? NSRegularExpression(
                pattern: #"\bfunc\s+(?:(decode|prefill)_c(\d+)_(\d+)|state_(\d+)|(head)|(decode|prefill)_(\d+))\s*[<(]"#
              )
        else { return nil }

        var chunks: [Phase: [Int: Set<Int>]] = [:]
        var stateSizes = Set<Int>()
        var hasHead = false
        var predates = false
        let full = NSRange(text.startIndex..<text.endIndex, in: text)
        re.enumerateMatches(in: text, range: full) { match, _, _ in
            guard let m = match else { return }
            func group(_ i: Int) -> String? {
                Range(m.range(at: i), in: text).map { String(text[$0]) }
            }
            if let phase = group(1).flatMap(Phase.init(rawValue:)),
               let k = group(2).flatMap({ Int($0) }), let size = group(3).flatMap({ Int($0) }) {
                chunks[phase, default: [:]][size, default: []].insert(k)
            } else if let size = group(4).flatMap({ Int($0) }) {
                stateSizes.insert(size)
            } else if group(5) != nil {
                hasHead = true
            } else if group(6) != nil {
                predates = true
            }
        }
        return MaterializedFunctions(
            chunks: chunks, stateSizes: stateSizes, hasHead: hasHead, predatesLayerChunks: predates
        )
    }

    // MARK: - Function Resolution

    /// The loaded function `name`, or nil if it hasn't been loaded yet.
    private func loadedFunction(named name: String) -> SerialFunction? {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if case .loaded(let model) = functions[name] { return model }
        return nil
    }

    /// The loaded chunk functions of `phase` at `size`, in order, or a clear
    /// error naming the call the caller skipped.
    private func chunkModels(_ phase: Phase, size: Int) throws -> [SerialFunction] {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        return try (0..<layerChunkCount).map { k in
            guard case .loaded(let model) = functions[Self.chunkFunctionName(phase, chunk: k, size: size)] else {
                throw KVCacheError.functionNotLoaded(size: size)
            }
            return model
        }
    }

    /// Pre-load *and specialize* every function for a given cache size. Call
    /// from an async context before sync `prefill()`/`decode()` calls; every
    /// path that is about to predict at a new size goes through here, which is
    /// what keeps ``specialize(size:)`` off the token loop.
    public func ensureLoaded(forGlobalCacheSize cacheSize: Int) async throws {
        let size = cacheSizePolicy.size(forNeeded: cacheSize)
        try await withThrowingTaskGroup(of: Void.self) { group in
            for name in functionNames(size: size) {
                group.addTask { _ = try await self.loadIfNeeded(name: name) }
            }
            try await group.waitForAll()
        }
        try specialize(size: size)
    }

    /// Run one throwaway step of each phase through a freshly loaded size,
    /// into a scratch `MLState` that no conversation owns.
    ///
    /// A GPU-backed CoreML function does not finish compiling when it loads:
    /// `MLModel.load` only builds the E5RT plan, and MPSGraph specializes the
    /// executable lazily inside the *first* prediction — seconds per function,
    /// redone in every process. So pay it here — at load, or at the moment a
    /// conversation grows into a new size — instead of inside the first token
    /// the user is waiting on. The scratch state matters: predictions mutate
    /// KV caches in place, so warming through the live cache would write a
    /// phantom token 0 into it.
    private func specialize(size: Int) throws {
        cacheLock.lock()
        let alreadyDone = !specializedSizes.insert(size).inserted
        cacheLock.unlock()
        guard !alreadyDone else { return }

        let start = CFAbsoluteTimeGetCurrent()
        try autoreleasepool {
            let scratch = try makeEmptyKVState(size: size)
            _ = try decode(token: 0, position: 0, kvState: scratch)
            if !isDecodeOnly {
                _ = try prefill(
                    tokens: [Int32](repeating: 0, count: chunkSize),
                    startPosition: 0, logitsRow: 0, kvState: scratch
                )
            }
        }
        Log.info("[CoreML] Specialized size \(size) in \(String(format: "%.1f", CFAbsoluteTimeGetCurrent() - start))s")
    }

    /// Result of checking the cache for `name`: an already-loaded model, or a
    /// load Task (either newly started by us or one a concurrent caller had
    /// already kicked off).
    private enum CacheLookup {
        case existing(SerialFunction)
        case pending(id: UInt64, task: Task<SerialFunction, Error>)
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
        let task: Task<SerialFunction, Error> = Task {
            try await Self.loadFunction(url: url, computeUnits: units, function: name)
        }
        nextLoadID += 1
        let id = nextLoadID
        functions[name] = .loading(id: id, task: task)
        return .pending(id: id, task: task)
    }

    private func markLoaded(name: String, model: SerialFunction) {
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
    private func loadIfNeeded(name: String) async throws -> SerialFunction {
        switch lookupOrStart(name: name) {
        case .existing(let model):
            return model
        case .pending(let id, let task):
            do {
                let model = try await task.value
                markLoaded(name: name, model: model)
                Log.info("[CoreML] Function '\(name)' loaded.")
                return model
            } catch {
                clearPending(name: name, id: id)
                throw error
            }
        }
    }

    /// Kick off background loads for every retained size's functions in
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

    /// Block until every retained function is loaded, reporting progress as
    /// each completes. On first-run installs this is what warms the ANE / E5RT
    /// cache before the first chat turn — otherwise the user hits multi-minute
    /// stalls mid-session. Safe to call even when the bg preload is running:
    /// both join the same in-flight tasks.
    public func warmSynchronously(
        concurrency: Int = 4,
        progress: @Sendable @escaping (_ completed: Int, _ total: Int) -> Void
    ) async {
        let allOK = await drainLoads(
            names: allFunctionNames, concurrency: concurrency, progress: progress
        )
        if allOK { markWarmed() }
    }

    /// Every per-size function this load retains, in ascending size order.
    /// (`head` is loaded by `load` itself.)
    private var allFunctionNames: [String] {
        materializedSizes.flatMap { functionNames(size: $0) }
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

    /// Run one prefill chunk and return the logits of token `logitsRow`.
    ///
    /// `tokens.count` must equal ``chunkSize``; the engine pads the prompt to a
    /// chunk boundary. Only one row of the chunk is ever wanted (the last
    /// *real* token), so only that row of the final hidden state goes through
    /// `head`. The returned logits live in the cache's rotating logits
    /// backings, like ``decode(token:position:kvState:)``'s.
    public func prefill(
        tokens: [Int32],
        startPosition: Int32,
        logitsRow: Int,
        kvState: KVCacheState
    ) throws -> MLMultiArray {
        // This chunk writes cache rows startPosition ..< +count, so every one
        // of them has to fit — checked before anything, the ring included, is
        // touched (a negative position would index the ring below slot 0).
        guard startPosition >= 0 else {
            throw CoreMLModelError.positionOutOfRange(
                position: Int(startPosition), cacheSize: kvState.size
            )
        }
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

        let step = try kvState.step(.prefill, io: prefillIO, headIO: headIO)
        // Padded positions get the pad token's rows, like any other token.
        let hidden = try runChunks(
            step, tokens: tokens, position: startPosition,
            models: chunkModels(.prefill, size: kvState.size), kvState: kvState
        )
        try PredictionBuffer.copyRow(logitsRow, of: hidden, into: step.headInput)
        return try runHead(step, kvState: kvState)
    }

    /// Per-token prefill via repeated `decode()` calls — the fallback used when
    /// only decode functions are loaded. Slower than a real prefill function
    /// (no fused chunk kernel), but keeps the resident function count at half,
    /// the only way to fit on iPhone 12 Pro / 6 GB.
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
                    // Copy: the decode logits live in a backing that a later
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
    /// backing: they stay valid across the *next* step and are overwritten by
    /// the one after, which is exactly the lifetime the sampling loop needs.
    public func decode(
        token: Int32,
        position: Int32,
        kvState: KVCacheState
    ) throws -> MLMultiArray {
        guard position >= 0, Int(position) < kvState.size else {
            throw CoreMLModelError.positionOutOfRange(
                position: Int(position), cacheSize: kvState.size
            )
        }
        let step = try kvState.step(.decode, io: decodeIO, headIO: headIO)
        // The decode head reads the last chunk's output buffer directly.
        _ = try runChunks(
            step, tokens: CollectionOfOne(token), position: position,
            models: chunkModels(.decode, size: kvState.size), kvState: kvState
        )
        return try runHead(step, kvState: kvState)
    }

    /// Look the tokens' rows up, record their positions in the ring, and run
    /// every layer chunk of one step in order. Returns the final (normed)
    /// hidden state.
    ///
    /// The KV caches never appear here — they are state, mutated in place
    /// inside `kvState.caches` by whichever chunks own them.
    private func runChunks(
        _ step: StepScratch,
        tokens: some Collection<Int32>,
        position: Int32,
        models: [SerialFunction],
        kvState: KVCacheState
    ) throws -> MLMultiArray {
        try embeddings.fill(tokens: tokens, tokenEmbed: step.tokenEmbed, pleRows: step.pleRows)
        kvState.setPosition(position)
        kvState.markRing(start: position, count: tokens.count)
        for (k, model) in models.enumerated() {
            let result = try model.prediction(
                from: step.chunkInputs[k], using: kvState.caches, options: step.chunkOptions[k]
            )
            guard let produced = result.featureValue(for: Feature.hiddenOut)?.multiArrayValue else {
                throw CoreMLModelError.missingOutput(Feature.hiddenOut)
            }
            // The next chunk's input provider holds the backing, so a
            // framework-allocated result has to land there.
            if produced !== step.hidden[k] {
                kvState.noteIgnoredBacking(Feature.hiddenOut)
                try PredictionBuffer.copyPrefix(from: produced, to: step.hidden[k], what: Feature.hiddenOut)
            }
        }
        return step.hidden[models.count - 1]
    }

    /// `head` on `step.headInput`, into the cache's next logits backing.
    private func runHead(_ step: StepScratch, kvState: KVCacheState) throws -> MLMultiArray {
        let (backing, options) = try kvState.nextLogits(headIO)
        let result = try head.prediction(from: step.headInputs, options: options)
        guard let logits = result.featureValue(for: Feature.logits)?.multiArrayValue else {
            throw CoreMLModelError.missingOutput(Feature.logits)
        }
        if logits !== backing {
            kvState.noteIgnoredBacking(Feature.logits)
        }
        return logits
    }

    // MARK: - I/O Classification

    /// Read one phase's chunk signatures, rejecting artifacts that predate
    /// stateful KV caches or host-side embedding lookups.
    ///
    /// Each chunk takes `token_embed` and its `ple_rows` columns (fp16
    /// `[1, L, …]`), the int32 `position`, the int32 `sliding_pos_ring` if it
    /// has a sliding layer, and — every chunk but the first — the previous
    /// chunk's `hidden`; it returns `hidden_out`. Anything named
    /// `k_<n>` / `v_<n>` on the signature means the caches still cross the
    /// boundary. Every state a chunk declares must be one `state_<N>` declares,
    /// since the chunks run on the `MLState` made from it.
    static func classifyPhase(
        models: [MLModel], names: [String], stateNames: Set<String>,
        ringShape: inout [NSNumber]?
    ) throws -> PhaseIO {
        var chunks: [ChunkIO] = []
        var hiddenShape: [NSNumber]?
        for (k, (model, function)) in zip(models, names).enumerated() {
            let description = model.modelDescription
            let inputs = description.inputDescriptionsByName
            let outputs = description.outputDescriptionsByName

            let cacheIO = (Array(inputs.keys) + Array(outputs.keys)).filter(isCacheName).sorted()
            let states = Set(description.stateDescriptionsByName.keys)
            guard cacheIO.isEmpty, !states.isEmpty else {
                throw CoreMLModelError.modelPredatesCacheStates(function: function, cacheFeatures: cacheIO)
            }
            guard states.isSubset(of: stateNames) else {
                throw CoreMLModelError.unexpectedSignature(
                    function: function,
                    detail: "declares states \(states.subtracting(stateNames).sorted()) that the state function does not"
                )
            }
            func fp16(_ name: String, in features: [String: MLFeatureDescription]) -> [NSNumber]? {
                guard let c = features[name]?.multiArrayConstraint, c.dataType == .float16,
                      c.shape.count == 3 else { return nil }
                return c.shape
            }
            guard let tokenEmbed = fp16(Feature.tokenEmbed, in: inputs),
                  let pleRows = fp16(Feature.pleRows, in: inputs) else {
                throw CoreMLModelError.modelPredatesHostEmbeddings(
                    function: function, inputs: inputs.keys.sorted()
                )
            }
            guard let out = fp16(Feature.hiddenOut, in: outputs), outputs.count == 1 else {
                throw CoreMLModelError.unexpectedSignature(
                    function: function,
                    detail: "expected the one output `\(Feature.hiddenOut)` (fp16, rank 3), got \(outputs.keys.sorted())"
                )
            }
            let takesHidden = k > 0
            let hidden = takesHidden ? fp16(Feature.hidden, in: inputs) : tokenEmbed
            let expected = hiddenShape ?? tokenEmbed
            guard let hidden, hidden == expected, tokenEmbed == expected, out == expected,
                  pleRows[1] == expected[1] else {
                throw CoreMLModelError.unexpectedSignature(
                    function: function,
                    detail: "hidden / token_embed / hidden_out / ple_rows shapes disagree (expected \(expected.map(\.intValue)))"
                )
            }
            hiddenShape = expected

            let takesRing = inputs[Feature.ring] != nil
            if let ring = inputs[Feature.ring]?.multiArrayConstraint {
                guard ring.dataType == .int32, ringShape == nil || ringShape == ring.shape else {
                    throw CoreMLModelError.unexpectedSignature(
                        function: function, detail: "`\(Feature.ring)` is not the int32 ring the other chunks take"
                    )
                }
                ringShape = ring.shape
            }
            let known = [Feature.tokenEmbed, Feature.pleRows, Feature.position]
                + (takesHidden ? [Feature.hidden] : []) + (takesRing ? [Feature.ring] : [])
            guard Set(inputs.keys) == Set(known),
                  inputs[Feature.position]?.multiArrayConstraint?.dataType == .int32 else {
                throw CoreMLModelError.unexpectedSignature(
                    function: function,
                    detail: "expected inputs \(known.sorted()) with an int32 `\(Feature.position)`, got \(inputs.keys.sorted())"
                )
            }
            chunks.append(ChunkIO(takesHidden: takesHidden, takesRing: takesRing, pleRowsShape: pleRows))
        }
        guard let hiddenShape, !chunks.isEmpty else {
            throw CoreMLModelError.unexpectedSignature(function: names.first ?? "?", detail: "no layer chunks")
        }
        return PhaseIO(chunks: chunks, hiddenShape: hiddenShape)
    }

    /// `head`: fp16 `hidden` `[1, 1, D]` in, float `logits` out.
    static func classifyHead(model: MLModel) throws -> HeadIO {
        let description = model.modelDescription
        guard let input = description.inputDescriptionsByName[Feature.hidden]?.multiArrayConstraint,
              input.dataType == .float16, input.shape.count == 3, input.shape[1] == 1,
              description.inputDescriptionsByName.count == 1,
              let logits = description.outputDescriptionsByName[Feature.logits]?.multiArrayConstraint
        else {
            throw CoreMLModelError.unexpectedSignature(
                function: headFunctionName,
                detail: "expected fp16 `\(Feature.hidden)` [1, 1, D] → `\(Feature.logits)`"
            )
        }
        return HeadIO(inputShape: input.shape, logitsShape: logits.shape, logitsDataType: logits.dataType)
    }

    /// The embedding inputs' widths must be the shipped tables' widths: the
    /// token table for `token_embed`, and the per-layer table split across the
    /// chunks' `ple_rows`, in order.
    private static func checkEmbeddingShapes(_ io: PhaseIO, against embeddings: HostEmbeddings) throws {
        let tokenWidth = io.hiddenShape[2].intValue
        let pleWidth = io.chunks.map { $0.pleRowsShape[2].intValue }.reduce(0, +)
        guard tokenWidth == embeddings.token.cols, pleWidth == embeddings.perLayer.cols else {
            throw CoreMLModelError.unexpectedSignature(
                function: "layer chunks",
                detail: "embedding inputs are \(tokenWidth) / \(pleWidth) wide in total, but the Embeddings/ tables have \(embeddings.token.cols) / \(embeddings.perLayer.cols) columns"
            )
        }
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
}

// MARK: - Serial functions

/// One loaded function, predicting one call at a time.
///
/// Core ML's synchronous `prediction` is not safe to call concurrently on one
/// `MLModel`, and every conversation shares the loaded functions (see
/// ``CoreMLModel``). The lock is uncontended in a single conversation — tens
/// of nanoseconds against milliseconds of prediction — and when two callers do
/// meet on a function, the second waits for the first's step to finish, which
/// is all the hardware underneath could offer it anyway.
final class SerialFunction: @unchecked Sendable {
    /// For reading the description only; predict through the methods below.
    let model: MLModel
    private let lock = NSLock()

    init(_ model: MLModel) {
        self.model = model
    }

    func prediction(
        from input: MLFeatureProvider, using state: MLState, options: MLPredictionOptions
    ) throws -> MLFeatureProvider {
        lock.lock()
        defer { lock.unlock() }
        return try model.prediction(from: input, using: state, options: options)
    }

    func prediction(
        from input: MLFeatureProvider, options: MLPredictionOptions
    ) throws -> MLFeatureProvider {
        lock.lock()
        defer { lock.unlock() }
        return try model.prediction(from: input, options: options)
    }

    func makeState() -> MLState {
        lock.lock()
        defer { lock.unlock() }
        return model.makeState()
    }
}

// MARK: - Input Provider

/// Minimal `MLFeatureProvider` over a name → array dictionary. The arrays are
/// wrapped once; refilling them in place feeds the next prediction.
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
    /// The artifact declares no materialized function set at all.
    case notMaterialized(String)
    /// The artifact declares the per-size `decode_<N>` / `prefill_<N>` pairs
    /// of the exports before the layer chunks.
    case modelPredatesLayerChunks(String)
    /// The artifact declares materialized functions, but none usable in the
    /// requested mode (e.g. prefill wanted, only decode chunks exported).
    case noUsableMaterializedFunctions(decodeSizes: [Int], prefillSizes: [Int])
    /// A KV cache still crosses the function signature, i.e. the artifact was
    /// exported before the caches became CoreML state.
    case modelPredatesCacheStates(function: String, cacheFeatures: [String])
    /// The function takes token ids rather than embedding rows, i.e. the
    /// artifact was exported before the lookups moved to the host.
    case modelPredatesHostEmbeddings(function: String, inputs: [String])
    /// The function's inputs/outputs are not the shape this runtime expects.
    case unexpectedSignature(function: String, detail: String)
    /// `MLModel.load` failed for one function.
    case functionLoadFailed(function: String, computeUnits: String, underlying: Error)
    /// A declared output was missing from a prediction result.
    case missingOutput(String)
    /// A token position doesn't fit the allocated KV cache.
    case positionOutOfRange(position: Int, cacheSize: Int)

    public var errorDescription: String? {
        switch self {
        case .notMaterialized(let name):
            "\(name) declares no `decode_c<k>_<N>` / `state_<N>` / `head` functions — export it with `uv run gemma-export`"
        case .modelPredatesLayerChunks(let name):
            "\(name) runs each step as one function per size — it predates the layer chunks the runtime now expects. Re-run `uv run gemma-export`."
        case .noUsableMaterializedFunctions(let decodeSizes, let prefillSizes):
            "No usable materialized sizes (decode sizes: \(decodeSizes), prefill sizes: \(prefillSizes))"
        case .modelPredatesCacheStates(let function, let cacheFeatures):
            "Function '\(function)' passes KV caches through its signature (\(cacheFeatures.isEmpty ? "no state features at all" : cacheFeatures.joined(separator: ", "))) — this model predates global-cache states. Re-run `uv run gemma-export`."
        case .modelPredatesHostEmbeddings(let function, let inputs):
            "Function '\(function)' takes \(inputs.joined(separator: ", ")) instead of the fp16 `token_embed` / `ple_rows` embedding rows — this model predates host-side embedding lookups. Re-run `uv run gemma-export`."
        case .unexpectedSignature(let function, let detail):
            "Function '\(function)' has an unexpected signature: \(detail)"
        case .functionLoadFailed(let function, let computeUnits, let underlying):
            "Could not load '\(function)' (\(computeUnits)): \(underlying.localizedDescription)"
                + (computeUnits.contains("ANE") || computeUnits == "all"
                    ? " — with the Neural Engine this usually means the ANE compiler rejected the function (details only in `log show --predicate 'process == \"ANECompilerService\"'`). It rejects cache sizes of 32768 and up: stay at --max-context 16384 or below, or use cpu-and-gpu."
                    : "")
        case .missingOutput(let name):
            "Prediction result is missing output '\(name)'"
        case .positionOutOfRange(let position, let cacheSize):
            "Token position \(position) does not fit a KV cache of size \(cacheSize)"
        }
    }
}
