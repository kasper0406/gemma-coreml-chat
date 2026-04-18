/// CoreML model wrapper for the multifunction Gemma4-E2B .mlpackage.
///
/// Supports two model layouts:
/// - **Standard**: Two functions named `decode` and `prefill`, with optional
///   RangeDim on global KV cache inputs.
/// - **Materialized**: Concrete-shape functions named `decode_64`, `decode_128`,
///   …, `decode_65536` and `prefill_64`, …, `prefill_65536`. Each function is
///   specialized to a specific global KV cache size (no dynamic shape ops).
///   The runtime selects the function matching the current cache size.
///
/// Materialized models are produced by `gemma-materialize` and work on all
/// backends including ANE and iPhone, whereas standard RangeDim models only
/// work on GPU.

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

    /// Shapes of each prefill KV input, extracted once so the prefill model
    /// can be released without losing the metadata needed to re-seed KV caches.
    public let prefillKVShapes: [String: [NSNumber]]
    /// Dtypes of each prefill KV input (companion to `prefillKVShapes`).
    public let prefillKVDtypes: [String: MLMultiArrayDataType]

    /// Available materialized sizes (sorted ascending), or nil for standard models.
    public let materializedSizes: [Int]?

    /// URL of the compiled .mlmodelc (for lazy function loading).
    private let modelURL: URL
    /// URL the caller originally passed to `load(from:)` (.mlpackage or .mlmodelc).
    /// Used as a stable identity for the warm-cache sentinel; its mod-time
    /// invalidates the sentinel when the source is replaced or re-exported.
    private let sourceURL: URL
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
    private enum LoadState {
        case loaded(MLModel)
        case loading(Task<SendableMLModel, Error>)
    }

    /// Function state keyed by function name (e.g. "decode" or "decode_512").
    private var functions: [String: LoadState]
    private let cacheLock = NSLock()

    private init(
        prefillIO: ClassifiedIO,
        prefillKVShapes: [String: [NSNumber]],
        prefillKVDtypes: [String: MLMultiArrayDataType],
        decodeIO: ClassifiedIO,
        globalKVInputNames: Set<String>,
        materializedSizes: [Int]?,
        modelURL: URL,
        sourceURL: URL,
        computeUnits: MLComputeUnits,
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
        self.materializedSizes = materializedSizes
        self.modelURL = modelURL
        self.sourceURL = sourceURL
        self.computeUnits = computeUnits
        self.functions = initialFunctions.mapValues { .loaded($0) }
    }

    /// Load the multifunction model from a .mlpackage or .mlmodelc URL.
    ///
    /// Auto-detects whether the model uses standard (`decode`/`prefill`)
    /// or materialized (`decode_64`/`prefill_64`/…) function names.
    ///
    /// For .mlpackage files, the model is compiled and cached as .mlmodelc
    /// next to the source for fast subsequent loads (E5RT cache reuse).
    /// For .mlmodelc files, loads directly without recompilation.
    public static func load(
        from url: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CoreMLModel {
        let compiledURL: URL

        if url.pathExtension == "mlpackage" {
            // Compile and cache alongside the .mlpackage
            let cachedURL = url.deletingPathExtension().appendingPathExtension("mlmodelc")
            compiledURL = try await compileAndCache(source: url, cached: cachedURL)
        } else {
            // Already compiled (.mlmodelc)
            compiledURL = url
        }

        return try await loadCompiled(from: compiledURL, sourceURL: url, computeUnits: computeUnits)
    }

    /// Compile .mlpackage → .mlmodelc, caching at `cached` path.
    ///
    /// Invalidates via a content hash of the source's `model.mlmodel` stored
    /// in a sidecar. Mtime comparison is unreliable here: swapping in a
    /// different `.mlpackage` build (e.g. non-materialized ↔ materialized) can
    /// leave the source older than an existing cache, masking a real change.
    private static func compileAndCache(source: URL, cached: URL) async throws -> URL {
        let sidecar = cached.appendingPathExtension("src-sha256")
        let currentHash = sourceModelHash(source: source)

        if FileManager.default.fileExists(atPath: cached.path) {
            let cachedHash = (try? String(contentsOf: sidecar, encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if let c = currentHash, let s = cachedHash, c == s {
                Log.info("[CoreML] Using cached compiled model at \(cached.path)")
                return cached
            }
            Log.info("[CoreML] Cache hash \(cachedHash == nil ? "missing" : "mismatch") — recompiling")
            try? FileManager.default.removeItem(at: cached)
            try? FileManager.default.removeItem(at: sidecar)
        }

        Log.info("[CoreML] Compiling \(source.lastPathComponent)...")
        let compiledURL = try await MLModel.compileModel(at: source)
        Log.info("[CoreML] Compiled to \(compiledURL.path)")

        try? FileManager.default.removeItem(at: cached)
        try? FileManager.default.moveItem(at: compiledURL, to: cached)
        let finalURL = FileManager.default.fileExists(atPath: cached.path) ? cached : compiledURL
        if finalURL == cached, let hash = currentHash {
            try? hash.write(to: sidecar, atomically: true, encoding: .utf8)
        }
        return finalURL
    }

    /// SHA-256 of the source package's `model.mlmodel` (structure/metadata,
    /// excluding the multi-GB weights blob). Sufficient to detect shape
    /// changes — materialization, function-set edits, I/O renames — which are
    /// what would invalidate a compiled cache.
    private static func sourceModelHash(source: URL) -> String? {
        let mlmodel = source.appendingPathComponent("Data/com.apple.CoreML/model.mlmodel")
        guard let data = try? Data(contentsOf: mlmodel) else { return nil }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// Load a pre-compiled multifunction .mlmodelc.
    ///
    /// Tries standard function names (`decode`/`prefill`) first.
    /// If that fails, falls back to materialized names (`decode_64`/`prefill_64`/…).
    private static func loadCompiled(
        from url: URL,
        sourceURL: URL,
        computeUnits: MLComputeUnits
    ) async throws -> CoreMLModel {
        Log.info("[CoreML] Loading decode + prefill functions from \(url.lastPathComponent)...")

        // Try standard model first
        do {
            return try await loadStandard(from: url, sourceURL: sourceURL, computeUnits: computeUnits)
        } catch {
            Log.info("[CoreML] Standard function load failed (\(error.localizedDescription)), trying materialized...")
        }
        return try await loadMaterialized(from: url, sourceURL: sourceURL, computeUnits: computeUnits)
    }

    /// Load a standard two-function model (decode + prefill).
    private static func loadStandard(
        from url: URL,
        sourceURL: URL,
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
            modelURL: url,
            sourceURL: sourceURL,
            computeUnits: computeUnits,
            initialFunctions: ["decode": decodeModel, "prefill": prefillModel]
        )
    }

    /// Load a materialized model with concrete function names.
    ///
    /// Enumerates available sizes from the compiled model.mil manifest (cheap)
    /// and loads only the three functions needed to bootstrap inference:
    /// `decode_{smallest}`, `decode_{second}`, `prefill_{smallest}`. The two
    /// decode models are needed to discriminate global caches (dim-1 scales
    /// with cache size) from sliding caches (dim-1 fixed at window size) —
    /// without two sizes we can't distinguish them when the window size
    /// coincides with the smallest materialized size.
    ///
    /// Remaining function pairs load in the background via `preloadAllSizes()`.
    /// If mil enumeration fails, falls back to probing a power-of-2 range and
    /// keeping whatever decodes succeed.
    private static func loadMaterialized(
        from url: URL,
        sourceURL: URL,
        computeUnits: MLComputeUnits
    ) async throws -> CoreMLModel {
        let probeDecodes: [(size: Int, model: MLModel)]
        let sizes: [Int]
        let prefillModel: MLModel
        let prefillName: String

        if let discovered = enumerateMaterializedSizes(compiledURL: url), !discovered.isEmpty {
            sizes = discovered
            Log.info("[CoreML] Materialized sizes (from manifest): \(sizes)")

            let smallest = sizes[0]
            prefillName = "prefill_\(smallest)"
            let secondSize: Int? = sizes.count >= 2 ? sizes[1] : nil

            async let decode1Task = Self.loadFunction(
                url: url, computeUnits: computeUnits, function: "decode_\(smallest)"
            )
            async let prefillTask = Self.loadFunction(
                url: url, computeUnits: computeUnits, function: prefillName
            )
            async let decode2Task: MLModel? = {
                guard let s = secondSize else { return nil }
                return try? await Self.loadFunction(
                    url: url, computeUnits: computeUnits, function: "decode_\(s)"
                )
            }()

            let decode1 = try await decode1Task
            let prefill = try await prefillTask
            let decode2 = await decode2Task
            prefillModel = prefill

            var probes: [(size: Int, model: MLModel)] = [(smallest, decode1)]
            if let s = secondSize, let m = decode2 {
                probes.append((s, m))
            }
            probeDecodes = probes
            Log.info("[CoreML] Loaded initial pair: decode_\(smallest), \(prefillName)\(secondSize.map { " (+ decode_\($0) for classification)" } ?? "")")
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
            probeDecodes = loaded

            prefillName = "prefill_\(sizes[0])"
            prefillModel = try await Self.loadFunction(
                url: url, computeUnits: computeUnits, function: prefillName
            )
            Log.info("[CoreML] Materialized model with sizes: \(sizes) (probed)")
        }

        let decodeModel = probeDecodes[0].model
        let decodeIO = classifyIO(model: decodeModel)
        let prefillIO = classifyIO(model: prefillModel)
        let (prefillKVShapes, prefillKVDtypes) = extractKVMetadata(
            model: prefillModel, kvInputNames: prefillIO.kvInputNames
        )
        logIOSummary(decodeIO: decodeIO, prefillIO: prefillIO)

        // Globals are the KV caches whose dim-1 scales with the materialized size;
        // sliding caches keep a fixed dim-1 (= sliding window). Distinguishing them
        // requires comparing two different sizes — the old "dim-1 == smallest"
        // heuristic misclassifies sliding caches whenever the window equals the
        // smallest materialized size (e.g. window=512 and sizes start at 512).
        let globalNames: Set<String>
        if probeDecodes.count >= 2 {
            globalNames = detectGlobalsByShape(
                modelA: probeDecodes[0].model,
                modelB: probeDecodes[1].model,
                kvInputNames: decodeIO.kvInputNames
            )
            Log.info("[CoreML] Materialized global KV caches (\(globalNames.count)): \(globalNames.sorted())")
        } else {
            // Only one size available — growth is a no-op anyway since the cache
            // never exceeds the sole materialized size.
            globalNames = []
            Log.info("[CoreML] Only one materialized size; skipping global detection")
        }

        var initialFunctions: [String: MLModel] = [:]
        for (size, model) in probeDecodes {
            initialFunctions["decode_\(size)"] = model
        }
        initialFunctions[prefillName] = prefillModel

        let instance = CoreMLModel(
            prefillIO: prefillIO,
            prefillKVShapes: prefillKVShapes,
            prefillKVDtypes: prefillKVDtypes,
            decodeIO: decodeIO,
            globalKVInputNames: globalNames,
            materializedSizes: sizes,
            modelURL: url,
            sourceURL: sourceURL,
            computeUnits: computeUnits,
            initialFunctions: initialFunctions
        )
        instance.preloadAllSizes()
        return instance
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

    /// Discover materialized sizes by scanning the compiled model.mil manifest
    /// for `func decode_N<…>` / `func prefill_N<…>` declarations. Returns the
    /// sorted list of sizes that have BOTH a decode and prefill function, or
    /// nil if the manifest is missing/unparseable.
    static func enumerateMaterializedSizes(compiledURL: URL) -> [Int]? {
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
        let common = decodeSizes.intersection(prefillSizes)
        return common.isEmpty ? nil : common.sorted()
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
        Log.info("[CoreML] Decode: logits=\(decodeIO.logitsOutputName), token=\(decodeIO.tokenInputName), pos=\(decodeIO.positionInputName), kvIn=\(decodeIO.kvInputNames.count), kvOut=\(decodeIO.kvOutputNames.count)")
        Log.info("[CoreML] Prefill: logits=\(prefillIO.logitsOutputName), token=\(prefillIO.tokenInputName), pos=\(prefillIO.positionInputName), kvIn=\(prefillIO.kvInputNames.count), kvOut=\(prefillIO.kvOutputNames.count)")
        if decodeIO.nInputName != nil {
            Log.info("[CoreML] Dynamic context: N input detected (decode=\(decodeIO.nInputName!), prefill=\(prefillIO.nInputName ?? "none"))")
        }
    }

    // MARK: - Function Resolution

    /// Round a cache size up to the nearest materialized size.
    /// Returns nil for standard (non-materialized) models.
    public func materializedSize(forCacheSize cacheSize: Int) -> Int? {
        guard let sizes = materializedSizes else { return nil }
        return sizes.first { $0 >= cacheSize } ?? sizes.last!
    }

    /// Resolve the function name for a given prefix and cache size.
    private func functionName(prefix: String, cacheSize: Int?) -> String {
        if let cacheSize, let size = materializedSize(forCacheSize: cacheSize) {
            return "\(prefix)_\(size)"
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
            group.addTask { _ = try await self.loadIfNeeded(name: prefillName) }
            try await group.waitForAll()
        }
    }

    /// Result of checking the cache for `name`: an already-loaded model, or a
    /// load Task (either newly started by us or one a concurrent caller had
    /// already kicked off).
    private enum CacheLookup {
        case existing(MLModel)
        case pending(Task<SendableMLModel, Error>)
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
            case .loading(let task): return .pending(task)
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
        functions[name] = .loading(task)
        return .pending(task)
    }

    private func markLoaded(name: String, model: MLModel) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        functions[name] = .loaded(model)
    }

    private func clearPending(name: String) {
        cacheLock.lock()
        defer { cacheLock.unlock() }
        if case .loading = functions[name] { functions.removeValue(forKey: name) }
    }

    /// Load a single function by name. Concurrent callers for the same name
    /// share one in-flight Task instead of issuing duplicate loads.
    @discardableResult
    private func loadIfNeeded(name: String) async throws -> MLModel {
        switch lookupOrStart(name: name) {
        case .existing(let model):
            return model
        case .pending(let task):
            do {
                let model = try await task.value.model
                markLoaded(name: name, model: model)
                Log.info("[CoreML] Function '\(name)' loaded.")
                return model
            } catch {
                clearPending(name: name)
                throw error
            }
        }
    }

    /// Kick off background loads for every materialized function pair in
    /// ascending size order. Non-blocking: later calls to `ensureLoaded` join
    /// in-flight tasks rather than issuing duplicate loads.
    public func preloadAllSizes(concurrency: Int = 2) {
        guard let sizes = materializedSizes else { return }
        let names = sizes.flatMap { ["decode_\($0)", "prefill_\($0)"] }
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
        let names = sizes.flatMap { ["decode_\($0)", "prefill_\($0)"] }
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

    /// Whether this model's functions have previously been compiled to the
    /// ANE / E5RT cache for the current compute units. When false, the first
    /// run will pay multi-minute compilation on each new function; the app
    /// should call `warmSynchronously(progress:)` before entering the chat.
    public var isWarmed: Bool {
        guard let sentinel = Self.warmSentinelURL(
            sourceName: sourceURL.lastPathComponent,
            computeUnits: computeUnits
        ) else { return false }
        guard let sentinelDate = attrDate(sentinel, key: .modificationDate) else { return false }
        // Sentinel is only valid if at least as new as the source file.
        let sourceDate = attrDate(sourceURL, key: .modificationDate) ?? .distantPast
        return sentinelDate >= sourceDate
    }

    private func markWarmed() {
        guard let sentinel = Self.warmSentinelURL(
            sourceName: sourceURL.lastPathComponent,
            computeUnits: computeUnits
        ) else { return }
        let dir = sentinel.deletingLastPathComponent()
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: sentinel.path, contents: nil)
        try? FileManager.default.setAttributes(
            [.modificationDate: Date()], ofItemAtPath: sentinel.path
        )
        Log.info("[CoreML] Marked warm cache: \(sentinel.lastPathComponent)")
    }

    /// Sentinel path in Application Support keyed by source file name + CU.
    /// Lives outside the .mlmodelc so it survives re-compilation and works on
    /// iOS where the bundle (and thus .mlmodelc next to it) is read-only.
    private static func warmSentinelURL(
        sourceName: String,
        computeUnits: MLComputeUnits
    ) -> URL? {
        guard let appSupport = try? FileManager.default.url(
            for: .applicationSupportDirectory, in: .userDomainMask,
            appropriateFor: nil, create: true
        ) else { return nil }
        let dir = appSupport.appendingPathComponent("GemmaCore", isDirectory: true)
        let safe = sourceName.replacingOccurrences(of: "/", with: "_")
        let cu = computeUnitsTag(computeUnits)
        return dir.appendingPathComponent("warmed-\(safe)-\(cu).marker")
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

    private func attrDate(_ url: URL, key: FileAttributeKey) -> Date? {
        (try? FileManager.default.attributesOfItem(atPath: url.path)[key]) as? Date
    }

    // MARK: - Prediction

    /// Run one prefill chunk.
    public func prefill(
        tokens: MLMultiArray,
        seqLen: Int32,
        kvState: KVCacheState,
        globalCacheSize: Int32? = nil
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
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
        let result = try activeModel.prediction(from: provider)
        let logits = result.featureValue(for: prefillLogitsName)!.multiArrayValue!
        let newKV = try KVCacheState.from(
            prediction: result,
            outputNames: prefillKVOutputNames,
            inputNames: prefillKVInputNames,
            globalNames: kvState.globalNames
        )
        return (logits, newKV)
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
        let result = try activeModel.prediction(from: provider)
        let logits = result.featureValue(for: decodeLogitsName)!.multiArrayValue!
        let newKV = try KVCacheState.from(
            prediction: result,
            outputNames: decodeKVOutputNames,
            inputNames: decodeKVInputNames,
            globalNames: kvState.globalNames
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
        let kvInputNames: [String]
        let kvOutputNames: [String]
    }

    /// Classify model I/O using name matching with positional fallback.
    static func classifyIO(model: MLModel) -> ClassifiedIO {
        let inputDescs = model.modelDescription.inputDescriptionsByName
        let outputDescs = model.modelDescription.outputDescriptionsByName

        // Outputs: float32 = logits, everything else = state
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

        // Try name matching: input name ∈ output names → state
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
               "State input count (\(kvInputs.count)) != output count (\(kvOutputs.count))")

        let (tokenName, posName, nName) = identifyControlInputs(
            controlInputs, inputDescs: inputDescs
        )

        return ClassifiedIO(
            logitsOutputName: logitsName,
            tokenInputName: tokenName,
            positionInputName: posName,
            nInputName: nName,
            kvInputNames: kvInputs,
            kvOutputNames: kvOutputs
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

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "Model not found at the specified path"
        case .missingKVInput(let name):
            "KV cache is missing the array for input '\(name)'"
        }
    }
}
