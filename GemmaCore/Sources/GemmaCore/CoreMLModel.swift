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
    /// Compute units used for all function loads.
    private let computeUnits: MLComputeUnits

    /// Loaded MLModel instances, keyed by function name (e.g. "decode" or "decode_512").
    private var functionCache: [String: MLModel]
    private let cacheLock = NSLock()

    private init(
        prefillIO: ClassifiedIO,
        prefillKVShapes: [String: [NSNumber]],
        prefillKVDtypes: [String: MLMultiArrayDataType],
        decodeIO: ClassifiedIO,
        globalKVInputNames: Set<String>,
        materializedSizes: [Int]?,
        modelURL: URL,
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
        self.computeUnits = computeUnits
        self.functionCache = initialFunctions
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

        return try await loadCompiled(from: compiledURL, computeUnits: computeUnits)
    }

    /// Compile .mlpackage → .mlmodelc, caching at `cached` path.
    private static func compileAndCache(source: URL, cached: URL) async throws -> URL {
        // Invalidate cache if source model is newer
        if FileManager.default.fileExists(atPath: cached.path) {
            let srcDate = (try? FileManager.default.attributesOfItem(atPath: source.path)[.modificationDate] as? Date) ?? .distantPast
            let cacheDate = (try? FileManager.default.attributesOfItem(atPath: cached.path)[.modificationDate] as? Date) ?? .distantFuture
            if srcDate > cacheDate {
                Log.info("[CoreML] Source model newer than cache — recompiling")
                try? FileManager.default.removeItem(at: cached)
            }
        }

        if FileManager.default.fileExists(atPath: cached.path) {
            Log.info("[CoreML] Using cached compiled model at \(cached.path)")
            return cached
        }

        Log.info("[CoreML] Compiling \(source.lastPathComponent)...")
        let compiledURL = try await MLModel.compileModel(at: source)
        Log.info("[CoreML] Compiled to \(compiledURL.path)")

        // Move compiled model to cache location
        try? FileManager.default.removeItem(at: cached)
        try? FileManager.default.moveItem(at: compiledURL, to: cached)
        return FileManager.default.fileExists(atPath: cached.path) ? cached : compiledURL
    }

    /// Load a pre-compiled multifunction .mlmodelc.
    ///
    /// Tries standard function names (`decode`/`prefill`) first.
    /// If that fails, falls back to materialized names (`decode_64`/`prefill_64`/…).
    private static func loadCompiled(
        from url: URL,
        computeUnits: MLComputeUnits
    ) async throws -> CoreMLModel {
        Log.info("[CoreML] Loading decode + prefill functions from \(url.lastPathComponent)...")

        // Try standard model first
        do {
            return try await loadStandard(from: url, computeUnits: computeUnits)
        } catch {
            Log.info("[CoreML] Standard function load failed (\(error.localizedDescription)), trying materialized...")
        }
        return try await loadMaterialized(from: url, computeUnits: computeUnits)
    }

    /// Load a standard two-function model (decode + prefill).
    private static func loadStandard(
        from url: URL,
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
            computeUnits: computeUnits,
            initialFunctions: ["decode": decodeModel, "prefill": prefillModel]
        )
    }

    /// Load a materialized model with concrete function names.
    ///
    /// Discovers available sizes by probing `decode_N` for powers of 2 from
    /// 64 to 65536. Loads only the smallest pair at init; larger sizes are
    /// loaded on demand via `ensureLoaded(forGlobalCacheSize:)`.
    private static func loadMaterialized(
        from url: URL,
        computeUnits: MLComputeUnits
    ) async throws -> CoreMLModel {
        // Probe which sizes exist
        let candidateSizes = (6...16).map { 1 << $0 }  // 64, 128, ..., 65536
        var sizes: [Int] = []
        for s in candidateSizes {
            let config = MLModelConfiguration()
            config.computeUnits = computeUnits
            config.functionName = "decode_\(s)"
            do {
                let _ = try await MLModel.load(contentsOf: url, configuration: config)
                sizes.append(s)
            } catch {
                // Size not available, skip
            }
        }
        guard !sizes.isEmpty else {
            throw CoreMLModelError.modelNotFound
        }
        sizes.sort()
        Log.info("[CoreML] Materialized model with sizes: \(sizes)")

        let smallest = sizes.first!
        let decodeName = "decode_\(smallest)"
        let prefillName = "prefill_\(smallest)"

        let decodeConfig = MLModelConfiguration()
        decodeConfig.computeUnits = computeUnits
        decodeConfig.functionName = decodeName

        let prefillConfig = MLModelConfiguration()
        prefillConfig.computeUnits = computeUnits
        prefillConfig.functionName = prefillName

        async let decodeTask = MLModel.load(contentsOf: url, configuration: decodeConfig)
        async let prefillTask = MLModel.load(contentsOf: url, configuration: prefillConfig)

        let decodeModel = try await decodeTask
        let prefillModel = try await prefillTask
        Log.info("[CoreML] Loaded initial pair: \(decodeName), \(prefillName)")

        let decodeIO = classifyIO(model: decodeModel)
        let prefillIO = classifyIO(model: prefillModel)
        let (prefillKVShapes, prefillKVDtypes) = extractKVMetadata(
            model: prefillModel, kvInputNames: prefillIO.kvInputNames
        )
        logIOSummary(decodeIO: decodeIO, prefillIO: prefillIO)

        // For materialized models, global KV inputs are those with dim-1 == smallest
        // (sliding caches have fixed dim-1 == 512).
        let decodeDescs = decodeModel.modelDescription.inputDescriptionsByName
        var globalNames = Set<String>()
        for name in decodeIO.kvInputNames {
            guard let c = decodeDescs[name]?.multiArrayConstraint else { continue }
            let shape = c.shape.map { $0.intValue }
            if shape.count >= 2 && shape[1] == smallest {
                globalNames.insert(name)
            }
        }
        // Also find matching prefill KV names
        let prefillDescs = prefillModel.modelDescription.inputDescriptionsByName
        for name in prefillIO.kvInputNames {
            guard let c = prefillDescs[name]?.multiArrayConstraint else { continue }
            let shape = c.shape.map { $0.intValue }
            if shape.count >= 2 && shape[1] == smallest {
                globalNames.insert(name)
            }
        }
        if !globalNames.isEmpty {
            Log.info("[CoreML] Materialized global KV caches (\(globalNames.count)): \(globalNames.sorted().prefix(4))...")
        }

        return CoreMLModel(
            prefillIO: prefillIO,
            prefillKVShapes: prefillKVShapes,
            prefillKVDtypes: prefillKVDtypes,
            decodeIO: decodeIO,
            globalKVInputNames: globalNames,
            materializedSizes: sizes,
            modelURL: url,
            computeUnits: computeUnits,
            initialFunctions: [decodeName: decodeModel, prefillName: prefillModel]
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
        guard let model = functionCache[name] else {
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

        cacheLock.lock()
        let needDecode = functionCache[decodeName] == nil
        let needPrefill = functionCache[prefillName] == nil
        cacheLock.unlock()

        if !needDecode && !needPrefill { return }
        Log.info("[CoreML] Loading functions: \(decodeName), \(prefillName)...")

        if needDecode { try await loadIfNeeded(name: decodeName) }
        if needPrefill { try await loadIfNeeded(name: prefillName) }
    }

    /// Load a single function by name into the cache.
    private func loadIfNeeded(name: String) async throws {
        cacheLock.lock()
        if functionCache[name] != nil { cacheLock.unlock(); return }
        cacheLock.unlock()

        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.functionName = name
        let loaded = try await MLModel.load(contentsOf: modelURL, configuration: config)

        cacheLock.lock()
        functionCache[name] = loaded
        cacheLock.unlock()
        Log.info("[CoreML] Function '\(name)' loaded.")
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
