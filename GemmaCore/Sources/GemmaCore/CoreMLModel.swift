/// CoreML model wrapper for the multifunction Gemma4-E2B .mlpackage.
///
/// Loads the prefill and decode functions from a single .mlpackage using
/// `functionName` on `MLModelConfiguration`. Supports both .mlpackage
/// (compiles and caches .mlmodelc) and pre-compiled .mlmodelc.

import CoreML
import Foundation

/// Holds the decode MLModel permanently and the prefill MLModel on demand.
///
/// CoreML allocates separate weight buffers for each loaded function, roughly
/// doubling memory when both are loaded. To avoid this, the prefill model is
/// loaded only when needed (via ``loadPrefill()``) and released after the
/// prefill phase (via ``releasePrefill()``).
public final class CoreMLModel: @unchecked Sendable {
    public let decodeModel: MLModel

    /// Prefill model — loaded on demand, released after use.
    public private(set) var prefillModel: MLModel?

    /// Stored for on-demand prefill reloading.
    private let modelURL: URL
    private let computeUnits: MLComputeUnits

    /// Logits output name for each function.
    public let decodeLogitsName: String
    public let prefillLogitsName: String

    /// Token input name for each function.
    public let decodeTokenInputName: String
    public let prefillTokenInputName: String

    /// Position input name for each function.
    public let decodePositionInputName: String
    public let prefillPositionInputName: String

    /// "N" phantom input name (global cache dim), nil for fixed-shape models.
    public let decodeNInputName: String?
    public let prefillNInputName: String?

    /// KV cache input/output names, in matched order.
    public let decodeKVInputNames: [String]
    public let decodeKVOutputNames: [String]
    public let prefillKVInputNames: [String]
    public let prefillKVOutputNames: [String]

    /// Global KV input names that have flexible (RangeDim) shapes.
    public let globalKVInputNames: Set<String>

    /// Shapes of each prefill KV input, extracted once so the prefill model
    /// can be released without losing the metadata needed to re-seed KV caches.
    public let prefillKVShapes: [String: [NSNumber]]
    /// Dtypes of each prefill KV input (companion to `prefillKVShapes`).
    public let prefillKVDtypes: [String: MLMultiArrayDataType]

    private init(
        decodeModel: MLModel,
        prefillModel: MLModel?,
        modelURL: URL,
        computeUnits: MLComputeUnits,
        prefillIO: ClassifiedIO,
        prefillKVShapes: [String: [NSNumber]],
        prefillKVDtypes: [String: MLMultiArrayDataType],
        decodeIO: ClassifiedIO,
        globalKVInputNames: Set<String>
    ) {
        self.decodeModel = decodeModel
        self.prefillModel = prefillModel
        self.modelURL = modelURL
        self.computeUnits = computeUnits
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
    }

    /// Load the multifunction model from a .mlpackage or .mlmodelc URL.
    ///
    /// For .mlpackage files, the model is compiled and cached as .mlmodelc
    /// next to the source for fast subsequent loads (E5RT cache reuse).
    /// For .mlmodelc files, loads directly without recompilation.
    public static func load(
        from url: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CoreMLModel {
        let modelURL: URL

        if url.pathExtension == "mlpackage" {
            // Compile and cache alongside the .mlpackage
            let cachedURL = url.deletingPathExtension().appendingPathExtension("mlmodelc")
            modelURL = try await compileAndCache(source: url, cached: cachedURL)
        } else {
            // Already compiled (.mlmodelc)
            modelURL = url
        }

        return try await loadCompiled(from: modelURL, computeUnits: computeUnits)
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
    /// Both decode and prefill functions are loaded with the requested
    /// compute units.  The prefill model is kept in memory initially but
    /// can be released via ``releasePrefill()`` and reloaded on demand
    /// with ``loadPrefill()``.
    private static func loadCompiled(
        from url: URL,
        computeUnits: MLComputeUnits
    ) async throws -> CoreMLModel {
        Log.info("[CoreML] Loading decode + prefill functions from \(url.lastPathComponent)...")

        // Load both functions in parallel
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
        Log.info("[CoreML] Both functions loaded.")

        let decodeIO = classifyIO(model: decodeModel)
        let prefillIO = classifyIO(model: prefillModel)
        let prefillInputDescs = prefillModel.modelDescription.inputDescriptionsByName
        var prefillKVShapes: [String: [NSNumber]] = [:]
        var prefillKVDtypes: [String: MLMultiArrayDataType] = [:]
        for name in prefillIO.kvInputNames {
            guard let c = prefillInputDescs[name]?.multiArrayConstraint else { continue }
            prefillKVShapes[name] = c.shape
            prefillKVDtypes[name] = c.dataType
        }

        Log.info("[CoreML] Decode: logits=\(decodeIO.logitsOutputName), token=\(decodeIO.tokenInputName), pos=\(decodeIO.positionInputName), kvIn=\(decodeIO.kvInputNames.count), kvOut=\(decodeIO.kvOutputNames.count)")
        Log.info("[CoreML] Prefill: logits=\(prefillIO.logitsOutputName), token=\(prefillIO.tokenInputName), pos=\(prefillIO.positionInputName), kvIn=\(prefillIO.kvInputNames.count), kvOut=\(prefillIO.kvOutputNames.count)")
        if decodeIO.nInputName != nil {
            Log.info("[CoreML] Dynamic context: N input detected (decode=\(decodeIO.nInputName!), prefill=\(prefillIO.nInputName ?? "none"))")
        }

        // Detect global KV inputs with flexible shapes (RangeDim).
        let globalNames = detectFlexibleGlobalKV(
            model: decodeModel, kvInputNames: decodeIO.kvInputNames
        )
        if !globalNames.isEmpty {
            Log.info("[CoreML] Flexible global KV caches: \(globalNames.sorted())")
        }

        // Both models kept — prefill avoids recompilation when needed
        return CoreMLModel(
            decodeModel: decodeModel,
            prefillModel: prefillModel,
            modelURL: url,
            computeUnits: computeUnits,
            prefillIO: prefillIO,
            prefillKVShapes: prefillKVShapes,
            prefillKVDtypes: prefillKVDtypes,
            decodeIO: decodeIO,
            globalKVInputNames: globalNames
        )
    }

    /// Load the prefill model with target compute units.
    /// Call before using ``prefill()``.  No-op if already loaded.
    public func loadPrefill() async throws {
        guard prefillModel == nil else { return }
        Log.info("[CoreML] Loading prefill model on demand...")
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.functionName = "prefill"
        prefillModel = try await MLModel.load(contentsOf: modelURL, configuration: config)
        Log.info("[CoreML] Prefill model loaded.")
    }

    /// Release the prefill model to free memory (~4-5 GB).
    /// Call after the prefill phase when only decode is needed.
    public func releasePrefill() {
        guard prefillModel != nil else { return }
        prefillModel = nil
        Log.info("[CoreML] Prefill model released.")
    }

    // MARK: - Prediction

    /// Run one prefill chunk.
    public func prefill(
        tokens: MLMultiArray,
        seqLen: Int32,
        kvState: KVCacheState,
        globalCacheSize: Int32? = nil
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        guard let prefillModel else {
            throw CoreMLModelError.prefillNotLoaded
        }
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
        let result = try prefillModel.prediction(from: provider)
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
        let result = try decodeModel.prediction(from: provider)
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
    case prefillNotLoaded
    case missingKVInput(String)

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "Model not found at the specified path"
        case .prefillNotLoaded:
            "Prefill model not loaded. Call loadPrefill() first."
        case .missingKVInput(let name):
            "KV cache is missing the array for input '\(name)'"
        }
    }
}
