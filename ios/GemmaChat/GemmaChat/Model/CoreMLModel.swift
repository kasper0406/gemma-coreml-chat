 /// CoreML model wrapper for the multifunction Gemma4-E2B .mlpackage.
///
/// Loads the prefill and decode functions from a single .mlpackage using
/// iOS 18's `functionName` parameter on `MLModelConfiguration`.

import CoreML
import Foundation

/// Holds the decode MLModel permanently and the prefill MLModel on demand.
///
/// CoreML allocates separate weight buffers for each loaded function, roughly
/// doubling memory when both are loaded. To avoid this, the prefill model is
/// loaded only when needed (via ``loadPrefill()``) and released after the
/// prefill phase (via ``releasePrefill()``).
final class CoreMLModel: @unchecked Sendable {
    let decodeModel: MLModel

    /// Prefill model — loaded on demand, released after use.
    private(set) var prefillModel: MLModel?

    /// Stored for on-demand prefill reloading.
    private let modelURL: URL
    private let computeUnits: MLComputeUnits

    /// Logits output name for each function.
    let decodeLogitsName: String
    let prefillLogitsName: String

    /// Token input name for each function.
    let decodeTokenInputName: String
    let prefillTokenInputName: String

    /// Position input name for each function.
    let decodePositionInputName: String
    let prefillPositionInputName: String

    /// "N" phantom input name (global cache dim), nil for fixed-shape models.
    let decodeNInputName: String?
    let prefillNInputName: String?

    /// KV cache input/output names, in matched order.
    let decodeKVInputNames: [String]
    let decodeKVOutputNames: [String]
    let prefillKVInputNames: [String]
    let prefillKVOutputNames: [String]

    /// Global KV input names that have flexible (RangeDim) shapes.
    let globalKVInputNames: Set<String>

    /// Prefill input descriptions for KV cache initialization.
    /// Extracted once so the prefill model can be released between uses.
    let prefillInputDescriptions: [String: MLFeatureDescription]

    private init(
        decodeModel: MLModel,
        modelURL: URL,
        computeUnits: MLComputeUnits,
        prefillIO: ClassifiedIO,
        prefillInputDescriptions: [String: MLFeatureDescription],
        decodeIO: ClassifiedIO,
        globalKVInputNames: Set<String>
    ) {
        self.decodeModel = decodeModel
        self.prefillModel = nil
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
        self.prefillInputDescriptions = prefillInputDescriptions
        self.globalKVInputNames = globalKVInputNames
    }

    /// Load the multifunction model from the app bundle.
    ///
    /// The .mlpackage is bundled as a folder reference (not compiled by Xcode)
    /// because `coremlc` strips multifunction metadata. We compile at runtime
    /// via `MLModel.compileModel(at:)`, cache the result, and load with `functionName`.
    static func load(
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CoreMLModel {
        guard let url = Bundle.main.url(forResource: "gemma4-e2b", withExtension: "mlpackage") else {
            throw CoreMLModelError.modelNotFound
        }

        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let cachedURL = appSupport.appendingPathComponent("gemma4-e2b.mlmodelc")

        // Invalidate cache if source model is newer than cached compilation
        if FileManager.default.fileExists(atPath: cachedURL.path) {
            let srcDate = (try? FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date) ?? .distantPast
            let cacheDate = (try? FileManager.default.attributesOfItem(atPath: cachedURL.path)[.modificationDate] as? Date) ?? .distantFuture
            if srcDate > cacheDate {
                print("[CoreML] Source model newer than cache — recompiling")
                try? FileManager.default.removeItem(at: cachedURL)
            }
        }

        let modelURL: URL
        if FileManager.default.fileExists(atPath: cachedURL.path) {
            print("[CoreML] Using cached compiled model at \(cachedURL.path)")
            modelURL = cachedURL
        } else {
            print("[CoreML] Compiling \(url.lastPathComponent)...")
            let compiledURL = try await MLModel.compileModel(at: url)
            print("[CoreML] Compiled to \(compiledURL.path)")

            try? FileManager.default.createDirectory(at: appSupport, withIntermediateDirectories: true)
            try? FileManager.default.moveItem(at: compiledURL, to: cachedURL)
            modelURL = FileManager.default.fileExists(atPath: cachedURL.path) ? cachedURL : compiledURL
        }

        return try await load(from: modelURL, computeUnits: computeUnits)
    }

    /// Load a pre-compiled multifunction .mlmodelc.
    ///
    /// Only the decode model is loaded with full compute units.  The prefill
    /// model is loaded temporarily with `.cpuOnly` to extract I/O metadata,
    /// then released immediately.  Call ``loadPrefill()`` before ``prefill()``.
    static func load(
        from url: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CoreMLModel {
        print("[CoreML] Loading decode function from \(url.lastPathComponent)...")

        // Load decode function with target compute units
        let decodeConfig = MLModelConfiguration()
        decodeConfig.computeUnits = computeUnits
        decodeConfig.functionName = "decode"
        let decodeModel = try await MLModel.load(contentsOf: url, configuration: decodeConfig)
        print("[CoreML] Decode loaded. Inputs: \(Array(decodeModel.modelDescription.inputDescriptionsByName.keys).sorted())")

        // Load prefill with cpuOnly just to extract I/O metadata.
        // This avoids the memory cost of GPU/ANE compilation for a model
        // we won't run predictions on until loadPrefill() is called.
        print("[CoreML] Extracting prefill I/O metadata (cpuOnly)...")
        let prefillConfig = MLModelConfiguration()
        prefillConfig.computeUnits = .cpuOnly
        prefillConfig.functionName = "prefill"
        let tempPrefill = try await MLModel.load(contentsOf: url, configuration: prefillConfig)

        let decodeIO = Self.classifyIO(model: decodeModel)
        let prefillIO = Self.classifyIO(model: tempPrefill)
        let prefillInputDescs = tempPrefill.modelDescription.inputDescriptionsByName

        print("[CoreML] Decode: logits=\(decodeIO.logitsOutputName), token=\(decodeIO.tokenInputName), pos=\(decodeIO.positionInputName), kvIn=\(decodeIO.kvInputNames.count), kvOut=\(decodeIO.kvOutputNames.count)")
        print("[CoreML] Prefill: logits=\(prefillIO.logitsOutputName), token=\(prefillIO.tokenInputName), pos=\(prefillIO.positionInputName), kvIn=\(prefillIO.kvInputNames.count), kvOut=\(prefillIO.kvOutputNames.count)")
        if decodeIO.nInputName != nil {
            print("[CoreML] Dynamic context: N input detected (decode=\(decodeIO.nInputName!), prefill=\(prefillIO.nInputName ?? "none"))")
        }

        // Detect global KV inputs with flexible shapes (RangeDim).
        let globalNames = Self.detectFlexibleGlobalKV(
            model: decodeModel, kvInputNames: decodeIO.kvInputNames
        )
        if !globalNames.isEmpty {
            print("[CoreML] Flexible global KV caches: \(globalNames.sorted())")
        }

        // tempPrefill released here — only metadata is kept
        return CoreMLModel(
            decodeModel: decodeModel,
            modelURL: url,
            computeUnits: computeUnits,
            prefillIO: prefillIO,
            prefillInputDescriptions: prefillInputDescs,
            decodeIO: decodeIO,
            globalKVInputNames: globalNames
        )
    }

    /// Load the prefill model with target compute units.
    /// Call before using ``prefill()``.  No-op if already loaded.
    func loadPrefill() async throws {
        guard prefillModel == nil else { return }
        print("[CoreML] Loading prefill model on demand...")
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        config.functionName = "prefill"
        prefillModel = try await MLModel.load(contentsOf: modelURL, configuration: config)
        print("[CoreML] Prefill model loaded.")
    }

    /// Release the prefill model to free memory (~4-5 GB).
    /// Call after the prefill phase when only decode is needed.
    func releasePrefill() {
        guard prefillModel != nil else { return }
        prefillModel = nil
        print("[CoreML] Prefill model released.")
    }

    // MARK: - Prediction

    /// Run one prefill chunk.
    /// - Parameters:
    ///   - tokens: Int32 array of shape (1, CHUNK_SIZE)
    ///   - seqLen: start position for this chunk
    ///   - kvState: current KV cache state
    ///   - globalCacheSize: current global KV cache dim (for N input), nil for fixed models
    /// - Returns: (logits as MLMultiArray, updated KVCacheState)
    func prefill(
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
        let newKV = KVCacheState.from(
            prediction: result,
            outputNames: prefillKVOutputNames,
            inputNames: prefillKVInputNames,
            globalNames: kvState.globalNames
        )
        return (logits, newKV)
    }

    /// Run one decode step.
    /// - Parameters:
    ///   - token: single token ID
    ///   - position: absolute position in the sequence
    ///   - kvState: current KV cache state
    ///   - globalCacheSize: current global KV cache dim (for N input), nil for fixed models
    /// - Returns: (logits as MLMultiArray, updated KVCacheState)
    func decode(
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
        let newKV = KVCacheState.from(
            prediction: result,
            outputNames: decodeKVOutputNames,
            inputNames: decodeKVInputNames,
            globalNames: kvState.globalNames
        )
        return (logits, newKV)
    }

    // MARK: - I/O Classification

    /// Classified I/O names for a model function.
    private struct ClassifiedIO {
        let logitsOutputName: String
        let tokenInputName: String      // "tokens" (prefill) or "token_id" (decode)
        let positionInputName: String   // "start_position" (prefill) or "position" (decode)
        let nInputName: String?         // "N" for dynamic-shape models, nil otherwise
        let kvInputNames: [String]      // KV cache + ring tracker inputs, naturally sorted
        let kvOutputNames: [String]     // KV cache + ring tracker outputs, naturally sorted
    }

    /// Classify model I/O using name matching with positional fallback.
    ///
    /// Primary: if an input name also appears as a non-logits output, it's state.
    /// Fallback (old models with _argN naming): positional split based on
    /// state output count.
    private static func classifyIO(model: MLModel) -> ClassifiedIO {
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

        // Try name matching: input name ∈ output names (exact or with _out suffix) → state
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

        // Fallback: if name matching found no state inputs (old _argN naming),
        // use positional split
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

        // Identify token vs position among control inputs,
        // and detect the optional N (global cache dim) input.
        let (tokenName, posName, nName) = Self.identifyControlInputs(
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

        // Detect N input by name first (shape (1,) Int32, named "N")
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
    /// These are global attention caches that can grow dynamically.
    private static func detectFlexibleGlobalKV(
        model: MLModel,
        kvInputNames: [String]
    ) -> Set<String> {
        var flex: Set<String> = []
        for name in kvInputNames {
            guard let desc = model.modelDescription.inputDescriptionsByName[name],
                  let constraint = desc.multiArrayConstraint else { continue }
            // RangeDim → shapeConstraint.type == .range
            if constraint.shapeConstraint.type == .range {
                flex.insert(name)
            }
        }
        return flex
    }

    /// Natural string comparison: numeric segments are compared by value,
    /// so "_arg2" < "_arg10".
    private static func naturalCompare(_ a: String, _ b: String) -> Bool {
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
private final class CoreMLInputProvider: MLFeatureProvider {
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
                fatalError("KV cache missing array for input '\(name)'")
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
    static func int32Scalar(_ value: Int32) -> MLMultiArray {
        let array = try! MLMultiArray(shape: [1], dataType: .int32)
        array[0] = NSNumber(value: value)
        return array
    }

    /// Create an Int32 array of shape (1, length) from a Swift array.
    static func int32Row(_ values: [Int32]) -> MLMultiArray {
        let array = try! MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
        let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
        for (i, v) in values.enumerated() { ptr[i] = v }
        return array
    }
}

// MARK: - Errors

enum CoreMLModelError: Error, LocalizedError {
    case modelNotFound
    case prefillNotLoaded

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "gemma4-e2b.mlpackage not found in app bundle"
        case .prefillNotLoaded:
            "Prefill model not loaded. Call loadPrefill() first."
        }
    }
}
