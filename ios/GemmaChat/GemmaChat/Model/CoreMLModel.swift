 /// CoreML model wrapper for the multifunction Gemma4-E2B .mlpackage.
///
/// Loads the prefill and decode functions from a single .mlpackage using
/// iOS 18's `functionName` parameter on `MLModelConfiguration`.

import CoreML
import Foundation

/// Holds both prefill and decode MLModel instances plus their I/O metadata.
final class CoreMLModel: @unchecked Sendable {
    let prefillModel: MLModel
    let decodeModel: MLModel

    /// Logits output name for each function.
    let decodeLogitsName: String
    let prefillLogitsName: String

    /// Token input name for each function.
    let decodeTokenInputName: String
    let prefillTokenInputName: String

    /// Position input name for each function.
    let decodePositionInputName: String
    let prefillPositionInputName: String

    /// KV cache input/output names, in matched order.
    let decodeKVInputNames: [String]
    let decodeKVOutputNames: [String]
    let prefillKVInputNames: [String]
    let prefillKVOutputNames: [String]

    /// Pre-allocated reusable scalars to avoid per-step allocation.
    private let decodeTokenScalar: MLMultiArray
    private let decodePositionScalar: MLMultiArray
    private let prefillPositionScalar: MLMultiArray

    private init(
        prefillModel: MLModel,
        decodeModel: MLModel,
        prefillIO: ClassifiedIO,
        decodeIO: ClassifiedIO
    ) {
        self.prefillModel = prefillModel
        self.decodeModel = decodeModel
        self.prefillLogitsName = prefillIO.logitsOutputName
        self.decodeLogitsName = decodeIO.logitsOutputName
        self.prefillTokenInputName = prefillIO.tokenInputName
        self.decodeTokenInputName = decodeIO.tokenInputName
        self.prefillPositionInputName = prefillIO.positionInputName
        self.decodePositionInputName = decodeIO.positionInputName
        self.prefillKVInputNames = prefillIO.kvInputNames
        self.prefillKVOutputNames = prefillIO.kvOutputNames
        self.decodeKVInputNames = decodeIO.kvInputNames
        self.decodeKVOutputNames = decodeIO.kvOutputNames
        self.decodeTokenScalar = try! MLMultiArray(shape: [1], dataType: .int32)
        self.decodePositionScalar = try! MLMultiArray(shape: [1], dataType: .int32)
        self.prefillPositionScalar = try! MLMultiArray(shape: [1], dataType: .int32)
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
    static func load(
        from url: URL,
        computeUnits: MLComputeUnits = .cpuAndGPU
    ) async throws -> CoreMLModel {
        print("[CoreML] Loading decode function from \(url.lastPathComponent)...")

        // Load decode function
        let decodeConfig = MLModelConfiguration()
        decodeConfig.computeUnits = computeUnits
        decodeConfig.functionName = "decode"
        let decodeModel = try await MLModel.load(contentsOf: url, configuration: decodeConfig)
        print("[CoreML] Decode loaded. Inputs: \(Array(decodeModel.modelDescription.inputDescriptionsByName.keys).sorted())")

        print("[CoreML] Loading prefill function...")
        // Load prefill function
        let prefillConfig = MLModelConfiguration()
        prefillConfig.computeUnits = computeUnits
        prefillConfig.functionName = "prefill"
        let prefillModel = try await MLModel.load(contentsOf: url, configuration: prefillConfig)
        print("[CoreML] Prefill loaded. Inputs: \(Array(prefillModel.modelDescription.inputDescriptionsByName.keys).sorted())")

        // Classify I/O
        let decodeIO = Self.classifyIO(model: decodeModel)
        let prefillIO = Self.classifyIO(model: prefillModel)

        print("[CoreML] Decode: logits=\(decodeIO.logitsOutputName), token=\(decodeIO.tokenInputName), pos=\(decodeIO.positionInputName), kvIn=\(decodeIO.kvInputNames.count), kvOut=\(decodeIO.kvOutputNames.count)")
        print("[CoreML] Prefill: logits=\(prefillIO.logitsOutputName), token=\(prefillIO.tokenInputName), pos=\(prefillIO.positionInputName), kvIn=\(prefillIO.kvInputNames.count), kvOut=\(prefillIO.kvOutputNames.count)")

        return CoreMLModel(
            prefillModel: prefillModel,
            decodeModel: decodeModel,
            prefillIO: prefillIO,
            decodeIO: decodeIO
        )
    }

    // MARK: - Prediction

    /// Run one prefill chunk.
    /// - Parameters:
    ///   - tokens: Int32 array of shape (1, CHUNK_SIZE)
    ///   - seqLen: start position for this chunk
    ///   - kvState: current KV cache state
    /// - Returns: (logits as MLMultiArray, updated KVCacheState)
    func prefill(
        tokens: MLMultiArray,
        seqLen: Int32,
        kvState: KVCacheState
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        prefillPositionScalar.dataPointer.bindMemory(to: Int32.self, capacity: 1).pointee = seqLen

        var features: [String: MLMultiArray] = [:]
        features[prefillTokenInputName] = tokens
        features[prefillPositionInputName] = prefillPositionScalar

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
            inputNames: prefillKVInputNames
        )
        return (logits, newKV)
    }

    /// Run one decode step.
    /// - Parameters:
    ///   - token: single token ID
    ///   - position: absolute position in the sequence
    ///   - kvState: current KV cache state
    /// - Returns: (logits as MLMultiArray, updated KVCacheState)
    func decode(
        token: Int32,
        position: Int32,
        kvState: KVCacheState
    ) throws -> (logits: MLMultiArray, kvState: KVCacheState) {
        decodeTokenScalar.dataPointer.bindMemory(to: Int32.self, capacity: 1).pointee = token
        decodePositionScalar.dataPointer.bindMemory(to: Int32.self, capacity: 1).pointee = position

        var features: [String: MLMultiArray] = [:]
        features[decodeTokenInputName] = decodeTokenScalar
        features[decodePositionInputName] = decodePositionScalar

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
            inputNames: decodeKVInputNames
        )
        return (logits, newKV)
    }

    // MARK: - I/O Classification

    /// Classified I/O names for a model function.
    private struct ClassifiedIO {
        let logitsOutputName: String
        let tokenInputName: String      // "tokens" (prefill) or "token_id" (decode)
        let positionInputName: String   // "start_position" (prefill) or "position" (decode)
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

        // Identify token vs position among control inputs.
        // Token input has more elements (prefill: [1,8] vs [1]) or contains "token" in name.
        let (tokenName, posName) = Self.identifyControlInputs(
            controlInputs, inputDescs: inputDescs
        )

        return ClassifiedIO(
            logitsOutputName: logitsName,
            tokenInputName: tokenName,
            positionInputName: posName,
            kvInputNames: kvInputs,
            kvOutputNames: kvOutputs
        )
    }

    /// Distinguish the token input from the position input among control inputs.
    private static func identifyControlInputs(
        _ names: [String],
        inputDescs: [String: MLFeatureDescription]
    ) -> (tokenName: String, positionName: String) {
        precondition(names.count == 2, "Expected exactly 2 control inputs, got \(names.count)")

        // By element count: token input has more elements (prefill: 8 vs 1)
        let count0 = inputDescs[names[0]]?.multiArrayConstraint?.shape
            .map { $0.intValue }.reduce(1, *) ?? 1
        let count1 = inputDescs[names[1]]?.multiArrayConstraint?.shape
            .map { $0.intValue }.reduce(1, *) ?? 1
        if count0 != count1 {
            return count0 > count1 ? (names[0], names[1]) : (names[1], names[0])
        }

        // By name: "token" in name → token input
        if names[0].contains("token") { return (names[0], names[1]) }
        if names[1].contains("token") { return (names[1], names[0]) }

        // Fallback: first naturally-sorted name is token (works for _arg0/_arg1)
        let sorted = names.sorted { naturalCompare($0, $1) }
        return (sorted[0], sorted[1])
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

    var errorDescription: String? {
        switch self {
        case .modelNotFound:
            "gemma4-e2b.mlpackage not found in app bundle"
        }
    }
}
