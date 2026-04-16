/// KV cache state for Gemma4-E2B CoreML inference.
///
/// Holds KV arrays keyed by model input name, matching the multifunction
/// model's I/O contract. Uses positional matching between outputs and inputs
/// since input names (_argN) differ from output names (cast_N/slice_update_N).

import CoreML
import Foundation

/// Immutable snapshot of the KV cache state.
/// Each prediction returns a new KVCacheState with updated arrays.
public struct KVCacheState: @unchecked Sendable {
    /// KV arrays keyed by their model input name.
    public let arraysByName: [String: MLMultiArray]

    /// Ordered input names (declaration order from model spec).
    public let inputNames: [String]

    /// Number of tokens that have been processed into this cache.
    public let processedTokens: Int

    /// Names of KV inputs with flexible (RangeDim) shapes — the global caches.
    public let globalNames: Set<String>

    /// Access arrays in input-name order (for positional compatibility).
    public var arrays: [MLMultiArray] {
        inputNames.map { arraysByName[$0]! }
    }

    /// Current dim-1 size of global caches, or nil if none.
    public var currentGlobalCacheSize: Int? {
        for name in globalNames {
            if let arr = arraysByName[name] {
                return arr.shape[1].intValue
            }
        }
        return nil
    }

    public init(
        arraysByName: [String: MLMultiArray],
        inputNames: [String],
        processedTokens: Int,
        globalNames: Set<String>
    ) {
        self.arraysByName = arraysByName
        self.inputNames = inputNames
        self.processedTokens = processedTokens
        self.globalNames = globalNames
    }

    /// Create an empty state from the model's input descriptions.
    /// Global caches are sized to `initialGlobalSize` instead of the spec shape.
    /// Float16 arrays are zero-filled; Int32 arrays are filled with -1.
    public static func empty(
        kvInputNames: [String],
        inputDescriptions: [String: MLFeatureDescription],
        globalNames: Set<String> = [],
        initialGlobalSize: Int? = nil
    ) -> KVCacheState {
        var dict: [String: MLMultiArray] = [:]
        for name in kvInputNames {
            if let desc = inputDescriptions[name],
               let constraint = desc.multiArrayConstraint
            {
                var shape = constraint.shape.map { $0.intValue }
                // Override dim-1 for global caches when dynamic sizing is active
                if let size = initialGlobalSize, globalNames.contains(name), shape.count == 4 {
                    shape[1] = size
                }
                let nsShape = shape.map { NSNumber(value: $0) }
                let dtype = constraint.dataType
                let array = try! MLMultiArray(shape: nsShape, dataType: dtype)
                if dtype == .int32 {
                    let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
                    for i in 0..<array.count { ptr[i] = -1 }
                }
                dict[name] = array
            } else {
                let array = try! MLMultiArray(shape: [1], dataType: .float16)
                dict[name] = array
            }
        }
        return KVCacheState(
            arraysByName: dict,
            inputNames: kvInputNames,
            processedTokens: 0,
            globalNames: globalNames
        )
    }

    /// Extract updated KV state from a prediction result.
    ///
    /// Maps output names → input names by position, rebuilding
    /// the name-keyed dictionary for the next call.
    /// Deep-copies each array because CoreML reuses output buffers.
    public static func from(
        prediction: MLFeatureProvider,
        outputNames: [String],
        inputNames: [String],
        processedTokens: Int = 0,
        globalNames: Set<String> = []
    ) -> KVCacheState {
        precondition(outputNames.count == inputNames.count,
                     "KV output count (\(outputNames.count)) != input count (\(inputNames.count))")

        var dict: [String: MLMultiArray] = [:]
        dict.reserveCapacity(inputNames.count)

        // Probe first name to decide matching strategy
        let useNameMatching = !inputNames.isEmpty
            && prediction.featureValue(for: inputNames[0])?.multiArrayValue != nil

        if useNameMatching {
            for name in inputNames {
                dict[name] = prediction.featureValue(for: name)!.multiArrayValue!.deepCopy()
            }
        } else {
            // Positional: output[i] → input[i]
            for (outName, inName) in zip(outputNames, inputNames) {
                guard let value = prediction.featureValue(for: outName)?.multiArrayValue else {
                    fatalError("Missing KV output tensor: \(outName)")
                }
                dict[inName] = value.deepCopy()
            }
        }

        return KVCacheState(
            arraysByName: dict,
            inputNames: inputNames,
            processedTokens: processedTokens,
            globalNames: globalNames
        )
    }

    /// Create a new KVCacheState with an updated processedTokens count.
    public func withProcessedTokens(_ count: Int) -> KVCacheState {
        KVCacheState(arraysByName: arraysByName, inputNames: inputNames,
                     processedTokens: count, globalNames: globalNames)
    }

    /// Grow global caches so dim-1 >= `needed` using a doubling strategy.
    /// Returns self unchanged if no growth is required or if there are no global caches.
    public func grownToFit(needed: Int, maxLen: Int) -> KVCacheState {
        guard !globalNames.isEmpty else { return self }
        guard let curSize = currentGlobalCacheSize, curSize < needed else { return self }

        let newLen = min(max(curSize * 2, needed), maxLen)
        Log.info("[KV] Growing global caches: \(curSize) → \(newLen) (needed \(needed))")

        var newDict = arraysByName
        for name in globalNames {
            guard let old = arraysByName[name] else { continue }
            let oldShape = old.shape.map { $0.intValue }
            guard oldShape.count == 4 else { continue }

            let newShape = [oldShape[0], newLen, oldShape[2], oldShape[3]]
            let nsShape = newShape.map { NSNumber(value: $0) }
            let grown = try! MLMultiArray(shape: nsShape, dataType: old.dataType)

            // Copy old data into the front of the grown array
            let oldBytes = old.count * MemoryLayout<Float16>.stride
            memcpy(grown.dataPointer, old.dataPointer, oldBytes)

            newDict[name] = grown
        }

        return KVCacheState(
            arraysByName: newDict,
            inputNames: inputNames,
            processedTokens: processedTokens,
            globalNames: globalNames
        )
    }
}
