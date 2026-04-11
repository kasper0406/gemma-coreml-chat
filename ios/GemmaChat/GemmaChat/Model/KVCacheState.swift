/// KV cache state for Gemma4-E2B CoreML inference.
///
/// Holds KV arrays keyed by model input name, matching the multifunction
/// model's I/O contract. Uses positional matching between outputs and inputs
/// since input names (_argN) differ from output names (cast_N/slice_update_N).
///
/// Uses name-based keying (matching Python's dict approach) to avoid
/// dependency on dictionary ordering, which isn't guaranteed in Swift.

import CoreML
import Foundation

/// Immutable snapshot of the KV cache state.
/// Each prediction returns a new KVCacheState with updated arrays.
struct KVCacheState: @unchecked Sendable {
    /// KV arrays keyed by their model input name.
    /// E.g. {"flat_kv_0": array, "flat_kv_1": array, ...}
    let arraysByName: [String: MLMultiArray]

    /// Ordered input names (declaration order from model spec).
    let inputNames: [String]

    /// Number of tokens that have been processed into this cache.
    let processedTokens: Int

    /// Access arrays in input-name order (for positional compatibility).
    var arrays: [MLMultiArray] {
        inputNames.map { arraysByName[$0]! }
    }

    /// Create an empty state from the model's input descriptions.
    /// Reads actual shapes and dtypes from the model spec so it works with
    /// any model variant (uniform 1024 or heterogeneous sliding/global).
    /// Float16 arrays are zero-filled; Int32 arrays are filled with -1.
    static func empty(
        kvInputNames: [String],
        inputDescriptions: [String: MLFeatureDescription]
    ) -> KVCacheState {
        var dict: [String: MLMultiArray] = [:]
        for name in kvInputNames {
            if let desc = inputDescriptions[name],
               let constraint = desc.multiArrayConstraint
            {
                let shape = constraint.shape
                let dtype = constraint.dataType
                let array = try! MLMultiArray(shape: shape, dataType: dtype)
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
            processedTokens: 0
        )
    }

    /// Extract updated KV state from a prediction result.
    ///
    /// Maps output names → input names by position, rebuilding
    /// the name-keyed dictionary for the next call.
    static func from(
        prediction: MLFeatureProvider,
        outputNames: [String],
        inputNames: [String],
        processedTokens: Int = 0
    ) -> KVCacheState {
        precondition(outputNames.count == inputNames.count,
                     "KV output count (\(outputNames.count)) != input count (\(inputNames.count))")

        var dict: [String: MLMultiArray] = [:]
        dict.reserveCapacity(inputNames.count)

        // Probe first name to decide matching strategy (avoid full allSatisfy scan)
        let useNameMatching = !inputNames.isEmpty
            && prediction.featureValue(for: inputNames[0])?.multiArrayValue != nil

        if useNameMatching {
            for name in inputNames {
                dict[name] = prediction.featureValue(for: name)!.multiArrayValue!
            }
        } else {
            // Positional: output[i] → input[i]
            for (outName, inName) in zip(outputNames, inputNames) {
                guard let value = prediction.featureValue(for: outName)?.multiArrayValue else {
                    fatalError("Missing KV output tensor: \(outName)")
                }
                dict[inName] = value
            }
        }

        return KVCacheState(
            arraysByName: dict,
            inputNames: inputNames,
            processedTokens: processedTokens
        )
    }

    /// Create a new KVCacheState with an updated processedTokens count.
    func withProcessedTokens(_ count: Int) -> KVCacheState {
        KVCacheState(arraysByName: arraysByName, inputNames: inputNames, processedTokens: count)
    }

    /// Deep-copy all MLMultiArray buffers so this snapshot is independent
    /// of any future CoreML predictions (which may reuse output buffers).
    func deepCopy() -> KVCacheState {
        var copied: [String: MLMultiArray] = [:]
        copied.reserveCapacity(arraysByName.count)
        for (name, array) in arraysByName {
            let new = try! MLMultiArray(shape: array.shape, dataType: array.dataType)
            let bytesPerElement: Int
            switch array.dataType {
            case .float16: bytesPerElement = 2
            case .float32: bytesPerElement = 4
            case .int32:   bytesPerElement = 4
            case .double:  bytesPerElement = 8
            case .int8:    bytesPerElement = 1
            @unknown default: bytesPerElement = 4
            }
            memcpy(new.dataPointer, array.dataPointer, array.count * bytesPerElement)
            copied[name] = new
        }
        return KVCacheState(arraysByName: copied, inputNames: inputNames, processedTokens: processedTokens)
    }
}
