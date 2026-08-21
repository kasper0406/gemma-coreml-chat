/// KV cache state for Gemma4-E2B CoreML inference.
///
/// Holds KV arrays keyed by model input name, matching the multifunction
/// model's I/O contract. Uses positional matching between outputs and inputs
/// since input names (_argN) differ from output names (cast_N/slice_update_N).

import CoreML
import Foundation

/// Errors raised while building or updating a KV cache state.
public enum KVCacheError: Error, LocalizedError {
    case missingOutputTensor(String)
    case missingShapeOrDtype(String)

    public var errorDescription: String? {
        switch self {
        case .missingOutputTensor(let name):
            "Missing KV output tensor: \(name)"
        case .missingShapeOrDtype(let name):
            "Missing shape or dtype for KV input '\(name)'"
        }
    }
}

/// The single source of truth for how big a global KV cache may be.
///
/// A materialized model can only run the concrete sizes it was exported with,
/// so **every** place that allocates or grows a global cache has to round
/// through this one policy. Rounding independently — "next power of two" — is
/// only accidentally correct for the default contiguous power-of-two export:
/// with `--materialize-sizes 512,2048`, crossing 512 tokens grows the cache to
/// 1024 while `CoreMLModel.functionName(prefix:cacheSize:)` resolves
/// `decode_2048`, and every turn then fails on a shape mismatch.
///
/// Vend one from ``CoreMLModel/cacheSizePolicy`` rather than constructing it
/// ad hoc: the model wrapper is what knows the sizes that were actually loaded.
public struct KVCacheSizePolicy: Sendable {
    /// Concrete materialized sizes in ascending order, or nil for standard
    /// (RangeDim) models, which accept any dim-1.
    public let materializedSizes: [Int]?

    /// Largest cache size the loaded model can serve.
    public let maxLen: Int

    public init(materializedSizes: [Int]?, maxLen: Int) {
        self.materializedSizes = materializedSizes?.sorted()
        self.maxLen = maxLen
    }

    /// Smallest runnable cache size that holds `needed` tokens.
    ///
    /// When `needed` exceeds `maxLen` the result is clamped to `maxLen`, so
    /// callers must independently cap how many token positions they feed the
    /// model — a clamped cache cannot hold every requested position.
    public func size(forNeeded needed: Int) -> Int {
        let clamped = min(max(needed, 1), maxLen)
        if let sizes = materializedSizes, let largest = sizes.last {
            return sizes.first { $0 >= clamped } ?? largest
        }
        var pow2 = 1
        while pow2 < clamped { pow2 *= 2 }
        return min(pow2, maxLen)
    }
}

/// Immutable snapshot of the KV cache state.
/// Each prediction returns a new KVCacheState with updated arrays.
public struct KVCacheState: @unchecked Sendable {
    /// KV arrays keyed by their model input name.
    public let arraysByName: [String: MLMultiArray]

    /// Ordered input names (declaration order from model spec).
    public let inputNames: [String]

    /// Names of KV inputs with flexible (RangeDim) shapes — the global caches.
    public let globalNames: Set<String>

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
        globalNames: Set<String>
    ) {
        self.arraysByName = arraysByName
        self.inputNames = inputNames
        self.globalNames = globalNames
    }

    /// Create an empty state from pre-extracted KV shapes/dtypes.
    /// Global caches are sized to `initialGlobalSize` instead of the spec shape.
    /// Float* arrays are zero-filled; Int32 arrays are filled with -1.
    public static func empty(
        kvInputNames: [String],
        shapes: [String: [NSNumber]],
        dtypes: [String: MLMultiArrayDataType],
        globalNames: Set<String> = [],
        initialGlobalSize: Int? = nil
    ) throws -> KVCacheState {
        var dict: [String: MLMultiArray] = [:]
        for name in kvInputNames {
            guard let specShape = shapes[name], let dtype = dtypes[name] else {
                throw KVCacheError.missingOutputTensor(name)
            }
            var shape = specShape.map { $0.intValue }
            // Override dim-1 for global caches when dynamic sizing is active
            if let size = initialGlobalSize, globalNames.contains(name), shape.count == 4 {
                shape[1] = size
            }
            let nsShape = shape.map { NSNumber(value: $0) }
            dict[name] = try allocateZeroedKV(shape: nsShape, dtype: dtype)
        }
        return KVCacheState(
            arraysByName: dict,
            inputNames: kvInputNames,
            globalNames: globalNames
        )
    }

    /// Allocate an MLMultiArray and initialize it for KV-cache use.
    /// MLMultiArray does NOT guarantee zero initialization, so we explicitly
    /// zero float arrays and fill int32 arrays with -1 (the "empty slot" sentinel).
    static func allocateZeroedKV(
        shape: [NSNumber], dtype: MLMultiArrayDataType
    ) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: dtype)
        if dtype == .int32 {
            let ptr = array.dataPointer.bindMemory(to: Int32.self, capacity: array.count)
            for i in 0..<array.count { ptr[i] = -1 }
        } else {
            memset(array.dataPointer, 0, array.count * bytesPerElement(of: dtype))
        }
        return array
    }

    /// Bytes per element for the dtypes this package uses.
    static func bytesPerElement(of dtype: MLMultiArrayDataType) -> Int {
        switch dtype {
        case .float16: return 2
        case .float32: return 4
        case .float64: return 8
        case .int32:   return 4
        default:       return 2
        }
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
        globalNames: Set<String> = []
    ) throws -> KVCacheState {
        precondition(outputNames.count == inputNames.count,
                     "KV output count (\(outputNames.count)) != input count (\(inputNames.count))")

        var dict: [String: MLMultiArray] = [:]
        dict.reserveCapacity(inputNames.count)

        // Probe first name to decide matching strategy
        let useNameMatching = !inputNames.isEmpty
            && prediction.featureValue(for: inputNames[0])?.multiArrayValue != nil

        if useNameMatching {
            for name in inputNames {
                guard let value = prediction.featureValue(for: name)?.multiArrayValue else {
                    throw KVCacheError.missingOutputTensor(name)
                }
                dict[name] = try value.deepCopy()
            }
        } else {
            // Positional: output[i] → input[i]
            for (outName, inName) in zip(outputNames, inputNames) {
                guard let value = prediction.featureValue(for: outName)?.multiArrayValue else {
                    throw KVCacheError.missingOutputTensor(outName)
                }
                dict[inName] = try value.deepCopy()
            }
        }

        return KVCacheState(
            arraysByName: dict,
            inputNames: inputNames,
            globalNames: globalNames
        )
    }

    /// Grow global caches so dim-1 >= `needed`, using `policy` to pick the new
    /// size. Returns self unchanged if no growth is required, if there are no
    /// global caches, or if the policy cannot offer anything larger.
    ///
    /// The size MUST come from the policy: the resolved `{decode,prefill}_N`
    /// function and the cache have to agree on dim-1, and only the model knows
    /// which `N` were exported.
    public func grownToFit(needed: Int, policy: KVCacheSizePolicy) throws -> KVCacheState {
        guard !globalNames.isEmpty else { return self }
        guard let curSize = currentGlobalCacheSize, curSize < needed else { return self }

        let newLen = policy.size(forNeeded: needed)
        guard newLen > curSize else { return self }
        Log.info("[KV] Growing global caches: \(curSize) → \(newLen) (needed \(needed))")

        var newDict = arraysByName
        for name in globalNames {
            guard let old = arraysByName[name] else { continue }
            let oldShape = old.shape.map { $0.intValue }
            guard oldShape.count == 4 else { continue }

            let newShape = [oldShape[0], newLen, oldShape[2], oldShape[3]]
            let nsShape = newShape.map { NSNumber(value: $0) }
            let grown = try Self.allocateZeroedKV(shape: nsShape, dtype: old.dataType)
            memcpy(grown.dataPointer, old.dataPointer,
                   old.count * Self.bytesPerElement(of: old.dataType))

            newDict[name] = grown
        }

        return KVCacheState(
            arraysByName: newDict,
            inputNames: inputNames,
            globalNames: globalNames
        )
    }
}
