/// Standalone Swift benchmark for CoreML model loading and inference.
///
/// Tests multifunction Gemma4-E2B .mlpackage with different compute backends.
/// Measures: compilation, loading, first prediction, cached loading.
///
/// Build:  swiftc -O -parse-as-library model_bench.swift -o model_bench
/// Usage:  ./model_bench /path/to/gemma4-e2b.mlpackage [cpu|cpu-gpu|all]

import CoreML
import Foundation

// MARK: - Configuration

let chunkSize = 8
let vocabSize = 262_144

// MARK: - Timing

func measure(_ label: String, _ block: () throws -> Void) rethrows -> Double {
    let start = CFAbsoluteTimeGetCurrent()
    try block()
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    print("  \(label): \(String(format: "%.2f", elapsed))s")
    return elapsed
}

func measureAsync(_ label: String, _ block: () async throws -> Void) async rethrows -> Double {
    let start = CFAbsoluteTimeGetCurrent()
    try await block()
    let elapsed = CFAbsoluteTimeGetCurrent() - start
    print("  \(label): \(String(format: "%.2f", elapsed))s")
    return elapsed
}

// MARK: - I/O Classification (mirrors CoreMLModel.classifyIO)

struct ClassifiedIO {
    let logitsOutputName: String
    let tokenInputName: String
    let positionInputName: String
    let nInputName: String?
    let kvInputNames: [String]
    let kvOutputNames: [String]
}

func naturalCompare(_ a: String, _ b: String) -> Bool {
    a.compare(b, options: .numeric) == .orderedAscending
}

func classifyIO(model: MLModel) -> ClassifiedIO {
    let inputDescs = model.modelDescription.inputDescriptionsByName
    let outputDescs = model.modelDescription.outputDescriptionsByName

    var logitsName = ""
    var kvOutputs: [String] = []
    for (name, desc) in outputDescs {
        if let c = desc.multiArrayConstraint, c.dataType == .float32 {
            logitsName = name
        } else {
            kvOutputs.append(name)
        }
    }
    kvOutputs.sort(by: naturalCompare)

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

    if kvInputs.isEmpty && !kvOutputs.isEmpty {
        let allInputs = Array(inputDescs.keys).sorted(by: naturalCompare)
        let stateCount = kvOutputs.count
        let controlCount = allInputs.count - stateCount
        controlInputs = Array(allInputs.prefix(controlCount))
        kvInputs = Array(allInputs.suffix(stateCount))
    } else {
        controlInputs.sort(by: naturalCompare)
        kvInputs.sort(by: naturalCompare)
    }

    precondition(!logitsName.isEmpty, "No Float32 output found (logits)")
    precondition(kvInputs.count == kvOutputs.count,
                 "State in (\(kvInputs.count)) != out (\(kvOutputs.count))")

    // Identify control inputs
    var nName: String? = nil
    var remaining = controlInputs
    if let idx = remaining.firstIndex(of: "N") {
        nName = "N"
        remaining.remove(at: idx)
    }
    precondition(remaining.count == 2,
                 "Expected 2 control inputs after N, got \(remaining.count): \(remaining)")

    let count0 = inputDescs[remaining[0]]?.multiArrayConstraint?.shape
        .map { $0.intValue }.reduce(1, *) ?? 1
    let count1 = inputDescs[remaining[1]]?.multiArrayConstraint?.shape
        .map { $0.intValue }.reduce(1, *) ?? 1

    let tokenName: String
    let posName: String
    if count0 != count1 {
        (tokenName, posName) = count0 > count1
            ? (remaining[0], remaining[1])
            : (remaining[1], remaining[0])
    } else if remaining[0].contains("token") {
        (tokenName, posName) = (remaining[0], remaining[1])
    } else if remaining[1].contains("token") {
        (tokenName, posName) = (remaining[1], remaining[0])
    } else {
        let sorted = remaining.sorted(by: naturalCompare)
        (tokenName, posName) = (sorted[0], sorted[1])
    }

    return ClassifiedIO(
        logitsOutputName: logitsName,
        tokenInputName: tokenName,
        positionInputName: posName,
        nInputName: nName,
        kvInputNames: kvInputs,
        kvOutputNames: kvOutputs
    )
}

func detectFlexibleGlobalKV(model: MLModel, kvInputNames: [String]) -> Set<String> {
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

// MARK: - MLMultiArray helpers

func int32Scalar(_ v: Int32) -> MLMultiArray {
    let a = try! MLMultiArray(shape: [1], dataType: .int32)
    a[0] = NSNumber(value: v)
    return a
}

func int32Row(_ values: [Int32]) -> MLMultiArray {
    let a = try! MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: .int32)
    let ptr = a.dataPointer.bindMemory(to: Int32.self, capacity: values.count)
    for (i, v) in values.enumerated() { ptr[i] = v }
    return a
}

// MARK: - Feature Provider

final class InputProvider: MLFeatureProvider {
    let featureNames: Set<String>
    private let values: [String: MLFeatureValue]

    init(_ features: [String: MLMultiArray]) {
        var vals: [String: MLFeatureValue] = [:]
        vals.reserveCapacity(features.count)
        for (k, v) in features { vals[k] = MLFeatureValue(multiArray: v) }
        self.values = vals
        self.featureNames = Set(vals.keys)
    }

    func featureValue(for name: String) -> MLFeatureValue? { values[name] }
}

// MARK: - KV Cache

func makeEmptyKV(
    names: [String],
    descs: [String: MLFeatureDescription],
    globalNames: Set<String>,
    globalSize: Int
) -> [String: MLMultiArray] {
    var dict: [String: MLMultiArray] = [:]
    for name in names {
        guard let desc = descs[name], let c = desc.multiArrayConstraint else {
            dict[name] = try! MLMultiArray(shape: [1], dataType: .float16)
            continue
        }
        var shape = c.shape.map { $0.intValue }
        if globalNames.contains(name), shape.count == 4 {
            shape[1] = globalSize
        }
        let nsShape = shape.map { NSNumber(value: $0) }
        let arr = try! MLMultiArray(shape: nsShape, dataType: c.dataType)
        if c.dataType == .int32 {
            let ptr = arr.dataPointer.bindMemory(to: Int32.self, capacity: arr.count)
            for i in 0..<arr.count { ptr[i] = -1 }
        }
        dict[name] = arr
    }
    return dict
}

// MARK: - Prediction

func runPrefillChunk(
    model: MLModel,
    io: ClassifiedIO,
    tokens: [Int32],
    position: Int32,
    kvDict: [String: MLMultiArray],
    globalSize: Int32?
) throws -> (logits: MLMultiArray, kvOut: [String: MLMultiArray]) {
    var features: [String: MLMultiArray] = [:]
    features[io.tokenInputName] = int32Row(tokens)
    features[io.positionInputName] = int32Scalar(position)
    if let nName = io.nInputName, let n = globalSize {
        features[nName] = int32Scalar(n)
    }
    for (name, arr) in kvDict {
        features[name] = arr
    }

    let result = try model.prediction(from: InputProvider(features))
    let logits = result.featureValue(for: io.logitsOutputName)!.multiArrayValue!

    // Extract updated KV
    var newKV: [String: MLMultiArray] = [:]
    // Try name matching first (input name in outputs)
    let useNameMatch = !io.kvInputNames.isEmpty
        && result.featureValue(for: io.kvInputNames[0])?.multiArrayValue != nil
    if useNameMatch {
        for name in io.kvInputNames {
            newKV[name] = result.featureValue(for: name)!.multiArrayValue!
        }
    } else {
        for (outName, inName) in zip(io.kvOutputNames, io.kvInputNames) {
            newKV[inName] = result.featureValue(for: outName)!.multiArrayValue!
        }
    }
    return (logits, newKV)
}

func runDecode(
    model: MLModel,
    io: ClassifiedIO,
    token: Int32,
    position: Int32,
    kvDict: [String: MLMultiArray],
    globalSize: Int32?
) throws -> (logits: MLMultiArray, kvOut: [String: MLMultiArray]) {
    var features: [String: MLMultiArray] = [:]
    features[io.tokenInputName] = int32Scalar(token)
    features[io.positionInputName] = int32Scalar(position)
    if let nName = io.nInputName, let n = globalSize {
        features[nName] = int32Scalar(n)
    }
    for (name, arr) in kvDict {
        features[name] = arr
    }

    let result = try model.prediction(from: InputProvider(features))
    let logits = result.featureValue(for: io.logitsOutputName)!.multiArrayValue!

    var newKV: [String: MLMultiArray] = [:]
    let useNameMatch = !io.kvInputNames.isEmpty
        && result.featureValue(for: io.kvInputNames[0])?.multiArrayValue != nil
    if useNameMatch {
        for name in io.kvInputNames {
            newKV[name] = result.featureValue(for: name)!.multiArrayValue!
        }
    } else {
        for (outName, inName) in zip(io.kvOutputNames, io.kvInputNames) {
            newKV[inName] = result.featureValue(for: outName)!.multiArrayValue!
        }
    }
    return (logits, newKV)
}

/// Greedy argmax over Float32 logits.
func argmax(_ logits: MLMultiArray) -> Int32 {
    let count = logits.count
    let ptr = logits.dataPointer.bindMemory(to: Float32.self, capacity: count)
    var bestIdx = 0
    var bestVal = ptr[0]
    for i in 1..<count {
        if ptr[i] > bestVal {
            bestVal = ptr[i]
            bestIdx = i
        }
    }
    return Int32(bestIdx)
}

/// Extract logits for a single position from (CHUNK_SIZE, vocabSize) output.
func extractLogits(at position: Int, from logits: MLMultiArray, vocabSize: Int) -> MLMultiArray {
    let result = try! MLMultiArray(shape: [NSNumber(value: vocabSize)], dataType: .float32)
    let src = logits.dataPointer.bindMemory(to: Float32.self, capacity: logits.count)
    let dst = result.dataPointer.bindMemory(to: Float32.self, capacity: vocabSize)
    dst.update(from: src.advanced(by: position * vocabSize), count: vocabSize)
    return result
}

// MARK: - Benchmark

func parseComputeUnits(_ s: String) -> MLComputeUnits {
    switch s.lowercased() {
    case "cpu":      return .cpuOnly
    case "cpu-gpu":  return .cpuAndGPU
    case "all":      return .all
    default:
        print("Unknown compute unit '\(s)', using cpu-gpu")
        return .cpuAndGPU
    }
}

@main
struct ModelBench {
    static func main() async {
        let args = CommandLine.arguments
        guard args.count >= 2 else {
            print("Usage: model_bench <model.mlpackage> [cpu|cpu-gpu|all] [--generate N]")
            return
        }

        let modelPath = args[1]
        let computeUnits = args.count >= 3 ? parseComputeUnits(args[2]) : .cpuAndGPU
        var generateTokens = 5  // default: 5 tokens for quick smoke test
        if let genIdx = args.firstIndex(of: "--generate"), genIdx + 1 < args.count {
            generateTokens = Int(args[genIdx + 1]) ?? 5
        }

        let unitName: String
        switch computeUnits {
        case .cpuOnly:     unitName = "CPU_ONLY"
        case .cpuAndGPU:   unitName = "CPU_AND_GPU"
        case .all:         unitName = "ALL (ANE)"
        default:           unitName = "unknown"
        }

        let modelURL = URL(fileURLWithPath: modelPath)
        let cacheDir = FileManager.default.temporaryDirectory.appendingPathComponent("model_bench_cache")

        print("╔══════════════════════════════════════════════════╗")
        print("║  CoreML Model Benchmark (Swift)                 ║")
        print("╚══════════════════════════════════════════════════╝")
        print()
        print("Model:         \(modelPath)")
        print("Compute units: \(unitName)")
        print("Generate:      \(generateTokens) tokens")
        print()

        // MARK: Step 1 — Compile .mlpackage → .mlmodelc
        print("━━━ Step 1: Compile .mlpackage → .mlmodelc ━━━")
        let compiledURL: URL
        let cachedModelcURL = cacheDir.appendingPathComponent("gemma4-e2b.mlmodelc")

        if FileManager.default.fileExists(atPath: cachedModelcURL.path) {
            print("  Using cached .mlmodelc at \(cachedModelcURL.path)")
            compiledURL = cachedModelcURL
        } else {
            do {
                var tempCompiled: URL!
                _ = try await measureAsync("Compile") {
                    tempCompiled = try await MLModel.compileModel(at: modelURL)
                }
                try FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true)
                try? FileManager.default.removeItem(at: cachedModelcURL)
                try FileManager.default.moveItem(at: tempCompiled, to: cachedModelcURL)
                compiledURL = cachedModelcURL
                print("  Cached to: \(cachedModelcURL.path)")
            } catch {
                print("  ❌ Compile failed: \(error)")
                return
            }
        }
        print()

        // MARK: Step 2 — Load decode function
        print("━━━ Step 2: Load decode (\(unitName)) ━━━")
        let decodeModel: MLModel
        do {
            var model: MLModel!
            _ = try await measureAsync("Load decode") {
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits
                config.functionName = "decode"
                model = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            }
            decodeModel = model
        } catch {
            print("  ❌ Load failed: \(error)")
            return
        }

        let decodeIO = classifyIO(model: decodeModel)
        let globalNames = detectFlexibleGlobalKV(model: decodeModel, kvInputNames: decodeIO.kvInputNames)
        print("  Logits:    \(decodeIO.logitsOutputName)")
        print("  Token:     \(decodeIO.tokenInputName)")
        print("  Position:  \(decodeIO.positionInputName)")
        print("  N input:   \(decodeIO.nInputName ?? "none")")
        print("  KV pairs:  \(decodeIO.kvInputNames.count)")
        print("  Global KV: \(globalNames.count) flexible inputs")
        print()

        // MARK: Step 3 — Load prefill function
        print("━━━ Step 3: Load prefill (\(unitName)) ━━━")
        let prefillModel: MLModel
        do {
            var model: MLModel!
            _ = try await measureAsync("Load prefill") {
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits
                config.functionName = "prefill"
                model = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            }
            prefillModel = model
        } catch {
            print("  ❌ Load failed: \(error)")
            return
        }

        let prefillIO = classifyIO(model: prefillModel)
        let prefillDescs = prefillModel.modelDescription.inputDescriptionsByName
        print("  Logits:    \(prefillIO.logitsOutputName)")
        print("  Token:     \(prefillIO.tokenInputName)")
        print("  Position:  \(prefillIO.positionInputName)")
        print("  N input:   \(prefillIO.nInputName ?? "none")")
        print("  KV pairs:  \(prefillIO.kvInputNames.count)")

        // Check if prefill/decode KV names match
        let namesMatch = prefillIO.kvInputNames == decodeIO.kvInputNames
        print("  KV names match decode: \(namesMatch ? "✅ yes" : "⚠️ NO")")
        if !namesMatch {
            print("  Prefill KV: \(prefillIO.kvInputNames.prefix(3))...")
            print("  Decode KV:  \(decodeIO.kvInputNames.prefix(3))...")
        }
        print()

        // MARK: Step 4 — Prefill prediction
        print("━━━ Step 4: Prefill prediction (1 chunk) ━━━")
        // Use token ID 2 (BOS) followed by padding
        let promptTokens: [Int32] = [2, 106, 1645, 108, 2299, 106, 2516, 108]  // ~"<bos>Hi"
        let nReal = promptTokens.count
        let globalSize = chunkSize

        var kvDict = makeEmptyKV(
            names: prefillIO.kvInputNames,
            descs: prefillDescs,
            globalNames: globalNames,
            globalSize: globalSize
        )

        var lastLogits: MLMultiArray!
        do {
            _ = try measure("Prefill chunk 0") {
                let (logits, newKV) = try runPrefillChunk(
                    model: prefillModel,
                    io: prefillIO,
                    tokens: promptTokens,
                    position: 0,
                    kvDict: kvDict,
                    globalSize: globalNames.isEmpty ? nil : Int32(globalSize)
                )
                kvDict = newKV
                lastLogits = logits
            }
        } catch {
            print("  ❌ Prefill failed: \(error)")
            return
        }

        // Extract logits at last real token
        let lastPosInChunk = nReal - 1
        if lastLogits.shape.count > 1 && lastLogits.shape[0].intValue > 1 {
            lastLogits = extractLogits(at: lastPosInChunk, from: lastLogits, vocabSize: vocabSize)
        }

        let firstToken = argmax(lastLogits)
        let topLogit = lastLogits.dataPointer.bindMemory(to: Float32.self, capacity: vocabSize)[Int(firstToken)]
        print("  First predicted token: \(firstToken) (logit=\(String(format: "%.3f", topLogit)))")
        print()

        // MARK: Step 5 — Decode predictions
        print("━━━ Step 5: Decode \(generateTokens) tokens ━━━")
        var generatedTokens: [Int32] = []
        var totalDecodeTime = 0.0
        var currentToken = firstToken
        var currentLogits = lastLogits!

        do {
            for step in 0..<generateTokens {
                let t = try measure("Decode step \(step) (token=\(currentToken), pos=\(nReal + step))") {
                    let (logits, newKV) = try runDecode(
                        model: decodeModel,
                        io: decodeIO,
                        token: currentToken,
                        position: Int32(nReal + step),
                        kvDict: kvDict,
                        globalSize: globalNames.isEmpty ? nil : Int32(globalSize)
                    )
                    kvDict = newKV
                    currentLogits = logits
                }
                totalDecodeTime += t
                generatedTokens.append(currentToken)
                currentToken = argmax(currentLogits)

                if currentToken == 1 || currentToken == 106 {
                    print("  → Stop token \(currentToken), ending generation")
                    break
                }
            }
        } catch {
            print("  ❌ Decode failed: \(error)")
            return
        }

        if !generatedTokens.isEmpty {
            let tokPerSec = Double(generatedTokens.count) / totalDecodeTime
            print("  Generated \(generatedTokens.count) tokens in \(String(format: "%.2f", totalDecodeTime))s")
            print("  → \(String(format: "%.2f", tokPerSec)) tok/s")
            print("  Token IDs: \(generatedTokens)")
        }
        print()

        // MARK: Step 6 — Cached reload benchmark
        print("━━━ Step 6: Cached reload (2nd load from .mlmodelc) ━━━")
        do {
            _ = try await measureAsync("Re-load decode") {
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits
                config.functionName = "decode"
                let _ = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            }
            _ = try await measureAsync("Re-load prefill") {
                let config = MLModelConfiguration()
                config.computeUnits = computeUnits
                config.functionName = "prefill"
                let _ = try await MLModel.load(contentsOf: compiledURL, configuration: config)
            }
        } catch {
            print("  ❌ Reload failed: \(error)")
        }
        print()

        // MARK: Summary
        print("━━━ Summary ━━━")
        print("  Backend:   \(unitName)")
        print("  KV pairs:  \(decodeIO.kvInputNames.count)")
        print("  Dynamic N: \(decodeIO.nInputName != nil ? "yes" : "no")")
        print("  Global KV: \(globalNames.count) flexible")
        print("  Tokens:    \(generatedTokens)")
        print("  ✅ All steps completed")
    }
}
