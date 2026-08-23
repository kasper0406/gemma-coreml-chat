/// Swift benchmark runner for Gemma4-E2B CoreML models.
///
/// Links the shared `GemmaCore` library so it exercises the same inference
/// path as the CLI and iOS app, and benefits from the compiled-model cache
/// (`.mlmodelc` under $TMPDIR/gemma-bench-cache/).  Python's coremltools
/// re-compiles on every load, which is why these measurements live in Swift.
///
/// Usage:
///   swift run -c release GemmaBench \
///     --model ./gemma4-e2b.mlpackage \
///     --compute-units all \
///     --context-length 1024 \
///     --decode-tokens 64 \
///     --run-index 0
///
/// Writes one JSON object per run to stdout.  The orchestrator that invokes
/// this binary (benchmarks/runner.py) is responsible for looping over the
/// backend × context-length × run matrix and aggregating the output.

import CoreML
import Foundation
import GemmaCore


// ── Args ─────────────────────────────────────────────────────────────────


struct BenchArgs {
    var modelPath: String = "./gemma4-e2b.mlpackage"
    var computeUnitsName: String = "cpu-gpu"
    var contextLength: Int = 128
    var decodeTokens: Int = 32
    var runIndex: Int = 0
    var warmup: Bool = true                  // do one prefill+decode to prime caches
}


func parseArgs() -> BenchArgs {
    let av = CommandLine.arguments
    var a = BenchArgs()
    var i = 1
    while i < av.count {
        let flag = av[i]
        let next: String? = (i + 1 < av.count) ? av[i + 1] : nil
        switch flag {
        case "--model":            a.modelPath = next!; i += 2
        case "--compute-units":    a.computeUnitsName = next!; i += 2
        case "--context-length":   a.contextLength = Int(next!)!; i += 2
        case "--decode-tokens":    a.decodeTokens = Int(next!)!; i += 2
        case "--run-index":        a.runIndex = Int(next!)!; i += 2
        case "--no-warmup":        a.warmup = false; i += 1
        case "--help", "-h":
            printUsage()
            exit(0)
        default:
            FileHandle.standardError.write(Data("Unknown flag: \(flag)\n".utf8))
            printUsage()
            exit(2)
        }
    }
    return a
}


func printUsage() {
    let msg = """
    Usage: GemmaBench --model <path> [options]

      --model PATH              .mlpackage or .mlmodelc (required)
      --compute-units UNITS     cpu | cpu-gpu | cpu-ane | all   (default: cpu-gpu)
      --context-length N        tokens in the fake prompt       (default: 128)
      --decode-tokens N         decode steps to time            (default: 32)
      --run-index N             metadata only, echoed back      (default: 0)
      --no-warmup               skip the priming prefill+decode

    Emits one JSON object on stdout.

    """
    FileHandle.standardError.write(Data(msg.utf8))
}


func parseComputeUnits(_ s: String) -> MLComputeUnits {
    switch s.lowercased() {
    case "cpu":      return .cpuOnly
    case "cpu-gpu":  return .cpuAndGPU
    case "cpu-ane", "ane":  return .cpuAndNeuralEngine
    case "all":      return .all
    default:
        FileHandle.standardError.write(Data("Unknown compute unit '\(s)', using cpu-gpu\n".utf8))
        return .cpuAndGPU
    }
}


// ── Utility ──────────────────────────────────────────────────────────────


func now() -> Double { CFAbsoluteTimeGetCurrent() }


/// Build a synthetic prompt of `length` token IDs.  We use a rolling
/// sequence seeded by `seed` so different runs touch different KV rows
/// (defeats any fast-path caching of identical prompts inside CoreML).
func syntheticPrompt(length: Int, seed: Int) -> [Int32] {
    var out = [Int32](); out.reserveCapacity(length)
    // Gemma vocab is 262144; stay in a simple mid range of "safe" IDs.
    var rng: UInt64 = 0x9E37_79B9_7F4A_7C15 &+ UInt64(bitPattern: Int64(seed))
    for _ in 0..<length {
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        // Keep IDs well away from 0 (padding) and special tokens.
        let id = Int32(1024 + Int(rng >> 48) % 100_000)
        out.append(id)
    }
    return out
}


/// Global KV cache sizes the engine will touch for a run of `promptLen` prompt
/// tokens generating `newTokens` tokens: the padded-prefill bucket and the
/// decode-target bucket, both put through the model's own bucketing policy.
///
/// The bench pre-loads exactly these before starting a timer. Leaving it to the
/// background preload instead means a multi-GB `MLModel.load` can land inside
/// the measured window — and `ensureLoaded` for the decode bucket then races
/// that preload, so `prefill_time_s` randomly includes a whole model load.
func requiredCacheSizes(model: CoreMLModel, promptLen: Int, newTokens: Int) -> [Int] {
    let nChunks = (promptLen + GemmaConfig.chunkSize - 1) / GemmaConfig.chunkSize
    let paddedLen = nChunks * GemmaConfig.chunkSize
    let maxSteps = min(newTokens, max(model.effectiveMaxSeqLen - promptLen, 0))
    let decodeTarget = min(promptLen + maxSteps, model.effectiveMaxSeqLen)
    let policy = model.cacheSizePolicy
    return [policy.size(forNeeded: paddedLen), policy.size(forNeeded: decodeTarget)]
}


struct RunJSON: Encodable {
    let run_index: Int
    let context_length: Int
    let compute_units: String
    let decode_tokens_requested: Int
    let prefill_time_s: Double
    let decode_time_s: Double
    let decode_tokens_generated: Int
    let prefill_tokens_per_sec: Double
    let decode_tokens_per_sec: Double
    let warmup: Bool
    let load_time_s: Double
    let total_time_s: Double
}


func emitJSON(_ r: RunJSON) {
    let enc = JSONEncoder()
    enc.outputFormatting = [.sortedKeys]
    let data = try! enc.encode(r)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write(Data("\n".utf8))
}


// ── Benchmark ────────────────────────────────────────────────────────────


@main
struct GemmaBenchMain {
    static func main() async {
        let args = parseArgs()
        let units = parseComputeUnits(args.computeUnitsName)
        let modelURL = URL(fileURLWithPath: args.modelPath).standardizedFileURL

        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            FileHandle.standardError.write(Data("error: model not found at \(modelURL.path)\n".utf8))
            exit(2)
        }

        // A bench run's context length is fixed and known up front, so cap the
        // materialized function pairs at the largest cache this run can touch
        // (prompt + decode + one sampled token, plus warmup slack). Without the
        // cap every exported size stays resident, which on a 16 GB machine
        // swaps hard enough to distort the timings we're here to measure.
        let warmupSlack = args.warmup ? 32 : 0
        let maxCacheNeeded = args.contextLength + args.decodeTokens + 1 + warmupSlack
        var maxContext = 64
        while maxContext < maxCacheNeeded { maxContext *= 2 }
        maxContext = min(maxContext, GemmaConfig.maxSeqLen)

        // --- Load (compile + load from .mlmodelc cache) ---
        let loadStart = now()
        let model: CoreMLModel
        do {
            model = try await CoreMLModel.load(
                from: modelURL,
                computeUnits: units,
                maxContextSize: maxContext,
                // No background preload: it would run multi-GB MLModel loads
                // underneath the measured window. We load what we need below,
                // deterministically, before any timer starts.
                backgroundPreload: false
            )
        } catch {
            FileHandle.standardError.write(Data("error: load failed: \(error)\n".utf8))
            exit(3)
        }

        // Bring up exactly the function pairs this run will touch, so nothing
        // loads lazily once we are measuring.
        var neededSizes = Set<Int>()
        if args.warmup {
            neededSizes.formUnion(
                requiredCacheSizes(model: model, promptLen: 16, newTokens: 2)
            )
        }
        neededSizes.formUnion(requiredCacheSizes(
            model: model, promptLen: args.contextLength, newTokens: args.decodeTokens + 1
        ))
        do {
            for size in neededSizes.sorted() {
                try await model.ensureLoaded(forGlobalCacheSize: size)
            }
        } catch {
            FileHandle.standardError.write(Data("error: preload failed: \(error)\n".utf8))
            exit(3)
        }
        // Counts the full "model is ready to run" cost, compile + preload.
        let loadTime = now() - loadStart

        let engine = InferenceEngine(model: model, temperature: 0.0, topP: 1.0)

        // --- Optional warmup: short prefill + 2 decode steps ---
        if args.warmup {
            let primer = syntheticPrompt(length: 16, seed: -1)
            var warmedUp = 0
            // respectStopTokens: false — the synthetic prompt makes the model
            // sample a stop token immediately, which would cut the run short.
            let stream = engine.generate(
                promptIDs: primer, maxNewTokens: 2, respectStopTokens: false
            )
            do {
                for try await _ in stream { warmedUp += 1 }
            } catch {
                FileHandle.standardError.write(Data("error: warmup failed: \(error)\n".utf8))
                exit(4)
            }
            _ = warmedUp
        }

        // --- Measured run ---
        let prompt = syntheticPrompt(length: args.contextLength, seed: args.runIndex + 1)
        let totalStart = now()
        var firstTokenAt: Double = 0
        var lastTokenAt: Double = 0
        var decodeTokensGenerated = 0
        let stream = engine.generate(
            promptIDs: prompt,
            maxNewTokens: args.decodeTokens + 1,       // +1 because engine yields the sample
                                                       // from prefill logits before any decode.
            respectStopTokens: false                   // keep decoding through stop tokens so
                                                       // every cell measures the same step count.
        )
        do {
            var first = true
            for try await _ in stream {
                if first {
                    firstTokenAt = now()
                    first = false
                } else {
                    lastTokenAt = now()
                    decodeTokensGenerated += 1
                }
                if decodeTokensGenerated >= args.decodeTokens { break }
            }
        } catch {
            FileHandle.standardError.write(Data("error: generate failed: \(error)\n".utf8))
            exit(5)
        }
        let totalEnd = now()

        // Guard against degenerate cases where the engine yielded nothing.
        guard firstTokenAt > 0 else {
            FileHandle.standardError.write(Data("error: no tokens produced\n".utf8))
            exit(6)
        }
        let prefillTime = firstTokenAt - totalStart
        let decodeTime = max(lastTokenAt - firstTokenAt, 0.0)

        let prefillTPS = Double(args.contextLength) / max(prefillTime, 1e-9)
        let decodeTPS = (decodeTokensGenerated > 0)
            ? Double(decodeTokensGenerated) / max(decodeTime, 1e-9)
            : 0.0

        emitJSON(RunJSON(
            run_index: args.runIndex,
            context_length: args.contextLength,
            compute_units: args.computeUnitsName,
            decode_tokens_requested: args.decodeTokens,
            prefill_time_s: prefillTime,
            decode_time_s: decodeTime,
            decode_tokens_generated: decodeTokensGenerated,
            prefill_tokens_per_sec: prefillTPS,
            decode_tokens_per_sec: decodeTPS,
            warmup: args.warmup,
            load_time_s: loadTime,
            total_time_s: totalEnd - totalStart
        ))
    }
}
