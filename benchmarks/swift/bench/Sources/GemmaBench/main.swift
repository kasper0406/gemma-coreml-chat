/// One measured run of the Gemma model for `benchmarks/runner.py`.
///
/// Links `GemmaCore` and drives ``CoreMLModel`` directly — the same prefill,
/// decode and head calls the CLI and iOS app make — with greedy sampling.
/// Loading, compilation and warm-up happen first and are not measured. Then,
/// in this order, it runs and timestamps
///
///   idle_pre   `idleSeconds` of sleep, the model loaded
///   prefill    `prefillTokens` prompt tokens: prompts of `contextLength`
///              tokens (a whole cache), each into a fresh cache, back to back
///   decode     `decodeTokens` greedy tokens, after a `decodePrompt`-token
///              prompt, into a cache of `contextLength` rows
///   idle_post  `idleSeconds` of sleep
///
/// with a `settleSeconds` pause after prefill and after decode, so neither
/// phase's power tail lands in the next window. Every boundary is a
/// `CLOCK_UPTIME_RAW` timestamp in nanoseconds; the runner maps them onto its
/// powermetrics samples. Emits one JSON object on stdout.
///
/// Usage:
///   GemmaBench --model <path> --compute-units cpu-and-ne --context-length 512

import CoreML
import Foundation
import GemmaCore

// Fixed workload, identical for every configuration.
let prefillTokens = 4096        // per window: 8 × 512 or 2 × 2048 tokens
let decodePrompt = 128
let decodeTokens = 256
let idleSeconds = 4.0
let settleSeconds = 1.0

struct Args {
    var modelPath = ""
    var computeUnits = ""
    var contextLength = 0
}

func fail(_ message: String, code: Int32) -> Never {
    FileHandle.standardError.write(Data("error: \(message)\n".utf8))
    exit(code)
}

func parseArgs() -> Args {
    let av = CommandLine.arguments
    var a = Args()
    var i = 1
    while i + 1 < av.count {
        switch av[i] {
        case "--model": a.modelPath = av[i + 1]
        case "--compute-units": a.computeUnits = av[i + 1]
        case "--context-length": a.contextLength = Int(av[i + 1]) ?? 0
        default: fail("unknown flag \(av[i])", code: 2)
        }
        i += 2
    }
    guard i == av.count, !a.modelPath.isEmpty, a.contextLength > 0 else {
        fail("usage: GemmaBench --model <path> --compute-units <cpu-and-gpu|cpu-and-ne|all|cpu-only> --context-length <N>", code: 2)
    }
    return a
}

func computeUnits(_ s: String) -> MLComputeUnits {
    switch s {
    case "cpu-only": return .cpuOnly
    case "cpu-and-gpu": return .cpuAndGPU
    case "cpu-and-ne": return .cpuAndNeuralEngine
    case "all": return .all
    default: fail("unknown compute units '\(s)'", code: 2)
    }
}

func uptimeNs() -> UInt64 { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }

/// A reproducible synthetic prompt; ids stay clear of 0 (padding) and the
/// special tokens.
func syntheticPrompt(length: Int, seed: UInt64) -> [Int32] {
    var rng = 0x9E37_79B9_7F4A_7C15 &+ seed
    return (0..<length).map { _ in
        rng = rng &* 6364136223846793005 &+ 1442695040888963407
        return Int32(1024 + Int(rng >> 48) % 100_000)
    }
}

/// Prefill `prompt` into `kv` chunk by chunk; returns the last real token's logits.
func prefill(_ model: CoreMLModel, _ prompt: [Int32], _ kv: KVCacheState) throws -> MLMultiArray {
    let chunk = model.chunkSize
    var logits: MLMultiArray?
    for start in stride(from: 0, to: prompt.count, by: chunk) {
        let real = min(chunk, prompt.count - start)
        let tokens = Array(prompt[start..<start + real]) + [Int32](repeating: 0, count: chunk - real)
        logits = try autoreleasepool {
            try model.prefill(tokens: tokens, startPosition: Int32(start), logitsRow: real - 1, kvState: kv)
        }
    }
    return logits!
}

struct Window: Encodable {
    let start_ns: UInt64
    let end_ns: UInt64
}

struct Phase: Encodable {
    let tokens: Int
    let start_ns: UInt64
    let end_ns: UInt64
}

struct RunJSON: Encodable {
    let clock = "CLOCK_UPTIME_RAW"
    let compute_units: String
    let context_length: Int
    let load_time_s: Double
    let idle_pre: Window
    let prefill: Phase
    let prefill_prompts: Int
    let decode: Phase
    let decode_prompt: Int
    let idle_post: Window
}

@main
struct GemmaBenchMain {
    static func main() async {
        let args = parseArgs()
        let units = computeUnits(args.computeUnits)
        let n = args.contextLength
        guard n >= decodePrompt + decodeTokens + 1 else {
            fail("--context-length must hold \(decodePrompt) + \(decodeTokens) + 1 tokens", code: 2)
        }

        let loadStart = uptimeNs()
        let model: CoreMLModel
        do {
            model = try await CoreMLModel.load(
                from: URL(fileURLWithPath: args.modelPath),
                computeUnits: units,
                maxContextSize: n,
                // Nothing may load underneath a measured window; this run's
                // one size is loaded (and specialized) right below.
                backgroundPreload: false
            )
            guard model.materializedSizes.contains(n) else {
                fail("the model has no size \(n) (has \(model.materializedSizes))", code: 3)
            }
            try await model.ensureLoaded(forGlobalCacheSize: n)
        } catch {
            fail("load failed: \(error.localizedDescription)", code: 3)
        }
        let loadTime = Double(uptimeNs() - loadStart) / 1e9

        let prompts = max(1, prefillTokens / n)
        do {
            // Warm-up, unmeasured: one prompt chunk and a few decode steps.
            let scratch = try model.makeEmptyKVState(size: n)
            var logits = try prefill(model, syntheticPrompt(length: model.chunkSize, seed: 0), scratch)
            for p in 0..<4 {
                let t = Sampling.sampleNextToken(logits: logits, temperature: 0)
                logits = try model.decode(token: t, position: Int32(model.chunkSize + p), kvState: scratch)
            }

            // Every cache the windows use exists before the first timestamp.
            let prefillCaches = try (0..<prompts).map { _ in try model.makeEmptyKVState(size: n) }
            let decodeCache = try model.makeEmptyKVState(size: n)
            let prefillPrompts = (0..<prompts).map { syntheticPrompt(length: n, seed: UInt64($0 + 1)) }
            let decodePromptIDs = syntheticPrompt(length: decodePrompt, seed: 1000)

            let idle0 = uptimeNs()
            try await Task.sleep(for: .seconds(idleSeconds))
            let idle1 = uptimeNs()

            let p0 = uptimeNs()
            for (prompt, kv) in zip(prefillPrompts, prefillCaches) {
                logits = try prefill(model, prompt, kv)
                _ = Sampling.sampleNextToken(logits: logits, temperature: 0)
            }
            let p1 = uptimeNs()
            try await Task.sleep(for: .seconds(settleSeconds))

            logits = try prefill(model, decodePromptIDs, decodeCache)
            var token = Sampling.sampleNextToken(logits: logits, temperature: 0)
            let d0 = uptimeNs()
            for step in 0..<decodeTokens {
                logits = try autoreleasepool {
                    try model.decode(token: token, position: Int32(decodePrompt + step), kvState: decodeCache)
                }
                token = Sampling.sampleNextToken(logits: logits, temperature: 0)
            }
            let d1 = uptimeNs()
            try await Task.sleep(for: .seconds(settleSeconds))

            let idle2 = uptimeNs()
            try await Task.sleep(for: .seconds(idleSeconds))
            let idle3 = uptimeNs()

            let out = RunJSON(
                compute_units: args.computeUnits,
                context_length: n,
                load_time_s: loadTime,
                idle_pre: Window(start_ns: idle0, end_ns: idle1),
                prefill: Phase(tokens: prompts * n, start_ns: p0, end_ns: p1),
                prefill_prompts: prompts,
                decode: Phase(tokens: decodeTokens, start_ns: d0, end_ns: d1),
                decode_prompt: decodePrompt,
                idle_post: Window(start_ns: idle2, end_ns: idle3)
            )
            let enc = JSONEncoder()
            enc.outputFormatting = [.sortedKeys]
            FileHandle.standardOutput.write(try enc.encode(out))
            FileHandle.standardOutput.write(Data("\n".utf8))
        } catch {
            fail("run failed: \(error)", code: 5)
        }
    }
}
