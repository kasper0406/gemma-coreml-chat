/// Gemma4 CLI chat powered by CoreML via the GemmaCore library.
///
/// Usage:
///   swift run GemmaChatCLI --model ./gemma4-e2b.mlpackage
///   swift run GemmaChatCLI --model ./gemma4-e2b.mlpackage --compute-units all
///   swift run GemmaChatCLI --model ./gemma4-e2b.mlpackage --verbose
///   swift run GemmaChatCLI --model ./gemma4-e2b.mlpackage --log-file /tmp/gemma.log
///
/// Commands: /reset  /quit  /help

import CoreML
import Foundation
import GemmaCore

/// Known CLI flags and which ones consume the next argument as a value.
private let knownFlags: Set<String> = ["--model", "--compute-units", "--verbose", "--log-file"]
private let flagsWithValue: Set<String> = ["--model", "--compute-units", "--log-file"]

@main
struct GemmaChatCLI {
    static func main() async {
        let args = ProcessInfo.processInfo.arguments
        let modelPath = parseArg(args, flag: "--model") ?? "./gemma4-e2b.mlpackage"
        let computeUnits = parseComputeUnits(args)

        // --- Configure logging ---
        configureLogging(args)
        warnUnknownFlags(args)

        printHeader()

        // --- Load model ---
        let modelURL = URL(fileURLWithPath: modelPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("Error: model not found at \(modelPath)")
            return
        }

        print("Loading model from \(modelPath) (compute: \(computeUnitsLabel(computeUnits)))...")
        let loadStart = CFAbsoluteTimeGetCurrent()

        let model: CoreMLModel
        do {
            model = try await CoreMLModel.load(from: modelURL, computeUnits: computeUnits)
        } catch {
            print("Error loading model: \(error)")
            return
        }

        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print("Model loaded in \(String(format: "%.1f", loadTime))s")

        // --- Load tokenizer (embedded in .mlpackage, or from HuggingFace) ---
        print("Loading tokenizer...")
        let tokenizer: GemmaTokenizer
        do {
            tokenizer = try await GemmaTokenizer(fromModelPackage: modelURL)
            print("Tokenizer loaded (embedded in model).\n")
        } catch {
            print("No embedded tokenizer found, downloading from HuggingFace...")
            do {
                tokenizer = try await GemmaTokenizer(pretrained: "google/gemma-4-E2B-it")
                print("Tokenizer downloaded.\n")
            } catch {
                print("Error loading tokenizer: \(error)")
                print("Hint: re-export the model with `uv run gemma-export` to embed the tokenizer.")
                return
            }
        }

        let engine = InferenceEngine(model: model, temperature: 1.0, topP: 0.9)
        let genContext = GenerationContext()
        var history: [ChatMessage] = []

        // --- Chat loop ---
        print("Type a message to chat. Commands: /help /reset /quit\n")

        while true {
            print("> ", terminator: "")
            fflush(stdout)

            guard let line = readLine(strippingNewline: true) else {
                break // EOF
            }

            let input = line.trimmingCharacters(in: .whitespaces)
            if input.isEmpty { continue }

            // Commands
            if input.hasPrefix("/") {
                switch input.lowercased() {
                case "/quit", "/exit", "/q":
                    print("Goodbye!")
                    return
                case "/reset", "/clear":
                    history.removeAll()
                    genContext.reset()
                    print("Conversation reset.\n")
                    continue
                case "/help", "/h", "/?":
                    printHelp()
                    continue
                default:
                    print("Unknown command: \(input). Type /help for commands.\n")
                    continue
                }
            }

            // Add user message
            history.append(ChatMessage(role: .user, content: input))

            // Encode conversation
            let promptIDs = tokenizer.encodeChatPrompt(history: history).map { Int32($0) }

            // Check if we can reuse KV cache from the previous turn
            let (existingKV, prefillOffset) = resolveKVReuse(
                promptIDs: promptIDs, context: genContext
            )

            // Generate
            print("\ngemma> ", terminator: "")
            fflush(stdout)

            var responseTokens: [Int32] = []
            let genStart = CFAbsoluteTimeGetCurrent()
            let stream = engine.generate(
                promptIDs: promptIDs,
                maxNewTokens: 1024,
                existingKVState: existingKV,
                prefillOffset: prefillOffset,
                context: genContext
            )

            do {
                for try await tokenID in stream {
                    if GemmaConfig.stopTokenIDs.contains(tokenID) { break }
                    responseTokens.append(tokenID)
                    let text = tokenizer.decode(responseTokens.map { Int($0) })
                    // Print incremental text by computing delta
                    let prevText = responseTokens.count > 1
                        ? tokenizer.decode(responseTokens.dropLast().map { Int($0) })
                        : ""
                    let delta = String(text.dropFirst(prevText.count))
                    if !delta.isEmpty {
                        print(delta, terminator: "")
                        fflush(stdout)
                    }
                }
            } catch {
                print("\n[error] \(error.localizedDescription)\n")
                continue
            }

            let genTime = CFAbsoluteTimeGetCurrent() - genStart
            let tokPerSec = genTime > 0 ? Double(responseTokens.count) / genTime : 0

            let responseText = tokenizer.decode(responseTokens.map { Int($0) })
            print("\n[\(responseTokens.count) tokens, \(String(format: "%.1f", tokPerSec)) tok/s]\n")

            // Add assistant response to history
            history.append(ChatMessage(role: .assistant, content: responseText))
        }
    }

    // MARK: - KV Reuse

    /// Check if the previous KV cache can be reused for the new prompt.
    ///
    /// If `context.cachedTokens` is a prefix of `promptIDs`, returns the
    /// existing KV state and the offset to resume prefilling from.
    /// Otherwise returns `(nil, 0)` for a full prefill from scratch.
    static func resolveKVReuse(
        promptIDs: [Int32], context: GenerationContext
    ) -> (KVCacheState?, Int) {
        guard let kv = context.kvState else { return (nil, 0) }
        let cached = context.cachedTokens
        guard !cached.isEmpty, cached.count <= promptIDs.count else {
            Log.info("[KV] No reusable prefix (cached=\(cached.count), prompt=\(promptIDs.count))")
            return (nil, 0)
        }

        // Find the longest common prefix. BPE round-trips (decode → re-encode
        // via chat template) can change tokens at turn boundaries, so we
        // fall back to a partial match rather than throwing out the cache.
        var matchLen = 0
        for i in 0..<cached.count {
            if cached[i] != promptIDs[i] { break }
            matchLen = i + 1
        }
        guard matchLen > 0 else {
            Log.info("[KV] Prefix mismatch at position 0 — full prefill (cached[0]=\(cached[0]) prompt[0]=\(promptIDs[0]))")
            return (nil, 0)
        }
        if matchLen < cached.count {
            Log.info("[KV] Partial prefix match: reusing \(matchLen)/\(promptIDs.count) tokens (cache had \(cached.count); first mismatch at \(matchLen): cached=\(cached[matchLen]) prompt=\(promptIDs[matchLen]))")
        } else {
            Log.info("[KV] Reusing \(matchLen)/\(promptIDs.count) tokens from cache")
        }
        return (kv, matchLen)
    }

    // MARK: - Logging

    /// Configure `Log.destination` from CLI flags.
    /// Default: suppressed. `--verbose`: stderr. `--log-file <path>`: file.
    static func configureLogging(_ args: [String]) {
        if let logPath = parseArg(args, flag: "--log-file") {
            let url = URL(fileURLWithPath: logPath)
            // Create or truncate the log file
            FileManager.default.createFile(atPath: url.path, contents: nil)
            if let handle = FileHandle(forWritingAtPath: url.path) {
                handle.seekToEndOfFile()
                Log.destination = .file(handle)
                return
            } else {
                print("Warning: cannot open log file '\(logPath)', logging disabled")
            }
        }

        if args.contains("--verbose") {
            Log.destination = .stderr
        } else {
            Log.destination = .none
        }
    }

    /// Warn about unrecognized `--flags` to catch typos.
    static func warnUnknownFlags(_ args: [String]) {
        var i = 1  // skip argv[0]
        while i < args.count {
            let arg = args[i]
            if arg.hasPrefix("--") {
                if knownFlags.contains(arg) {
                    if flagsWithValue.contains(arg) { i += 1 }  // skip value
                } else {
                    print("Warning: unknown flag '\(arg)'")
                }
            }
            i += 1
        }
    }

    // MARK: - Helpers

    static func parseArg(_ args: [String], flag: String) -> String? {
        guard let idx = args.firstIndex(of: flag), idx + 1 < args.count else { return nil }
        return args[idx + 1]
    }

    static func parseComputeUnits(_ args: [String]) -> MLComputeUnits {
        guard let value = parseArg(args, flag: "--compute-units") else {
            return .cpuAndGPU
        }
        switch value.lowercased() {
        case "all": return .all
        case "cpu-only", "cpu": return .cpuOnly
        case "cpu-and-gpu", "cpu-gpu": return .cpuAndGPU
        case "cpu-and-ne", "cpu-ane": return .cpuAndNeuralEngine
        default:
            print("Warning: unknown compute units '\(value)', using cpu-and-gpu")
            return .cpuAndGPU
        }
    }

    static func computeUnitsLabel(_ units: MLComputeUnits) -> String {
        switch units {
        case .all: "all"
        case .cpuOnly: "cpu-only"
        case .cpuAndGPU: "cpu-and-gpu"
        case .cpuAndNeuralEngine: "cpu-and-ne"
        @unknown default: "unknown"
        }
    }

    static func printHeader() {
        print("""
        ╔══════════════════════════════════╗
        ║   Gemma4 Chat (CoreML + Swift)   ║
        ╚══════════════════════════════════╝
        """)
    }

    static func printHelp() {
        print("""
        Commands:
          /reset  — Clear conversation history
          /quit   — Exit
          /help   — Show this message

        CLI flags:
          --model <path>           Path to .mlpackage or .mlmodelc
          --compute-units <units>  all | cpu-and-gpu (default) | cpu-only | cpu-and-ne
          --verbose                Show diagnostic logs on stderr
          --log-file <path>        Write diagnostic logs to a file
        
        The tokenizer is loaded from the .mlpackage (embedded during export).
        Re-export with `uv run gemma-export` if missing.
        
        """)
    }
}
