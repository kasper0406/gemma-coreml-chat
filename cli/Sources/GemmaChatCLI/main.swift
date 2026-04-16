/// Gemma4 CLI chat powered by CoreML via the GemmaCore library.
///
/// Usage:
///   swift run GemmaChatCLI --model ./gemma4-e2b.mlpackage
///   swift run GemmaChatCLI --model ./gemma4-e2b.mlpackage --compute-units all
///
/// Commands: /reset  /quit  /help

import CoreML
import Foundation
import GemmaCore

@main
struct GemmaChatCLI {
    static func main() async {
        let args = ProcessInfo.processInfo.arguments
        let modelPath = parseArg(args, flag: "--model") ?? "./gemma4-e2b.mlpackage"
        let computeUnits = parseComputeUnits(args)

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
            let promptIDs = tokenizer.encodeChatPrompt(history: history)

            // Generate
            print("\ngemma> ", terminator: "")
            fflush(stdout)

            var responseTokens: [Int32] = []
            let stream = engine.generate(promptIDs: promptIDs.map { Int32($0) }, maxNewTokens: 1024)

            for await tokenID in stream {
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

            let responseText = tokenizer.decode(responseTokens.map { Int($0) })
            print("\n")

            // Add assistant response to history
            history.append(ChatMessage(role: .assistant, content: responseText))
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
        
        The tokenizer is loaded from the .mlpackage (embedded during export).
        Re-export with `uv run gemma-export` if missing.
        
        """)
    }
}
