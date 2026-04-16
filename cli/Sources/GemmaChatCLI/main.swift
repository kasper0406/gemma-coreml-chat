/// Gemma4 CLI chat powered by CoreML via the GemmaCore library.
///
/// Usage:
///   swift run GemmaChatCLI --model ./gemma4-e2b.mlpackage
///
/// Commands: /reset  /quit  /help

import Foundation
import GemmaCore

@main
struct GemmaChatCLI {
    static func main() async {
        let args = ProcessInfo.processInfo.arguments
        let modelPath = parseArg(args, flag: "--model") ?? "./gemma4-e2b.mlpackage"
        let tokenizerID = parseArg(args, flag: "--tokenizer") ?? "google/gemma-4-E2B-it"

        printHeader()

        // --- Load model ---
        let modelURL = URL(fileURLWithPath: modelPath)
        guard FileManager.default.fileExists(atPath: modelURL.path) else {
            print("Error: model not found at \(modelPath)")
            return
        }

        print("Loading model from \(modelPath)...")
        let loadStart = CFAbsoluteTimeGetCurrent()

        let model: CoreMLModel
        do {
            model = try await CoreMLModel.load(from: modelURL)
        } catch {
            print("Error loading model: \(error)")
            return
        }

        let loadTime = CFAbsoluteTimeGetCurrent() - loadStart
        print("Model loaded in \(String(format: "%.1f", loadTime))s")

        // --- Load tokenizer ---
        print("Loading tokenizer (\(tokenizerID))...")
        let tokenizer: GemmaTokenizer
        do {
            tokenizer = try await GemmaTokenizer(pretrained: tokenizerID)
        } catch {
            print("Error loading tokenizer: \(error)")
            return
        }
        print("Tokenizer ready.\n")

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
        
        """)
    }
}
