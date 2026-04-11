/// Chat view model: orchestrates tokenization, eager prefill, and generation.
///
/// Bridges the UI (SwiftUI) with the inference engine and eager prefill manager.

import Combine
import CoreML
import Foundation
import Observation
import UIKit

/// App-wide loading state.
enum AppState: Equatable {
    case loadingModel
    case ready
    case error(String)
}

@Observable
@MainActor
final class ChatViewModel {
    // MARK: - Published State

    var messages: [ChatMessage] = []
    var inputText: String = "" {
        didSet { inputTextDidChange() }
    }
    var isGenerating: Bool = false
    var streamingText: String = ""
    var appState: AppState = .loadingModel
    var prefillStatus: PrefillStatus = .idle
    var contextTokenCount: Int = 0
    var generatedTokenCount: Int = 0

    // MARK: - Settings

    var temperature: Float = 1.0
    var topP: Float = 0.9
    var maxNewTokens: Int = 256
    var systemPrompt: String? = "You are a helpful assistant running on an iPhone. Keep your answers concise and to the point — the user is reading on a small screen in a chat interface."

    // MARK: - Private

    private var model: CoreMLModel?
    private var tokenizer: GemmaTokenizer?
    private var engine: InferenceEngine?
    private var eagerPrefill: EagerPrefillManager?
    private var generateTask: Task<Void, Never>?
    private var prefillDebounceTask: Task<Void, Never>?

    private func handleMemoryWarning() {
        print("[Memory] Received memory warning")
        if isGenerating {
            cancelGeneration()
            messages.append(ChatMessage(
                role: .system,
                content: "⚠️ Generation stopped due to low memory."
            ))
        }
        Task { await eagerPrefill?.reset() }
    }

    // MARK: - Model Loading

    func loadModel() async {
        appState = .loadingModel

        do {
            let tok = try await GemmaTokenizer()
            tokenizer = tok

            #if targetEnvironment(simulator)
            let units: MLComputeUnits = .cpuOnly
            #else
            let units: MLComputeUnits = .all
            #endif

            let coreml = try await CoreMLModel.load(computeUnits: units)
            model = coreml

            let eng = InferenceEngine(model: coreml, temperature: temperature, topP: topP)
            engine = eng

            eagerPrefill = EagerPrefillManager(
                engine: eng,
                tokenizer: tok,
                kvInputNames: coreml.prefillKVInputNames,
                inputDescriptions: coreml.prefillModel.modelDescription.inputDescriptionsByName
            )

            // Listen for memory warnings (view model lives for app lifetime)
            NotificationCenter.default.addObserver(
                forName: UIApplication.didReceiveMemoryWarningNotification,
                object: nil,
                queue: .main
            ) { [weak self] _ in
                guard let self else { return }
                Task { @MainActor in
                    self.handleMemoryWarning()
                }
            }

            appState = .ready
        } catch {
            appState = .error(error.localizedDescription)
        }
    }

    // MARK: - Chat

    func sendMessage() {
        let text = inputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !isGenerating else { return }

        // Handle commands
        if text.hasPrefix("/") {
            handleCommand(text)
            return
        }

        inputText = ""
        messages.append(ChatMessage(role: .user, content: text))
        startGeneration(userText: text)
    }

    func cancelGeneration() {
        generateTask?.cancel()
        generateTask = nil
        isGenerating = false
    }

    func resetConversation() {
        messages.removeAll()
        streamingText = ""
        contextTokenCount = 0
        generatedTokenCount = 0
        isGenerating = false
        generateTask?.cancel()
        generateTask = nil
        Task { await eagerPrefill?.reset() }
    }

    // MARK: - Private: Generation

    private func startGeneration(userText: String) {
        guard let engine, let eagerPrefill, let tokenizer else { return }

        isGenerating = true
        streamingText = ""
        generatedTokenCount = 0

        generateTask = Task { [weak self] in
            guard let self else { return }

            do {
                // Get prefill state (finishes any pending eager prefill)
                let history = messages.filter { $0.role != .system }
                let (promptIDs, kvState, prefillOffset) = try await eagerPrefill.finishPrefill(
                    finalText: userText,
                    history: Array(history.dropLast()),  // history without the just-added user msg
                    systemPrompt: systemPrompt
                )

                contextTokenCount = promptIDs.count

                // Run generation
                let stream = engine.generate(
                    promptIDs: promptIDs,
                    maxNewTokens: maxNewTokens,
                    existingKVState: prefillOffset > 0 ? kvState : nil,
                    prefillOffset: prefillOffset
                )

                var genIDs: [Int32] = []
                for await tokenID in stream {
                    if Task.isCancelled { break }

                    if GemmaConfig.stopTokenIDs.contains(tokenID) { break }
                    genIDs.append(tokenID)

                    let decoded = tokenizer.decode(genIDs.map { Int($0) })
                    streamingText = decoded
                    generatedTokenCount = genIDs.count
                }

                // Finalize
                let reply = tokenizer.decode(genIDs.map { Int($0) })
                messages.append(ChatMessage(role: .assistant, content: reply))
                streamingText = ""

                // Reset eager prefill for next turn
                await eagerPrefill.reset()

            } catch {
                messages.append(ChatMessage(
                    role: .system,
                    content: "Error: \(error.localizedDescription)"
                ))
            }

            isGenerating = false
        }
    }

    // MARK: - Private: Eager Prefill

    private func inputTextDidChange() {
        // Debounce: wait 150ms after last keystroke before tokenizing + prefilling
        prefillDebounceTask?.cancel()
        guard !isGenerating, appState == .ready else { return }

        prefillDebounceTask = Task { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .milliseconds(150))
            if Task.isCancelled { return }

            guard let eagerPrefill else { return }
            let text = inputText
            let history = messages.filter { $0.role != .system }

            await eagerPrefill.textChanged(
                currentText: text,
                history: history,
                systemPrompt: systemPrompt
            )

            let status = await eagerPrefill.status
            prefillStatus = status
        }
    }

    // MARK: - Private: Commands

    private func handleCommand(_ text: String) {
        let cmd = text.lowercased()
        switch cmd {
        case "/reset":
            resetConversation()
            messages.append(ChatMessage(role: .system, content: "History cleared."))
        case "/quit", "/exit":
            // No-op in iOS (user closes app)
            break
        case "/help":
            messages.append(ChatMessage(role: .system, content: """
                Commands:
                  /reset — clear conversation history
                  /help  — show this message
                """))
        default:
            messages.append(ChatMessage(
                role: .system,
                content: "Unknown command: \(text). Try /help"
            ))
        }
        inputText = ""
    }
}
