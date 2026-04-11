/// Main chat interface with message list, streaming display, and input bar.

import SwiftUI

struct ChatView: View {
    @Environment(ChatViewModel.self) private var viewModel

    var body: some View {
        switch viewModel.appState {
        case .loadingModel:
            ProgressView("Loading model…")
                .font(.headline)
                .task { await viewModel.loadModel() }
        case .error(let msg):
            VStack(spacing: 16) {
                Image(systemName: "exclamationmark.triangle")
                    .font(.largeTitle)
                    .foregroundStyle(.red)
                Text(msg)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                Button("Retry") {
                    Task { await viewModel.loadModel() }
                }
            }
        case .ready:
            chatContent
        }
    }

    private var chatContent: some View {
        @Bindable var vm = viewModel
        return VStack(spacing: 0) {
            // Stats bar
            statsBar
                .padding(.horizontal)
                .padding(.vertical, 4)

            Divider()

            // Messages
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(viewModel.messages) { message in
                            MessageBubble(message: message)
                                .id(message.id)
                        }

                        // Streaming response
                        if !viewModel.streamingText.isEmpty {
                            MessageBubble(message: ChatMessage(
                                role: .assistant,
                                content: viewModel.streamingText
                            ))
                            .id("streaming")
                        }
                    }
                    .padding()
                }
                .onChange(of: viewModel.messages.count) {
                    if let last = viewModel.messages.last {
                        withAnimation {
                            proxy.scrollTo(last.id, anchor: .bottom)
                        }
                    }
                }
                .onChange(of: viewModel.streamingText) {
                    withAnimation {
                        proxy.scrollTo("streaming", anchor: .bottom)
                    }
                }
            }

            Divider()

            // Prefill indicator
            if case .prefilling(let completed, let total) = viewModel.prefillStatus {
                HStack(spacing: 6) {
                    ProgressView()
                        .scaleEffect(0.7)
                    Text("Prefilling \(completed)/\(total) chunks…")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal)
                .padding(.vertical, 2)
            } else if case .ready(let chunks) = viewModel.prefillStatus, chunks > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "bolt.fill")
                        .font(.caption2)
                    Text("\(chunks) chunks prefilled")
                        .font(.caption)
                }
                .foregroundStyle(.green)
                .padding(.horizontal)
                .padding(.vertical, 2)
            }

            // Input bar
            inputBar
        }
        .navigationTitle("Gemma Chat")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var statsBar: some View {
        HStack {
            Text("ctx \(viewModel.contextTokenCount)/\(GemmaConfig.maxSeqLen)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Spacer()
            Text("gen \(viewModel.generatedTokenCount)/\(viewModel.maxNewTokens)")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var inputBar: some View {
        @Bindable var vm = viewModel
        return HStack(spacing: 8) {
            TextField("Message…", text: $vm.inputText, axis: .vertical)
                .textFieldStyle(.roundedBorder)
                .lineLimit(1...5)
                .disabled(viewModel.isGenerating)
                .onSubmit {
                    viewModel.sendMessage()
                }

            if viewModel.isGenerating {
                Button {
                    viewModel.cancelGeneration()
                } label: {
                    Image(systemName: "stop.circle.fill")
                        .font(.title2)
                        .foregroundStyle(.red)
                }
            } else {
                Button {
                    viewModel.sendMessage()
                } label: {
                    Image(systemName: "arrow.up.circle.fill")
                        .font(.title2)
                        .foregroundStyle(
                            viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty
                                ? .gray : .blue
                        )
                }
                .disabled(viewModel.inputText.trimmingCharacters(in: .whitespaces).isEmpty)
            }
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
}

#Preview {
    NavigationStack {
        ChatView()
            .environment(ChatViewModel())
    }
}
