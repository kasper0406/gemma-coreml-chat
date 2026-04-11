import SwiftUI

@main
struct GemmaChatApp: App {
    @State private var viewModel = ChatViewModel()

    var body: some Scene {
        WindowGroup {
            ChatView()
                .environment(viewModel)
        }
    }
}
