/// Configurable logging utility for diagnostics.
///
/// All library output uses this instead of `print()` so CLI apps
/// can cleanly separate model output (stdout) from diagnostics.
/// Set ``destination`` before loading the model to control output.

import Foundation

public enum Log {
    /// Where diagnostic messages are written.
    public enum Destination: Sendable {
        /// Write to stderr (default).
        case stderr
        /// Write to an open file handle (caller manages lifetime).
        case file(FileHandle)
        /// Suppress all diagnostic output.
        case none
    }

    /// Controls where ``info(_:)`` writes. Default is `.stderr`.
    public nonisolated(unsafe) static var destination: Destination = .stderr

    /// Write a diagnostic message to the configured destination.
    public static func info(_ message: String) {
        switch destination {
        case .stderr:
            let data = Data((message + "\n").utf8)
            FileHandle.standardError.write(data)
        case .file(let handle):
            let data = Data((message + "\n").utf8)
            handle.write(data)
        case .none:
            break
        }
    }
}
