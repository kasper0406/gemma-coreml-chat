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
    ///
    /// Uses the throwing `write(contentsOf:)` API instead of legacy
    /// `write(_:)`. The legacy variant throws `NSFileHandleOperationException`
    /// (Obj-C, uncatchable from Swift) when the destination fd hits an I/O
    /// error — observed on iOS when stderr breaks under memory pressure mid-
    /// load, taking the whole app down inside an otherwise-harmless log call.
    /// We silently drop write failures: diagnostics must never crash the app.
    public static func info(_ message: String) {
        let handle: FileHandle
        switch destination {
        case .stderr: handle = FileHandle.standardError
        case .file(let h): handle = h
        case .none: return
        }
        let data = Data((message + "\n").utf8)
        try? handle.write(contentsOf: data)
    }
}
