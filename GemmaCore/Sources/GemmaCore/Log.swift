/// Logging utility that writes to stderr to avoid corrupting stdout.
///
/// All library output uses this instead of `print()` so CLI apps
/// can cleanly separate model output (stdout) from diagnostics (stderr).

import Foundation

public enum Log {
    /// Write a diagnostic message to stderr.
    public static func info(_ message: String) {
        let data = Data((message + "\n").utf8)
        FileHandle.standardError.write(data)
    }
}
