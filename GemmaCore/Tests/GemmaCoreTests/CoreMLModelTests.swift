/// `CoreMLModel` against `TinyModel.mlpackage`: a package with the exported
/// function set and signatures but no model (see `tests/test_runtime_fixture.py`,
/// which writes it). Its chunks mark the global-cache rows their write
/// selection picks and fold the count and a digest of every host input into
/// the hidden state, so a step on the wrong state or a torn input changes the
/// logits.

import CoreML
import Foundation
import XCTest

@testable import GemmaCore

final class CoreMLModelTests: XCTestCase {
    private static func load() async throws -> CoreMLModel {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "TinyModel", withExtension: "mlpackage"))
        let model = try await CoreMLModel.load(
            from: url, computeUnits: .cpuOnly, backgroundPreload: false
        )
        for size in model.materializedSizes {
            try await model.ensureLoaded(forGlobalCacheSize: size)
        }
        return model
    }

    private static func ring(_ kv: KVCacheState) -> [Int32] { kv.ring }

    private static func floats(_ logits: MLMultiArray) -> [Float] {
        logits.withUnsafeBufferPointer(ofType: Float16.self) { $0.map(Float.init) }
    }

    /// A conversation: one prefill chunk, then `steps` decode steps; every
    /// step's logits, copied out of the rotating backings.
    private static func conversation(
        _ model: CoreMLModel, size: Int, seed: Int32, steps: Int
    ) throws -> [[Float]] {
        let kv = try model.makeEmptyKVState(size: size)
        let chunk = (0..<Int32(model.chunkSize)).map { ($0 + seed) % 8 }
        var out = [floats(try model.prefill(
            tokens: chunk, startPosition: 0, logitsRow: chunk.count - 1, kvState: kv
        ))]
        for i in 0..<steps {
            let position = Int32(model.chunkSize + i)
            out.append(floats(try model.decode(
                token: (position * 3 + seed) % 8, position: position, kvState: kv
            )))
        }
        return out
    }

    /// A negative position used to pass the upper-bound check and index
    /// `ring[-1]` (Swift's `%` keeps the sign). It must be rejected before
    /// the ring — or anything else — is touched.
    func testNegativePositionsAreRejectedBeforeTheRingIsTouched() async throws {
        let model = try await Self.load()
        let kv = try model.makeEmptyKVState()
        let empty = Self.ring(kv)
        XCTAssertTrue(empty.allSatisfy { $0 == -1 })

        XCTAssertThrowsError(try model.decode(token: 1, position: -1, kvState: kv)) {
            guard case CoreMLModelError.positionOutOfRange(let position, _) = $0 else {
                return XCTFail("unexpected error \($0)")
            }
            XCTAssertEqual(position, -1)
        }
        XCTAssertThrowsError(try model.prefill(
            tokens: [Int32](repeating: 1, count: model.chunkSize),
            startPosition: -2, logitsRow: 0, kvState: kv
        ))
        XCTAssertEqual(Self.ring(kv), empty)

        // The same cache still works from position 0.
        _ = try model.decode(token: 1, position: 0, kvState: kv)
        XCTAssertEqual(Self.ring(kv).filter { $0 >= 0 }, [0])
    }

    /// Every conversation shares the loaded functions — the chunks of its
    /// size, and `head` across sizes — and Core ML's synchronous prediction
    /// is not safe to call concurrently on one `MLModel`. Conversations at
    /// both sizes running at once must produce exactly what they produce one
    /// at a time.
    func testConcurrentConversationsMatchSerialOnes() async throws {
        let model = try await Self.load()
        let sizes = model.materializedSizes
        XCTAssertEqual(sizes.count, 2)
        let runs = 8, steps = 40
        let expected = try (0..<runs).map {
            try Self.conversation(model, size: sizes[$0 % 2], seed: Int32($0), steps: steps)
        }

        let results = Results(count: runs)
        DispatchQueue.concurrentPerform(iterations: runs) { i in
            results.set(i, Result {
                try Self.conversation(model, size: sizes[i % 2], seed: Int32(i), steps: steps)
            })
        }
        for i in 0..<runs {
            XCTAssertEqual(try results.get(i).get(), expected[i], "conversation \(i)")
        }
    }
}

/// Decode-only mode (the iOS path below 7 GiB) prefills through `decode`,
/// one token at a time, and hands the engine the last token's logits.
final class DecodeOnlyPrefillTests: XCTestCase {
    private static func load(decodeOnly: Bool) async throws -> CoreMLModel {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "TinyModel", withExtension: "mlpackage"))
        let model = try await CoreMLModel.load(
            from: url, computeUnits: .cpuOnly, decodeOnly: decodeOnly, backgroundPreload: false
        )
        for size in model.materializedSizes {
            try await model.ensureLoaded(forGlobalCacheSize: size)
        }
        return model
    }

    private static func floats(_ logits: MLMultiArray) -> [Float] {
        logits.withUnsafeBufferPointer(ofType: Float16.self) { $0.map(Float.init) }
    }

    /// The prefill logits are all of head's `(slices, rows)`, equal to those
    /// of the same decode steps run directly — the fixture's head is two
    /// slices, so a copy of the first row alone loses half the vocabulary —
    /// and the engine's first token is their argmax.
    func testDecodeOnlyPrefillReturnsTheWholeVocabulary() async throws {
        let decodeOnly = try await Self.load(decodeOnly: true)
        let full = try await Self.load(decodeOnly: false)
        XCTAssertTrue(decodeOnly.isDecodeOnly)
        XCTAssertEqual(decodeOnly.chunkSize, 1)
        let prompt: [Int32] = [3, 1, 4, 1, 5, 7, 2]

        let engine = InferenceEngine(model: decodeOnly, temperature: 0)
        let (logits, kv) = try await engine.fullPrefill(ids: prompt)

        let reference = try full.makeEmptyKVState(size: kv.size)
        var want: MLMultiArray?
        for (i, token) in prompt.enumerated() {
            want = try full.decode(token: token, position: Int32(i), kvState: reference)
        }
        let expected = try XCTUnwrap(want)
        XCTAssertEqual(logits.shape, expected.shape)
        XCTAssertEqual(logits.count, 8)
        XCTAssertEqual(Self.floats(logits), Self.floats(expected))

        var generated: [Int32] = []
        for try await id in engine.generate(promptIDs: prompt, maxNewTokens: 3, respectStopTokens: false) {
            generated.append(id)
        }
        XCTAssertEqual(generated.count, 3)
        XCTAssertEqual(generated.first, Sampling.sampleNextToken(logits: expected, temperature: 0))
    }
}

final class SerialFunctionTests: XCTestCase {
    /// Two threads predicting through one `SerialFunction` never overlap:
    /// the input provider below sees how many predictions are reading it at
    /// once, and lingers long enough for an unserialized second one to join.
    func testPredictionsThroughOneFunctionNeverOverlap() async throws {
        let package = try XCTUnwrap(Bundle.module.url(forResource: "TinyModel", withExtension: "mlpackage"))
        let compiled = try await MLModel.compileModel(at: package)
        defer { try? FileManager.default.removeItem(at: compiled) }
        let config = MLModelConfiguration()
        config.computeUnits = .cpuOnly
        config.functionName = CoreMLModel.headFunctionName
        let head = SerialFunction(try await MLModel.load(contentsOf: compiled, configuration: config))

        let gauge = Gauge()
        let failures = Results(count: 4)
        DispatchQueue.concurrentPerform(iterations: 4) { i in
            failures.set(i, Result {
                let input = try GaugedInput(gauge: gauge)
                _ = try head.prediction(from: input, options: MLPredictionOptions())
                return []
            })
        }
        for i in 0..<4 { XCTAssertNoThrow(try failures.get(i).get()) }
        XCTAssertEqual(gauge.peak, 1)
    }
}

/// Counts the predictions inside a `GaugedInput` at once.
private final class Gauge: @unchecked Sendable {
    private let lock = NSLock()
    private var active = 0
    private(set) var peak = 0

    func enter() {
        lock.lock(); defer { lock.unlock() }
        active += 1
        peak = max(peak, active)
    }

    func leave() {
        lock.lock(); defer { lock.unlock() }
        active -= 1
    }
}

private final class GaugedInput: MLFeatureProvider {
    let featureNames: Set<String> = [CoreMLModel.Feature.hidden]
    private let hidden: MLFeatureValue
    private let gauge: Gauge

    init(gauge: Gauge) throws {
        hidden = MLFeatureValue(multiArray: try MLMultiArray(shape: [1, 1, 64], dataType: .float16))
        self.gauge = gauge
    }

    func featureValue(for featureName: String) -> MLFeatureValue? {
        gauge.enter()
        defer { gauge.leave() }
        Thread.sleep(forTimeInterval: 0.05)
        return featureName == CoreMLModel.Feature.hidden ? hidden : nil
    }
}

private final class Results: @unchecked Sendable {
    private var values: [Result<[[Float]], Error>?]
    private let lock = NSLock()

    init(count: Int) { values = Array(repeating: nil, count: count) }

    func set(_ i: Int, _ value: Result<[[Float]], Error>) {
        lock.lock(); defer { lock.unlock() }
        values[i] = value
    }

    func get(_ i: Int) -> Result<[[Float]], Error> {
        lock.lock(); defer { lock.unlock() }
        return values[i]!
    }
}

final class PromptLimitTests: XCTestCase {
    private static func engine() async throws -> InferenceEngine {
        let url = try XCTUnwrap(Bundle.module.url(forResource: "TinyModel", withExtension: "mlpackage"))
        let model = try await CoreMLModel.load(from: url, computeUnits: .cpuOnly, backgroundPreload: false)
        return InferenceEngine(model: model)
    }

    private static func run(_ engine: InferenceEngine, prompt: Int) async throws -> [Int32] {
        var out: [Int32] = []
        let ids = (0..<prompt).map { Int32(1 + $0 % 7) }
        for try await id in engine.generate(promptIDs: ids, maxNewTokens: 4, respectStopTokens: false) {
            out.append(id)
        }
        return out
    }

    /// The engine never cuts a prompt (a suffix loses the chat framing): one
    /// that fits runs, with room for its reply's first token; one that does
    /// not is refused.
    func testAPromptLongerThanTheContextIsRefusedNotCut() async throws {
        let engine = try await Self.engine()
        let limit = engine.maxPromptTokens
        XCTAssertEqual(limit, engine.model.effectiveMaxSeqLen - 1)
        // A reply budget never takes more than half the context from the history.
        XCTAssertEqual(engine.promptBudget(reservingForReply: 100), limit - 100)
        XCTAssertEqual(engine.promptBudget(reservingForReply: 4 * limit), limit - limit / 2)
        let generated = try await Self.run(engine, prompt: limit)
        XCTAssertEqual(generated.count, 1)
        do {
            _ = try await Self.run(engine, prompt: limit + 1)
            XCTFail("a prompt past the context ran")
        } catch InferenceError.promptTooLong(let tokens, let reported) {
            XCTAssertEqual(tokens, limit + 1)
            XCTAssertEqual(reported, limit)
        }
    }
}
