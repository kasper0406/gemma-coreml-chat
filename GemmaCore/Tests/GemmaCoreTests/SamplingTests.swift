/// The banded nucleus sampler against a straightforward reference that sorts
/// the whole vocabulary: same nucleus (ids, order and renormalized
/// probabilities, bit for bit) and same token for a fixed uniform draw.

import Accelerate
import CoreML
import Synchronization
import XCTest

@testable import GemmaCore

final class SamplingTests: XCTestCase {
    // MARK: - Reference

    /// The sampler's semantics, spelled out: probabilities through the same
    /// vDSP steps, a full sort by (probability desc, token id asc), the
    /// shortest prefix whose running Float sum reaches topP (the whole
    /// vocabulary if it never does), renormalized. A distribution with no
    /// finite maximum or sum is a point mass on the largest non-NaN logit.
    private static func referenceNucleus(
        _ logits: [Float], temperature: Float, topP: Float
    ) -> [(id: Int32, prob: Float)] {
        let pointMass = [(id: Int32(firstLargest(logits)), prob: Float(1))]
        let n = vDSP_Length(logits.count)
        var probs = logits
        var invTemp = 1.0 / temperature
        vDSP_vsmul(probs, 1, &invTemp, &probs, 1, n)
        var maxVal: Float = 0
        vDSP_maxv(probs, 1, &maxVal, n)
        guard maxVal.isFinite else { return pointMass }
        var negMax = -maxVal
        vDSP_vsadd(probs, 1, &negMax, &probs, 1, n)
        var count32 = Int32(logits.count)
        probs.withUnsafeMutableBufferPointer { vvexpf($0.baseAddress!, $0.baseAddress!, &count32) }
        var sum: Float = 0
        vDSP_sve(probs, 1, &sum, n)
        guard sum.isFinite else { return pointMass }
        var invSum = 1.0 / sum
        vDSP_vsmul(probs, 1, &invSum, &probs, 1, n)

        let order = probs.indices.sorted { probs[$0] != probs[$1] ? probs[$0] > probs[$1] : $0 < $1 }
        var cutoff = order.count
        var cumulative: Float = 0
        for (i, idx) in order.enumerated() {
            cumulative += probs[idx]
            if cumulative >= topP {
                cutoff = i + 1
                break
            }
        }
        var top = order.prefix(cutoff).map { probs[$0] }
        var topSum: Float = 0
        vDSP_sve(top, 1, &topSum, vDSP_Length(top.count))
        var invTopSum = 1.0 / topSum
        vDSP_vsmul(top, 1, &invTopSum, &top, 1, vDSP_Length(top.count))
        return zip(order.prefix(cutoff), top).map { (Int32($0), $1) }
    }

    /// First index of the largest non-NaN value (+inf is largest); 0 if none.
    private static func firstLargest(_ values: [Float]) -> Int {
        var best = -1
        for (i, v) in values.enumerated() where !v.isNaN && (best < 0 || v > values[best]) { best = i }
        return max(best, 0)
    }

    private static func referenceDraw(_ nucleus: [(id: Int32, prob: Float)], uniform: Float) -> Int32 {
        var accum: Float = 0
        for (id, prob) in nucleus {
            accum += prob
            if accum >= uniform { return id }
        }
        return nucleus[0].id
    }

    // MARK: - Inputs

    /// SplitMix64: reproducible without depending on the system generator.
    private struct Rng {
        var state: UInt64
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
        mutating func uniform() -> Float { Float(next() >> 40) / Float(1 << 24) }
        mutating func normal() -> Float {
            let u1 = max(uniform(), 1e-7), u2 = uniform()
            return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
        }
    }

    private static func array(_ values: [Float], fp16: Bool) throws -> MLMultiArray {
        let a = try MLMultiArray(shape: [1, NSNumber(value: values.count)], dataType: fp16 ? .float16 : .float32)
        if fp16 {
            a.withUnsafeMutableBufferPointer(ofType: Float16.self) { buf, _ in
                for (i, v) in values.enumerated() { buf[i] = Float16(v) }
            }
        } else {
            a.withUnsafeMutableBufferPointer(ofType: Float.self) { buf, _ in
                for (i, v) in values.enumerated() { buf[i] = v }
            }
        }
        return a
    }

    /// Named logit vectors covering peaked, flat, tied and masked shapes.
    private static func distributions(vocab: Int, seed: UInt64) -> [(String, [Float])] {
        var rng = Rng(state: seed)
        let normal = (0..<vocab).map { _ in rng.normal() }
        // Gemma-like: heavy spread, soft-capped at 30.
        let softcapped = normal.map { 30 * tanh(4 * $0 / 30) }
        var masked = [Float](repeating: -.infinity, count: vocab)
        for i in stride(from: 0, to: vocab, by: 97) { masked[i] = normal[i] }
        var oneHot = [Float](repeating: -.infinity, count: vocab)
        oneHot[vocab / 3] = 0
        var fewPeaks = normal.map { $0 * 0.5 }
        for i in [7, 1000 % vocab, vocab - 1] { fewPeaks[i] = 12 }
        return [
            ("normal", normal),
            ("peaked", normal.map { $0 * 8 }),
            ("flat", normal.map { $0 * 0.05 }),
            ("softcapped", softcapped),
            ("allEqual", [Float](repeating: 1.5, count: vocab)),
            ("ties", normal.map { ($0 * 2).rounded() / 2 }),
            ("twoLevels", (0..<vocab).map { $0 % 5 == 0 ? 3 : 0 }),
            ("masked", masked),
            ("oneHot", oneHot),
            ("fewPeaks", fewPeaks),
            ("extremeRange", normal.map { $0 * 60 }),
            // At T=1 these land exactly on the sampler's band edges (6, 12
            // and 24 nats below the top token).
            ("bandEdges", (0..<vocab).map { [0, -6, -12, -24, -30][$0 % 5] }),
            // Logits a few ulps either side of the band edges, so neighbouring
            // probabilities straddle each edge's rounded threshold.
            ("adjacentToEdges", (0..<vocab).map { i -> Float in
                guard i % 29 != 0 else { return 0 }
                var x: Float = -[6, 12, 24][i % 3]
                for _ in 0..<(i / 3 % 7) { x = i % 2 == 0 ? x.nextUp : x.nextDown }
                return x
            }),
        ]
    }

    private static let topPs: [Float] = [1e-6, 0.1, 0.5, 0.9, 0.95, 0.999, 1.0]
    private static let temperatures: [Float] = [0.1, 0.7, 1.0, 1.5, 3.0]
    /// Includes temperatures whose reciprocal overflows (a point mass on the
    /// argmax) or underflows to 0 (uniform).
    private static let extremeTemperatures: [Float] = [
        1e-39, 1e-30, 1e-6, 1e6, 1e30, .greatestFiniteMagnitude, .infinity,
    ]
    private static let uniforms: [Float] = [0, 1e-6, 0.1, 0.25, 0.5, 0.75, 0.9, 0.999, 0.9999999]

    private func assertMatchesReference(
        _ name: String, _ logits: [Float], fp16: Bool, temperature: Float, topP: Float,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let input = try Self.array(logits, fp16: fp16)
        let reference = Self.referenceNucleus(
            fp16 ? logits.map { Float(Float16($0)) } : logits, temperature: temperature, topP: topP
        )
        let got = Sampling.nucleus(logits: input, temperature: temperature, topP: topP)
        let context = "\(name) fp16=\(fp16) T=\(temperature) topP=\(topP)"
        XCTAssertEqual(got.map(\.id), reference.map(\.id), "nucleus ids, \(context)", file: file, line: line)
        XCTAssertEqual(
            got.map(\.prob.bitPattern), reference.map(\.prob.bitPattern),
            "nucleus probabilities, \(context)", file: file, line: line
        )
        for u in Self.uniforms {
            XCTAssertEqual(
                Sampling.sampleNextToken(logits: input, temperature: temperature, topP: topP, uniform: u),
                Self.referenceDraw(reference, uniform: u),
                "draw u=\(u), \(context)", file: file, line: line
            )
        }
    }

    // MARK: - Tests

    func testNucleusMatchesFullSortReference() throws {
        for (name, logits) in Self.distributions(vocab: 3001, seed: 1) {
            for temperature in Self.temperatures {
                for topP in Self.topPs {
                    try assertMatchesReference(name, logits, fp16: false, temperature: temperature, topP: topP)
                }
            }
        }
    }

    func testFloat16LogitsMatchReference() throws {
        // fp16 quantization adds plenty of exact ties.
        for (name, logits) in Self.distributions(vocab: 2048, seed: 2) {
            for topP: Float in [0.5, 0.9, 1.0] {
                try assertMatchesReference(name, logits, fp16: true, temperature: 1.0, topP: topP)
            }
        }
    }

    func testRandomDistributionsMatchReference() throws {
        var rng = Rng(state: 3)
        for trial in 0..<200 {
            let vocab = 1 + Int(rng.next() % 5000)
            let scale = exp(rng.uniform() * 8 - 4)  // e^-4 ... e^4
            let tieStep: Float = rng.uniform() < 0.3 ? 0.25 : 0
            var logits = (0..<vocab).map { _ in rng.normal() * scale }
            if tieStep > 0 { logits = logits.map { ($0 / tieStep).rounded() * tieStep } }
            if rng.uniform() < 0.2 {
                for i in 0..<vocab where rng.uniform() < 0.5 { logits[i] = -.infinity }
                logits[Int(rng.next() % UInt64(vocab))] = 0
            }
            let temperature = 0.05 + rng.uniform() * 3
            let topP = rng.uniform() < 0.1 ? 1.0 : max(rng.uniform(), 1e-6)
            try assertMatchesReference(
                "random#\(trial) V=\(vocab)", logits, fp16: rng.uniform() < 0.25,
                temperature: temperature, topP: topP
            )
        }
    }

    func testFullVocabularyMatchesReference() throws {
        // Gemma's vocabulary size, so the band edges see realistic counts.
        let vocab = 262_144
        let cases: [String: [Float]] = ["softcapped": [0.9, 0.999], "flat": [0.9], "ties": [0.9]]
        for (name, logits) in Self.distributions(vocab: vocab, seed: 4) {
            for topP in cases[name] ?? [] {
                try assertMatchesReference(name, logits, fp16: false, temperature: 1.0, topP: topP)
            }
        }
    }

    func testExtremeTemperaturesMatchReference() throws {
        for (name, logits) in Self.distributions(vocab: 1500, seed: 6) {
            for temperature in Self.extremeTemperatures {
                for topP: Float in [1e-6, 0.5, 0.9, 1.0] {
                    try assertMatchesReference(name, logits, fp16: false, temperature: temperature, topP: topP)
                }
            }
        }
    }

    func testReciprocalOverflowIsGreedyAndInfiniteTemperatureIsUniform() throws {
        var rng = Rng(state: 7)
        let logits = (0..<1000).map { _ in rng.normal() }
        let input = try Self.array(logits, fp16: false)
        let top = Int32(Self.firstLargest(logits))
        for temperature: Float in [1e-39, .leastNonzeroMagnitude] {
            XCTAssertEqual(Sampling.nucleus(logits: input, temperature: temperature, topP: 0.9).map(\.id), [top])
            XCTAssertEqual(Sampling.sampleNextToken(logits: input, temperature: temperature, topP: 0.9, uniform: 0.7), top)
        }
        // 1/T == 0: every token equally likely, so the nucleus is the lowest
        // ids (half of them, give or take the Float running sum's rounding).
        let uniform = Sampling.nucleus(logits: input, temperature: .infinity, topP: 0.5)
        XCTAssertEqual(uniform.map(\.id), (0..<Int32(uniform.count)).map { $0 })
        XCTAssert((499...501).contains(uniform.count), "\(uniform.count)")
    }

    func testZeroAndSubnormalTailIsWalkedInIdOrder() throws {
        // Probabilities 1, e^-80 (normal), e^-95 (subnormal) and exactly 0;
        // topP 2 is never reached, forcing the walk through every band.
        let levels: [Float] = [-80, -95, -150, -.infinity]
        var logits = (0..<4000).map { levels[$0 % levels.count] }
        logits[1234] = 0
        for topP: Float in [0.9, 1.0, 2.0] {
            try assertMatchesReference("tail", logits, fp16: false, temperature: 1.0, topP: topP)
        }
        let nucleus = Sampling.nucleus(logits: try Self.array(logits, fp16: false), temperature: 1.0, topP: 2.0)
        XCTAssertEqual(nucleus.count, logits.count)
        XCTAssertEqual(nucleus.first?.id, 1234)
        XCTAssert(nucleus.contains { $0.prob.isSubnormal })
        let zeros = nucleus.filter { $0.prob == 0 }.map(\.id)
        XCTAssertEqual(zeros.count, logits.filter { $0 <= -150 }.count)
        XCTAssertEqual(zeros, zeros.sorted())
        XCTAssertEqual(Array(nucleus.suffix(zeros.count)).map(\.id), zeros)
    }

    func testFlatFullVocabularyWithTinyTopP() throws {
        // Every token lands in the first band; the nucleus is a single token.
        var rng = Rng(state: 8)
        let logits = (0..<262_144).map { _ in rng.uniform() * 0.01 }
        for topP: Float in [1e-6, 0.01] {
            try assertMatchesReference("flat262k", logits, fp16: false, temperature: 1.0, topP: topP)
        }
    }

    func testConcurrentCallsMatchSerialResults() throws {
        var rng = Rng(state: 9)
        let inputs = try (0..<16).map { i -> (MLMultiArray, Float, Float) in
            let logits = (0..<1_000).map { _ in rng.normal() * Float(1 + i % 4) }
            return (try Self.array(logits, fp16: i % 3 == 0), [0, 0.7, 1.0, 1.5][i % 4], [0.5, 0.9, 0.99, 1.0][i / 4])
        }
        let uniforms: [Float] = (0..<10).map { Float($0) / 10 }
        let serial = inputs.map { input, temperature, topP in
            uniforms.map { Sampling.sampleNextToken(logits: input, temperature: temperature, topP: topP, uniform: $0) }
        }
        let results = Mutex([[Int32]](repeating: [], count: inputs.count))
        DispatchQueue.concurrentPerform(iterations: inputs.count * 16) { job in
            let (input, temperature, topP) = inputs[job % inputs.count]
            let got = uniforms.map {
                Sampling.sampleNextToken(logits: input, temperature: temperature, topP: topP, uniform: $0)
            }
            if job < inputs.count { results.withLock { $0[job] = got } } else {
                XCTAssertEqual(got, serial[job % inputs.count], "job \(job)")
            }
        }
        XCTAssertEqual(results.withLock { $0 }, serial)
    }

    func testGreedyIsFirstArgmax() throws {
        var rng = Rng(state: 5)
        for fp16 in [false, true] {
            var logits = (0..<4096).map { _ in (rng.normal() * 4).rounded() }
            logits[100] = 50
            logits[3000] = 50
            let input = try Self.array(logits, fp16: fp16)
            for temperature: Float in [0, -1] {
                XCTAssertEqual(
                    Sampling.sampleNextToken(logits: input, temperature: temperature, topP: 0.9, uniform: 0.5), 100
                )
            }
        }
    }

    func testNonFiniteLogitsArePointMassOnLargestLogit() throws {
        var rng = Rng(state: 10)
        let base = (0..<1000).map { _ in rng.normal() }
        let finiteTop = Self.firstLargest(base)
        var cases: [(String, [Float], Int)] = [
            ("all -inf", [Float](repeating: -.infinity, count: 1000), 0),
            ("all NaN", [Float](repeating: .nan, count: 1000), 0),
        ]
        for k in [0, 10, 500, 999] {
            var posInf = base
            posInf[k] = .infinity
            posInf[(k + 7) % 1000] = .infinity
            cases.append(("+inf at \(k)", posInf, min(k, (k + 7) % 1000)))
        }
        for k in stride(from: 0, to: 1000, by: 37) where k != finiteTop {
            var nan = base
            nan[k] = .nan
            cases.append(("NaN at \(k)", nan, finiteTop))
        }
        for (name, logits, expected) in cases {
            let input = try Self.array(logits, fp16: false)
            for temperature: Float in [0, 1.0] {
                for u: Float in [0, 0.5, 0.999] {
                    XCTAssertEqual(
                        Sampling.sampleNextToken(logits: input, temperature: temperature, topP: 0.9, uniform: u),
                        Int32(expected), "\(name) T=\(temperature) u=\(u)"
                    )
                }
            }
            if temperature1Nucleus(input) != [Int32(expected)] { XCTFail("nucleus, \(name)") }
        }
    }

    private func temperature1Nucleus(_ input: MLMultiArray) -> [Int32] {
        Sampling.nucleus(logits: input, temperature: 1.0, topP: 0.9).map(\.id)
    }
}
