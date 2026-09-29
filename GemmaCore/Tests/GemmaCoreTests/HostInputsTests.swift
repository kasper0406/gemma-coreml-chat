/// The Swift host inputs against `decode_coreml.host_inputs`, the reference.
/// The fixture is written by `tests/test_host_inputs.py`, which also checks it
/// is current.

import CoreML
import Foundation
import XCTest

@testable import GemmaCore

final class HostInputsTests: XCTestCase {
    private struct Case: Decodable {
        struct Input: Decodable {
            let shape: [Int]
            let fp16: String
        }
        let start: Int
        let rows: Int
        let cache_length: Int
        let inputs: [String: Input]
    }

    func testInputsMatchTheReference() throws {
        let fixture = try XCTUnwrap(Bundle.module.url(forResource: "HostInputs", withExtension: "json"))
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: fixture))
        XCTAssertFalse(cases.isEmpty)
        for c in cases {
            let ringLength = try XCTUnwrap(c.inputs[HostInputs.writeSliding]).shape[1]
            // The ring after a conversation that wrote every position from 0.
            var ring = [Int32](repeating: -1, count: ringLength)
            for p in 0..<(c.start + c.rows) { ring[p % ringLength] = Int32(p) }

            for (name, input) in c.inputs {
                let array = try MLMultiArray(shape: input.shape.map { NSNumber(value: $0) }, dataType: .float16)
                switch name {
                case HostInputs.ropeSliding, HostInputs.ropeGlobal:
                    let timescale = HostInputs.timescales(for: name, half: input.shape[3] / 2)
                    try HostInputs.fillRope(array, start: c.start, timescale: timescale)
                case HostInputs.maskSliding:
                    try HostInputs.fillSlidingMask(array, start: c.start, ring: ring)
                case HostInputs.maskGlobal:
                    try HostInputs.fillGlobalMask(array, start: c.start)
                default:
                    try HostInputs.fillWrite(array, start: c.start, wraps: name == HostInputs.writeSliding)
                }
                let got = array.withUnsafeBufferPointer(ofType: Float16.self) { Array($0) }
                let want = try XCTUnwrap(Data(base64Encoded: input.fp16)).withUnsafeBytes {
                    Array($0.bindMemory(to: Float16.self))
                }
                XCTAssertEqual(got.count, want.count, "\(name) at \(c.start)")
                // Bit for bit, except that the RoPE rows may round a value on an
                // fp16 boundary the other way (fp64 cos/sin differ by an ulp
                // between libraries): at most one fp16 step apart.
                let mismatches = zip(got, want).filter { $0.bitPattern != $1.bitPattern }
                if name.hasPrefix("rope_") {
                    XCTAssertLessThanOrEqual(mismatches.count, got.count / 1000, "\(name) at \(c.start)")
                    for (g, w) in mismatches {
                        XCTAssertEqual(Float(g), Float(w), accuracy: Float(w.ulp), "\(name) at \(c.start)")
                    }
                } else {
                    XCTAssertTrue(mismatches.isEmpty, "\(name) at \(c.start): \(mismatches.count) values differ")
                }
            }
        }
    }
}
