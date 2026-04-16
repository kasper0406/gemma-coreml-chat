#!/usr/bin/env swift
//
// Swift reproducer: multifunction CoreML model with RangeDim.
//
// Requires Xcode (not just CommandLineTools) to compile.
//
// Build & run:
//   1. Generate test models with Python:
//        python benchmarks/multifunction_rangedim_bug.py
//      Note the "Working dir:" path in the output.
//
//   2. Compile and run:
//        swiftc -framework CoreML -framework Foundation \
//          benchmarks/multifunction_rangedim_bug.swift -o /tmp/mf_test
//        /tmp/mf_test /path/to/temp/dir
//
// Expected: both single-function and multifunction models load and predict.
// Actual:   multifunction model may fail to compile/load.

import CoreML
import Foundation

func loadAndPredict(modelURL: URL, label: String, functionName: String? = nil) {
    let config = MLModelConfiguration()
    config.computeUnits = .cpuOnly

    if let fn = functionName {
        config.functionName = fn
    }

    do {
        let compiledURL = try MLModel.compileModel(at: modelURL)
        let model = try MLModel(contentsOf: compiledURL, configuration: config)

        // Build input: position=[0], cache=ones(1,8,4)
        let position = try MLMultiArray(shape: [1], dataType: .int32)
        position[0] = 0

        let cache = try MLMultiArray(shape: [1, 8, 4], dataType: .float16)
        for i in 0..<cache.count {
            cache[i] = 1.0
        }

        let inputProvider = try MLDictionaryFeatureProvider(
            dictionary: ["position": position, "cache": cache]
        )
        let prediction = try model.prediction(from: inputProvider)
        let outputName = prediction.featureNames.first ?? "?"
        let output = prediction.featureValue(for: outputName)?.multiArrayValue
        print("  ✅ \(label): loaded OK, output shape = \(output?.shape ?? [])")

        // Clean up compiled model
        try? FileManager.default.removeItem(at: compiledURL)
    } catch {
        print("  ❌ \(label): FAILED — \(error)")
    }
}

// ── Main ──

guard CommandLine.arguments.count > 1 else {
    print("Usage: swift \(CommandLine.arguments[0]) <temp-dir-from-python-reproducer>")
    print("\nFirst run:  python benchmarks/multifunction_rangedim_bug.py")
    exit(1)
}

let tmpDir = CommandLine.arguments[1]
let funcA = URL(fileURLWithPath: "\(tmpDir)/func_a.mlpackage")
let funcB = URL(fileURLWithPath: "\(tmpDir)/func_b.mlpackage")
let merged = URL(fileURLWithPath: "\(tmpDir)/merged.mlpackage")

print("Step 1: Loading single-function models (should work)…")
loadAndPredict(modelURL: funcA, label: "func_a")
loadAndPredict(modelURL: funcB, label: "func_b")

print("\nStep 2: Loading multifunction model (this is where it fails)…")
loadAndPredict(modelURL: merged, label: "merged/decode", functionName: "decode")
loadAndPredict(modelURL: merged, label: "merged/prefill", functionName: "prefill")

print("\nDone.")
