import CoreML
import Foundation

@available(macOS 14.4, *)
@main
struct CoreMLComputePlanInspector {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            FileHandle.standardError.write(
                Data("usage: inspect_coreml_compute_plan.swift MODEL.mlpackage\n".utf8))
            Foundation.exit(64)
        }

        let source = URL(fileURLWithPath: CommandLine.arguments[1])
        let compiled = try await MLModel.compileModel(at: source)
        defer { try? FileManager.default.removeItem(at: compiled) }

        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        configuration.optimizationHints.specializationStrategy = .fastPrediction
        let plan = try await MLComputePlan.load(
            contentsOf: compiled, configuration: configuration)

        guard case let .program(program) = plan.modelStructure,
              let main = program.functions["main"] else {
            throw InspectorError.notMLProgram
        }

        var counts: [String: Int] = [:]
        var costByDevice: [String: Double] = [:]
        var operatorsByDevice: [String: [String: Int]] = [:]
        var unsupportedByNeuralEngine: [String: Int] = [:]
        var costlyOperations: [(Double, String, String)] = []

        func label(_ device: MLComputeDevice) -> String {
            switch device {
            case .cpu: "cpu"
            case .gpu: "gpu"
            case .neuralEngine: "neural-engine"
            @unknown default: "unknown-device"
            }
        }

        func inspect(_ block: MLModelStructure.Program.Block) {
            for operation in block.operations {
                let usage = plan.deviceUsage(for: operation)
                let preferred = usage.map { label($0.preferred) } ?? "unknown"
                counts[preferred, default: 0] += 1
                operatorsByDevice[preferred, default: [:]][operation.operatorName, default: 0] += 1
                if let usage, !usage.supported.contains(where: {
                    if case .neuralEngine = $0 { return true }
                    return false
                }) {
                    unsupportedByNeuralEngine[operation.operatorName, default: 0] += 1
                }
                let cost = plan.estimatedCost(of: operation)?.weight ?? 0
                costByDevice[preferred, default: 0] += cost
                costlyOperations.append((cost, preferred, operation.operatorName))
                operation.blocks.forEach(inspect)
            }
        }
        inspect(main.block)

        print("Preferred device operation counts:")
        for (device, count) in counts.sorted(by: { $0.key < $1.key }) {
            print("  \(device): \(count) operations, estimated cost \(costByDevice[device, default: 0])")
        }
        print("Operators by preferred device:")
        for (device, operators) in operatorsByDevice.sorted(by: { $0.key < $1.key }) {
            let summary = operators.sorted {
                $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value
            }.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
            print("  \(device): \(summary)")
        }
        print("Operations without Neural Engine support:")
        for (name, count) in unsupportedByNeuralEngine.sorted(by: {
            $0.value == $1.value ? $0.key < $1.key : $0.value > $1.value
        }) {
            print("  \(name): \(count)")
        }
        print("Highest estimated-cost operations:")
        for (cost, device, name) in costlyOperations.sorted(by: { $0.0 > $1.0 }).prefix(30) {
            print("  \(String(format: "%.8f", cost))  \(device)  \(name)")
        }
    }
}

enum InspectorError: Error {
    case notMLProgram
}
