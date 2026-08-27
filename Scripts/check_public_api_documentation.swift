#!/usr/bin/env swift

import Foundation

guard CommandLine.arguments.count == 2 else {
    FileHandle.standardError.write(
        Data("usage: check_public_api_documentation.swift <symbol-graph.json>\n".utf8))
    exit(2)
}

let graphURL = URL(fileURLWithPath: CommandLine.arguments[1])
let data = try Data(contentsOf: graphURL)
guard let graph = try JSONSerialization.jsonObject(with: data) as? [String: Any],
      let symbols = graph["symbols"] as? [[String: Any]] else {
    FileHandle.standardError.write(Data("Invalid Swift symbol graph: \(graphURL.path)\n".utf8))
    exit(2)
}

let callableKinds: Set<String> = [
    "swift.init", "swift.method", "swift.type.method", "swift.subscript",
]
let returnDocumentedKinds: Set<String> = [
    "swift.method", "swift.type.method", "swift.subscript",
]
var failures: [String] = []
var checkedSymbolCount = 0

for symbol in symbols {
    guard symbol["accessLevel"] as? String == "public",
          let location = symbol["location"] as? [String: Any],
          let uri = location["uri"] as? String else { continue }

    checkedSymbolCount += 1

    let path = (symbol["pathComponents"] as? [String])?.joined(separator: ".")
        ?? "<unknown symbol>"
    let position = location["position"] as? [String: Any]
    let line = ((position?["line"] as? Int) ?? 0) + 1
    let source = "\(URL(string: uri)?.path ?? uri):\(line)"
    let doc = symbol["docComment"] as? [String: Any]
    let docLines = (doc?["lines"] as? [[String: Any]])?
        .compactMap { $0["text"] as? String } ?? []
    let meaningfulLines = docLines.map {
        $0.trimmingCharacters(in: .whitespacesAndNewlines)
    }.filter { !$0.isEmpty }

    if meaningfulLines.isEmpty {
        failures.append("\(source): \(path) has no /// documentation")
        continue
    }

    guard let kind = (symbol["kind"] as? [String: Any])?["identifier"] as? String else {
        continue
    }

    if kind == "swift.enum.case" {
        let declaration = symbol["declarationFragments"] as? [[String: Any]] ?? []
        let associatedNames = declaration.compactMap { fragment -> String? in
            guard fragment["kind"] as? String == "externalParam",
                  let name = fragment["spelling"] as? String,
                  name != "_" else { return nil }
            return name
        }
        for name in associatedNames {
            let hasDescription = meaningfulLines.contains {
                $0.hasPrefix("- Parameter \(name):") || $0.hasPrefix("- \(name):")
            }
            if !hasDescription {
                failures.append(
                    "\(source): \(path) does not document associated value `\(name)`")
            }
        }
    }

    guard callableKinds.contains(kind) else { continue }

    let signature = symbol["functionSignature"] as? [String: Any]
    let parameters = signature?["parameters"] as? [[String: Any]] ?? []
    for parameter in parameters {
        guard let name = (parameter["internalName"] as? String)
                ?? (parameter["name"] as? String) else { continue }
        let hasDescription = meaningfulLines.contains {
            $0.hasPrefix("- Parameter \(name):") || $0.hasPrefix("- \(name):")
        }
        if !hasDescription {
            failures.append("\(source): \(path) does not document parameter `\(name)`")
        }
    }

    if returnDocumentedKinds.contains(kind) {
        let returns = signature?["returns"] as? [[String: Any]] ?? []
        let returnSpelling = returns.compactMap { $0["spelling"] as? String }.joined()
        if !returnSpelling.isEmpty, returnSpelling != "()", returnSpelling != "Void" {
            let hasReturns = meaningfulLines.contains {
                $0.hasPrefix("- Returns:") || $0.hasPrefix("- Return:")
            }
            if !hasReturns {
                failures.append("\(source): \(path) does not document its return value")
            }
        }
    }

    let declaration = symbol["declarationFragments"] as? [[String: Any]] ?? []
    let canThrow = declaration.contains {
        let spelling = $0["spelling"] as? String
        return spelling == "throws" || spelling == "rethrows"
    }
    if canThrow, !meaningfulLines.contains(where: { $0.hasPrefix("- Throws:") }) {
        failures.append("\(source): \(path) does not document thrown errors")
    }
}

if failures.isEmpty {
    print(
        "Public API documentation contract passed for \(checkedSymbolCount) "
            + "source-defined public symbols.")
} else {
    FileHandle.standardError.write(Data(
        ("Public API documentation contract failed:\n"
            + failures.sorted().map { "- \($0)" }.joined(separator: "\n") + "\n").utf8))
    exit(1)
}
