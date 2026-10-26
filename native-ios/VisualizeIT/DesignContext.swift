import Foundation
import CryptoKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

// SOURCE ONLY: native compilation, filesystem durability and device memory gates remain open.
// JSON dictionaries deliberately retain inert foreign recipes/extensions without translating them.
typealias ContextObject = [String: Any]

enum ContextFailure: LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let reason) = self { return reason }; return nil }
}

enum ContextSnapshotGate {
    static let reason = "Design Context snapshot import and export are unavailable in this source preview. Lossless numeric and Unicode-key interoperability is unresolved. Your local projects and PNG artwork are preserved; Photos import and PNG sharing remain available."
    static func requireAvailable() throws { throw ContextFailure.invalid(reason) }
}

enum ContextCodec {
    static let null = NSNull()
    static func fail(_ reason: String) throws -> Never { throw ContextFailure.invalid(reason) }
    static func id() -> String { UUID().uuidString }
    static func bytes(_ value: Any) throws -> Data {
        try JSONSerialization.data(withJSONObject: value, options: [.sortedKeys, .fragmentsAllowed, .withoutEscapingSlashes])
    }
    static func equal(_ a: Any, _ b: Any) -> Bool { (try? bytes(a)) == (try? bytes(b)) }
    static func hash(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }
    static func object(_ value: Any?) throws -> ContextObject {
        guard let result = value as? ContextObject else { try fail("Expected an object.") }; return result
    }
    static func keys(_ value: ContextObject, _ names: String) throws {
        guard Set(value.keys) == Set(names.split(separator: " ").map(String.init)) else { try fail("Unknown or missing context fields.") }
    }
    static func string(_ value: Any?) throws -> String {
        guard let value = value as? String else { try fail("Expected text.") }; return value
    }
    static func identifier(_ value: Any?, nullable: Bool = false) throws {
        if nullable && value is NSNull { return }
        let text = try string(value)
        guard !text.isEmpty, text.utf8.count <= 128, text.utf8.allSatisfy({ (65...90).contains($0) || (97...122).contains($0) || (48...57).contains($0) || [46,95,58,45].contains($0) }) else { try fail("Invalid identifier.") }
    }
    static func integer(_ value: Any?) throws -> Int {
        guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite,
              number.doubleValue.rounded() == number.doubleValue, abs(number.doubleValue) <= Double(Int.max / 2) else { try fail("Expected a bounded integer.") }
        return number.intValue
    }
    static func effect(_ value: ContextObject) throws {
        try keys(value,"type payload"); let payload = try object(value["payload"])
        switch try string(value["type"]) {
        case "draft":
            guard !payload.isEmpty, Set(payload.keys).isSubset(of:["text","capability","settings"]) else { try fail("Invalid draft effect.") }
            if payload["text"] != nil { _ = try string(payload["text"]) }
            if let capability = payload["capability"], !(capability is NSNull) { _ = try string(capability) }
            if payload["settings"] != nil { _ = try object(payload["settings"]) }
        case "clarify": try keys(payload,"question"); _ = try string(payload["question"])
        case "import": try keys(payload,"")
        default: try fail("Unknown effect type.")
        }
    }
    static func brief(_ value: Any?) throws {
        if value is NSNull { return }
        let value = try object(value); try keys(value, "revisionId text"); try identifier(value["revisionId"]); _ = try string(value["text"])
    }
    static func png(_ data: Data) throws -> (Int, Int) {
        guard data.count <= 16_777_216, data.count >= 33, Array(data.prefix(8)) == [137,80,78,71,13,10,26,10] else { try fail("PNG size/signature rejected.") }
        let b = [UInt8](data)
        func u32(_ i: Int) -> UInt32 { b[i..<i+4].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) } }
        func crc(_ range: Range<Int>) -> UInt32 {
            var c: UInt32 = 0xffffffff
            for byte in b[range] { c ^= UInt32(byte); for _ in 0..<8 { c = (c >> 1) ^ ((c & 1) == 1 ? 0xedb88320 : 0) } }
            return c ^ 0xffffffff
        }
        var offset = 8, width = 0, height = 0, header = false, image = false, ended = false, idatClosed = false
        while offset < b.count {
            guard offset <= b.count - 12 else { try fail("Truncated PNG chunk.") }
            let n = Int(u32(offset)); guard n <= b.count - offset - 12 else { try fail("Truncated PNG payload.") }
            let kindBytes = Array(b[offset+4..<offset+8])
            guard kindBytes.allSatisfy({ (65...90).contains($0) || (97...122).contains($0) }), kindBytes[2] & 32 == 0 else { try fail("Invalid PNG chunk type.") }
            let kind = String(bytes: kindBytes, encoding: .ascii)!
            guard crc(offset+4..<offset+8+n) == u32(offset+8+n) else { try fail("PNG checksum rejected.") }
            if !header && kind != "IHDR" { try fail("PNG header must be first.") }
            if kind == "IHDR" {
                guard !header, n == 13 else { try fail("Invalid PNG header.") }
                width = Int(u32(offset+8)); height = Int(u32(offset+12)); header = true
                guard width > 0, height > 0, width <= 4096, height <= 4096, width * height <= 16_777_216 else { try fail("PNG dimensions exceed the snapshot limit.") }
            } else if kind == "IDAT" { guard !idatClosed else { try fail("Noncontiguous PNG image chunks.") }; image = true }
            else { if image { idatClosed = true }; if kind == "IEND" { guard n == 0, image else { try fail("Invalid PNG end.") }; ended = true }
                else if kindBytes[0] & 32 == 0 && kind != "PLTE" { try fail("Unknown critical PNG chunk.") }
            }
            offset += n + 12
            if ended { break }
        }
        guard ended, offset == b.count, let source = CGImageSourceCreateWithData(data as CFData, nil), CGImageSourceGetCount(source) == 1,
              let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
              (properties[kCGImagePropertyPixelWidth] as? NSNumber)?.intValue == width,
              (properties[kCGImagePropertyPixelHeight] as? NSNumber)?.intValue == height,
              let decoded = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCacheImmediately: true] as CFDictionary),
              decoded.width == width, decoded.height == height, CGImageSourceGetStatus(source) == .statusComplete else { try fail("PNG decoding failed.") }
        return (width, height)
    }
    static func artwork(_ value: Any?) throws {
        if value is NSNull { return }
        let art = try object(value); try keys(art, "id png provenance"); try identifier(art["id"])
        let p = try object(art["png"]); try keys(p, "sha256 byteCount width height base64")
        let encoded = try string(p["base64"])
        guard encoded.utf8.count <= 22_369_624, let data = Data(base64Encoded: encoded), data.base64EncodedString() == encoded else { try fail("Noncanonical base64.") }
        let (w,h) = try png(data)
        guard try integer(p["byteCount"]) == data.count, try integer(p["width"]) == w, try integer(p["height"]) == h,
              try string(p["sha256"]) == hash(data) else { try fail("PNG identity mismatch.") }
        let provenance = try object(art["provenance"])
        try keys(provenance, "kind producer brief prompt effectiveRequest source")
        guard ["generated","imported","legacy"].contains(try string(provenance["kind"])) else { try fail("Unknown provenance kind.") }
        try brief(provenance["brief"])
        for key in ["prompt", "source"] { if !(provenance[key] is NSNull) { _ = try string(provenance[key]) } }
        if !(provenance["effectiveRequest"] is NSNull) { _ = try object(provenance["effectiveRequest"]) }
        if !(provenance["producer"] is NSNull) {
            let producer = try object(provenance["producer"]); try keys(producer, "app version capability"); _ = try string(producer["app"])
            for key in ["version","capability"] { if !(producer[key] is NSNull) { _ = try string(producer[key]) } }
        }
        if provenance["kind"] as? String == "generated" {
            let producer = try object(provenance["producer"]); _ = try string(producer["capability"])
            guard !(provenance["brief"] is NSNull), !(provenance["prompt"] is NSNull), !(provenance["effectiveRequest"] is NSNull) else { try fail("Incomplete creation provenance.") }
        }
    }
    static func decode(_ data: Data) throws -> ContextObject {
        guard data.count <= 25_165_824, String(data: data, encoding: .utf8) != nil else { try fail("Snapshot size or UTF-8 rejected.") }
        var scanner = StrictContextJSON(data: data); try scanner.check()
        let doc = try object(JSONSerialization.jsonObject(with: data)); try validate(doc); return doc
    }
    static func validate(_ d: ContextObject) throws {
        try keys(d, "schemaVersion documentType projectId projectRevisionId parentRevisionId title origin messages acceptedBrief promptDraft chosenSuggestionEffects generationAttempt selectedArtwork extensions")
        guard try integer(d["schemaVersion"]) == 1, try string(d["documentType"]) == "visualizeit-project" else { try fail("Unsupported context schema.") }
        for key in ["projectId", "projectRevisionId"] { try identifier(d[key]) }; try identifier(d["parentRevisionId"], nullable: true); _ = try string(d["title"])
        if !(d["origin"] is NSNull) { let origin = try object(d["origin"]); try keys(origin, "projectId revisionId"); try identifier(origin["projectId"]); try identifier(origin["revisionId"]) }
        guard let messages = d["messages"] as? [ContextObject], let effects = d["chosenSuggestionEffects"] as? [ContextObject], messages.count <= 1000, effects.count <= 1000 else { try fail("Transcript/effect capacity exceeded.") }
        var ids = Set<String>()
        for m in messages {
            try keys(m, "id role source text artworkIds"); try identifier(m["id"])
            guard ids.insert(try string(m["id"])).inserted, ["user","assistant","system"].contains(try string(m["role"])) else { try fail("Duplicate message or invalid role.") }
            _ = try string(m["source"]); _ = try string(m["text"])
            guard let refs = m["artworkIds"] as? [String], Set(refs).count == refs.count else { try fail("Invalid artwork references.") }; for ref in refs { try identifier(ref) }
        }
        try brief(d["acceptedBrief"])
        if !(d["promptDraft"] is NSNull) {
            let draft = try object(d["promptDraft"]); try keys(draft, "revisionId baseBriefRevisionId text capability settings")
            try identifier(draft["revisionId"]); try identifier(draft["baseBriefRevisionId"], nullable: true); _ = try string(draft["text"])
            if !(draft["capability"] is NSNull) { _ = try string(draft["capability"]) }; _ = try object(draft["settings"])
        }
        ids.removeAll()
        for e in effects {
            try keys(e, "id actionId projectId contextRevisionId draftRevisionId briefRevisionId effect resultDraftRevisionId")
            for key in ["id","actionId","projectId","contextRevisionId"] { try identifier(e[key]) }
            for key in ["draftRevisionId","briefRevisionId","resultDraftRevisionId"] { try identifier(e[key], nullable: true) }
            guard ids.insert(try string(e["id"])).inserted else { try fail("Duplicate chosen effect.") }
            try effect(object(e["effect"]))
        }
        if !(d["generationAttempt"] is NSNull) {
            let a = try object(d["generationAttempt"]); try keys(a, "id projectId contextRevisionId brief prompt capability effectiveRequest")
            for key in ["id","projectId","contextRevisionId"] { try identifier(a[key]) }; try brief(a["brief"])
            guard !(a["brief"] is NSNull) else { try fail("Attempt requires accepted brief.") }
            _ = try string(a["prompt"]); _ = try string(a["capability"]); _ = try object(a["effectiveRequest"])
        }
        guard d["parentRevisionId"] as? String != d["projectRevisionId"] as? String else { try fail("Self-parent revision.") }
        let ext = try object(d["extensions"]); guard try bytes(ext).count <= 65_536 else { try fail("Extensions exceed capacity.") }
        var metadata = d
        if var art = d["selectedArtwork"] as? ContextObject, var p = art["png"] as? ContextObject { p["base64"] = ""; art["png"] = p; metadata["selectedArtwork"] = art }
        guard try bytes(metadata).count <= 1_048_576, try bytes(d).count <= 25_165_824 else { try fail("Snapshot metadata exceeds capacity.") }
        try artwork(d["selectedArtwork"])
        var scanner = StrictContextJSON(data: try bytes(d)); try scanner.check()
    }
    static func newProject() -> ContextObject {
        ["schemaVersion":1,"documentType":"visualizeit-project","projectId":id(),"projectRevisionId":id(),"parentRevisionId":null,"title":"Untitled","origin":null,"messages":[ContextObject](),"acceptedBrief":null,"promptDraft":null,"chosenSuggestionEffects":[ContextObject](),"generationAttempt":null,"selectedArtwork":null,"extensions":ContextObject()]
    }
    static func makeArtwork(_ data: Data, provenance: ContextObject, id: String = id()) throws -> ContextObject {
        let (w,h) = try png(data)
        let art: ContextObject = ["id":id,"png":["sha256":hash(data),"byteCount":data.count,"width":w,"height":h,"base64":data.base64EncodedString()],"provenance":provenance]
        try artwork(art); return art
    }
}

// Reject duplicate decoded member names, excess nesting and invalid escaped surrogate pairs
// before JSONSerialization creates a dictionary (which would otherwise erase duplicates).
private struct StrictContextJSON {
    let b: [UInt8]; var i = 0
    init(data: Data) { b = Array(data) }
    mutating func whitespace() { while i < b.count && [9,10,13,32].contains(b[i]) { i += 1 } }
    mutating func check() throws { try value(0); whitespace(); guard i == b.count else { try ContextCodec.fail("Trailing JSON.") } }
    mutating func string() throws -> String {
        let start = i; i += 1
        while i < b.count {
            let c = b[i]; i += 1
            if c == 34 { return try JSONSerialization.jsonObject(with: Data(b[start..<i]), options: .fragmentsAllowed) as! String }
            if c < 32 { try ContextCodec.fail("JSON control character.") }
            if c == 92 {
                guard i < b.count else { try ContextCodec.fail("Truncated escape.") }
                let e = b[i]; i += 1
                if e == 117 {
                    let first = try hex()
                    if (0xd800...0xdbff).contains(first) { guard i+2 <= b.count, b[i] == 92, b[i+1] == 117 else { try ContextCodec.fail("Unpaired Unicode surrogate.") }; i += 2; let second = try hex(); guard (0xdc00...0xdfff).contains(second) else { try ContextCodec.fail("Unpaired Unicode surrogate.") } }
                    else if (0xdc00...0xdfff).contains(first) { try ContextCodec.fail("Unpaired Unicode surrogate.") }
                } else if ![34,92,47,98,102,110,114,116].contains(e) { try ContextCodec.fail("Invalid JSON escape.") }
            }
        }
        try ContextCodec.fail("Unterminated string.")
    }
    mutating func hex() throws -> UInt16 {
        guard i+4 <= b.count, let s = String(bytes:b[i..<i+4], encoding:.ascii), let n = UInt16(s,radix:16) else { try ContextCodec.fail("Invalid Unicode escape.") }; i += 4; return n
    }
    mutating func value(_ depth: Int) throws {
        whitespace(); guard i < b.count else { try ContextCodec.fail("JSON truncation rejected.") }
        if b[i] == 34 { _ = try string(); return }
        if b[i] == 123 || b[i] == 91 {
            guard depth < 32 else { try ContextCodec.fail("JSON container nesting rejected.") }
            let object = b[i] == 123, end: UInt8 = object ? 125 : 93; i += 1; whitespace(); var keys = Set<String>()
            if i < b.count && b[i] == end { i += 1; return }
            while true {
                if object { guard i < b.count, b[i] == 34 else { try ContextCodec.fail("Expected JSON member.") }; let key = try string(); guard keys.insert(key).inserted else { try ContextCodec.fail("Duplicate JSON member.") }; whitespace(); guard i < b.count, b[i] == 58 else { try ContextCodec.fail("Missing colon.") }; i += 1 }
                try value(depth+1); whitespace(); guard i < b.count else { try ContextCodec.fail("Truncated container.") }
                if b[i] == end { i += 1; return }; guard b[i] == 44 else { try ContextCodec.fail("Missing comma.") }; i += 1; whitespace()
            }
        }
        let start = i
        while i < b.count && ![9,10,13,32,44,93,125].contains(b[i]) { i += 1 }
        guard i > start else { try ContextCodec.fail("Invalid JSON value.") }
        _ = try JSONSerialization.jsonObject(with: Data(b[start..<i]), options:.fragmentsAllowed)
    }
}

struct ContextExport: FileDocument {
    static var readableContentTypes: [UTType] { [.json] }
    var data: Data
    init(data: Data) { self.data = data }
    init(configuration: ReadConfiguration) throws { try ContextSnapshotGate.requireAvailable(); guard let data = configuration.file.regularFileContents else { throw ContextFailure.invalid("Missing snapshot.") }; self.data = data }
    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper { try ContextSnapshotGate.requireAvailable(); return FileWrapper(regularFileWithContents: data) }
}

@MainActor
final class DesignContext: ObservableObject {
    @Published private(set) var document: ContextObject?
    @Published var status: String?
    @Published private(set) var legacyArtwork: Artwork?
    private let root: URL
    private let fm = FileManager.default
    var projectID: String? { document?["projectId"] as? String }
    var revision: String? { document?["projectRevisionId"] as? String }
    var draft: ContextObject? { document?["promptDraft"] as? ContextObject }
    var accepted: ContextObject? { document?["acceptedBrief"] as? ContextObject }
    var messages: [ContextObject] { document?["messages"] as? [ContextObject] ?? [] }
    init() { root = FileManager.default.urls(for:.documentDirectory,in:.userDomainMask)[0].appendingPathComponent("DesignContext",isDirectory:true) }
    // Portable IDs are NEVER used as path components (even '.' and '..' are valid IDs).
    private func diskID(_ id: String) -> String { ContextCodec.hash(Data(id.utf8)) }
    private func project(_ id: String) -> URL { root.appendingPathComponent(diskID(id),isDirectory:true) }
    private func manifest(_ projectID: String, _ revision: String) -> URL { project(projectID).appendingPathComponent("\(diskID(revision)).json") }
    private func read(_ url: URL) throws -> Data {
        let attributes = try fm.attributesOfItem(atPath:url.path)
        guard ((attributes[.size] as? NSNumber)?.intValue ?? Int.max) <= 25_165_824 else { try ContextCodec.fail("Snapshot exceeds file limit.") }
        return try Data(contentsOf:url)
    }
    private func pointer(_ id: String) throws -> ContextObject { try ContextCodec.object(JSONSerialization.jsonObject(with:read(project(id).appendingPathComponent("active.json")))) }
    private func loadManifest(_ url: URL) throws -> ContextObject {
        let data = try read(url); var scanner = StrictContextJSON(data:data); try scanner.check()
        var d = try ContextCodec.object(JSONSerialization.jsonObject(with:data))
        if var art = d["selectedArtwork"] as? ContextObject, var p = art["png"] as? ContextObject {
            let hash = try ContextCodec.string(p["sha256"])
            guard hash.count == 64, hash.utf8.allSatisfy({ (48...57).contains($0) || (97...102).contains($0) }), p["base64"] as? String == "" else { try ContextCodec.fail("Invalid local asset reference.") }
            p["base64"] = try read(root.appendingPathComponent("assets/\(hash).png")).base64EncodedString()
            art["png"] = p; d["selectedArtwork"] = art
        }
        try ContextCodec.validate(d); return d
    }
    private func storedMetadata(_ d: ContextObject) -> ContextObject {
        var metadata = d
        if var art = d["selectedArtwork"] as? ContextObject, var p = art["png"] as? ContextObject { p["base64"] = ""; art["png"] = p; metadata["selectedArtwork"] = art }
        return metadata
    }
    // Catalog membership comes only from activated pointer ancestry or a separate
    // atomic delayed-output receipt. Merely staging a manifest never publishes it.
    private func committedManifestURLs(_ pid: String) -> [URL] {
        var pending: [String] = [], visited = Set<String>(), result: [URL] = []
        if let p = try? pointer(pid) { pending += [p["current"] as? String,p["previous"] as? String].compactMap { $0 } }
        let receipts = project(pid).appendingPathComponent("catalog-receipts",isDirectory:true)
        for url in (try? fm.contentsOfDirectory(at:receipts,includingPropertiesForKeys:nil)) ?? [] {
            guard let data = try? read(url), let receipt = try? ContextCodec.object(JSONSerialization.jsonObject(with:data)),
                  Set(receipt.keys) == Set(["projectId","revisionId","manifestSha256"]), receipt["projectId"] as? String == pid,
                  let rev = receipt["revisionId"] as? String, let hash = receipt["manifestSha256"] as? String,
                  let manifestData = try? read(manifest(pid,rev)), ContextCodec.hash(manifestData) == hash else { continue }
            pending.append(rev)
        }
        while let rev = pending.popLast() {
            guard visited.insert(rev).inserted else { continue }
            let url = manifest(pid,rev)
            guard let d = try? loadManifest(url), d["projectId"] as? String == pid, d["projectRevisionId"] as? String == rev else { continue }
            result.append(url)
            if let parent = d["parentRevisionId"] as? String { pending.append(parent) }
        }
        return result
    }
    private func recordDelayedMembership(_ pid: String, revision: String) throws {
        let url = manifest(pid,revision), d = try loadManifest(url)
        guard d["projectId"] as? String == pid, d["projectRevisionId"] as? String == revision else { try ContextCodec.fail("Delayed result identity mismatch.") }
        try verifyAssets(d)
        let receipts = project(pid).appendingPathComponent("catalog-receipts",isDirectory:true)
        try fm.createDirectory(at:receipts,withIntermediateDirectories:true)
        let receipt: ContextObject = ["projectId":pid,"revisionId":revision,"manifestSha256":ContextCodec.hash(try read(url))]
        try ContextCodec.bytes(receipt).write(to:receipts.appendingPathComponent(diskID(revision) + ".json"),options:.atomic)
    }
    func open(_ id: String) throws -> ContextObject {
        try ContextCodec.identifier(id)
        let p = try pointer(id)
        for key in ["current","previous"] {
            if let rev = p[key] as? String {
                try ContextCodec.identifier(rev)
                if let d = try? loadManifest(manifest(id,rev)), d["projectId"] as? String == id, d["projectRevisionId"] as? String == rev,
                   (try? verifyAssets(d)) != nil { if key == "previous" { status = "Recovered previous complete saved design. Current files were preserved." }; return d }
            }
        }
        try ContextCodec.fail("Current and previous saved states are unavailable. Files were preserved. Snapshot recovery is currently blocked by the interoperability gate; choose another valid saved project or retain these files for recovery.")
    }
    private func verifyAssets(_ d: ContextObject) throws {
        if let a = d["selectedArtwork"] as? ContextObject, let p = a["png"] as? ContextObject, let hash = p["sha256"] as? String {
            let data = try read(root.appendingPathComponent("assets/\(hash).png"))
            guard ContextCodec.hash(data) == hash, data.base64EncodedString() == p["base64"] as? String else { try ContextCodec.fail("Saved PNG is missing or corrupt.") }
        }
    }
    // Immutable files are staged before a single current+previous pointer activates them.
    // No old revisions/assets are deleted; orphan staging files cannot become active.
    private func commit(_ d: ContextObject, previous: String?) throws {
        try ContextCodec.validate(d)
        let pid = d["projectId"] as! String, rev = d["projectRevisionId"] as! String
        try checkConflicts(d)
        try fm.createDirectory(at:root.appendingPathComponent("assets"),withIntermediateDirectories:true)
        try fm.createDirectory(at:project(pid),withIntermediateDirectories:true)
        let identity = project(pid).appendingPathComponent("identity.json")
        if !fm.fileExists(atPath:identity.path) { try ContextCodec.bytes(["projectId":pid]).write(to:identity,options:.atomic) }
        if let a = d["selectedArtwork"] as? ContextObject, let p = a["png"] as? ContextObject {
            let data = Data(base64Encoded:p["base64"] as! String)!, url = root.appendingPathComponent("assets/\(p["sha256"] as! String).png")
            if fm.fileExists(atPath:url.path) { guard try read(url) == data else { try ContextCodec.fail("Immutable asset conflict.") } }
            else { try data.write(to:url,options:.atomic) }
        }
        let url = manifest(pid,rev), data = try ContextCodec.bytes(storedMetadata(d))
        if fm.fileExists(atPath:url.path) { guard ContextCodec.equal(try loadManifest(url),d) else { try ContextCodec.fail("Revision identity conflict.") } }
        else { try data.write(to:url,options:.atomic) }
        try verifyAssets(d)
        let p: ContextObject = ["current":rev,"previous":previous as Any? ?? ContextCodec.null]
        try ContextCodec.bytes(p).write(to:project(pid).appendingPathComponent("active.json"),options:.atomic)
    }
    private func checkConflicts(_ d: ContextObject) throws {
        guard fm.fileExists(atPath:root.path) else { return }
        let incoming = d["selectedArtwork"] as? ContextObject
        for pid in projects() {
            for file in committedManifestURLs(pid) {
                // A damaged historical/current manifest must not defeat previous-state recovery.
                // Valid records continue to enforce immutable identity conflicts.
                guard let old = try? loadManifest(file) else { continue }
                if old["projectRevisionId"] as? String == d["projectRevisionId"] as? String, !ContextCodec.equal(old,d) { try ContextCodec.fail("Same revision has conflicting content.") }
                if let a = old["selectedArtwork"] as? ContextObject, let incoming, a["id"] as? String == incoming["id"] as? String, !ContextCodec.equal(a,incoming) { try ContextCodec.fail("Artwork identity has conflicting bytes or provenance.") }
            }
        }
    }
    func activate(_ d: ContextObject) {
        document = d
        let id = d["projectId"] as? String
        legacyArtwork = UserDefaults.standard.string(forKey:"designContextLegacyFallback") == id && d["selectedArtwork"] is NSNull ? ArtworkFactory.restore() : nil
        UserDefaults.standard.set(id,forKey:"designContextProject")
    }
    func start() throws {
        if let id = UserDefaults.standard.string(forKey:"designContextProject") {
            activate(try open(id))
            if let marker = UserDefaults.standard.string(forKey:"designContextLegacyFallback"), marker == id, document?["selectedArtwork"] is NSNull {
                legacyArtwork = ArtworkFactory.restore(); status = "Legacy artwork remains local; snapshot export is unavailable until it validates."
            }
            return
        }
        let existing = (try? fm.contentsOfDirectory(at:root,includingPropertiesForKeys:nil)) ?? []
        if !existing.isEmpty { try ContextCodec.fail("Saved projects exist without an active selection. Choose one explicitly; files were preserved.") }
        let d = ContextCodec.newProject(); try commit(d,previous:nil); activate(d)
        // Legacy association is stable and original bytes stay untouched. Unknown history is null.
        if let legacy = ArtworkFactory.restore() {
            do {
                let data = try read(legacy.exportURL)
                let art = try ContextCodec.makeArtwork(data,provenance:unknownProvenance("legacy"),id:"legacy-" + ContextCodec.hash(data))
                try mutate(expected:revision) { $0["selectedArtwork"] = art }
            } catch {
                legacyArtwork = legacy; UserDefaults.standard.set(projectID,forKey:"designContextLegacyFallback")
                status = "Legacy PNG remains available locally; snapshot adoption failed: \(error.localizedDescription)"
            }
        }
    }
    func projects() -> [String] {
        ((try? fm.contentsOfDirectory(at:root,includingPropertiesForKeys:nil)) ?? []).compactMap { directory in
            guard let data = try? read(directory.appendingPathComponent("identity.json")), let identity = try? JSONSerialization.jsonObject(with:data) as? ContextObject else { return nil }
            guard let id = identity["projectId"] as? String, fm.fileExists(atPath:project(id).appendingPathComponent("active.json").path) else { return nil }
            return id
        }.sorted()
    }
    func create() throws { let d = ContextCodec.newProject(); try commit(d,previous:nil); activate(d) }
    func switchProject(_ id: String) throws {
        activate(try open(id))
        if UserDefaults.standard.string(forKey:"designContextLegacyFallback") == id, document?["selectedArtwork"] is NSNull { legacyArtwork = ArtworkFactory.restore() }
    }
    private func catalog() -> [String: URL] {
        guard let pid = projectID else { return [:] }
        var items: [String: URL] = [:]
        for url in committedManifestURLs(pid) {
            // Catalog listing does not decode/retain every historical PNG in memory.
            if let data = try? read(url), let d = try? JSONSerialization.jsonObject(with:data) as? ContextObject,
               let art = d["selectedArtwork"] as? ContextObject, let id = art["id"] as? String { items[id] = url }
        }
        return items
    }
    func catalogIDs() -> [String] { catalog().keys.sorted() }
    func artworkAvailable(_ id: String) -> Bool { catalog()[id] != nil }
    func selectCatalog(_ id: String, expected: String?) throws {
        guard let url = catalog()[id], let art = try loadManifest(url)["selectedArtwork"] as? ContextObject else { try ContextCodec.fail("Artwork is unavailable in this local project.") }
        try mutate(expected:expected) { $0["selectedArtwork"] = art }
    }
    func mutate(expected: String?, _ body: (inout ContextObject) throws -> Void) throws {
        guard var next = document, let pid = projectID, let expected, revision == expected,
              try open(pid)["projectRevisionId"] as? String == expected else { try ContextCodec.fail("Design changed. Reload it before applying this action.") }
        next["parentRevisionId"] = expected; next["projectRevisionId"] = ContextCodec.id()
        try body(&next)
        try commit(next,previous:expected); activate(next)
    }
    func edit(_ text: String, capability: String, expected: String?) throws {
        try mutate(expected:expected) { d in
            let old = d["promptDraft"] as? ContextObject
            d["promptDraft"] = ["revisionId":ContextCodec.id(),"baseBriefRevisionId":old?["baseBriefRevisionId"] ?? (d["acceptedBrief"] as? ContextObject)?["revisionId"] ?? ContextCodec.null,"text":text,"capability":capability,"settings":old?["settings"] ?? ContextObject()]
        }
    }
    private func accept(_ d: inout ContextObject, text: String, displayedDraft: Bool) {
        let old = d["acceptedBrief"] as? ContextObject
        let oldDraft = d["promptDraft"] as? ContextObject
        let clean = old != nil && oldDraft?["text"] as? String == old?["text"] as? String && ContextCodec.equal(oldDraft?["baseBriefRevisionId"] ?? ContextCodec.null,old?["revisionId"] ?? ContextCodec.null)
        let b: ContextObject = old?["text"] as? String == text ? old! : ["revisionId":ContextCodec.id(),"text":text]
        d["acceptedBrief"] = b
        if displayedDraft || oldDraft?["text"] as? String == text || oldDraft == nil || clean {
            var draft = oldDraft ?? ["capability":ContextCodec.null,"settings":ContextObject()]
            draft["revisionId"] = ContextCodec.id(); draft["baseBriefRevisionId"] = b["revisionId"]; draft["text"] = text; d["promptDraft"] = draft
        }
    }
    func acceptBrief(_ proposal: String? = nil, expected: String?) throws {
        guard let text = proposal ?? draft?["text"] as? String else { try ContextCodec.fail("Enter a draft first.") }
        try mutate(expected:expected) { accept(&$0,text:text,displayedDraft:proposal == nil) }
    }
    func loadAccepted(expected: String?) throws {
        guard let b = accepted else { try ContextCodec.fail("No accepted brief is saved.") }
        try mutate(expected:expected) { d in d["promptDraft"] = ["revisionId":ContextCodec.id(),"baseBriefRevisionId":b["revisionId"]!,"text":b["text"]!,"capability":(d["promptDraft"] as? ContextObject)?["capability"] ?? ContextCodec.null,"settings":(d["promptDraft"] as? ContextObject)?["settings"] ?? ContextObject()] }
    }
    func append(_ text: String, reply: String, expected: String?) throws {
        try mutate(expected:expected) { d in
            var m = d["messages"] as! [ContextObject]
            for (role, source, value) in [("user","native-user",text),("assistant","native-finite-helper",reply)] { m.append(["id":ContextCodec.id(),"role":role,"source":source,"text":value,"artworkIds":[String]()]) }
            d["messages"] = m
        }
    }
    // Each button captures these IDs when rendered. A repeated/stale chip cannot mutate state.
    func chip(_ capability: String) -> ContextObject {
        ["id":ContextCodec.id(),"actionId":"native-pattern-" + capability,"projectId":projectID as Any? ?? ContextCodec.null,"contextRevisionId":revision as Any? ?? ContextCodec.null,"draftRevisionId":draft?["revisionId"] ?? ContextCodec.null,"briefRevisionId":accepted?["revisionId"] ?? ContextCodec.null]
    }
    func applyChip(_ chip: ContextObject, capability: String) throws {
        guard ContextCodec.equal(chip["projectId"]!,document?["projectId"] ?? ContextCodec.null), ContextCodec.equal(chip["contextRevisionId"]!,document?["projectRevisionId"] ?? ContextCodec.null), ContextCodec.equal(chip["draftRevisionId"]!,draft?["revisionId"] ?? ContextCodec.null), ContextCodec.equal(chip["briefRevisionId"]!,accepted?["revisionId"] ?? ContextCodec.null) else { try ContextCodec.fail("Suggestion is stale.") }
        try mutate(expected:chip["contextRevisionId"] as? String) { d in
            var effects = d["chosenSuggestionEffects"] as! [ContextObject]
            guard !effects.contains(where:{ $0["id"] as? String == chip["id"] as? String }) else { try ContextCodec.fail("Suggestion already applied.") }
            var draft = d["promptDraft"] as? ContextObject ?? ["baseBriefRevisionId":(d["acceptedBrief"] as? ContextObject)?["revisionId"] ?? ContextCodec.null,"text":"","settings":ContextObject()]
            draft["revisionId"] = ContextCodec.id(); draft["capability"] = capability; d["promptDraft"] = draft
            var effect = chip; effect["effect"] = ["type":"draft","payload":["capability":capability]]; effect["resultDraftRevisionId"] = draft["revisionId"]; effects.append(effect); d["chosenSuggestionEffects"] = effects
        }
    }
    func applyNavigationChip(_ chip: ContextObject, type: String) throws {
        guard ["clarify","import"].contains(type), ContextCodec.equal(chip["projectId"]!,document?["projectId"] ?? ContextCodec.null), ContextCodec.equal(chip["contextRevisionId"]!,document?["projectRevisionId"] ?? ContextCodec.null), ContextCodec.equal(chip["draftRevisionId"]!,draft?["revisionId"] ?? ContextCodec.null), ContextCodec.equal(chip["briefRevisionId"]!,accepted?["revisionId"] ?? ContextCodec.null) else { try ContextCodec.fail("Suggestion is stale.") }
        try mutate(expected:chip["contextRevisionId"] as? String) { d in
            var effects = d["chosenSuggestionEffects"] as! [ContextObject]
            guard !effects.contains(where:{ $0["id"] as? String == chip["id"] as? String }) else { try ContextCodec.fail("Suggestion already applied.") }
            var entry = chip
            let payload: ContextObject = type == "clarify" ? ["question":"Choose a finite native pattern explicitly, Typography to render words, or Upload an existing image. What output do you want?"] : [:]
            entry["effect"] = ["type":type,"payload":payload]; entry["resultDraftRevisionId"] = (d["promptDraft"] as? ContextObject)?["revisionId"] ?? ContextCodec.null
            effects.append(entry); d["chosenSuggestionEffects"] = effects
        }
    }
    func prepare(pattern: PatternKind, expected: String?) throws -> ContextObject {
        guard let draft, let text = draft["text"] as? String else { try ContextCodec.fail("Enter a design draft first.") }
        guard draft["capability"] as? String == pattern.contextCapability else { try ContextCodec.fail("Select a native pattern explicitly before generating. Imported recipes cannot be regenerated here.") }
        if pattern != .text {
            let words = text.lowercased()
            let allowed = ["wood","slat","wave","dot","pattern","blue","green","pink","orange","color","colour","a","the","with","and","make","design", "", "minimalist"]
            let tokens = words.components(separatedBy:CharacterSet.letters.inverted).filter { !$0.isEmpty }
            guard tokens.allSatisfy({ allowed.contains($0) || allowed.contains(String($0.dropLast())) }),
                  words.unicodeScalars.allSatisfy({ CharacterSet.letters.contains($0) || CharacterSet.whitespacesAndNewlines.contains($0) || CharacterSet.punctuationCharacters.contains($0) }) else { try ContextCodec.fail("This text is outside the finite native pattern vocabulary. Save it as a brief, import an image, or explicitly select Typography to render the words.") }
        }
        guard ContextCodec.equal(draft["baseBriefRevisionId"] ?? ContextCodec.null,accepted?["revisionId"] ?? ContextCodec.null) else { try ContextCodec.fail("Draft is stale. Choose Save brief or Load accepted brief from Generate’s context menu.") }
        var attempt: ContextObject = [:]
        try mutate(expected:expected) { d in
            accept(&d,text:text,displayedDraft:true)
            let accent = text.lowercased().contains("blue") ? "blue" : text.lowercased().contains("green") ? "green" : text.lowercased().contains("pink") ? "pink" : "orange"
            var request: ContextObject = ["pattern":pattern.contextCapability,"width":768,"height":960]
            if pattern != .slats { request["accent"] = accent }; if pattern == .text { request["text"] = text.isEmpty ? "MAKE\nSOMETHING\nYOURS." : text }
            attempt = ["id":ContextCodec.id(),"projectId":d["projectId"]!,"contextRevisionId":d["projectRevisionId"]!,"brief":d["acceptedBrief"]!,"prompt":text,"capability":pattern.contextCapability,"effectiveRequest":request]
            d["generationAttempt"] = attempt
        }
        return attempt
    }
    func finish(_ generated: Artwork, attempt: ContextObject) throws {
        let pid = attempt["projectId"] as! String, expected = attempt["contextRevisionId"] as! String
        var origin = try open(pid)
        let data = try read(generated.exportURL)
        let art = try ContextCodec.makeArtwork(data,provenance:["kind":"generated","producer":["app":"VisualizeIT iOS","version":"0.1.0","capability":attempt["capability"]!],"brief":attempt["brief"]!,"prompt":attempt["prompt"]!,"effectiveRequest":attempt["effectiveRequest"]!,"source":ContextCodec.null])
        // Only locally generated results may use the native 3D slat preview. Imported
        // producer claims never authorize replacing their PNG with procedural geometry.
        let previews = root.appendingPathComponent("previews",isDirectory:true)
        try fm.createDirectory(at:previews,withIntermediateDirectories:true)
        try ContextCodec.bytes(["capability":attempt["capability"]!,"sha256":ContextCodec.hash(data)]).write(to:previews.appendingPathComponent(diskID(art["id"] as! String) + ".json"),options:.atomic)
        if projectID == pid, origin["projectRevisionId"] as? String == expected {
            try mutate(expected:expected) { $0["selectedArtwork"] = art }
        } else {
            // Retain delayed valid output in an inert local catalog revision; never activate it.
            origin["parentRevisionId"] = origin["projectRevisionId"]; origin["projectRevisionId"] = ContextCodec.id(); origin["selectedArtwork"] = art
            try ContextCodec.validate(origin); try checkConflicts(origin)
            try fm.createDirectory(at:project(pid),withIntermediateDirectories:true)
            let p = art["png"] as! ContextObject
            try data.write(to:root.appendingPathComponent("assets/\(p["sha256"] as! String).png"),options:.atomic)
            try ContextCodec.bytes(storedMetadata(origin)).write(to:manifest(pid,origin["projectRevisionId"] as! String),options:.atomic)
            try recordDelayedMembership(pid,revision:origin["projectRevisionId"] as! String)
            status = "Result retained in its originating project catalog; select it deliberately."
        }
    }
    func importPhoto(_ generated: Artwork, project: String?, expected: String?) throws {
        guard projectID == project else { try ContextCodec.fail("Project changed during Photos import. Current design was preserved.") }
        let art = try ContextCodec.makeArtwork(read(generated.exportURL),provenance:unknownProvenance("imported"))
        try mutate(expected:expected) { $0["selectedArtwork"] = art }
    }
    func unknownProvenance(_ kind: String) -> ContextObject { ["kind":kind,"producer":ContextCodec.null,"brief":ContextCodec.null,"prompt":ContextCodec.null,"effectiveRequest":ContextCodec.null,"source":ContextCodec.null] }
    func selected() throws -> Artwork? {
        guard let art = document?["selectedArtwork"] as? ContextObject, let p = art["png"] as? ContextObject else { return nil }
        let id = art["id"] as! String, url = root.appendingPathComponent("assets/\(p["sha256"] as! String).png")
        guard let image = UIImage(data:try read(url)) else { try ContextCodec.fail("Saved image cannot open.") }
        let previewURL = root.appendingPathComponent("previews/\(diskID(id)).json")
        let preview = (try? read(previewURL)).flatMap { try? JSONSerialization.jsonObject(with:$0) as? ContextObject }
        let pattern = preview?["sha256"] as? String == p["sha256"] as? String ? PatternKind.allCases.first { $0.contextCapability == preview?["capability"] as? String } : nil
        return Artwork(id:id,image:image,title:pattern?.rawValue ?? "Imported design",exportURL:url,patternKind:pattern)
    }
    func readImport(_ url: URL) throws -> ContextObject {
        try ContextSnapshotGate.requireAvailable()
        let access = url.startAccessingSecurityScopedResource(); defer { if access { url.stopAccessingSecurityScopedResource() } }
        let d = try ContextCodec.decode(read(url)); try checkConflicts(d); return d
    }
    func importSnapshot(_ incoming: ContextObject, choice: String, expectedProject: String?, expectedRevision: String?) throws {
        try ContextSnapshotGate.requireAvailable()
        guard choice != "cancel" else { return }
        guard projectID == expectedProject, revision == expectedRevision else { try ContextCodec.fail("Project changed while choosing import. Try again.") }
        var d = incoming; let pid = d["projectId"] as! String
        let exists = fm.fileExists(atPath:project(pid).appendingPathComponent("active.json").path)
        if exists, let old = try? open(pid), old["projectRevisionId"] as? String == d["projectRevisionId"] as? String {
            guard ContextCodec.equal(old,d) else { try ContextCodec.fail("Revision conflict.") }; activate(old); return
        }
        var previous: String?
        if choice == "copy" {
            d["origin"] = ["projectId":pid,"revisionId":d["projectRevisionId"]!]; d["projectId"] = ContextCodec.id(); d["projectRevisionId"] = ContextCodec.id(); d["parentRevisionId"] = ContextCodec.null
        } else if exists {
            // If both local states are corrupt an explicit valid import can recover without
            // deleting those files. A valid previous state remains the new recovery pointer.
            previous = (try? open(pid))?["projectRevisionId"] as? String
        }
        try commit(d,previous:previous); activate(d)
    }
    func export() throws -> Data {
        try ContextSnapshotGate.requireAvailable()
        guard legacyArtwork == nil else { try ContextCodec.fail("This legacy PNG exceeds or failed snapshot validation. It remains usable locally; export was refused without dropping or resizing it.") }
        guard let document else { try ContextCodec.fail("No saved project.") }; try ContextCodec.validate(document); try verifyAssets(document); return try ContextCodec.bytes(document)
    }
}

extension PatternKind {
    var contextCapability: String { switch self { case .slats: return "native.slats"; case .waves:return "native.waves"; case .dots:return "native.dots"; case .text:return "native.text" } }
}
