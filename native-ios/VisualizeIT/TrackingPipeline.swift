import Foundation
import CryptoKit
import SwiftUI
import ImageIO
import Network

private let requestedModelHead = "27e06a513713232f4ae542d50b8364a3bb319669451448e694bcd335a3fb9311"

/// This milestone admits a local computer only. Parsing uses Network's IP types,
/// with explicit ranges; a public HTTPS hostname does not imply local consent.
private enum LocalOperator {
    static func allows(host: String) -> Bool {
        var name = host.lowercased()
        if name.hasPrefix("["), name.hasSuffix("]") { name.removeFirst(); name.removeLast() }
        if let address = IPv4Address(name) { return allows(ipv4: address, presentation: name) }
        if let address = IPv6Address(name) {
            if let mapped = address.asIPv4 { return allows(ipv4: mapped) }
            let bytes = Array(address.rawValue)
            return address.isLoopback || address.isLinkLocal || (bytes.count == 16 && bytes[0] & 0xfe == 0xfc)
        }
        if name.hasSuffix(".") { name.removeLast() }
        guard name.count <= 253 else { return false }
        let labels = name.split(separator: ".", omittingEmptySubsequences: false)
        let valid = labels.allSatisfy { label in
            guard !label.isEmpty, label.count <= 63, label.first != "-", label.last != "-" else { return false }
            return label.utf8.allSatisfy { (97...122).contains($0) || (48...57).contains($0) || $0 == 45 }
        }
        return valid && (name == "localhost" || (labels.count >= 2 && labels.last == "local"))
    }

    private static func allows(ipv4: IPv4Address, presentation: String? = nil) -> Bool {
        let bytes = Array(ipv4.rawValue)
        guard bytes.count == 4 else { return false }
        // Reject ambiguous shorthand / leading-zero spellings even if a parser accepts them.
        if let presentation, presentation != bytes.map({ String($0) }).joined(separator: ".") { return false }
        return ipv4.isLoopback || ipv4.isLinkLocal || bytes[0] == 10
            || (bytes[0] == 172 && (16...31).contains(bytes[1])) || (bytes[0] == 192 && bytes[1] == 168)
    }
}

struct NativeCapabilities: Decodable {
    let schema_version: Int
    let model_head: String
    let installed_head_sha256: String?
    let admission_ready: Bool
    let live_camera_supported: Bool
    let blockers: [String]
}

struct CameraCalibration: Encodable {
    let calibration_revision: String
    let width: Int
    let height: Int
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let units = "pixels"
    let orientation: String
    let mirrored = false
    let crop: Crop
    struct Crop: Encodable { let x: Int; let y: Int; let width: Int; let height: Int }
}

private struct SessionRequest: Encodable {
    let schema_version = 1
    let expected_model_head = requestedModelHead
    let camera: CameraCalibration
    let asset_id: String
    let onboarding: Onboarding
    struct Onboarding: Encodable {
        let selection_tap: Tap
        let artwork_revision: String
    }
    struct Tap: Encodable { let x: Double; let y: Double }
}

private struct NativeSession: Decodable {
    let schema_version: Int
    let session_id: String
    let generation: Int
    let state: String
    let calibration_revision: String
    let artwork_revision: String
    let model_head: String
    let installed_head_sha256: String?
}

private struct FrameRequest: Encodable {
    let schema_version = 1
    let generation: Int
    let sequence: Int
    let timestamp_ns: Int64
    let jpeg_base64: String
    let jpeg_sha256: String
    let calibration_revision: String
    let artwork_revision: String
}

struct NativeFrameResponse: Decodable {
    let schema_version: Int
    let session_id: String
    let generation: Int
    let frame_id: String
    let sequence: Int
    let timestamp_ns: Int64
    let state: String
    let calibration_revision: String
    let artwork_revision: String
    let model_head: String
    let rgb_sha256: String
    let mask_sha256: String
    let outline_segments: [[[Double]]]
    let render: Render?
    let installed_head_sha256: String?
    let pose: Pose?
    let reason: String?
    struct Render: Decodable {
        let frame_id: String
        let rgb_sha256: String
        let artwork_revision: String
        let rgba_png_base64: String
    }
    struct Pose: Decodable {
        let object_to_camera: [[Double]]
        let units: String
        let camera_axes: String
        var isValid: Bool {
            object_to_camera.count == 4 && object_to_camera.allSatisfy { $0.count == 4 && $0.allSatisfy(\.isFinite) }
            && units == "metres" && camera_axes == "opencv_x_right_y_down_z_forward"
            && abs(object_to_camera[3][0]) < 0.000001 && abs(object_to_camera[3][1]) < 0.000001
            && abs(object_to_camera[3][2]) < 0.000001 && abs(object_to_camera[3][3] - 1) < 0.000001
        }
    }
}

struct ProcessedResult {
    let frame: CapturedFrame
    let response: NativeFrameResponse
    let renderedArtwork: Data?
}

/// Redirects are rejected: bearer credentials and consent never follow an operator redirect.
private final class NoRedirect: NSObject, URLSessionTaskDelegate {
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) { completionHandler(nil) }
}

@MainActor
final class TrackingPipeline: ObservableObject {
    @Published var operatorAddress = ""
    @Published var token = "" // In memory only; never logged or persisted.
    @Published var assetID = ""
    @Published var uploadConsent = false {
        didSet { if !uploadConsent { disconnect() } }
    }
    @Published private(set) var status = "Disconnected. Camera stays on this iPhone."
    @Published private(set) var busy = false
    @Published private(set) var admitted = false
    @Published private(set) var capabilities: NativeCapabilities?
    @Published private(set) var processedResult: ProcessedResult?
    var frozenFrame: CapturedFrame? { processedResult?.frame }
    var response: NativeFrameResponse? { processedResult?.response }
    @Published private(set) var selection: CGPoint?
    // No fallback intrinsics, guessed focal lengths or client-entered calibration.
    // A calibrated capture provider must set this only after device-level verification.
    private var calibration: CameraCalibration?
    private var currentSession: NativeSession?
    private var operatorURL: URL?
    private var epoch = UUID()
    private var sequence = 0
    private weak var activeCamera: CameraController?
    private var task: Task<Void, Never>?
    private var captureTimeout: Task<Void, Never>?
    private let redirectDelegate = NoRedirect()
    private lazy var transport: URLSession = {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.timeoutIntervalForRequest = 20
        configuration.timeoutIntervalForResource = 30
        configuration.httpCookieStorage = nil
        configuration.urlCache = nil
        return URLSession(configuration: configuration, delegate: redirectDelegate, delegateQueue: nil)
    }()

    func select(_ point: CGPoint) {
        selection = point
        disconnect()
        status = "Selection point recorded. Object selection is unavailable until calibrated capture and an observed-mask provider are connected."
    }

    func connect(camera: CameraController, artworkRevision: String) {
        disconnect()
        guard uploadConsent else { status = "Enable temporary frame upload consent first."; return }
        guard let url = URL(string: operatorAddress.trimmingCharacters(in: .whitespacesAndNewlines)),
              url.scheme?.lowercased() == "https", let host = url.host, !host.isEmpty,
              LocalOperator.allows(host: host),
              url.user == nil, url.password == nil, url.query == nil, url.fragment == nil,
              url.path.isEmpty || url.path == "/" else {
            status = "Enter a trusted local HTTPS origin: private, link-local or loopback IP, localhost, or a .local name. Public/cloud hosts, paths and URL credentials are unavailable."
            return
        }
        guard !token.isEmpty else { status = "Enter the operator's bearer token. It stays in memory."; return }
        operatorURL = url
        calibration = camera.calibration
        activeCamera = camera
        let generation = epoch
        busy = true
        status = "Checking tracking capabilities…"
        task = Task {
            defer { if epoch == generation, !admitted { busy = false } }
            do {
                let capability: NativeCapabilities = try await request(path: "v1/native/capabilities", method: "GET")
                guard epoch == generation, !Task.isCancelled, uploadConsent else { return }
                capabilities = capability
                guard capability.schema_version == 1, capability.model_head == requestedModelHead else { throw PipelineError.message("Operator model or schema does not match the pinned native contract.") }
                guard capability.installed_head_sha256 == nil || capability.installed_head_sha256 == requestedModelHead else { throw PipelineError.message("Operator installed head conflicts with the pinned native model.") }
                guard capability.admission_ready, capability.live_camera_supported else {
                    status = "Tracking blocked: " + (capability.blockers.isEmpty ? "verified live admission is unavailable." : capability.blockers.joined(separator: "; "))
                    return
                }
                guard let calibration else {
                    status = "Tracking blocked: camera intrinsic matrix delivery is unavailable. No camera frames were uploaded."
                    return
                }
                guard let selection, !assetID.isEmpty else { throw PipelineError.message("Select the object and configure an operator-admitted asset first.") }
                let body = SessionRequest(camera: calibration, asset_id: assetID,
                    onboarding: .init(selection_tap: .init(x: min(Double(calibration.width) - 0.000001, max(0, Double(selection.x) * Double(calibration.width))),
                        y: min(Double(calibration.height) - 0.000001, max(0, Double(selection.y) * Double(calibration.height)))), artwork_revision: artworkRevision))
                let session: NativeSession = try await request(path: "v1/native/sessions", method: "POST", body: JSONEncoder().encode(body))
                guard epoch == generation, !Task.isCancelled, uploadConsent else {
                    await release(session: session, origin: url, bearer: token)
                    return
                }
                currentSession = session
                guard session.schema_version == 1, session.model_head == requestedModelHead,
                      session.calibration_revision == calibration.calibration_revision,
                      session.artwork_revision == artworkRevision, session.generation > 0,
                      session.installed_head_sha256 == nil || session.installed_head_sha256 == requestedModelHead else { throw PipelineError.message("Operator returned a mismatched session.") }
                admitted = true
                status = "Admitted. Streaming one exact-frame request at a time."
                busy = false
                inspectFrame(camera: camera)
            } catch is CancellationError { }
            catch { if epoch == generation { disconnect(); status = error.localizedDescription } }
        }
    }

    func inspectFrame(camera: CameraController) {
        guard admitted, uploadConsent, !busy, currentSession != nil else { return }
        busy = true
        let generation = epoch
        captureTimeout = Task {
            try? await Task.sleep(for: .seconds(5))
            guard !Task.isCancelled, epoch == generation else { return }
            disconnect(); status = "Camera did not deliver a calibrated frame. Reconnect after camera recovery."
        }
        camera.requestFrame { [weak self] frame in
            guard let self, self.epoch == generation, self.admitted, self.uploadConsent else { return }
            self.send(frame, generation: generation)
        }
    }

    private func send(_ frame: CapturedFrame, generation: UUID) {
        captureTimeout?.cancel()
        captureTimeout = nil
        guard let session = currentSession, max(frame.width, frame.height) <= 640 else { busy = false; return }
        guard frame.calibrationRevision == session.calibration_revision,
              calibration?.width == frame.width, calibration?.height == frame.height else {
            disconnect(); status = "Camera calibration changed. Reconnect to admit a new session."; return
        }
        sequence += 1
        let requestSequence = sequence
        let payload = FrameRequest(generation: session.generation, sequence: requestSequence, timestamp_ns: frame.timestampNanoseconds,
            jpeg_base64: frame.jpeg.base64EncodedString(), jpeg_sha256: SHA256.hash(data: frame.jpeg).map { String(format: "%02x", $0) }.joined(),
            calibration_revision: session.calibration_revision, artwork_revision: session.artwork_revision)
        task = Task {
            defer {
                if epoch == generation {
                    busy = false
                    if admitted, uploadConsent, let camera = activeCamera {
                        // Next capture happens only after completion; no frame backlog.
                        inspectFrame(camera: camera)
                    }
                }
            }
            do {
                let reply: NativeFrameResponse = try await request(path: "v1/native/sessions/\(session.session_id)/frames", method: "POST", body: JSONEncoder().encode(payload))
                guard epoch == generation, !Task.isCancelled, uploadConsent, admitted else { return }
                guard reply.schema_version == 1, reply.session_id == session.session_id,
                      reply.generation == session.generation, reply.sequence == requestSequence,
                      reply.timestamp_ns == frame.timestampNanoseconds,
                      reply.calibration_revision == session.calibration_revision,
                      reply.artwork_revision == session.artwork_revision, reply.model_head == requestedModelHead,
                      reply.frame_id == "\(session.session_id):\(session.generation):\(requestSequence)",
                      ["recovering", "tracking", "lost", "unavailable"].contains(reply.state) else { throw PipelineError.message("Discarded a response for a different captured frame or invalid state.") }
                if reply.state == "tracking" {
                    guard reply.installed_head_sha256 == requestedModelHead, reply.pose?.isValid == true else { throw PipelineError.message("Discarded unverified tracking provenance or an invalid pose.") }
                } else if reply.pose != nil { throw PipelineError.message("Discarded a pose outside the tracking state.") }
                guard reply.outline_segments.count <= 32, reply.outline_segments.reduce(0, { $0 + $1.count }) <= 2048,
                      reply.outline_segments.allSatisfy({ segment in segment.count <= 512 && segment.allSatisfy { point in
                          point.count == 2 && point.allSatisfy { $0.isFinite && $0 >= 0 && $0 <= 1 }
                      } }) else { throw PipelineError.message("Discarded an invalid observed foreground contour.") }
                var rendered: Data?
                if let render = reply.render {
                    guard reply.state == "tracking", render.frame_id == reply.frame_id,
                          render.rgb_sha256 == reply.rgb_sha256, render.artwork_revision == session.artwork_revision,
                          let bytes = Data(base64Encoded: render.rgba_png_base64),
                          let source = CGImageSourceCreateWithData(bytes as CFData, nil),
                          let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
                          properties[kCGImagePropertyPixelWidth] as? Int == frame.width,
                          properties[kCGImagePropertyPixelHeight] as? Int == frame.height else { throw PipelineError.message("Discarded an artwork render for a different frame.") }
                    rendered = bytes
                }
                // Results are shown with this exact processed JPEG, never over live video.
                processedResult = ProcessedResult(frame: frame, response: reply, renderedArtwork: rendered)
                status = reply.reason ?? "Frame \(reply.sequence): \(reply.state). \(rendered == nil ? "Artwork surface rendering is unavailable." : "Registered artwork render received.")"
            } catch is CancellationError { }
            catch { if epoch == generation { disconnect(); status = error.localizedDescription } }
        }
    }

    /// Changing design projects never carries an unrelated object's onboarding.
    func resetObjectAssociation() {
        disconnect()
        selection = nil
        assetID = ""
        status = "Choose this project's object and operator asset before tracking."
    }

    func disconnect(clearCredentials: Bool = false) {
        let oldSession = currentSession
        let oldOrigin = operatorURL
        let oldToken = token
        epoch = UUID()
        task?.cancel()
        captureTimeout?.cancel()
        captureTimeout = nil
        task = nil
        admitted = false
        busy = false
        processedResult = nil
        currentSession = nil
        activeCamera = nil
        capabilities = nil
        sequence = 0
        status = "Disconnected. Camera stays on this iPhone."
        if clearCredentials { token = ""; uploadConsent = false }
        if let oldSession, let oldOrigin { Task { await release(session: oldSession, origin: oldOrigin, bearer: oldToken) } }
    }

    private func release(session: NativeSession, origin: URL, bearer: String) async {
        var request = URLRequest(url: origin.appendingPathComponent("v1/native/sessions/\(session.session_id)"))
        request.httpMethod = "DELETE"
        request.setValue("Bearer \(bearer)", forHTTPHeaderField: "Authorization")
        _ = try? await transport.data(for: request)
    }

    private func request<T: Decodable>(path: String, method: String, body: Data? = nil) async throws -> T {
        guard let operatorURL else { throw PipelineError.message("Configure a trusted operator first.") }
        var request = URLRequest(url: operatorURL.appendingPathComponent(path))
        request.httpMethod = method
        request.httpBody = body
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (stream, rawResponse) = try await transport.bytes(for: request)
        var data = Data()
        for try await byte in stream {
            guard data.count < 6 * 1024 * 1024 else { throw PipelineError.message("Operator response exceeded the native limit.") }
            data.append(byte)
        }
        try Task.checkCancellation()
        guard let response = rawResponse as? HTTPURLResponse, (200..<300).contains(response.statusCode) else {
            let code = (rawResponse as? HTTPURLResponse)?.statusCode ?? 0
            throw PipelineError.message("Tracking operator rejected the request (HTTP \(code)). Review operator admission and configuration.")
        }
        return try JSONDecoder().decode(T.self, from: data)
    }

    enum PipelineError: LocalizedError {
        case message(String)
        var errorDescription: String? { if case .message(let value) = self { return value }; return nil }
    }
}
