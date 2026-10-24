import AVFoundation
import CoreImage
import SwiftUI
import simd
import CryptoKit

struct CapturedFrame: Sendable {
    let id: UUID
    let timestampNanoseconds: Int64
    let jpeg: Data
    let width: Int
    let height: Int
    let calibrationRevision: String
}

/// Session, lifecycle intent and output callbacks are confined to `queue`.
/// Only immutable JPEG bytes cross to the main actor; no pixel buffers escape.
final class CameraController: NSObject, ObservableObject, @unchecked Sendable {
    enum State: Equatable { case starting, running, denied, unavailable(String), stopped }
    @Published private(set) var state: State = .starting
    @Published private(set) var calibration: CameraCalibration?
    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "app.visualizeit.camera")
    private let context = CIContext()
    private var configured = false
    private var wantsRunning = false
    private var captureNext = false
    private var publishedCalibrationRevision: String?
    private var frameHandler: (@MainActor @Sendable (CapturedFrame) -> Void)?

    func start() {
        queue.async { self.wantsRunning = true }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: queue.async { self.configureAndStart() }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] allowed in
                guard let self else { return }
                if allowed { self.queue.async { self.configureAndStart() } }
                else { self.publish(.denied) }
            }
        default: publish(.denied)
        }
    }

    func stop() {
        queue.async {
            self.wantsRunning = false
            self.captureNext = false
            self.frameHandler = nil
            self.publishedCalibrationRevision = nil
            if self.session.isRunning { self.session.stopRunning() }
            self.publish(.stopped)
            DispatchQueue.main.async { self.calibration = nil }
        }
    }

    /// Requested only by the admitted, consented pipeline. One pending capture maximum.
    func requestFrame(_ handler: @escaping @MainActor @Sendable (CapturedFrame) -> Void) {
        queue.async {
            guard self.wantsRunning, self.session.isRunning, !self.captureNext else { return }
            self.frameHandler = handler
            self.captureNext = true
        }
    }

    private func publish(_ newState: State) {
        DispatchQueue.main.async { self.state = newState }
    }

    private func configureAndStart() {
        guard wantsRunning else { return }
        if !configured {
            session.beginConfiguration()
            session.sessionPreset = .vga640x480
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) else {
                session.commitConfiguration()
                publish(.unavailable("A rear camera is unavailable. You can still create or import a design."))
                return
            }
            session.addInput(input)
            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
            guard session.canAddOutput(output) else {
                session.removeInput(input)
                session.commitConfiguration()
                publish(.unavailable("Camera output could not be configured."))
                return
            }
            session.addOutput(output)
            output.setSampleBufferDelegate(self, queue: queue)
            if let connection = output.connection(with: .video) {
                // Raw sensor landscape bytes: no rotation, crop or rescale of K.
                if connection.isVideoStabilizationSupported { connection.preferredVideoStabilizationMode = .off }
                if connection.isVideoMirroringSupported { connection.automaticallyAdjustsVideoMirroring = false; connection.isVideoMirrored = false }
                if connection.isCameraIntrinsicMatrixDeliverySupported { connection.isCameraIntrinsicMatrixDeliveryEnabled = true }
            }
            session.commitConfiguration()
            configured = true
        }
        session.startRunning()
        publish(.running)
    }
}

extension CameraController: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard wantsRunning, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        guard max(width, height) <= 640,
              let intrinsicData = CMGetAttachment(sampleBuffer, key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix, attachmentModeOut: nil) as? Data,
              intrinsicData.count == MemoryLayout<simd_float3x3>.size else {
            if captureNext { captureNext = false; frameHandler = nil }
            return
        }
        let matrix = intrinsicData.withUnsafeBytes { $0.loadUnaligned(as: simd_float3x3.self) }
        let values = [matrix.columns.0.x, matrix.columns.1.y, matrix.columns.2.x, matrix.columns.2.y]
        guard values.allSatisfy(\.isFinite), values[0] > 0, values[1] > 0,
              abs(matrix.columns.0.y) < 0.000001, abs(matrix.columns.0.z) < 0.000001,
              abs(matrix.columns.1.x) < 0.000001, abs(matrix.columns.1.z) < 0.000001,
              abs(matrix.columns.2.z - 1) < 0.000001,
              values[2] >= 0, values[2] < Float(width), values[3] >= 0, values[3] < Float(height) else { return }
        // Hash meaningful values, not SIMD padding bytes which need not be stable.
        let identity = Data(("\(width)x\(height):" + values.map { String($0) }.joined(separator: ":")).utf8)
        let revision = SHA256.hash(data: identity).map { String(format: "%02x", $0) }.joined() + "-\(width)x\(height)"
        let measured = CameraCalibration(calibration_revision: revision, width: width, height: height,
            fx: Double(values[0]), fy: Double(values[1]), cx: Double(values[2]), cy: Double(values[3]),
            orientation: "landscape_right", crop: .init(x: 0, y: 0, width: width, height: height))
        if publishedCalibrationRevision != revision {
            publishedCalibrationRevision = revision
            DispatchQueue.main.async { self.calibration = measured }
        }
        guard captureNext else { return }
        captureNext = false
        let handler = frameHandler
        frameHandler = nil
        let image = CIImage(cvPixelBuffer: buffer)
        let extent = image.extent
        guard let bytes = context.jpegRepresentation(of: image, colorSpace: CGColorSpaceCreateDeviceRGB(), options: [:]) else { return }
        let time = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)
        let frame = CapturedFrame(id: UUID(), timestampNanoseconds: Int64(CMTimeGetSeconds(time) * 1_000_000_000), jpeg: bytes,
                                  width: Int(extent.width), height: Int(extent.height), calibrationRevision: revision)
        Task { @MainActor in handler?(frame) }
    }
}

final class PreviewSurface: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    var onSelection: ((CGPoint) -> Void)?
    override init(frame: CGRect) {
        super.init(frame: frame)
        addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(selectPoint(_:))))
    }
    required init?(coder: NSCoder) { fatalError("Programmatic view only") }
    @objc private func selectPoint(_ gesture: UITapGestureRecognizer) {
        // AVFoundation converts aspect-fill / crop coordinates into the capture-device space.
        onSelection?(previewLayer.captureDevicePointConverted(fromLayerPoint: gesture.location(in: self)))
    }
    override func layoutSubviews() {
        super.layoutSubviews()
        if let connection = previewLayer.connection, connection.isVideoRotationAngleSupported(90) { connection.videoRotationAngle = 90 }
    }
}

struct CameraPreview: UIViewRepresentable {
    let camera: CameraController
    let onSelection: (CGPoint) -> Void
    func makeUIView(context: Context) -> PreviewSurface {
        let view = PreviewSurface()
        view.previewLayer.videoGravity = .resizeAspectFill
        view.previewLayer.session = camera.session
        view.onSelection = onSelection
        return view
    }
    func updateUIView(_ view: PreviewSurface, context: Context) { view.onSelection = onSelection }
}
