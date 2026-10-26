import SwiftUI
import PhotosUI
import ImageIO
import SceneKit

enum PatternKind: String, CaseIterable, Identifiable {
    case slats = "Wood slats", waves = "Color waves", dots = "Dot pattern", text = "Typography"
    var id: String { rawValue }
}

struct Artwork: Identifiable {
    let id: String
    let image: UIImage
    let title: String
    let exportURL: URL
    let patternKind: PatternKind?
    init(id: String = UUID().uuidString, image: UIImage, title: String, exportURL: URL, patternKind: PatternKind?) {
        self.id = id; self.image = image; self.title = title; self.exportURL = exportURL; self.patternKind = patternKind
    }
}

@MainActor
enum ArtworkFactory {
    static func generate(prompt: String, pattern: PatternKind) throws -> Artwork {
        let size = CGSize(width: 768, height: 960)
        let format = UIGraphicsImageRendererFormat()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: size, format: format)
        let words = prompt.lowercased()
        let accent: UIColor = words.contains("blue") ? .systemBlue : words.contains("green") ? .systemGreen : words.contains("pink") ? .systemPink : .systemOrange
        let image = renderer.image { output in
            let context = output.cgContext
            context.scaleBy(x: 0.8, y: 0.8)
            UIColor(white: 0.06, alpha: 1).setFill()
            context.fill(CGRect(x: 0, y: 0, width: 960, height: 1200))
            switch pattern {
            case .slats:
                for index in 0..<18 {
                    let x = CGFloat(index) * 55
                    let colors = [UIColor(red: 0.28, green: 0.12, blue: 0.055, alpha: 1).cgColor,
                                  UIColor(red: 0.82, green: 0.57, blue: 0.32, alpha: 1).cgColor,
                                  UIColor(red: 0.4, green: 0.22, blue: 0.1, alpha: 1).cgColor] as CFArray
                    if let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0, 0.3, 1]) {
                        context.saveGState()
                        context.clip(to: CGRect(x: x, y: 0, width: 38, height: 1200))
                        context.drawLinearGradient(gradient, start: CGPoint(x: x, y: 0), end: CGPoint(x: x + 38, y: 0), options: [])
                        context.restoreGState()
                    }
                }
            case .waves:
                for index in 0..<16 {
                    let path = UIBezierPath()
                    let y = CGFloat(index) * 90 - 150
                    path.move(to: CGPoint(x: -100, y: y))
                    path.addCurve(to: CGPoint(x: 1060, y: y + 280), controlPoint1: CGPoint(x: 300, y: y - 180), controlPoint2: CGPoint(x: 500, y: y + 600))
                    path.lineWidth = 72
                    accent.withAlphaComponent(CGFloat(index % 3 + 1) / 3).setStroke()
                    path.stroke()
                }
            case .dots:
                for row in 0..<13 { for column in 0..<10 {
                    accent.withAlphaComponent(Double((row + column) % 3 + 1) / 3).setFill()
                    UIBezierPath(ovalIn: CGRect(x: CGFloat(column * 100 + 18), y: CGFloat(row * 100 + 18), width: 62, height: 62)).fill()
                } }
            case .text:
                let content = prompt.isEmpty ? "MAKE\nSOMETHING\nYOURS." : prompt
                let paragraph = NSMutableParagraphStyle()
                paragraph.alignment = .center
                (content as NSString).draw(in: CGRect(x: 80, y: 230, width: 800, height: 820), withAttributes: [.font: UIFont.systemFont(ofSize: 84, weight: .bold), .foregroundColor: accent, .paragraphStyle: paragraph])
            }
        }
        return try save(image: image, title: pattern.rawValue, pattern: pattern)
    }

    static func imported(_ data: Data) throws -> Artwork {
        guard data.count <= 32 * 1024 * 1024,
              let source = CGImageSourceCreateWithData(data as CFData, nil),
              let thumbnail = CGImageSourceCreateThumbnailAtIndex(source, 0, [kCGImageSourceCreateThumbnailFromImageAlways: true,
                  kCGImageSourceCreateThumbnailWithTransform: true, kCGImageSourceThumbnailMaxPixelSize: 1024] as CFDictionary) else {
            throw ArtworkError.invalidImage
        }
        return try save(image: UIImage(cgImage: thumbnail), title: "Imported design")
    }

    private static func save(image: UIImage, title: String, pattern: PatternKind? = nil) throws -> Artwork {
        guard let data = image.pngData() else { throw ArtworkError.invalidImage }
        let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Artwork", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("design-\(UUID().uuidString).png")
        try data.write(to: url, options: .atomic)
        // DesignContext commits selection only after PNG/provenance persistence succeeds.
        // Keep legacy defaults exclusively for adopting pre-context installations.
        return Artwork(image: image, title: title, exportURL: url, patternKind: pattern)
    }

    static func restore() -> Artwork? {
        guard let name = UserDefaults.standard.string(forKey: "lastArtwork"), name == URL(fileURLWithPath:name).lastPathComponent else { return nil }
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("Artwork").appendingPathComponent(name)
        guard let image = UIImage(contentsOfFile: url.path) else { return nil }
        let pattern = UserDefaults.standard.string(forKey: "lastArtworkPattern").flatMap(PatternKind.init(rawValue:))
        return Artwork(id:"legacy-file-" + ContextCodec.hash(Data(name.utf8)), image: image, title: pattern?.rawValue ?? "Saved design", exportURL: url, patternKind: pattern)
    }

    enum ArtworkError: LocalizedError {
        case invalidImage
        var errorDescription: String? { "Choose a supported image smaller than 32 MB." }
    }
}

@MainActor
struct ArtworkStage: View {
    let artwork: Artwork
    let reduceMotion: Bool
    @State private var translation = CGSize.zero
    @State private var scale: CGFloat = 1
    @State private var angle: Angle = .zero
    @State private var appeared = false
    @GestureState private var drag = CGSize.zero
    @GestureState private var magnification: CGFloat = 1
    @GestureState private var rotation: Angle = .zero

    var body: some View {
        VStack(spacing: 16) {
            if artwork.patternKind == .slats {
                SlattedPanelStage(reduceMotion: reduceMotion).frame(width: 300, height: 390)
            } else {
            Image(uiImage: artwork.image).resizable().scaledToFit()
                .frame(maxWidth: 235, maxHeight: 335)
                .clipShape(RoundedRectangle(cornerRadius: 3))
                .shadow(color: .pink.opacity(0.45), radius: 24, y: 20)
                .rotation3DEffect(.degrees(appeared ? -12 : (reduceMotion ? -12 : 150)), axis: (x: 0, y: 1, z: 0))
                .rotationEffect(angle + rotation)
                .scaleEffect(max(0.25, min(3, scale * magnification)))
                .offset(x: translation.width + drag.width, y: translation.height + drag.height + (appeared ? 0 : (reduceMotion ? 0 : 500)))
                .gesture(DragGesture().updating($drag) { value, state, _ in state = value.translation }.onEnded { value in
                    translation.width += value.translation.width; translation.height += value.translation.height
                })
                .simultaneousGesture(MagnificationGesture().updating($magnification) { value, state, _ in state = value }.onEnded { scale = max(0.25, min(3, scale * $0)) })
                .simultaneousGesture(RotationGesture().updating($rotation) { value, state, _ in state = value }.onEnded { angle += $0 })
                .accessibilityLabel("\(artwork.title), movable artwork preview")
                .accessibilityHint("Adjust to resize. Custom actions move or reset the artwork. Long press opens artwork controls.")
                .accessibilityAdjustableAction { direction in
                    switch direction {
                    case .increment: scale = min(3, scale + 0.15)
                    case .decrement: scale = max(0.25, scale - 0.15)
                    @unknown default: break
                    }
                }
                .accessibilityAction(named: "Move left") { translation.width -= 20 }
                .accessibilityAction(named: "Move right") { translation.width += 20 }
                .accessibilityAction(named: "Move up") { translation.height -= 20 }
                .accessibilityAction(named: "Move down") { translation.height += 20 }
                .accessibilityAction(named: "Reset artwork") { translation = .zero; scale = 1; angle = .zero }
                .contextMenu {
                    Button("Smaller", systemImage: "minus.magnifyingglass") { scale = max(0.25, scale - 0.15) }
                    Button("Larger", systemImage: "plus.magnifyingglass") { scale = min(3, scale + 0.15) }
                    Button("Rotate", systemImage: "rotate.right") { angle += .degrees(15) }
                    Button("Reset", systemImage: "arrow.counterclockwise") { translation = .zero; scale = 1; angle = .zero }
                }
            }
        }
        .onAppear { withAnimation(reduceMotion ? nil : .spring(response: 0.85, dampingFraction: 0.8)) { appeared = true } }
    }
}

/// Finite native preview geometry. This is not reconstruction or a tracked attachment.
@MainActor
private final class SlattedPanelModel: NSObject, ObservableObject {
    let scene = SCNScene()
    let panel = SCNNode()
    let camera = SCNNode()
    private var entered = false
    private var panOrigin = SCNVector3Zero
    private var moveOrigin = SCNVector3Zero
    private var pinchOrigin: Float = 1
    private var twistOrigin: Float = 0

    override init() {
        super.init()
        scene.background.contents = UIColor.clear
        camera.camera = SCNCamera()
        camera.camera?.usesOrthographicProjection = true
        camera.camera?.orthographicScale = 3.1
        camera.camera?.zNear = 0.01
        camera.camera?.zFar = 40
        camera.position = SCNVector3(0, 1.5, 8)
        camera.look(at: SCNVector3(0, -0.1, 0))
        scene.rootNode.addChildNode(camera)
        scene.rootNode.addChildNode(panel)
        let backing = SCNBox(width: 3.30, height: 4.55, length: 0.055, chamferRadius: 0.005)
        backing.firstMaterial?.diffuse.contents = UIColor(red: 0.075, green: 0.035, blue: 0.02, alpha: 1)
        backing.firstMaterial?.roughness.contents = 0.9
        let back = SCNNode(geometry: backing)
        back.position.z = -0.10
        panel.addChildNode(back)
        for index in 0..<18 {
            let box = SCNBox(width: 0.14, height: 4.55, length: 0.22, chamferRadius: 0.008)
            let face = SCNMaterial()
            face.lightingModel = .physicallyBased
            face.diffuse.contents = Self.grain(index: index)
            face.roughness.contents = 0.62
            face.metalness.contents = 0.0
            let edge = SCNMaterial()
            edge.lightingModel = .physicallyBased
            edge.diffuse.contents = UIColor(red: 0.30, green: 0.15, blue: 0.065, alpha: 1)
            edge.roughness.contents = 0.78
            box.materials = [face, edge, edge, edge, face, edge]
            let slat = SCNNode(geometry: box)
            slat.position.x = (Float(index) - 8.5) * 0.185
            panel.addChildNode(slat)
        }
        addLight(type: .ambient, position: SCNVector3Zero, color: UIColor(red: 0.78, green: 0.73, blue: 0.83, alpha: 1), intensity: 280)
        addLight(type: .omni, position: SCNVector3(-3, 5, 6), color: UIColor(red: 1, green: 0.87, blue: 0.68, alpha: 1), intensity: 1000)
        addLight(type: .omni, position: SCNVector3(-3, 0.8, -0.8), color: .systemPink, intensity: 900)
        addLight(type: .omni, position: SCNVector3(3, 0, -0.5), color: .systemPurple, intensity: 850)
        let shadow = SCNPlane(width: 4.5, height: 2.3)
        let material = SCNMaterial()
        material.lightingModel = .constant
        material.diffuse.contents = Self.shadowTexture()
        material.isDoubleSided = true
        material.writesToDepthBuffer = false
        shadow.firstMaterial = material
        let floor = SCNNode(geometry: shadow)
        floor.eulerAngles.x = -.pi / 2
        floor.position = SCNVector3(0, -2.55, 0)
        scene.rootNode.addChildNode(floor)
        reset()
    }

    private func addLight(type: SCNLight.LightType, position: SCNVector3, color: UIColor, intensity: CGFloat) {
        let node = SCNNode()
        node.light = SCNLight()
        node.light?.type = type
        node.light?.color = color
        node.light?.intensity = intensity
        node.position = position
        scene.rootNode.addChildNode(node)
    }

    private static func grain(index: Int) -> UIImage {
        let format = UIGraphicsImageRendererFormat(); format.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: 96, height: 512), format: format).image { renderer in
            let context = renderer.cgContext
            UIColor(red: 0.64 + CGFloat(index % 4) * 0.025, green: 0.39 + CGFloat(index % 3) * 0.02, blue: 0.20, alpha: 1).setFill()
            context.fill(CGRect(x: 0, y: 0, width: 96, height: 512))
            for line in 0..<65 {
                let path = UIBezierPath()
                let x = CGFloat(line) * 1.6
                path.move(to: CGPoint(x: x, y: 0))
                path.addCurve(to: CGPoint(x: x + 2, y: 512), controlPoint1: CGPoint(x: x + CGFloat((line + index) % 7), y: 170), controlPoint2: CGPoint(x: x - 3, y: 330))
                path.lineWidth = line % 4 == 0 ? 0.8 : 0.35
                UIColor(red: 0.25, green: 0.10, blue: 0.035, alpha: line % 4 == 0 ? 0.28 : 0.12).setStroke()
                path.stroke()
            }
        }
    }

    private static func shadowTexture() -> UIImage {
        let format = UIGraphicsImageRendererFormat(); format.scale = 1
        return UIGraphicsImageRenderer(size: CGSize(width: 256, height: 256), format: format).image { renderer in
            let colors = [UIColor.black.withAlphaComponent(0.32).cgColor,
                          UIColor.systemPink.withAlphaComponent(0.21).cgColor,
                          UIColor.systemPurple.withAlphaComponent(0.10).cgColor, UIColor.clear.cgColor] as CFArray
            if let gradient = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: colors, locations: [0, 0.4, 0.7, 1]) {
                renderer.cgContext.drawRadialGradient(gradient, startCenter: CGPoint(x: 128, y: 128), startRadius: 0, endCenter: CGPoint(x: 128, y: 128), endRadius: 125, options: [])
            }
        }
    }

    func enter(reduceMotion: Bool) {
        guard !entered else { return }
        entered = true
        guard !reduceMotion else { return }
        panel.position.y = -5
        panel.eulerAngles.y = -2.7
        let rise = SCNAction.move(to: SCNVector3Zero, duration: 0.85)
        let turn = SCNAction.rotateTo(x: 0.015, y: -0.24, z: 0.055, duration: 0.85, usesShortestUnitArc: true)
        rise.timingMode = .easeOut; turn.timingMode = .easeOut
        panel.runAction(.group([rise, turn]), forKey: "entrance")
    }

    func reset() { panel.removeAllActions(); panel.position = SCNVector3Zero; panel.scale = SCNVector3(1, 1, 1); panel.eulerAngles = SCNVector3(0.015, -0.24, 0.055) }
    func resize(_ factor: Float) { let value = min(2, max(0.4, panel.scale.x * factor)); panel.scale = SCNVector3(value, value, value) }
    func rotateStep() { panel.removeAllActions(); panel.eulerAngles.y += .pi / 12 }
    @objc func pan(_ gesture: UIPanGestureRecognizer) {
        guard let view = gesture.view else { return }
        if gesture.state == .began { panel.removeAllActions(); panOrigin = panel.eulerAngles; moveOrigin = panel.position }
        let delta = gesture.translation(in: view)
        if gesture.numberOfTouches > 1 {
            panel.position = SCNVector3(moveOrigin.x + Float(delta.x) * 0.012, moveOrigin.y - Float(delta.y) * 0.012, moveOrigin.z)
        } else {
            panel.eulerAngles.x = min(0.7, max(-0.7, panOrigin.x + Float(delta.y) * 0.008))
            panel.eulerAngles.y = panOrigin.y + Float(delta.x) * 0.008
        }
    }
    @objc func pinch(_ gesture: UIPinchGestureRecognizer) {
        if gesture.state == .began { pinchOrigin = panel.scale.x }
        let value = min(2, max(0.4, pinchOrigin * Float(gesture.scale)))
        panel.scale = SCNVector3(value, value, value)
    }
    @objc func twist(_ gesture: UIRotationGestureRecognizer) {
        if gesture.state == .began { twistOrigin = panel.eulerAngles.z }
        panel.eulerAngles.z = twistOrigin - Float(gesture.rotation)
    }
}

@MainActor
private struct SlattedPanelStage: View {
    let reduceMotion: Bool
    @StateObject private var model = SlattedPanelModel()
    @Environment(\.scenePhase) private var phase
    var body: some View {
        SlattedPanelViewport(model: model, active: phase == .active)
            .onAppear { model.enter(reduceMotion: reduceMotion) }
            .accessibilityLabel("Wood slat panel, finite 3D preview")
            .accessibilityHint("Drag to turn, use two fingers to move, pinch to resize. Additional controls are available in the context menu.")
            .accessibilityAdjustableAction { direction in if direction == .increment { model.resize(1.1) } else if direction == .decrement { model.resize(0.9) } }
            .accessibilityAction(named: "Rotate panel") { model.rotateStep() }
            .accessibilityAction(named: "Reset panel") { model.reset() }
            .contextMenu {
                Button("Smaller", systemImage: "minus.magnifyingglass") { model.resize(0.9) }
                Button("Larger", systemImage: "plus.magnifyingglass") { model.resize(1.1) }
                Button("Rotate", systemImage: "rotate.right") { model.rotateStep() }
                Button("Reset", systemImage: "arrow.counterclockwise") { model.reset() }
            }
    }
}

@MainActor
private struct SlattedPanelViewport: UIViewRepresentable {
    let model: SlattedPanelModel
    let active: Bool
    func makeUIView(context: Context) -> SCNView {
        let view = SCNView()
        view.backgroundColor = .clear
        view.isOpaque = false
        view.scene = model.scene
        view.pointOfView = model.camera
        view.autoenablesDefaultLighting = false
        view.antialiasingMode = .multisampling4X
        view.preferredFramesPerSecond = 30
        view.addGestureRecognizer(UIPanGestureRecognizer(target: model, action: #selector(SlattedPanelModel.pan(_:))))
        view.addGestureRecognizer(UIPinchGestureRecognizer(target: model, action: #selector(SlattedPanelModel.pinch(_:))))
        view.addGestureRecognizer(UIRotationGestureRecognizer(target: model, action: #selector(SlattedPanelModel.twist(_:))))
        return view
    }
    func updateUIView(_ view: SCNView, context: Context) { view.isPlaying = active }
    static func dismantleUIView(_ view: SCNView, coordinator: ()) { view.isPlaying = false; view.scene = nil }
}
