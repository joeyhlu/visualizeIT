import SwiftUI
import PhotosUI
import AVFoundation

private let spectrum = [Color(red: 0.43, green: 0.24, blue: 1), .purple, .pink, Color(red: 1, green: 0.62, blue: 0.36)]

struct Glass: ViewModifier {
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    var radius: CGFloat = 26
    func body(content: Content) -> some View {
        content.background {
            if reduceTransparency { RoundedRectangle(cornerRadius: radius).fill(Color(white: 0.16)) }
            else { RoundedRectangle(cornerRadius: radius).fill(.ultraThinMaterial).overlay(RoundedRectangle(cornerRadius: radius).fill(.white.opacity(0.07))) }
        }
        .overlay(RoundedRectangle(cornerRadius: radius).strokeBorder(.white.opacity(0.5), lineWidth: 0.8))
    }
}

struct StarMark: View {
    var size: CGFloat = 28
    var body: some View {
        Image(systemName: "sparkle").font(.system(size: size, weight: .medium))
            .foregroundStyle(LinearGradient(colors: spectrum, startPoint: .bottomLeading, endPoint: .topTrailing))
            .shadow(color: .pink.opacity(0.65), radius: 10).accessibilityHidden(true)
    }
}

private struct RoundControl: View {
    let icon: String
    let label: String
    let action: () -> Void
    var body: some View {
        Button(action: action) { Image(systemName: icon).font(.system(size: 23, weight: .light)).frame(width: 44, height: 44).modifier(Glass(radius: 50)) }
            .foregroundStyle(.white).accessibilityLabel(label)
    }
}

private struct GradientAction: View {
    let title: String
    let icon: String
    let action: () -> Void
    var body: some View {
        Button(action: action) {
            Label(title, systemImage: icon).font(.title3.weight(.medium)).frame(maxWidth: .infinity).frame(minHeight: 54)
                .background(LinearGradient(colors: spectrum, startPoint: .leading, endPoint: .trailing), in: Capsule())
                .overlay(Capsule().strokeBorder(.white.opacity(0.8), lineWidth: 1))
                .shadow(color: .pink.opacity(0.4), radius: 15)
        }.foregroundStyle(.white)
    }
}

private struct AnalyzePressStyle: ButtonStyle {
    let reduceMotion: Bool
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .scaleEffect(configuration.isPressed && !reduceMotion ? 1.06 : 1)
            .shadow(color: .pink.opacity(configuration.isPressed ? 0.8 : 0), radius: 16)
            .animation(reduceMotion ? nil : .easeOut(duration: 0.18), value: configuration.isPressed)
    }
}

private struct ChatMessage: Identifiable {
    let id = UUID()
    let text: String
    let isUser: Bool
}

@MainActor
struct StudioView: View {
    private enum Panel { case home, chat, chooser, generate, result }
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @StateObject private var camera = CameraController()
    @StateObject private var tracking = TrackingPipeline()
    @State private var panel: Panel = .home
    @State private var settings = false
    @State private var prompt = ""
    @State private var pattern: PatternKind = .slats
    @State private var artwork: Artwork?
    @State private var pickerItem: PhotosPickerItem?
    @State private var importing = false
    @State private var notice: String?
    @State private var chatInput = ""
    @State private var messages = [ChatMessage(text: "Hi! I’m VisualizeIT.\nTell me about your design. I can help you choose a local pattern, color, or typography. Object image generation is not connected yet.", isUser: false)]

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            CameraPreview(camera: camera) { tracking.select($0); notice = tracking.status }
                .ignoresSafeArea().accessibilityLabel("Live rear camera. Tap an object to record an onboarding point.")
            if let result = tracking.processedResult {
                ProcessedCameraView(result: result).ignoresSafeArea().allowsHitTesting(false)
                VStack { Spacer(); Text("Processed frame \(result.response.sequence) · \(result.response.state) · \(result.renderedArtwork == nil ? "surface artwork unavailable" : "registered artwork")").font(.caption).padding(10).modifier(Glass()).padding(.bottom, 160) }.allowsHitTesting(false)
            }
            if camera.state != .running { Color.black.ignoresSafeArea() }
            LinearGradient(colors: [.black.opacity(0.16), .clear, .black.opacity(0.12)], startPoint: .top, endPoint: .bottom).ignoresSafeArea().allowsHitTesting(false)

            VStack(spacing: 0) {
                header.padding(.horizontal, 17).padding(.top, 7)
                if panel == .chat { chat.padding(.top, 25) }
                else {
                    Spacer(minLength: 12)
                    if let artwork, !tracking.admitted, panel == .home || panel == .result {
                        ArtworkStage(artwork: artwork, reduceMotion: reduceMotion).id(artwork.id).padding(.horizontal, 30)
                    } else if camera.state != .running { cameraMessage.padding(30) }
                    Spacer(minLength: 16)
                    if let notice { noticeCard(notice).padding(.horizontal, 24).padding(.bottom, 16) }
                    switch panel {
                    case .chooser: chooser
                    case .generate: generator
                    case .result: result
                    default: navigation.padding(.bottom, 26)
                    }
                }
            }
            rainbowRim
            if importing { ProgressView("Importing design…").padding(24).modifier(Glass()).accessibilityAddTraits(.updatesFrequently) }
        }
        .foregroundStyle(.white)
        .sheet(isPresented: $settings) { TrackingSettings(pipeline: tracking, camera: camera, artworkRevision: artwork?.id.uuidString ?? "none", pattern: $pattern, artwork: artwork) }
        .task { artwork = ArtworkFactory.restore(); camera.start() }
        .task(id: pickerItem) {
            guard let pickerItem else { return }
            importing = true
            defer { importing = false }
            do {
                guard let data = try await pickerItem.loadTransferable(type: Data.self) else { throw ArtworkFactory.ArtworkError.invalidImage }
                try Task.checkCancellation()
                let imported = try ArtworkFactory.imported(data)
                tracking.disconnect()
                artwork = imported
                panel = .result
                notice = nil
            } catch is CancellationError { }
            catch { notice = error.localizedDescription }
        }
        .onChange(of: scenePhase) { _, phase in
            if phase == .active { camera.start() }
            else { camera.stop(); tracking.disconnect(clearCredentials: phase == .background) }
        }
        .onChange(of: camera.state) { _, state in if state != .running && state != .starting { tracking.disconnect() } }
        .onReceive(NotificationCenter.default.publisher(for: AVCaptureSession.wasInterruptedNotification, object: camera.session)) { _ in
            tracking.disconnect(); camera.stop(); notice = "Camera interrupted. Resume when the camera becomes available."
        }
        .onReceive(NotificationCenter.default.publisher(for: AVCaptureSession.runtimeErrorNotification, object: camera.session)) { _ in
            tracking.disconnect(); camera.stop(); notice = "Camera stopped after a system error. Tap Resume camera to retry."
        }
        .onDisappear { camera.stop(); tracking.disconnect(clearCredentials: true) }
    }

    private var header: some View {
        HStack {
            RoundControl(icon: panel == .chat ? "chevron.left" : "text.bubble", label: panel == .chat ? "Back to camera" : "Open design chat") {
                panel = panel == .chat ? .home : .chat; notice = nil
            }
            Spacer(minLength: 6)
            HStack(spacing: 9) { StarMark(size: 29); Text("VisualizeIT").font(.system(.body, design: .default).weight(.medium)) }.accessibilityElement(children: .combine)
            Spacer(minLength: 6)
            RoundControl(icon: "gearshape", label: "Settings and tracking pipeline") { settings = true }
        }
    }

    private var cameraMessage: some View {
        VStack(spacing: 16) {
            switch camera.state {
            case .starting: ProgressView(); Text("Starting camera…").font(.callout)
            case .denied:
                Image(systemName: "camera.fill").font(.largeTitle)
                Text("Allow camera access").font(.title2.weight(.medium))
                Text("Open Settings to enable your rear camera. You can create and import designs while access is off.").multilineTextAlignment(.center).foregroundStyle(.secondary)
                Button("Open Settings") { if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) } }.buttonStyle(.borderedProminent).tint(.purple)
            case .unavailable(let reason): Text(reason).multilineTextAlignment(.center)
            case .stopped: Text("Camera paused").font(.title3); Button("Resume camera") { camera.start() }.buttonStyle(.bordered)
            case .running: EmptyView()
            }
        }.padding(24).modifier(Glass())
    }

    private var navigation: some View {
        HStack(alignment: .center) {
            bottomButton(icon: "text.bubble", title: "Ask") { panel = .chat; notice = nil }
            Spacer()
            Button {
                notice = tracking.selection == nil ? "Tap an object to record its selection point. Object analysis needs calibrated capture and an observed-mask provider." : tracking.status
            } label: {
                VStack(spacing: 9) {
                    Circle().fill(reduceTransparency ? AnyShapeStyle(Color(white: 0.2)) : AnyShapeStyle(.ultraThinMaterial))
                        .overlay(Circle().fill(LinearGradient(colors: [.purple.opacity(0.5), .pink.opacity(0.35), .orange.opacity(0.45)], startPoint: .topLeading, endPoint: .bottomTrailing)))
                        .overlay(Circle().stroke(.white.opacity(0.9), lineWidth: 1.4).padding(6))
                        .overlay(Circle().stroke(AngularGradient(colors: spectrum + [spectrum[0]], center: .center), lineWidth: 3))
                        .shadow(color: .pink.opacity(0.85), radius: 13)
                        .frame(width: 70.2, height: 70.2)
                    Text("Analyze").font(.subheadline)
                }
            }.buttonStyle(AnalyzePressStyle(reduceMotion: reduceMotion)).accessibilityHint("Shows selection and analysis availability. Tap an object in the camera to record its onboarding point.")
            Spacer()
            bottomButton(icon: "pencil.line", title: "Design") { panel = .chooser; notice = nil }
        }.padding(.horizontal, 52)
    }

    private func bottomButton(icon: String, title: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            VStack(spacing: 9) { Image(systemName: icon).font(.system(size: 27, weight: .light)).frame(width: 54, height: 54).modifier(Glass(radius: 50)); Text(title).font(.subheadline) }
        }.accessibilityLabel(title)
    }

    private var chooser: some View {
        VStack(alignment: .leading, spacing: 22) {
            Capsule().fill(.white).frame(width: 44, height: 5).frame(maxWidth: .infinity)
            HStack { VStack(alignment: .leading, spacing: 4) { Text("Add a design").font(.title.bold()); Text("Choose how you want to create.").font(.subheadline) }; Spacer(); RoundControl(icon: "xmark", label: "Close design chooser") { panel = .home } }
            HStack(spacing: 12) {
                GradientAction(title: "Generate", icon: "sparkles") { panel = .generate }
                uploadButton
            }.padding(.bottom, 26)
        }.padding(24).modifier(Glass(radius: 38)).overlay(RoundedRectangle(cornerRadius: 38).stroke(LinearGradient(colors: spectrum, startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1.2)).padding(.horizontal, 3)
    }

    private var generator: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                HStack(spacing: 14) { StarMark(size: 34); Text("Generate design").font(.title2.weight(.semibold)); Spacer(); RoundControl(icon: "xmark", label: "Close generator") { panel = .home } }
                VStack(alignment: .trailing, spacing: 3) {
                    TextEditor(text: $prompt).scrollContentBackground(.hidden).frame(minHeight: 90, maxHeight: 130)
                        .overlay(alignment: .topLeading) { if prompt.isEmpty { Text("Describe the design you want to see in this space…").foregroundStyle(.white.opacity(0.65)).padding(8).allowsHitTesting(false) } }
                        .accessibilityLabel("Design description, up to 500 characters")
                        .onChange(of: prompt) { _, value in if value.count > 500 { prompt = String(value.prefix(500)) } }
                    Text("\(prompt.count)/500").font(.caption).foregroundStyle(.white.opacity(0.7))
                }.padding(12).modifier(Glass(radius: 22))
                Text("Prompt ideas").font(.headline)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) { ForEach(["A minimalist chair", "A ceramic vase", "A pair of sneakers", "Headphones", "A plant"], id: \.self) { idea in
                        Button { prompt = idea } label: {
                            Text(idea).font(.subheadline).frame(width: 76, alignment: .leading).padding(.horizontal, 12).frame(minHeight: 52).modifier(Glass(radius: 18))
                                .overlay(RoundedRectangle(cornerRadius: 18).stroke(prompt == idea ? Color.pink : Color.clear, lineWidth: 2))
                        }.accessibilityAddTraits(prompt == idea ? .isSelected : [])
                    } }
                }
                GradientAction(title: "Generate image", icon: "sparkles") { generate() }
            }.padding(24)
        }.frame(maxHeight: 430).modifier(Glass(radius: 34))
            .overlay(RoundedRectangle(cornerRadius: 34).stroke(LinearGradient(colors: spectrum, startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1.2))
            .padding(.horizontal, 5).padding(.bottom, 8)
    }

    private var result: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 14) { StarMark(size: 32); Text(artwork?.title == "Imported design" ? "Design imported" : "Design generated").font(.title2.weight(.medium)); Spacer(); RoundControl(icon: "xmark", label: "Close result panel") { panel = .home } }
            Text("You can move, resize, or generate a new design.").font(.subheadline).foregroundStyle(.white.opacity(0.8))
            HStack(spacing: 12) { GradientAction(title: "Regenerate", icon: "sparkles") { panel = .generate }; uploadButton }
        }.padding(24).modifier(Glass(radius: 36)).padding(.horizontal, 3)
    }

    private var uploadButton: some View {
        PhotosPicker(selection: $pickerItem, matching: .images, photoLibrary: .shared()) {
            Label("Upload", systemImage: "square.and.arrow.up").font(.title3.weight(.medium)).frame(maxWidth: .infinity, minHeight: 54).modifier(Glass(radius: 50))
        }.accessibilityHint("Imports an image locally from Photos. No network upload.")
    }

    private var chat: some View {
        VStack(spacing: 12) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 12) {
                        ForEach(messages) { message in
                            HStack(alignment: .top, spacing: 9) {
                                if message.isUser { Spacer(minLength: 42) }
                                else { StarMark(size: 24).frame(width: 34, height: 40).modifier(Glass(radius: 30)) }
                                Text(message.text).font(.body).lineSpacing(3).padding(16)
                                    .background { if message.isUser { RoundedRectangle(cornerRadius: 24).fill(LinearGradient(colors: spectrum.map { $0.opacity(0.65) }, startPoint: .leading, endPoint: .trailing)) } }
                                    .modifier(Glass(radius: 24))
                                if !message.isUser { Spacer(minLength: 20) }
                            }.id(message.id)
                        }
                    }.padding(.horizontal, 18)
                }.onChange(of: messages.count) { _, _ in if let id = messages.last?.id { withAnimation(reduceMotion ? nil : .easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .bottom) } } }
            }
            HStack(spacing: 12) {
                StarMark(size: 25)
                TextField("Tell me more about your idea…", text: $chatInput, axis: .vertical).lineLimit(1...4).submitLabel(.send).onSubmit(sendMessage).accessibilityLabel("Message local design helper")
                    .onChange(of: chatInput) { _, value in if value.count > 500 { chatInput = String(value.prefix(500)) } }
                Button(action: sendMessage) { Image(systemName: "paperplane").font(.title2).frame(width: 44, height: 44).background(LinearGradient(colors: spectrum, startPoint: .bottomLeading, endPoint: .topTrailing), in: Circle()).overlay(Circle().stroke(.white.opacity(0.8))) }.accessibilityLabel("Send message").disabled(chatInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }.padding(14).modifier(Glass(radius: 28)).padding(.horizontal, 14).padding(.bottom, 8)
        }
    }

    private func noticeCard(_ value: String) -> some View {
        HStack(alignment: .top) { Text(value).font(.callout); Button { notice = nil } label: { Image(systemName: "xmark").frame(width: 44, height: 44) }.accessibilityLabel("Dismiss message") }
            .padding(14).modifier(Glass(radius: 20)).accessibilityElement(children: .contain)
    }

    private var rainbowRim: some View {
        RoundedRectangle(cornerRadius: 29).stroke(AngularGradient(colors: [.pink, .orange, .purple, .pink, .orange, .pink], center: .center), lineWidth: 2)
            .shadow(color: .pink.opacity(0.8), radius: 8).padding(2).ignoresSafeArea().allowsHitTesting(false).accessibilityHidden(true)
    }

    private func generate() {
        let unsupported = ["chair", "vase", "sneaker", "headphone", "plant", "lamp"]
        if pattern != .text && unsupported.contains(where: { prompt.lowercased().contains($0) }) {
            notice = "Object image generation is not connected. Import an object image, or describe wood slats, color waves, dots or typography to create a local design."
            return
        }
        do { tracking.disconnect(); artwork = try ArtworkFactory.generate(prompt: prompt, pattern: pattern); panel = .result; notice = nil }
        catch { notice = error.localizedDescription }
    }

    private func sendMessage() {
        let text = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        messages.append(ChatMessage(text: text, isUser: true))
        chatInput = ""
        let lower = text.lowercased()
        let reply: String
        if lower.contains("wood") || lower.contains("slat") { pattern = .slats; reply = "Wood slats are available as a local pattern. Tap Design, then Generate to preview it. You can drag, resize, rotate and export the result." }
        else if lower.contains("text") || lower.contains("word") { pattern = .text; reply = "Typography is selected. Enter the words you want in Generate design. You can change the local pattern in Settings." }
        else { reply = "Try Wood slats, Color waves, Dot pattern, or Typography. Blue, green and pink descriptions change the accent color. For an object concept, import an image using Design → Upload." }
        messages.append(ChatMessage(text: reply, isUser: false))
        if messages.count > 30 { messages.removeFirst(messages.count - 30) }
    }
}

private struct ProcessedCameraView: View {
    let result: ProcessedResult
    var body: some View {
        GeometryReader { geometry in
            let rawWidth = CGFloat(result.frame.width)
            let rawHeight = CGFloat(result.frame.height)
            let scale = max(geometry.size.width / rawHeight, geometry.size.height / rawWidth)
            let width = rawHeight * scale
            let height = rawWidth * scale
            let origin = CGPoint(x: (geometry.size.width - width) / 2, y: (geometry.size.height - height) / 2)
            ZStack(alignment: .topLeading) {
                if let image = UIImage(data: result.frame.jpeg)?.cgImage {
                    Image(uiImage: UIImage(cgImage: image, scale: 1, orientation: .right)).resizable().frame(width: width, height: height).offset(x: origin.x, y: origin.y)
                }
                if let bytes = result.renderedArtwork, let image = UIImage(data: bytes)?.cgImage {
                    Image(uiImage: UIImage(cgImage: image, scale: 1, orientation: .right)).resizable().frame(width: width, height: height).offset(x: origin.x, y: origin.y)
                }
                Path { path in
                    for segment in result.response.outline_segments {
                        for (index, point) in segment.enumerated() {
                            // Raw x/y normalized contour -> portrait 90-degree clockwise raster.
                            let transformed = CGPoint(x: origin.x + ((rawHeight - 1) * (1 - point[1]) + 0.5) * scale,
                                y: origin.y + ((rawWidth - 1) * point[0] + 0.5) * scale)
                            if index == 0 { path.move(to: transformed) } else { path.addLine(to: transformed) }
                        }
                    }
                }.stroke(LinearGradient(colors: [.mint, .cyan, .purple, .pink], startPoint: .topLeading, endPoint: .bottomTrailing), style: StrokeStyle(lineWidth: 2, lineCap: .round, lineJoin: .round))
            }.frame(width: geometry.size.width, height: geometry.size.height).clipped()
        }.accessibilityLabel("Processed camera image with current observed foreground contour. \(result.response.state)")
    }
}

@MainActor
private struct TrackingSettings: View {
    @ObservedObject var pipeline: TrackingPipeline
    let camera: CameraController
    let artworkRevision: String
    @Binding var pattern: PatternKind
    let artwork: Artwork?
    @Environment(\.dismiss) private var dismiss
    var body: some View {
        NavigationStack {
            Form {
                Section("Local design preview") {
                    Picker("Pattern", selection: $pattern) { ForEach(PatternKind.allCases) { Text($0.rawValue).tag($0) } }
                    Text("Generate image creates a finite local pattern. Wood slats use an 18-slat native 3D preview; other patterns and imports use flat artwork. This is preview geometry, not reconstruction or a tracked attachment. General object image generation is unavailable.").font(.caption)
                    if let artwork { ShareLink(item: artwork.exportURL) { Label("Export design PNG", systemImage: "square.and.arrow.up") } }
                }
                Section("Tracking pipeline") {
                    Text("Use your trusted local computer only. When admitted and consented, captured camera frames leave this iPhone for that computer. Public/cloud operators are unavailable in this milestone.").font(.callout)
                    TextField("Trusted local HTTPS origin", text: $pipeline.operatorAddress).textContentType(.URL).keyboardType(.URL).textInputAutocapitalization(.never).autocorrectionDisabled()
                    SecureField("Bearer token (memory only)", text: $pipeline.token).textInputAutocapitalization(.never).autocorrectionDisabled()
                    TextField("Operator-admitted asset ID", text: $pipeline.assetID).textInputAutocapitalization(.never).autocorrectionDisabled()
                    Toggle("Allow temporary camera frame uploads", isOn: $pipeline.uploadConsent)
                    Text("Consent resets on backgrounding. No frames are uploaded before a configured live session passes admission. Artwork import and generation stay local.").font(.caption)
                    Button("Check and connect") { pipeline.connect(camera: camera, artworkRevision: artworkRevision) }.disabled(pipeline.busy)
                    Button("Disconnect", role: .destructive) { pipeline.disconnect() }
                    if pipeline.busy { ProgressView() }
                    Text(pipeline.status).font(.callout).textSelection(.enabled)
                }
                Section("Onboarding") {
                    Text(pipeline.selection == nil ? "Tap the object in the camera to record its selection point." : "Selection point recorded.")
                    Text(camera.calibration == nil ? "Waiting for camera-provided intrinsic calibration. Tracking stays blocked if the device cannot deliver it." : "Camera intrinsics received for the raw VGA frame. A trusted operator must provide observed foreground masks; a tap is not a mask.").font(.caption)
                    Button("Inspect one admitted frame") { pipeline.inspectFrame(camera: camera) }.disabled(!pipeline.admitted || pipeline.busy || !pipeline.uploadConsent)
                }
                if let frame = pipeline.frozenFrame, let image = UIImage(data: frame.jpeg) {
                    Section("Latest processed frame") {
                        Image(uiImage: image).resizable().scaledToFit()
                        Text("Frame \(pipeline.response?.sequence ?? 0): \(pipeline.response?.state ?? "unavailable")").font(.caption)
                        Text("This response belongs only to the image above. The camera continues capturing; results replace the image only as a matching pair. Surface artwork rendering is not integrated.").font(.caption)
                    }
                }
                Section("Privacy and availability") {
                    Text("Rear camera preview is local. No microphone access. Imported images and generated designs are saved in this app’s Documents folder. Export uses the system share sheet.")
                    Text("Native iOS source preview. Compilation and physical-device validation are pending. Android native app is pending.").font(.caption)
                }
            }.navigationTitle("Settings").navigationBarTitleDisplayMode(.inline)
                .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
        }.onChange(of: pipeline.operatorAddress) { _, _ in pipeline.disconnect() }
            .onChange(of: pipeline.token) { _, _ in pipeline.disconnect() }
            .onChange(of: pipeline.assetID) { _, _ in pipeline.disconnect() }
    }
}
