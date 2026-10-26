import SwiftUI
import PhotosUI
import AVFoundation
import UniformTypeIdentifiers

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
    let id: String
    let text: String
    let isUser: Bool
    let artworkIDs: [String]
    init(id: String = UUID().uuidString, text: String, isUser: Bool, artworkIDs: [String] = []) { self.id = id; self.text = text; self.isUser = isUser; self.artworkIDs = artworkIDs }
}

@MainActor
struct StudioView: View {
    private enum Panel { case home, chat, chooser, generate, result }
    @Environment(\.scenePhase) private var scenePhase
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @Environment(\.accessibilityReduceTransparency) private var reduceTransparency
    @StateObject private var camera = CameraController()
    @StateObject private var tracking = TrackingPipeline()
    @StateObject private var context = DesignContext()
    @State private var panel: Panel = .home
    @State private var settings = false
    @State private var prompt = ""
    @State private var pattern: PatternKind = .slats
    @State private var artwork: Artwork?
    @State private var pickerItem: PhotosPickerItem?
    @State private var importing = false
    @State private var notice: String?
    @State private var chatInput = ""
    @State private var contextError: String?
    @State private var snapshotPicker = false
    @State private var chipPhotoPicker = false
    @State private var snapshotExporter = false
    @State private var exportDisclosure = false
    @State private var importChoice = false
    @State private var pendingSnapshot: ContextObject?
    @State private var pendingProject: String?
    @State private var pendingRevision: String?
    @State private var exportDocument = ContextExport(data: Data())
    private var messages: [ChatMessage] {
        context.messages.map { ChatMessage(id: $0["id"] as! String, text: $0["text"] as! String, isUser: $0["role"] as? String == "user", artworkIDs: $0["artworkIds"] as! [String]) }
    }

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
        .sheet(isPresented: $settings) { TrackingSettings(pipeline: tracking, camera: camera, artworkRevision: artwork?.id ?? "none", pattern: patternBinding, artwork: artwork) }
        .task { performContext { try context.start(); try syncContext() }; camera.start() }
        .task(id: pickerItem) {
            guard let pickerItem else { return }
            let originProject = context.projectID, originRevision = context.revision
            importing = true
            defer { importing = false }
            do {
                guard let data = try await pickerItem.loadTransferable(type: Data.self) else { throw ArtworkFactory.ArtworkError.invalidImage }
                try Task.checkCancellation()
                let imported = try ArtworkFactory.imported(data)
                try context.importPhoto(imported, project: originProject, expected: originRevision)
                tracking.disconnect()
                artwork = try context.selected()
                panel = .result
                notice = nil
            } catch is CancellationError { }
            catch { notice = error.localizedDescription }
        }
        .fileImporter(isPresented: $snapshotPicker, allowedContentTypes: [.json]) { result in
            performContext {
                try ContextSnapshotGate.requireAvailable()
                let url = try result.get()
                pendingSnapshot = try context.readImport(url)
                pendingProject = context.projectID; pendingRevision = context.revision
                importChoice = true
            }
        }
        .photosPicker(isPresented: $chipPhotoPicker, selection: $pickerItem, matching: .images, photoLibrary: .shared())
        .confirmationDialog("Import saved design context", isPresented: $importChoice, titleVisibility: .visible) {
            Button("Update from this snapshot") { finishImport("update") }
            Button("Open as a copy") { finishImport("copy") }
            Button("Cancel", role: .cancel) { pendingSnapshot = nil }
        } message: { Text("Import retains the full saved transcript and selected PNG. Existing images stay local. Imported recipes are descriptive and will not run.") }
        .alert("Export saved design context?", isPresented: $exportDisclosure) {
            Button("Choose file destination") { performContext { exportDocument = ContextExport(data: try context.export()); snapshotExporter = true } }
            Button("Cancel", role: .cancel) { }
        } message: { Text("This file contains the FULL saved transcript (\(context.messages.count) messages), accepted brief, unaccepted draft and creation records, plus \(artwork == nil ? "no selected PNG" : "the selected PNG only"). It may contain sensitive text you entered. Older local artwork is excluded.") }
        .fileExporter(isPresented: $snapshotExporter, document: exportDocument, contentType: .json, defaultFilename: "design.visualizeit-project") { result in
            if case .failure(let error) = result { contextError = error.localizedDescription }
        }
        .alert("Design context", isPresented: Binding(get: { contextError != nil }, set: { if !$0 { contextError = nil } })) { Button("OK") { contextError = nil } } message: { Text(contextError ?? "") }
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
            }.contextMenu { projectActions }
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
            bottomButton(icon: "text.bubble", title: "Ask") { panel = .chat; notice = nil }.contextMenu { projectActions }
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
                    TextEditor(text: promptBinding).scrollContentBackground(.hidden).frame(minHeight: 90, maxHeight: 130)
                        .overlay(alignment: .topLeading) { if prompt.isEmpty { Text("Describe the design you want to see in this space…").foregroundStyle(.white.opacity(0.65)).padding(8).allowsHitTesting(false) } }
                        .accessibilityLabel("Design description. Edits are saved as an unaccepted draft.")
                    Text("\(prompt.count) characters · \(draftLabel)").font(.caption).foregroundStyle(.white.opacity(0.7))
                }.padding(12).modifier(Glass(radius: 22))
                Text("Prompt ideas").font(.headline)
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 8) {
                      ForEach(suggestedPatterns) { idea in
                        let chip = context.chip(idea.contextCapability)
                        Button { performContext { try context.applyChip(chip, capability: idea.contextCapability); try syncContext() } } label: {
                            Text(idea.rawValue).font(.subheadline).frame(width: 76, alignment: .leading).padding(.horizontal, 12).frame(minHeight: 52).modifier(Glass(radius: 18))
                                .overlay(RoundedRectangle(cornerRadius: 18).stroke(pattern == idea ? Color.pink : Color.clear, lineWidth: 2))
                        }.accessibilityAddTraits(pattern == idea ? .isSelected : [])
                      }
                      let navigationType = needsClarification ? "clarify" : "import"
                      let navigationChip = context.chip(navigationType)
                      Button {
                          performContext {
                              try context.applyNavigationChip(navigationChip, type: navigationType)
                              if navigationType == "import" { chipPhotoPicker = true }
                              else { contextError = "Choose a finite native pattern explicitly, Typography to render words, or Upload an existing image. Your draft and brief were preserved." }
                          }
                      } label: { Text(needsClarification ? "Choose output" : "Import image").font(.subheadline).frame(width: 76, alignment: .leading).padding(.horizontal, 12).frame(minHeight: 52).modifier(Glass(radius: 18)) }
                    }
                }
                GradientAction(title: "Generate image", icon: "sparkles") { generate() }
                    .contextMenu {
                        let expectedRevision = context.revision
                        Button("Save brief") { performContext { try context.acceptBrief(expected: expectedRevision); try syncContext() } }
                        Button("Load accepted brief") { performContext { try context.loadAccepted(expected: expectedRevision); try syncContext() } }
                    }
            }.padding(24)
        }.frame(maxHeight: 430).modifier(Glass(radius: 34))
            .overlay(RoundedRectangle(cornerRadius: 34).stroke(LinearGradient(colors: spectrum, startPoint: .topLeading, endPoint: .bottomTrailing), lineWidth: 1.2))
            .padding(.horizontal, 5).padding(.bottom, 8)
    }

    private var result: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 14) { StarMark(size: 32); Text(artwork?.title == "Imported design" ? "Design imported" : "Design generated").font(.title2.weight(.medium)); Spacer(); RoundControl(icon: "xmark", label: "Close result panel") { panel = .home } }
            Text("You can move, resize, or generate a new design.").font(.subheadline).foregroundStyle(.white.opacity(0.8))
            HStack(spacing: 12) { GradientAction(title: "Regenerate", icon: "sparkles") {
                if artwork?.patternKind == nil { contextError = "This image’s source recipe is unavailable here. Its PNG remains unchanged. Choose a native pattern explicitly under Design to create a separate result." }
                else { panel = .generate }
            }; uploadButton }
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
                                    .contextMenu {
                                        let expectedRevision = context.revision
                                        Button("Use as brief") { performContext { try context.acceptBrief(message.text, expected: expectedRevision); try syncContext() } }
                                        ForEach(message.artworkIDs, id: \.self) { id in
                                            Button(context.artworkAvailable(id) ? "Select artwork \(id)" : "Artwork \(id) unavailable locally") { performContext { try context.selectCatalog(id, expected: expectedRevision); try syncContext() } }.disabled(!context.artworkAvailable(id))
                                        }
                                    }
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
        performContext {
            let attempt = try context.prepare(pattern: pattern, expected: context.revision)
            // Rendering starts ONLY after accepted intent and exact request commit successfully.
            let generated = try ArtworkFactory.generate(prompt: attempt["prompt"] as! String, pattern: pattern)
            try context.finish(generated, attempt: attempt)
            tracking.disconnect(); try syncContext(); panel = .result; notice = nil
        }
    }

    private func sendMessage() {
        let text = chatInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let lower = text.lowercased()
        let reply: String
        if lower.contains("wood") || lower.contains("slat") { reply = "Wood slats are available as a finite local pattern. Select Wood slats in Generate design to change the draft capability. This message does not change your saved brief or draft." }
        else if lower.contains("text") || lower.contains("word") { reply = "Typography can render your words. Select Typography in Generate design explicitly. Arbitrary object image generation is unavailable." }
        else { reply = "Try Wood slats, Color waves, Dot pattern, or Typography. Blue, green and pink descriptions change the accent color. For an object concept, import an image using Design → Upload." }
        performContext { try context.append(text, reply: reply, expected: context.revision); chatInput = "" }
    }

    private var draftLabel: String {
        guard let draft = context.draft else { return "no saved draft" }
        if !ContextCodec.equal(draft["baseBriefRevisionId"] ?? ContextCodec.null, context.accepted?["revisionId"] ?? ContextCodec.null) { return "stale draft" }
        return draft["text"] as? String == context.accepted?["text"] as? String ? "accepted" : "unaccepted draft"
    }
    private var promptBinding: Binding<String> {
        Binding(get: { prompt }, set: { text in performContext { try context.edit(text, capability: context.draft?["capability"] as? String ?? pattern.contextCapability, expected: context.revision); prompt = text } })
    }
    private var patternBinding: Binding<PatternKind> {
        Binding(get: { pattern }, set: { selected in let chip = context.chip(selected.contextCapability); performContext { try context.applyChip(chip, capability: selected.contextCapability); try syncContext() } })
    }
    private var suggestedPatterns: [PatternKind] {
        // Deterministic ranking only; four existing capabilities fit the existing five slots.
        let text = ((context.accepted?["text"] as? String ?? "") + " " + prompt).lowercased()
        let first: PatternKind = text.contains("wood") || text.contains("slat") ? .slats : text.contains("wave") ? .waves : text.contains("dot") ? .dots : .text
        return [first] + PatternKind.allCases.filter { $0 != first }
    }
    private var needsClarification: Bool {
        let text = prompt.lowercased()
        return !text.isEmpty && !["wood","slat","wave","dot","text","typograph"].contains(where: text.contains)
    }
    @ViewBuilder private var projectActions: some View {
        let expectedRevision = context.revision
        Button("New local project") { performContext { try context.create(); tracking.resetObjectAssociation(); try syncContext(); panel = .home } }
        Menu("Switch project") { ForEach(context.projects(), id: \.self) { id in Button(id) { performContext { let prior = context.projectID; try context.switchProject(id); if prior != context.projectID { tracking.resetObjectAssociation() }; try syncContext(); panel = .home } } } }
        Menu("Select saved artwork") { ForEach(context.catalogIDs(), id: \.self) { id in Button(id) { performContext { try context.selectCatalog(id, expected: expectedRevision); try syncContext() } } } }
        Button("Import design context…") { performContext { try ContextSnapshotGate.requireAvailable(); snapshotPicker = true } }
        Button("Export design context…") { performContext { try ContextSnapshotGate.requireAvailable(); exportDisclosure = true } }.disabled(context.document == nil)
    }
    private func syncContext() throws {
        let selected = try context.selected()
        tracking.disconnect()
        artwork = selected ?? context.legacyArtwork
        prompt = context.draft?["text"] as? String ?? ""
        if let capability = context.draft?["capability"] as? String, let kind = PatternKind.allCases.first(where: { $0.contextCapability == capability }) { pattern = kind }
        chatInput = ""
        if let status = context.status { notice = status }
    }
    private func performContext(_ action: () throws -> Void) {
        do { try action() } catch { contextError = error.localizedDescription }
    }
    private func finishImport(_ choice: String) {
        guard let snapshot = pendingSnapshot else { return }
        performContext { try context.importSnapshot(snapshot, choice: choice, expectedProject: pendingProject, expectedRevision: pendingRevision); tracking.resetObjectAssociation(); try syncContext(); panel = .home }
        pendingSnapshot = nil
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
