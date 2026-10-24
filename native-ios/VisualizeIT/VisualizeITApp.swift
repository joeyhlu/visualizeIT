import SwiftUI

@main
@MainActor
struct VisualizeITApp: App {
    var body: some Scene {
        WindowGroup { StudioView().preferredColorScheme(.dark) }
    }
}
