using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.InputSystem.XR;
using UnityEngine.Rendering;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.ARSubsystems;
using Unity.XR.CoreUtils;
using VisualizeIt.Core;

namespace VisualizeIt
{
    public sealed class VisualizeItApp : MonoBehaviour
    {
        public Shader SurfaceShader;
        public FixtureProject Settings { get; private set; } = new FixtureProject();
        public string Message { get; private set; } = "Move your phone slowly to find a surface.";
        public Camera ARCamera { get; private set; }
        public bool EditorPreview { get; private set; }
        public bool Placed => surface != null;
        public bool Placing => placing;
        public string Capabilities { get; private set; } = "Checking AR capabilities…";
        public DesignImporter Importer { get; private set; }
        public event Action SettingsChanged;

        private ARSession session;
        private ARRaycastManager raycasts;
        private ARAnchorManager anchors;
        private AROcclusionManager occlusion;
        private ARAnchor anchor;
        private SceneLighting lighting;
        private FixtureSurface surface;
        private AppControls controls;
        private Texture2D checkerboard, design;
        private byte[] designPng;
        private readonly TrackingVisibility visibility = new TrackingVisibility();
        private readonly List<ARRaycastHit> hits = new List<ARRaycastHit>();
        private readonly List<InputAction> poseActions = new List<InputAction>();
        private bool foreground = true, placing, pendingResumeCheck;
        private int placementGeneration;
        private float statusClock;
        private FrameDiagnostics diagnostics;

        private void Awake()
        {
            Application.targetFrameRate = 60;
            Screen.sleepTimeout = SleepTimeout.NeverSleep;
            SurfaceShader = SurfaceShader != null ? SurfaceShader : Resources.Load<Shader>("SurfacePattern");
            if (SurfaceShader == null) { Debug.LogError("VisualizeIt surface shader missing."); enabled = false; return; }
            checkerboard = MakeCheckerboard();
            SetupAR();
            Importer = new GameObject("VisualizeItImageImport").AddComponent<DesignImporter>();
            Importer.transform.SetParent(transform);
            Importer.Imported += OnDesignImported;
            Importer.Failed += SetMessage;
            diagnostics = gameObject.AddComponent<FrameDiagnostics>();
            controls = gameObject.AddComponent<AppControls>();
            controls.Initialize(this);
        }

        private void SetupAR()
        {
            var sessionObject = new GameObject("AR Session");
            sessionObject.transform.SetParent(transform);
            session = sessionObject.AddComponent<ARSession>();
            sessionObject.AddComponent<ARInputManager>();
            var originObject = new GameObject("XR Origin");
            originObject.transform.SetParent(transform);
            var origin = originObject.AddComponent<XROrigin>();
            var offset = new GameObject("Camera Offset");
            offset.transform.SetParent(originObject.transform,false);
            origin.CameraFloorOffsetObject = offset;
            var cameraObject = new GameObject("AR Camera");
            cameraObject.tag = "MainCamera";
            cameraObject.transform.SetParent(offset.transform,false);
            ARCamera = cameraObject.AddComponent<Camera>();
            ARCamera.nearClipPlane = 0.03f;
            ARCamera.farClipPlane = 20;
            ARCamera.clearFlags = CameraClearFlags.SolidColor;
            ARCamera.backgroundColor = new Color(0.035f,0.05f,0.075f);
            ARCamera.allowHDR = false;
            origin.Camera = ARCamera;
            origin.RequestedTrackingOriginMode = XROrigin.TrackingOriginMode.Device;
            origin.CameraYOffset = 0;
            var poseDriver = cameraObject.AddComponent<TrackedPoseDriver>();
            poseDriver.positionInput = PoseAction("Position","Vector3","centerEyePosition","devicePosition");
            poseDriver.rotationInput = PoseAction("Rotation","Quaternion","centerEyeRotation","deviceRotation");
            poseDriver.trackingStateInput = PoseAction("Tracking state","Integer","trackingState","trackingState");
            var cameraManager = cameraObject.AddComponent<ARCameraManager>();
            cameraManager.requestedFacingDirection = CameraFacingDirection.World;
            cameraManager.autoFocusRequested = true;
            cameraManager.requestedLightEstimation = LightEstimation.AmbientIntensity | LightEstimation.AmbientColor
                | LightEstimation.AmbientSphericalHarmonics | LightEstimation.MainLightDirection | LightEstimation.MainLightIntensity;
            cameraObject.AddComponent<ARCameraBackground>();
            occlusion = cameraObject.AddComponent<AROcclusionManager>();
            // Providers resolve requests to supported modes; current modes and textures determine the UI status.
            occlusion.requestedEnvironmentDepthMode = EnvironmentDepthMode.Best;
            occlusion.environmentDepthTemporalSmoothingRequested = true;
            occlusion.requestedHumanDepthMode = HumanSegmentationDepthMode.Best;
            occlusion.requestedHumanStencilMode = HumanSegmentationStencilMode.Best;
            occlusion.requestedOcclusionPreferenceMode = OcclusionPreferenceMode.PreferEnvironmentOcclusion;
            originObject.AddComponent<ARPlaneManager>().requestedDetectionMode = PlaneDetectionMode.Horizontal | PlaneDetectionMode.Vertical;
            raycasts = originObject.AddComponent<ARRaycastManager>();
            anchors = originObject.AddComponent<ARAnchorManager>();
            originObject.AddComponent<AREnvironmentProbeManager>().automaticPlacementRequested = true;
            var lightObject = new GameObject("Estimated scene light");
            lightObject.transform.SetParent(transform);
            lightObject.transform.rotation = Quaternion.Euler(45,20,0);
            var mainLight = lightObject.AddComponent<Light>();
            mainLight.type = LightType.Directional;
            mainLight.intensity = 0.8f;
            mainLight.shadows = LightShadows.Soft;
            RenderSettings.ambientMode = AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.22f,0.22f,0.22f);
            lighting = lightObject.AddComponent<SceneLighting>();
            lighting.enabled = false;
            lighting.CameraManager = cameraManager;
            lighting.MainLight = mainLight;
            lighting.enabled = true;
#if UNITY_EDITOR
            // The editor preview is explicitly separate from device AR and never counts toward the AR exit gate.
            EditorPreview = true;
            session.enabled = false;
            poseDriver.enabled = false;
            cameraManager.enabled = false;
            cameraObject.GetComponent<ARCameraBackground>().enabled = false;
            occlusion.enabled = false;
            foreach (var behaviour in originObject.GetComponents<Behaviour>())
                if (!(behaviour is XROrigin)) behaviour.enabled = false;
            ARCamera.transform.position = new Vector3(0,0.25f,-0.45f);
            ARCamera.transform.LookAt(new Vector3(0,0.08f,0));
            Message = "Editor material preview. Phone AR must be tested on a device.";
#endif
        }

        private InputActionProperty PoseAction(string name, string layout, string hmdControl, string handheldControl)
        {
            var action = new InputAction(name,InputActionType.Value,expectedControlType:layout);
            action.AddBinding("<XRHMD>/" + hmdControl);
            action.AddBinding("<HandheldARInputDevice>/" + handheldControl);
            action.Enable();
            poseActions.Add(action);
            return new InputActionProperty(action);
        }

        public async void PlaceAtCrosshair()
        {
            if (placing) return;
            if (EditorPreview) { ReplaceSurface(null,Pose.identity); return; }
            if (ARSession.state != ARSessionState.SessionTracking)
            {
                SetMessage("Move your phone slowly until tracking is ready."); return;
            }
            if (!raycasts.Raycast(controls.PlacementScreenPoint,hits,TrackableType.PlaneWithinPolygon))
            {
                SetMessage("Aim at the object's supporting surface and move slowly to find it."); return;
            }
            var hit = hits[0];
            var normal = hit.pose.rotation * Vector3.up;
            var forward = Vector3.ProjectOnPlane(ARCamera.transform.forward,normal).normalized;
            if (forward.sqrMagnitude < 0.01f) forward = Vector3.ProjectOnPlane(ARCamera.transform.up,normal).normalized;
            var pose = new Pose(hit.pose.position,Quaternion.LookRotation(forward,normal));
            int generation = ++placementGeneration;
            placing = true;
            SetMessage("Attaching to the surface…");
            try
            {
                var result = await anchors.TryAddAnchorAsync(pose);
                if (this == null || generation != placementGeneration)
                {
                    if (result.status.IsSuccess() && anchors != null) anchors.TryRemoveAnchor(result.value);
                    return;
                }
                if (!result.status.IsSuccess()) { SetMessage("Could not anchor here. Move slightly and try again."); return; }
                ReplaceSurface(result.value,pose);
            }
            catch (Exception error) { if (this != null) SetMessage("Placement failed: " + error.Message); }
            finally { if (this != null && generation == placementGeneration) placing = false; }
        }

        private void ReplaceSurface(ARAnchor newAnchor, Pose pose)
        {
            ClearSurface();
            anchor = newAnchor;
            surface = new GameObject("Pattern surface").AddComponent<FixtureSurface>();
            surface.transform.SetParent(anchor != null ? anchor.transform : transform,false);
            surface.transform.SetPositionAndRotation(pose.position,pose.rotation);
            surface.Initialize(SurfaceShader,Settings.shape,Settings.dimensions,Settings.seamDegrees);
            // The anchor was requested at this pose; retain its actual local offset if the provider adjusted it.
            placementBasePosition = surface.transform.localPosition;
            placementBaseRotation = surface.transform.localRotation;
            ApplyAlignment();
            SetMessage(EditorPreview ? "Preview only — no real camera, tracking, or depth." : "Match the fixture's size and alignment to your object. Keep the object still.");
        }

        private Vector3 placementBasePosition;
        private Quaternion placementBaseRotation = Quaternion.identity;
        private void ApplyAlignment()
        {
            if (surface == null) return;
            surface.transform.localPosition = placementBasePosition + placementBaseRotation * Settings.alignmentOffset;
            surface.transform.localRotation = placementBaseRotation * Quaternion.Euler(0,Settings.yawDegrees,0);
        }

        public void ClearSurface()
        {
            if (surface != null) Destroy(surface.gameObject);
            surface = null;
            if (anchor != null && anchors != null) anchors.TryRemoveAnchor(anchor);
            anchor = null;
            visibility.Reset();
        }

        public void ResetPlacement()
        {
            ++placementGeneration;
            placing = false;
            ClearSurface();
            SetMessage("Place again at the object's supporting surface.");
        }

        public void ChangeShape(SurfaceShape shape)
        {
            Settings.shape = shape;
            if (surface != null) surface.Rebuild(shape,Settings.dimensions,Settings.seamDegrees);
            SettingsChanged?.Invoke();
        }

        public void SetDimension(int axis, float metres)
        {
            Vector3 size = Settings.dimensions;
            size[axis] = Mathf.Clamp(metres,0.005f,5);
            Settings.dimensions = size;
            if (surface != null) surface.Rebuild(Settings.shape,size,Settings.seamDegrees);
        }

        public void SetSeamAngle(float degrees)
        {
            Settings.seamDegrees = degrees;
            if (surface != null) surface.Rebuild(Settings.shape,Settings.dimensions,degrees);
        }

        public void FitCylinderWrap()
        {
            if (Settings.shape != SurfaceShape.Cylinder) return;
            Settings.patternMetres = PatternMapping.FitCylinderTileWidth(Settings.dimensions.x,Settings.patternMetres);
            Settings.patternRotation = 0;
            SettingsChanged?.Invoke();
            SetMessage("Repeat width fitted around the cylinder at 0°. Image borders and cap seams may still be visible.");
        }

        public void UpdateAlignment() => ApplyAlignment();
        public void UseCheckerboard() { Settings.useCheckerboard = true; SetMessage("Checkerboard: inspect curvature, stretch, and seams."); }

        private void OnDesignImported(Texture2D texture, byte[] png)
        {
            if (design != null) Destroy(design);
            design = texture;
            designPng = png;
            Settings.useCheckerboard = false;
            SetMessage("Design imported. Adjust repeat size, rotation, and roughness.");
        }

        public void SaveProject()
        {
            try { ProjectStore.Save(Settings,designPng); SetMessage("Design and settings saved on this phone."); }
            catch (Exception error) { SetMessage("Could not save: " + error.Message); }
        }

        public void LoadProject()
        {
            try
            {
                var loaded = ProjectStore.Load();
                if (loaded == null) { SetMessage("No saved project yet."); return; }
                if (!loaded.useCheckerboard)
                {
                    if (!File.Exists(ProjectStore.DesignPath)) throw new InvalidDataException("The saved design image is missing.");
                    if (!Importer.Load(ProjectStore.DesignPath)) throw new InvalidDataException("The saved image could not be opened.");
                }
                Settings = loaded;
                ResetPlacement();
                SettingsChanged?.Invoke();
                SetMessage("Settings loaded. Place again; room anchors are not saved between sessions.");
            }
            catch (Exception error) { SetMessage("Could not load: " + error.Message); }
        }

        public void SaveScreenshot() => StartCoroutine(CaptureScreenshot());
        private IEnumerator CaptureScreenshot()
        {
            controls.SetVisible(false);
            yield return new WaitForEndOfFrame();
            Texture2D screenshot = null;
            try
            {
                screenshot = ScreenCapture.CaptureScreenshotAsTexture();
                string directory = Path.Combine(Application.persistentDataPath,"screenshots");
                Directory.CreateDirectory(directory);
                string path = Path.Combine(directory,DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff") + ".png");
                File.WriteAllBytes(path,screenshot.EncodeToPNG());
                SetMessage("Screenshot saved on this phone.");
            }
            catch (Exception error) { SetMessage("Screenshot failed: " + error.Message); }
            finally { if (screenshot != null) Destroy(screenshot); controls.SetVisible(true); }
        }

        public void ExportDiagnostics()
        {
            try { SetMessage("Session report saved: " + diagnostics.SaveReport()); }
            catch (Exception error) { SetMessage("Report failed: " + error.Message); }
        }

        public void SetMessage(string message) => Message = message;

        private void Update()
        {
            bool tracked = EditorPreview || ARSession.state == ARSessionState.SessionTracking;
            bool anchorTracked = EditorPreview || (anchor != null && anchor.trackingState == TrackingState.Tracking);
            visibility.Tick(tracked,anchorTracked,foreground,Time.unscaledDeltaTime);
            if (surface != null)
                surface.Apply(Settings.useCheckerboard || design == null ? checkerboard : design,
                    Settings.patternMetres,Settings.patternRotation,Settings.roughness,visibility.Opacity,Settings.shellMetres);
            diagnostics.Record(Time.unscaledDeltaTime,tracked && anchorTracked && surface != null,EditorPreview);
            if (surface != null && !visibility.CanRender && !EditorPreview)
                Message = "Tracking paused. Keep the object still and return to the previous view.";
            if (pendingResumeCheck && tracked)
            {
                pendingResumeCheck = false;
                ResetPlacement();
                Message = "Camera resumed. Place again to confirm alignment.";
            }
            statusClock += Time.unscaledDeltaTime;
            if (statusClock >= 1)
            {
                statusClock = 0;
                bool depth = !EditorPreview && (occlusion.currentEnvironmentDepthMode != EnvironmentDepthMode.Disabled
                    && occlusion.TryGetEnvironmentDepthTexture(out _));
                bool people = !EditorPreview && occlusion.currentHumanDepthMode != HumanSegmentationDepthMode.Disabled
                    && occlusion.humanDepthTexture != null;
                Capabilities = EditorPreview ? "EDITOR PREVIEW · not device AR"
                    : (ARSession.state == ARSessionState.Unsupported ? "AR is unsupported on this phone"
                        : (tracked ? "AR tracking" : "Finding tracking") + " · "
                        + (depth ? "Scene depth active" : people ? "People occlusion active" : "Depth occlusion unavailable")
                        + "\n" + (lighting.HasEstimate ? "Light estimate active" : "Lighting estimate unavailable"));
                Capabilities += " · " + diagnostics.FramesPerSecond.ToString("F0") + " FPS";
            }
        }

        private void OnApplicationPause(bool paused)
        {
            foreground = !paused;
            visibility.Reset();
            if (!paused) pendingResumeCheck = true;
        }
        private void OnApplicationFocus(bool focused) { foreground = focused; if (!focused) visibility.Reset(); }

        private Texture2D MakeCheckerboard()
        {
            var texture = new Texture2D(256,256,TextureFormat.RGBA32,true,false) { name = "Metric checkerboard", wrapMode = TextureWrapMode.Repeat, filterMode = FilterMode.Trilinear, anisoLevel = 4 };
            var pixels = new Color32[256*256];
            for (int y=0; y<256; y++) for (int x=0; x<256; x++)
                pixels[y*256+x] = ((x/32+y/32)%2 == 0) ? new Color32(230,238,239,255) : new Color32(23,60,71,255);
            texture.SetPixels32(pixels); texture.Apply(true,false); return texture;
        }

        private void OnDestroy()
        {
            ++placementGeneration;
            foreach (var action in poseActions) action.Dispose();
            if (design != null) Destroy(design);
            if (checkerboard != null) Destroy(checkerboard);
        }
    }
}
