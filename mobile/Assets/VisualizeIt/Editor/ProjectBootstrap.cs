using System;
using System.IO;
using UnityEditor;
using UnityEditor.Build;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using UnityEngine.XR.ARFoundation;
using UnityEngine.XR.Management;
using UnityEditor.XR.Management;
using UnityEditor.XR.Management.Metadata;

namespace VisualizeIt.Editor
{
    [InitializeOnLoad]
    public static class ProjectBootstrap
    {
        public const string ScenePath = "Assets/VisualizeIt/Scenes/VisualizeIt.unity";
        private const string SettingsPath = "Assets/VisualizeIt/Settings";

        static ProjectBootstrap()
        {
            EditorApplication.delayCall += () =>
            {
                if (EditorApplication.isPlayingOrWillChangePlaymode || EditorApplication.isCompiling) return;
                if (!File.Exists(ScenePath)) Configure();
            };
        }

        [MenuItem("VisualizeIt/Configure project")]
        public static void Configure()
        {
            if (!Application.unityVersion.StartsWith("6000.3.",StringComparison.Ordinal))
                throw new InvalidOperationException("Open this project in Unity 6.3 LTS (6000.3.25f1). Do not downgrade it.");
            Directory.CreateDirectory(SettingsPath);
            Directory.CreateDirectory(Path.GetDirectoryName(ScenePath));
            AssetDatabase.Refresh();
            ConfigureRendering();
            ConfigureLoaders();
            ConfigurePlayer();
            if (!File.Exists(ScenePath))
            {
                var previousScene = EditorSceneManager.GetActiveScene();
                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene,NewSceneMode.Additive);
                EditorSceneManager.SetActiveScene(scene);
                var root = new GameObject("VisualizeIt");
                var app = root.AddComponent<VisualizeItApp>();
                app.SurfaceShader = AssetDatabase.LoadAssetAtPath<Shader>("Assets/VisualizeIt/Resources/SurfacePattern.shader");
                EditorSceneManager.SaveScene(scene,ScenePath);
                EditorSceneManager.CloseScene(scene,true);
                if (previousScene.IsValid()) EditorSceneManager.SetActiveScene(previousScene);
            }
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(ScenePath,true) };
            AssetDatabase.SaveAssets();
            Debug.Log("VisualizeIt configured. Open the VisualizeIt scene and build to a phone for AR validation.");
        }

        private static void ConfigureRendering()
        {
            string rendererPath = SettingsPath + "/ARForwardRenderer.asset";
            var renderer = AssetDatabase.LoadAssetAtPath<UniversalRendererData>(rendererPath);
            if (renderer == null)
            {
                renderer = ScriptableObject.CreateInstance<UniversalRendererData>();
                renderer.name = "AR Forward Renderer";
                AssetDatabase.CreateAsset(renderer,rendererPath);
            }
            bool hasBackground = renderer.rendererFeatures.Exists(feature => feature is ARBackgroundRendererFeature);
            if (!hasBackground)
            {
                var background = ScriptableObject.CreateInstance<ARBackgroundRendererFeature>();
                background.name = "AR Camera Background";
                AssetDatabase.AddObjectToAsset(background,renderer);
                renderer.rendererFeatures.Add(background);
                renderer.SetDirty();
            }
            string pipelinePath = SettingsPath + "/MobileARPipeline.asset";
            var pipeline = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(pipelinePath);
            if (pipeline == null)
            {
                pipeline = UniversalRenderPipelineAsset.Create(renderer);
                pipeline.name = "Mobile AR Pipeline";
                AssetDatabase.CreateAsset(pipeline,pipelinePath);
            }
            pipeline.supportsCameraDepthTexture = true;
            pipeline.supportsCameraOpaqueTexture = false;
            pipeline.msaaSampleCount = 2;
            pipeline.renderScale = 1;
            pipeline.reflectionProbeBlending = true;
            pipeline.reflectionProbeBoxProjection = true;
            GraphicsSettings.defaultRenderPipeline = pipeline;
            QualitySettings.renderPipeline = pipeline;
            EditorUtility.SetDirty(renderer);
            EditorUtility.SetDirty(pipeline);
        }

        private static void ConfigureLoaders()
        {
            if (!EditorBuildSettings.TryGetConfigObject<XRGeneralSettingsPerBuildTarget>(XRGeneralSettings.k_SettingsKey,out var targets))
            {
                targets = ScriptableObject.CreateInstance<XRGeneralSettingsPerBuildTarget>();
                AssetDatabase.CreateAsset(targets,SettingsPath + "/XRGeneralSettings.asset");
                EditorBuildSettings.AddConfigObject(XRGeneralSettings.k_SettingsKey,targets,true);
            }
            Assign(targets,BuildTargetGroup.Android,"UnityEngine.XR.ARCore.ARCoreLoader");
            Assign(targets,BuildTargetGroup.iOS,"UnityEngine.XR.ARKit.ARKitLoader");
            EditorUtility.SetDirty(targets);
        }

        private static void Assign(XRGeneralSettingsPerBuildTarget targets,BuildTargetGroup group,string loader)
        {
            if (!targets.HasSettingsForBuildTarget(group)) targets.CreateDefaultSettingsForBuildTarget(group);
            if (!targets.HasManagerSettingsForBuildTarget(group)) targets.CreateDefaultManagerSettingsForBuildTarget(group);
            var general = targets.SettingsForBuildTarget(group);
            general.InitManagerOnStart = true;
            if (!XRPackageMetadataStore.AssignLoader(general.Manager,loader,group))
                throw new InvalidOperationException("Could not assign " + loader + ". Wait for package resolution and configure again.");
            EditorUtility.SetDirty(general);
            EditorUtility.SetDirty(general.Manager);
        }

        private static void ConfigurePlayer()
        {
            PlayerSettings.companyName = "VisualizeIt";
            PlayerSettings.productName = "VisualizeIt";
            PlayerSettings.bundleVersion = "0.1.0";
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.Android,"com.visualizeit.mobile");
            PlayerSettings.SetApplicationIdentifier(NamedBuildTarget.iOS,"com.visualizeit.mobile");
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.Android,ScriptingImplementation.IL2CPP);
            PlayerSettings.SetScriptingBackend(NamedBuildTarget.iOS,ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel26;
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            PlayerSettings.iOS.targetOSVersionString = "15.0";
            PlayerSettings.iOS.cameraUsageDescription = "VisualizeIt uses the camera to place designs on physical surfaces.";
            PlayerSettings.defaultInterfaceOrientation = UIOrientation.Portrait;
            PlayerSettings.colorSpace = ColorSpace.Linear;
            PlayerSettings.SetUseDefaultGraphicsAPIs(BuildTarget.Android,false);
            PlayerSettings.SetGraphicsAPIs(BuildTarget.Android,new[] { GraphicsDeviceType.Vulkan,GraphicsDeviceType.OpenGLES3 });
            // Input System is required by pose tracking and the runtime UI.
            var player = new SerializedObject(AssetDatabase.LoadAllAssetsAtPath("ProjectSettings/ProjectSettings.asset")[0]);
            var input = player.FindProperty("activeInputHandler");
            if (input != null) { input.intValue = 1; player.ApplyModifiedPropertiesWithoutUndo(); }
        }

        [MenuItem("VisualizeIt/Open app scene")]
        public static void OpenScene()
        {
            if (!File.Exists(ScenePath)) Configure();
            if (EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) EditorSceneManager.OpenScene(ScenePath);
        }

        [MenuItem("VisualizeIt/Build Android APK")]
        public static void BuildAndroid()
        {
            Configure();
            if (!BuildPipeline.IsBuildTargetSupported(BuildTargetGroup.Android,BuildTarget.Android))
                throw new InvalidOperationException("Install Android Build Support, SDK/NDK, and OpenJDK for this Unity editor.");
            Build(BuildTarget.Android,"Builds/Android/VisualizeIt.apk");
        }

        [MenuItem("VisualizeIt/Export iOS project")]
        public static void BuildIOS()
        {
            Configure();
            if (!BuildPipeline.IsBuildTargetSupported(BuildTargetGroup.iOS,BuildTarget.iOS))
                throw new InvalidOperationException("Install iOS Build Support. Signing and device builds require macOS/Xcode.");
            Build(BuildTarget.iOS,"Builds/iOS");
        }

        private static void Build(BuildTarget target,string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { ScenePath }, target = target, locationPathName = path,
                options = BuildOptions.Development
            });
            if (report.summary.result != BuildResult.Succeeded)
                throw new InvalidOperationException("Build failed: " + report.summary.result);
        }
    }
}
