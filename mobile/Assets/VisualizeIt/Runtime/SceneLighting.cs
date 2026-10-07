using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.XR.ARFoundation;

namespace VisualizeIt
{
    public sealed class SceneLighting : MonoBehaviour
    {
        public ARCameraManager CameraManager;
        public Light MainLight;
        public bool HasEstimate { get; private set; }
        public bool HasDirectionalEstimate { get; private set; }
        private float lastEstimateTime;

        private void OnEnable()
        {
            if (CameraManager != null) CameraManager.frameReceived += OnFrame;
        }

        private void OnDisable()
        {
            if (CameraManager != null) CameraManager.frameReceived -= OnFrame;
        }

        private void Update()
        {
            if (Time.realtimeSinceStartup - lastEstimateTime > 3)
                HasEstimate = HasDirectionalEstimate = false;
        }

        private void OnFrame(ARCameraFrameEventArgs frame)
        {
            var estimate = frame.lightEstimation;
            float blend = 1 - Mathf.Exp(-Time.unscaledDeltaTime * 8);
            HasEstimate = estimate.averageBrightness.HasValue || estimate.ambientSphericalHarmonics.HasValue
                || estimate.averageMainLightBrightness.HasValue;
            HasDirectionalEstimate = estimate.mainLightDirection.HasValue;
            if (HasEstimate) lastEstimateTime = Time.realtimeSinceStartup;
            if (estimate.ambientSphericalHarmonics.HasValue)
            {
                RenderSettings.ambientMode = AmbientMode.Skybox;
                RenderSettings.ambientProbe = estimate.ambientSphericalHarmonics.Value;
            }
            else if (estimate.averageBrightness.HasValue)
            {
                RenderSettings.ambientMode = AmbientMode.Flat;
                Color ambient = estimate.colorCorrection ?? Color.white;
                RenderSettings.ambientLight = ambient * Mathf.Clamp(estimate.averageBrightness.Value * 0.35f, 0.05f, 1.5f);
            }
            if (estimate.mainLightDirection.HasValue)
                MainLight.transform.rotation = Quaternion.Slerp(MainLight.transform.rotation,
                    Quaternion.LookRotation(estimate.mainLightDirection.Value), blend);
            float? brightness = estimate.averageMainLightBrightness ?? estimate.averageBrightness;
            if (brightness.HasValue)
                MainLight.intensity = Mathf.Lerp(MainLight.intensity, Mathf.Clamp(brightness.Value,0.05f,4), blend);
            Color? color = estimate.mainLightColor ?? estimate.colorCorrection;
            if (color.HasValue) MainLight.color = Color.Lerp(MainLight.color,color.Value,blend);
            else if (estimate.averageColorTemperature.HasValue)
            {
                MainLight.useColorTemperature = true;
                MainLight.colorTemperature = Mathf.Clamp(estimate.averageColorTemperature.Value,1000,20000);
            }
        }
    }
}
