using System;
using System.IO;
using System.Runtime.InteropServices;
using UnityEngine;

namespace VisualizeIt
{
    public sealed class DesignImporter : MonoBehaviour
    {
        public event Action<Texture2D, byte[]> Imported;
        public event Action<string> Failed;
        public bool Busy { get; private set; }
        private float startedAt;

#if UNITY_IOS && !UNITY_EDITOR
        [DllImport("__Internal")] private static extern void VI_PickImage(string target);
#endif

        public void Pick()
        {
            if (Busy) return;
            Busy = true;
            startedAt = Time.realtimeSinceStartup;
            try
            {
#if UNITY_EDITOR
                string path = UnityEditor.EditorUtility.OpenFilePanel("Import a design", "", "png,jpg,jpeg");
                OnImagePicked(path);
#elif UNITY_ANDROID
                using (var unity = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
                using (var activity = unity.GetStatic<AndroidJavaObject>("currentActivity"))
                using (var importer = new AndroidJavaClass("com.visualizeit.importer.ImagePickerActivity"))
                    importer.CallStatic("pick", activity, gameObject.name);
#elif UNITY_IOS
                VI_PickImage(gameObject.name);
#else
                throw new PlatformNotSupportedException("Image import requires iOS, Android, or the editor.");
#endif
            }
            catch (Exception error) { Busy = false; Failed?.Invoke(error.Message); }
        }

        // Called by the native document/photo picker on Unity's main thread.
        public void OnImagePicked(string path)
        {
            Load(path);
        }

        public bool Load(string path)
        {
            Busy = false;
            if (string.IsNullOrEmpty(path)) return false;
            Texture2D texture = null;
            try
            {
                var file = new FileInfo(path);
                if (!file.Exists || file.Length > 16 * 1024 * 1024)
                    throw new InvalidDataException("Choose an image smaller than 16 MB.");
                byte[] bytes = File.ReadAllBytes(path);
                texture = new Texture2D(2, 2, TextureFormat.RGBA32, true, false);
                if (!ImageConversion.LoadImage(texture, bytes, false))
                    throw new InvalidDataException("Choose a PNG or JPEG image.");
                if (texture.width > 4096 || texture.height > 4096)
                    throw new InvalidDataException("Choose an image no larger than 4096 pixels per side.");
                texture.wrapMode = TextureWrapMode.Repeat;
                texture.filterMode = FilterMode.Trilinear;
                texture.anisoLevel = 4;
                texture.name = "Imported design";
                Imported?.Invoke(texture, texture.EncodeToPNG());
                texture = null; // Ownership transfers to the app.
                return true;
            }
            catch (Exception error) { Failed?.Invoke(error.Message); return false; }
            finally { if (texture != null) Destroy(texture); }
        }

        public void OnImagePickFailed(string message) { Busy = false; Failed?.Invoke(message); }

        private void Update()
        {
            if (Busy && Time.realtimeSinceStartup - startedAt > 120)
                OnImagePickFailed("Image selection timed out. Try importing again.");
        }
    }
}
