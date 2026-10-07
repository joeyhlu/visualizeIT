using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

namespace VisualizeIt
{
    public sealed class FrameDiagnostics : MonoBehaviour
    {
        private readonly List<float> frameMilliseconds = new List<float>(40000);
        private float elapsed;
        private float worstFrameMilliseconds;
        private int framesOver33Milliseconds;
        private int trackedFrames,totalFrames;
        private bool preview;
        public float FramesPerSecond { get; private set; }

        public void Record(float seconds, bool tracked, bool editorPreview)
        {
            if (seconds <= 0 || float.IsNaN(seconds) || float.IsInfinity(seconds)) return;
            elapsed += seconds;
            totalFrames++;
            worstFrameMilliseconds = Mathf.Max(worstFrameMilliseconds, seconds*1000);
            if (seconds > 1f/30) framesOver33Milliseconds++;
            FramesPerSecond = Mathf.Lerp(FramesPerSecond,1/seconds,0.04f);
            if (frameMilliseconds.Count < 40000) frameMilliseconds.Add(seconds*1000);
            if (tracked) trackedFrames++;
            preview = editorPreview;
        }

        public string SaveReport()
        {
            var sorted = frameMilliseconds.ToArray(); Array.Sort(sorted);
            var report = new Report
            {
                utc = DateTime.UtcNow.ToString("O"), device = SystemInfo.deviceModel,
                operatingSystem = SystemInfo.operatingSystem, graphics = SystemInfo.graphicsDeviceName,
                editorPreview = preview, durationSeconds = elapsed, sampledFrames = sorted.Length,
                totalFrames = totalFrames, worstFrameMilliseconds = worstFrameMilliseconds,
                framesOver33Milliseconds = framesOver33Milliseconds,
                averageFps = elapsed > 0 ? totalFrames/elapsed : 0,
                p95FrameMilliseconds = sorted.Length > 0 ? sorted[(int)((sorted.Length-1)*0.95f)] : 0,
                trackingAvailability = totalFrames > 0 ? (float)trackedFrames/totalFrames : 0,
                physicalValidationPassed = false
            };
            string directory = Path.Combine(Application.persistentDataPath,"diagnostics");
            Directory.CreateDirectory(directory);
            string path = Path.Combine(directory,DateTime.UtcNow.ToString("yyyyMMdd-HHmmss-fff") + ".json");
            File.WriteAllText(path,JsonUtility.ToJson(report,true)); return path;
        }

        [Serializable] private sealed class Report
        {
            public string utc,device,operatingSystem,graphics;
            public bool editorPreview,physicalValidationPassed;
            public float durationSeconds,averageFps,p95FrameMilliseconds,trackingAvailability,worstFrameMilliseconds;
            public int sampledFrames,totalFrames,framesOver33Milliseconds;
        }
    }
}
