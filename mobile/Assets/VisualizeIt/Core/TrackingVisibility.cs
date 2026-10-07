using System;

namespace VisualizeIt.Core
{
    // Session tracking alone is insufficient: a surface's anchor must also be tracking.
    public sealed class TrackingVisibility
    {
        public float Opacity { get; private set; }
        public bool CanRender { get; private set; }
        private float stableSeconds;

        public void Tick(bool sessionTracking, bool anchorTracking, bool foreground, float deltaSeconds)
        {
            if (float.IsNaN(deltaSeconds) || float.IsInfinity(deltaSeconds) || deltaSeconds < 0)
                throw new ArgumentOutOfRangeException(nameof(deltaSeconds));
            bool valid = sessionTracking && anchorTracking && foreground;
            stableSeconds = valid ? stableSeconds + deltaSeconds : 0;
            CanRender = valid && stableSeconds >= 0.35f;
            float target = CanRender ? 1 : 0;
            float step = deltaSeconds / (CanRender ? 0.25f : 0.12f);
            Opacity = target > Opacity ? Math.Min(target, Opacity + step) : Math.Max(target, Opacity - step);
        }

        public void Reset() { Opacity = 0; stableSeconds = 0; CanRender = false; }
    }
}
