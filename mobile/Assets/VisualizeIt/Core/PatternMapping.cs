using System;

namespace VisualizeIt.Core
{
    // Physical texture coordinates. Keep the shader equivalent in SurfacePattern.shader.
    public static class PatternMapping
    {
        public static Float2 TextureUV(Float2 metres, float tileWidth, float imageHeightOverWidth, float rotationDegrees)
        {
            Positive(tileWidth,nameof(tileWidth));
            Positive(imageHeightOverWidth,nameof(imageHeightOverWidth));
            if (!Finite(metres.X) || !Finite(metres.Y) || !Finite(rotationDegrees))
                throw new ArgumentOutOfRangeException(nameof(metres));
            double angle = (rotationDegrees % 360) * Math.PI / 180;
            double cosine = Math.Cos(angle), sine = Math.Sin(angle);
            return new Float2((float)((cosine*metres.X-sine*metres.Y)/tileWidth),
                (float)((sine*metres.X+cosine*metres.Y)/(tileWidth*imageHeightOverWidth)));
        }

        public static float FitCylinderTileWidth(float diameter, float requestedWidth, float minimumWidth = 0.005f, float maximumWidth = 1)
        {
            Positive(diameter,nameof(diameter)); Positive(requestedWidth,nameof(requestedWidth));
            Positive(minimumWidth,nameof(minimumWidth)); Positive(maximumWidth,nameof(maximumWidth));
            if (maximumWidth<minimumWidth) throw new ArgumentOutOfRangeException(nameof(maximumWidth));
            double circumference = Math.PI * diameter;
            double minimumRepeats=Math.Max(1,Math.Ceiling(circumference/maximumWidth));
            double maximumRepeats=Math.Floor(circumference/minimumWidth);
            if (maximumRepeats<minimumRepeats) throw new ArgumentOutOfRangeException(nameof(diameter),"No whole repeat fits the allowed tile widths.");
            double repeats = Math.Max(minimumRepeats,Math.Min(maximumRepeats,
                Math.Round(circumference/requestedWidth,MidpointRounding.AwayFromZero)));
            return (float)(circumference/repeats);
        }

        // A periodic image closes at the seam only when both UV jumps are integers.
        // Nonperiodic source images can still have visible borders even when this returns zero.
        public static float CylinderSeamPhaseError(float diameter, float tileWidth, float aspect, float rotationDegrees)
        {
            Positive(diameter,nameof(diameter));
            Float2 delta = TextureUV(new Float2((float)(Math.PI*diameter),0),tileWidth,aspect,rotationDegrees);
            double u = delta.X-Math.Round(delta.X), v = delta.Y-Math.Round(delta.Y);
            return (float)Math.Sqrt(u*u+v*v);
        }

        private static bool Finite(float value) => !float.IsNaN(value) && !float.IsInfinity(value);
        private static void Positive(float value,string name)
        {
            if (!Finite(value) || value <= 0) throw new ArgumentOutOfRangeException(name);
        }
    }
}
