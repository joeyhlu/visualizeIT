using System;

namespace VisualizeIt.Core
{
    public sealed class MappingQuality
    {
        public int TriangleCount, InvalidTriangleCount, WorstTriangle;
        // Singular values of the UV-metres -> surface-metres Jacobian. Ideal = 1.
        public double MinimumStretch = double.PositiveInfinity;
        public double MaximumStretch;
        public double MaximumAnisotropy = 1;
        public double MaximumScaleError;
        public bool Reliable => TriangleCount > 0 && InvalidTriangleCount == 0;
    }

    public static class MeshMappingAnalysis
    {
        public static MappingQuality Analyze(SurfaceMeshData mesh)
        {
            if (mesh == null || mesh.Positions.Count != mesh.MetricUV.Count || mesh.Triangles.Count % 3 != 0)
                throw new ArgumentException("Mesh positions, UVs and triangle indices must agree.");
            for (int i=0;i<mesh.Positions.Count;i++)
            {
                var p=mesh.Positions[i]; var uv=mesh.MetricUV[i];
                if (!Finite(p.X) || !Finite(p.Y) || !Finite(p.Z) || !Finite(uv.X) || !Finite(uv.Y))
                    throw new ArgumentException("Mesh coordinates must be finite.");
            }
            var result = new MappingQuality { TriangleCount = mesh.Triangles.Count/3, WorstTriangle = -1 };
            for (int i=0;i<mesh.Triangles.Count;i+=3)
            {
                int a=mesh.Triangles[i], b=mesh.Triangles[i+1], c=mesh.Triangles[i+2];
                if (a<0 || b<0 || c<0 || a>=mesh.Positions.Count || b>=mesh.Positions.Count || c>=mesh.Positions.Count)
                    throw new ArgumentException("Triangle index outside the mesh.");
                var t0=mesh.MetricUV[a]; var t1=mesh.MetricUV[b]; var t2=mesh.MetricUV[c];
                double u1=t1.X-t0.X, v1=t1.Y-t0.Y, u2=t2.X-t0.X, v2=t2.Y-t0.Y;
                double determinant=u1*v2-u2*v1;
                double uvScale=Math.Max(u1*u1+v1*v1,u2*u2+v2*v2);
                if (uvScale==0 || Math.Abs(determinant)<=uvScale*1e-12) { Invalid(result,i/3); continue; }
                var p0=mesh.Positions[a]; var p1=mesh.Positions[b]; var p2=mesh.Positions[c];
                double[] e1={p1.X-p0.X,p1.Y-p0.Y,p1.Z-p0.Z};
                double[] e2={p2.X-p0.X,p2.Y-p0.Y,p2.Z-p0.Z};
                double g00=0,g01=0,g11=0;
                for (int axis=0;axis<3;axis++)
                {
                    double du=(e1[axis]*v2-e2[axis]*v1)/determinant;
                    double dv=(-e1[axis]*u2+e2[axis]*u1)/determinant;
                    g00+=du*du; g01+=du*dv; g11+=dv*dv;
                }
                double discriminant=Math.Sqrt((g00-g11)*(g00-g11)+4*g01*g01);
                double high=Math.Sqrt(Math.Max(0,(g00+g11+discriminant)/2));
                // det/high avoids cancellation when the two eigenvalues differ greatly.
                double gramDet=Math.Max(0,g00*g11-g01*g01);
                double low=high>0 ? Math.Sqrt(gramDet)/high : 0;
                if (low<=high*1e-12 || !Finite(high)) { Invalid(result,i/3); continue; }
                result.MinimumStretch=Math.Min(result.MinimumStretch,low);
                result.MaximumStretch=Math.Max(result.MaximumStretch,high);
                result.MaximumAnisotropy=Math.Max(result.MaximumAnisotropy,high/low);
                double error=Math.Max(Math.Abs(low-1),Math.Abs(high-1));
                if (error>result.MaximumScaleError) { result.MaximumScaleError=error; result.WorstTriangle=i/3; }
            }
            return result;
        }

        private static void Invalid(MappingQuality quality,int triangle)
        {
            quality.InvalidTriangleCount++;
            quality.MinimumStretch=0; quality.MaximumAnisotropy=double.PositiveInfinity;
            quality.MaximumScaleError=double.PositiveInfinity; quality.WorstTriangle=triangle;
        }
        private static bool Finite(double value) => !double.IsNaN(value) && !double.IsInfinity(value);
    }
}
