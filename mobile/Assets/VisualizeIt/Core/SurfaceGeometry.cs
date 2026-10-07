using System;
using System.Collections.Generic;

namespace VisualizeIt.Core
{
    public enum SurfaceShape { Plane, Box, Cylinder }

    public struct Float2
    {
        public float X, Y;
        public Float2(float x, float y) { X = x; Y = y; }
    }

    public struct Float3
    {
        public float X, Y, Z;
        public Float3(float x, float y, float z) { X = x; Y = y; Z = z; }
    }

    public sealed class SurfaceMeshData
    {
        public readonly List<Float3> Positions = new List<Float3>();
        public readonly List<Float3> Normals = new List<Float3>();
        // UV values are physical distances in metres, not normalized texture coordinates.
        public readonly List<Float2> MetricUV = new List<Float2>();
        public readonly List<int> Triangles = new List<int>();
    }

    public static class SurfaceGeometry
    {
        public static SurfaceMeshData Create(SurfaceShape shape, float width, float height, float depth, int segments = 96, float seamDegrees = 0)
        {
            ValidateDimension(width);
            ValidateDimension(height);
            ValidateDimension(depth);
            if (float.IsNaN(seamDegrees) || float.IsInfinity(seamDegrees)) throw new ArgumentOutOfRangeException(nameof(seamDegrees));
            switch (shape)
            {
                case SurfaceShape.Plane: return Plane(width, depth);
                case SurfaceShape.Box: return Box(width, height, depth);
                case SurfaceShape.Cylinder:
                    if (segments < 12 || segments > 512) throw new ArgumentOutOfRangeException(nameof(segments));
                    return Cylinder(width * 0.5f, height, segments, seamDegrees);
                default: throw new ArgumentOutOfRangeException(nameof(shape));
            }
        }

        private static void ValidateDimension(float value)
        {
            if (float.IsNaN(value) || float.IsInfinity(value) || value < 0.005f || value > 5f)
                throw new ArgumentOutOfRangeException(nameof(value), "Fixture dimensions must be 0.005–5 metres.");
        }

        private static void Quad(SurfaceMeshData mesh, Float3 origin, Float3 u, Float3 v, Float3 normal,
            float width, float height, float uOffset = 0)
        {
            int start = mesh.Positions.Count;
            mesh.Positions.Add(origin);
            mesh.Positions.Add(Add(origin, u));
            mesh.Positions.Add(Add(Add(origin, u), v));
            mesh.Positions.Add(Add(origin, v));
            for (int i = 0; i < 4; i++) mesh.Normals.Add(normal);
            mesh.MetricUV.Add(new Float2(uOffset, 0));
            mesh.MetricUV.Add(new Float2(uOffset + width, 0));
            mesh.MetricUV.Add(new Float2(uOffset + width, height));
            mesh.MetricUV.Add(new Float2(uOffset, height));
            mesh.Triangles.AddRange(new[] { start, start + 1, start + 2, start, start + 2, start + 3 });
        }

        private static Float3 Add(Float3 a, Float3 b) => new Float3(a.X + b.X, a.Y + b.Y, a.Z + b.Z);

        private static SurfaceMeshData Plane(float width, float depth)
        {
            var mesh = new SurfaceMeshData();
            Quad(mesh, new Float3(-width / 2, 0, depth / 2), new Float3(width, 0, 0),
                new Float3(0, 0, -depth), new Float3(0, 1, 0), width, depth);
            return mesh;
        }

        private static SurfaceMeshData Box(float w, float h, float d)
        {
            var mesh = new SurfaceMeshData();
            // The pivot lies at the centre of the bottom face. Side UVs unwrap around the perimeter.
            Quad(mesh, new Float3(-w/2,0,d/2), new Float3(w,0,0), new Float3(0,h,0), new Float3(0,0,1), w,h);
            Quad(mesh, new Float3(w/2,0,d/2), new Float3(0,0,-d), new Float3(0,h,0), new Float3(1,0,0), d,h,w);
            Quad(mesh, new Float3(w/2,0,-d/2), new Float3(-w,0,0), new Float3(0,h,0), new Float3(0,0,-1), w,h,w+d);
            Quad(mesh, new Float3(-w/2,0,-d/2), new Float3(0,0,d), new Float3(0,h,0), new Float3(-1,0,0), d,h,2*w+d);
            Quad(mesh, new Float3(-w/2,h,d/2), new Float3(w,0,0), new Float3(0,0,-d), new Float3(0,1,0), w,d);
            Quad(mesh, new Float3(-w/2,0,-d/2), new Float3(w,0,0), new Float3(0,0,d), new Float3(0,-1,0), w,d);
            return mesh;
        }

        private static SurfaceMeshData Cylinder(float radius, float height, int segments, float seamDegrees)
        {
            var mesh = new SurfaceMeshData();
            float circumference = (float)(2 * Math.PI * radius);
            // Duplicate the seam vertices: the two UV coordinates differ by one circumference.
            for (int i = 0; i <= segments; i++)
            {
                float angle = (float)(2 * Math.PI * i / segments + (seamDegrees%360)*Math.PI/180);
                float x = (float)Math.Cos(angle), z = (float)Math.Sin(angle);
                for (int j = 0; j < 2; j++)
                {
                    mesh.Positions.Add(new Float3(radius*x, j*height, radius*z));
                    mesh.Normals.Add(new Float3(x,0,z));
                    mesh.MetricUV.Add(new Float2(circumference*i/segments, j*height));
                }
                if (i < segments)
                {
                    int a = i*2;
                    mesh.Triangles.AddRange(new[] { a,a+1,a+3, a,a+3,a+2 });
                }
            }
            Cap(mesh, radius, height, segments, true, seamDegrees);
            Cap(mesh, radius, 0, segments, false, seamDegrees);
            return mesh;
        }

        private static void Cap(SurfaceMeshData mesh, float radius, float y, int segments, bool top, float seamDegrees)
        {
            int centre = mesh.Positions.Count;
            var normal = new Float3(0, top ? 1 : -1, 0);
            mesh.Positions.Add(new Float3(0,y,0));
            mesh.Normals.Add(normal);
            mesh.MetricUV.Add(new Float2(radius,radius));
            for (int i = 0; i <= segments; i++)
            {
                float angle = (float)(2*Math.PI*i/segments + (seamDegrees%360)*Math.PI/180);
                float x = radius*(float)Math.Cos(angle), z = radius*(float)Math.Sin(angle);
                mesh.Positions.Add(new Float3(x,y,z));
                mesh.Normals.Add(normal);
                mesh.MetricUV.Add(new Float2(radius+x,radius+(top ? -z : z)));
                if (i < segments)
                {
                    int a = centre+1+i, b = a+1;
                    mesh.Triangles.AddRange(top ? new[] { centre,b,a } : new[] { centre,a,b });
                }
            }
        }
    }
}
