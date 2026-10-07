using System;
using System.Globalization;
using System.IO;
using System.Text;
using VisualizeIt.Core;

public static class ExportGeometry
{
    private static string F(float value) => value.ToString("R",CultureInfo.InvariantCulture);
    private static string D(double value) => value.ToString("R",CultureInfo.InvariantCulture);
    public static void Main(string[] args)
    {
        Directory.CreateDirectory(args[0]);
        foreach (SurfaceShape shape in Enum.GetValues(typeof(SurfaceShape)))
        {
            var mesh=SurfaceGeometry.Create(shape,0.1f,0.2f,0.15f);
            var json=new StringBuilder("{\"shape\":\"").Append(shape).Append("\",\"positions\":[");
            for (int i=0;i<mesh.Positions.Count;i++)
            {
                if (i>0) json.Append(','); var p=mesh.Positions[i];
                json.Append('[').Append(F(p.X)).Append(',').Append(F(p.Y)).Append(',').Append(F(p.Z)).Append(']');
            }
            json.Append("],\"normals\":[");
            for (int i=0;i<mesh.Normals.Count;i++)
            {
                if (i>0) json.Append(','); var n=mesh.Normals[i];
                json.Append('[').Append(F(n.X)).Append(',').Append(F(n.Y)).Append(',').Append(F(n.Z)).Append(']');
            }
            json.Append("],\"uvMetres\":[");
            for (int i=0;i<mesh.MetricUV.Count;i++)
            {
                if (i>0) json.Append(','); var uv=mesh.MetricUV[i];
                json.Append('[').Append(F(uv.X)).Append(',').Append(F(uv.Y)).Append(']');
            }
            json.Append("],\"triangles\":[").Append(string.Join(",",mesh.Triangles)).Append("]");
            var quality=MeshMappingAnalysis.Analyze(mesh);
            json.Append(",\"mappingQuality\":{\"invalidTriangles\":").Append(quality.InvalidTriangleCount)
                .Append(",\"minimumStretch\":").Append(D(quality.MinimumStretch))
                .Append(",\"maximumStretch\":").Append(D(quality.MaximumStretch))
                .Append(",\"maximumAnisotropy\":").Append(D(quality.MaximumAnisotropy))
                .Append(",\"maximumScaleError\":").Append(D(quality.MaximumScaleError)).Append("}");
            json.Append(",\"patternCases\":[");
            string[] names={"requested","fitted","rotated_rectangular"};
            for (int material=0;material<names.Length;material++)
            {
                if (material>0) json.Append(',');
                float width=material==1 && shape==SurfaceShape.Cylinder ? PatternMapping.FitCylinderTileWidth(0.1f,0.05f) : 0.05f;
                float aspect=material==2 ? 2 : 1, rotation=material==2 ? 35 : 0;
                json.Append("{\"name\":\"").Append(names[material]).Append("\",\"tileWidth\":").Append(F(width))
                    .Append(",\"aspect\":").Append(F(aspect)).Append(",\"rotationDegrees\":").Append(F(rotation));
                if (shape==SurfaceShape.Cylinder)
                    json.Append(",\"seamPhaseError\":").Append(F(PatternMapping.CylinderSeamPhaseError(0.1f,width,aspect,rotation)));
                json.Append(",\"textureUV\":[");
                for (int i=0;i<mesh.MetricUV.Count;i++)
                {
                    if (i>0) json.Append(',');
                    var uv=PatternMapping.TextureUV(mesh.MetricUV[i],width,aspect,rotation);
                    json.Append('[').Append(F(uv.X)).Append(',').Append(F(uv.Y)).Append(']');
                }
                json.Append("]}");
            }
            json.Append("]}");
            File.WriteAllText(Path.Combine(args[0],shape.ToString().ToLowerInvariant()+".json"),json.ToString());
            Console.WriteLine(shape+": "+mesh.Positions.Count+" vertices, "+mesh.Triangles.Count/3+" triangles");
        }
    }
}
