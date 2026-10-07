using System;
using VisualizeIt.Core;

namespace VisualizeIt.Tests
{
    public static class CoreVerification
    {
        private static int checks;
        public static int Run()
        {
            checks = 0;
            foreach (SurfaceShape shape in Enum.GetValues(typeof(SurfaceShape)))
            {
                var mesh = SurfaceGeometry.Create(shape,0.1f,0.2f,0.15f);
                Check(mesh.Positions.Count == mesh.Normals.Count && mesh.Positions.Count == mesh.MetricUV.Count,"Attribute counts");
                Check(mesh.Triangles.Count % 3 == 0,"Triangle count");
                for (int i=0; i<mesh.Triangles.Count; i+=3)
                {
                    int a=mesh.Triangles[i],b=mesh.Triangles[i+1],c=mesh.Triangles[i+2];
                    Check(a>=0 && b>=0 && c>=0 && a<mesh.Positions.Count && b<mesh.Positions.Count && c<mesh.Positions.Count,"Index bounds");
                    var p=mesh.Positions[a]; var q=mesh.Positions[b]; var r=mesh.Positions[c];
                    var u=new Float3(q.X-p.X,q.Y-p.Y,q.Z-p.Z); var v=new Float3(r.X-p.X,r.Y-p.Y,r.Z-p.Z);
                    var cross=new Float3(u.Y*v.Z-u.Z*v.Y,u.Z*v.X-u.X*v.Z,u.X*v.Y-u.Y*v.X);
                    var n=mesh.Normals[a];
                    Check(cross.X*n.X+cross.Y*n.Y+cross.Z*n.Z>1e-10,"Outward winding and nondegenerate faces");
                }
                foreach (var normal in mesh.Normals)
                    Check(Math.Abs(normal.X*normal.X+normal.Y*normal.Y+normal.Z*normal.Z-1)<1e-5,"Unit normals");
            }
            var box=SurfaceGeometry.Create(SurfaceShape.Box,0.1f,0.2f,0.15f);
            float minX=float.MaxValue,maxX=float.MinValue,minY=float.MaxValue,maxY=float.MinValue;
            foreach (var point in box.Positions)
            {
                minX=Math.Min(minX,point.X); maxX=Math.Max(maxX,point.X);
                minY=Math.Min(minY,point.Y); maxY=Math.Max(maxY,point.Y);
            }
            Check(Math.Abs(maxX-minX-0.1)<1e-6 && Math.Abs(maxY-minY-0.2)<1e-6 && minY==0,"Metric bounds and bottom pivot");
            Check(Math.Abs(box.MetricUV[2].X-0.1)<1e-6 && Math.Abs(box.MetricUV[2].Y-0.2)<1e-6,"Box texture distances in metres");
            var cylinder=SurfaceGeometry.Create(SurfaceShape.Cylinder,0.1f,0.2f,0.1f,96);
            Check(Math.Abs(cylinder.MetricUV[192].X-0.1*Math.PI)<1e-6,"Cylinder UV uses physical circumference");
            Check(Math.Abs(cylinder.Positions[0].X-cylinder.Positions[192].X)<1e-6 && Math.Abs(cylinder.Positions[0].Z-cylinder.Positions[192].Z)<1e-6,"Cylinder seam closes geometrically");
            for (int i=0; i<=192; i+=2)
            {
                var p=cylinder.Positions[i];
                Check(Math.Abs(p.X*p.X+p.Z*p.Z-0.05*0.05)<1e-7,"Cylinder surface radius");
            }
            Reject(() => SurfaceGeometry.Create(SurfaceShape.Box,float.NaN,0.1f,0.1f));
            Reject(() => SurfaceGeometry.Create(SurfaceShape.Box,0,0.1f,0.1f));
            Reject(() => SurfaceGeometry.Create(SurfaceShape.Box,0.1f,float.PositiveInfinity,0.1f));
            Reject(() => SurfaceGeometry.Create(SurfaceShape.Cylinder,0.1f,0.1f,0.1f,3));
            Reject(() => SurfaceGeometry.Create((SurfaceShape)99,0.1f,0.1f,0.1f));
            var visibility=new TrackingVisibility();
            visibility.Tick(true,true,true,0.2f);
            Check(visibility.Opacity==0,"Hold before initial attachment");
            visibility.Tick(true,true,true,0.2f);
            Check(visibility.CanRender && visibility.Opacity>0,"Recovery after stable tracking");
            visibility.Tick(true,true,true,0.3f);
            Check(visibility.Opacity==1,"Fully visible after recovery");
            visibility.Tick(true,false,true,0.12f);
            Check(visibility.Opacity==0 && !visibility.CanRender,"Anchor loss hides the surface even if session tracks");
            visibility.Tick(true,true,true,0.4f);
            visibility.Tick(false,true,true,0.12f);
            Check(visibility.Opacity==0,"Session loss hides the surface");
            visibility.Tick(true,true,true,0.4f);
            visibility.Tick(true,true,false,0.12f);
            Check(visibility.Opacity==0,"Backgrounded camera hides the surface");
            visibility.Reset();
            visibility.Tick(true,true,true,0.1f);
            Check(visibility.Opacity==0,"Reset clears recovery history");
            Reject(() => visibility.Tick(true,true,true,float.NaN));
            Reject(() => visibility.Tick(true,true,true,-1));
            MappingChecks();
            return checks;
        }

        private static void MappingChecks()
        {
            foreach (SurfaceShape shape in Enum.GetValues(typeof(SurfaceShape)))
            {
                var mesh=SurfaceGeometry.Create(shape,0.1f,0.2f,0.15f);
                var quality=MeshMappingAnalysis.Analyze(mesh);
                Check(quality.Reliable,"All fixture triangles have invertible surface mapping");
                Check(quality.MaximumScaleError<0.0002,"Metric texture scale within 0.02% on fixture triangles");
                Check(quality.MaximumAnisotropy<1.0002,"Fixture local texture anisotropy within 0.02%");
            }
            var stretched=SurfaceGeometry.Create(SurfaceShape.Plane,0.1f,0.2f,0.15f);
            for (int i=0;i<stretched.MetricUV.Count;i++)
            {
                var uv=stretched.MetricUV[i]; stretched.MetricUV[i]=new Float2(uv.X*2,uv.Y);
            }
            var bad=MeshMappingAnalysis.Analyze(stretched);
            Check(Math.Abs(bad.MinimumStretch-0.5)<1e-6 && Math.Abs(bad.MaximumStretch-1)<1e-6,"Detect deliberately doubled U coordinates");
            Check(Math.Abs(bad.MaximumAnisotropy-2)<1e-6,"Detect 2:1 local texture deformation");
            stretched.MetricUV[1]=stretched.MetricUV[0];
            Check(!MeshMappingAnalysis.Analyze(stretched).Reliable,"Collapsed UV triangle is unavailable, not accepted");

            Float2 mapped=PatternMapping.TextureUV(new Float2(0.04f,0.08f),0.04f,2,0);
            Check(Math.Abs(mapped.X-1)<1e-6 && Math.Abs(mapped.Y-1)<1e-6,"Rectangular image keeps its physical aspect ratio");
            mapped=PatternMapping.TextureUV(new Float2(0.04f,0),0.04f,1,90);
            Check(Math.Abs(mapped.X)<1e-6 && Math.Abs(mapped.Y-1)<1e-6,"Physical rotation precedes texture normalization");
            foreach (float diameter in new[] {0.005f,0.066f,0.1f,1f,5f})
            {
                float fitted=PatternMapping.FitCylinderTileWidth(diameter,0.037f);
                Check(PatternMapping.CylinderSeamPhaseError(diameter,fitted,1,0)<1e-4,"Fitted repeats close the side seam at zero rotation");
            }
            Check(PatternMapping.CylinderSeamPhaseError(0.1f,0.05f,1,0)>0.2f,"Unfitted circumference exposes repeat phase discontinuity");
            float width=PatternMapping.FitCylinderTileWidth(0.1f,0.05f);
            Check(PatternMapping.CylinderSeamPhaseError(0.1f,width,1,45)>0.1f,"Rotated repeat is not silently claimed seamless");
            float minimumFitted=PatternMapping.FitCylinderTileWidth(0.1f,0.005f);
            Check(minimumFitted>=0.005f && PatternMapping.CylinderSeamPhaseError(0.1f,minimumFitted,1,0)<1e-4,"Seam fit respects the shader's minimum tile width");
            float maximumFitted=PatternMapping.FitCylinderTileWidth(0.45f,1);
            Check(maximumFitted<=1 && PatternMapping.CylinderSeamPhaseError(0.45f,maximumFitted,1,0)<1e-4,"Seam fit respects the saved project's maximum tile width");
            foreach (float seam in new[] {-180f,-30f,0f,90f,180f})
            {
                var mesh=SurfaceGeometry.Create(SurfaceShape.Cylinder,0.1f,0.2f,0.1f,96,seam);
                Check(MeshMappingAnalysis.Analyze(mesh).MaximumScaleError<0.0002,"Seam relocation preserves local physical texture scale");
                var start=mesh.Positions[0]; var end=mesh.Positions[192];
                Check(Math.Abs(start.X-end.X)<1e-6 && Math.Abs(start.Z-end.Z)<1e-6,"Relocated cylinder seam still closes");
                Check(Math.Abs(start.X-0.05*Math.Cos(seam*Math.PI/180))<1e-6,"Seam position follows requested angle");
            }
            Reject(() => PatternMapping.TextureUV(new Float2(0,0),0,1,0));
            Reject(() => PatternMapping.TextureUV(new Float2(0,0),0.05f,float.NaN,0));
            Reject(() => PatternMapping.FitCylinderTileWidth(-1,0.05f));
            Reject(() => SurfaceGeometry.Create(SurfaceShape.Cylinder,0.1f,0.2f,0.1f,96,float.NaN));
        }

        private static void Check(bool valid,string description)
        {
            if (!valid) throw new Exception(description);
            checks++;
        }
        private static void Reject(Action action)
        {
            try { action(); }
            catch (ArgumentOutOfRangeException) { checks++; return; }
            throw new Exception("Invalid input was accepted");
        }
    }
}
