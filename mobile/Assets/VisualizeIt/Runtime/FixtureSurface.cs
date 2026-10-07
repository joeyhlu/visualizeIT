using UnityEngine;
using UnityEngine.Rendering;
using VisualizeIt.Core;

namespace VisualizeIt
{
    public sealed class FixtureSurface : MonoBehaviour
    {
        private Mesh mesh;
        private Material material;
        private MeshRenderer meshRenderer;

        public void Initialize(Shader shader, SurfaceShape shape, Vector3 size, float seamDegrees = 0)
        {
            if (shader == null) throw new System.InvalidOperationException("The surface shader is missing.");
            material = new Material(shader);
            gameObject.AddComponent<MeshFilter>();
            meshRenderer = gameObject.AddComponent<MeshRenderer>();
            meshRenderer.sharedMaterial = material;
            meshRenderer.shadowCastingMode = ShadowCastingMode.Off;
            meshRenderer.receiveShadows = true;
            Rebuild(shape, size, seamDegrees);
        }

        public void Rebuild(SurfaceShape shape, Vector3 size, float seamDegrees = 0)
        {
            var data = SurfaceGeometry.Create(shape, size.x, size.y, size.z,96,seamDegrees);
            if (mesh != null) Destroy(mesh);
            mesh = new Mesh { name = "Measured " + shape };
            mesh.vertices = data.Positions.ConvertAll(p => new Vector3(p.X,p.Y,p.Z)).ToArray();
            mesh.normals = data.Normals.ConvertAll(p => new Vector3(p.X,p.Y,p.Z)).ToArray();
            mesh.uv = data.MetricUV.ConvertAll(p => new Vector2(p.X,p.Y)).ToArray();
            mesh.triangles = data.Triangles.ToArray();
            mesh.RecalculateBounds();
            var bounds = mesh.bounds;
            bounds.Expand(0.02f); // Account for the maximum 10 mm shader surface lift.
            mesh.bounds = bounds;
            GetComponent<MeshFilter>().sharedMesh = mesh;
        }

        public void Apply(Texture texture, float patternMetres, float rotationDegrees,
            float roughness, float opacity, float shellMetres)
        {
            material.SetTexture("_BaseMap", texture);
            material.SetFloat("_PatternMetres", Mathf.Max(0.005f, patternMetres));
            material.SetFloat("_PatternRotation", (rotationDegrees%360) * Mathf.Deg2Rad);
            material.SetFloat("_Smoothness", 1 - Mathf.Clamp01(roughness));
            material.SetFloat("_Visibility", opacity);
            material.SetFloat("_ShellMetres", shellMetres);
            meshRenderer.enabled = opacity > 0;
        }

        private void OnDestroy()
        {
            if (mesh != null) Destroy(mesh);
            if (material != null) Destroy(material);
        }
    }
}
