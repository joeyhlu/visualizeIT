using System;
using System.IO;
using UnityEngine;
using VisualizeIt.Core;

namespace VisualizeIt
{
    [Serializable]
    public sealed class FixtureProject
    {
        public int version = 1;
        public SurfaceShape shape = SurfaceShape.Cylinder;
        public Vector3 dimensions = new Vector3(0.066f,0.122f,0.066f);
        public float patternMetres = 0.05f;
        public float patternRotation;
        public float seamDegrees;
        public float roughness = 0.6f;
        public float shellMetres = 0.002f;
        public float yawDegrees;
        public Vector3 alignmentOffset;
        public bool useCheckerboard = true;
    }

    public static class ProjectStore
    {
        private static string DirectoryPath => Path.Combine(Application.persistentDataPath,"project");
        public static string DesignPath => Path.Combine(DirectoryPath,"design.png");
        public static void Save(FixtureProject project, byte[] design)
        {
            Validate(project);
            Directory.CreateDirectory(DirectoryPath);
            if (design != null) AtomicWrite(DesignPath,design);
            AtomicWrite(Path.Combine(DirectoryPath,"project.json"),System.Text.Encoding.UTF8.GetBytes(JsonUtility.ToJson(project,true)));
        }

        public static FixtureProject Load()
        {
            string path = Path.Combine(DirectoryPath,"project.json");
            if (!File.Exists(path)) return null;
            var project = JsonUtility.FromJson<FixtureProject>(File.ReadAllText(path));
            Validate(project);
            return project;
        }

        private static void Validate(FixtureProject project)
        {
            if (project == null || project.version != 1) throw new InvalidDataException("Unsupported saved project.");
            SurfaceGeometry.Create(project.shape,project.dimensions.x,project.dimensions.y,project.dimensions.z);
            if (!Finite(project.patternMetres) || project.patternMetres < 0.005f || project.patternMetres > 1
                || !Finite(project.roughness) || project.roughness < 0 || project.roughness > 1
                || !Finite(project.shellMetres) || project.shellMetres < 0 || project.shellMetres > 0.01f
                || !Finite(project.yawDegrees) || !Finite(project.patternRotation)
                || !Finite(project.seamDegrees)
                || !Finite(project.alignmentOffset.x) || !Finite(project.alignmentOffset.y) || !Finite(project.alignmentOffset.z)
                || project.alignmentOffset.magnitude > 1)
                throw new InvalidDataException("Saved settings are outside the supported range.");
        }
        private static bool Finite(float x) => !float.IsNaN(x) && !float.IsInfinity(x);
        private static void AtomicWrite(string path, byte[] bytes)
        {
            string temporary = path + ".tmp";
            File.WriteAllBytes(temporary,bytes);
            if (File.Exists(path)) File.Replace(temporary,path,null);
            else File.Move(temporary,path);
        }
    }
}
