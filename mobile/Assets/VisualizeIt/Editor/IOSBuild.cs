#if UNITY_IOS
using System.IO;
using UnityEditor;
using UnityEditor.Callbacks;
using UnityEditor.iOS.Xcode;

namespace VisualizeIt.Editor
{
    public static class IOSBuild
    {
        [PostProcessBuild(100)]
        public static void LinkPhotoPicker(BuildTarget target,string path)
        {
            if (target != BuildTarget.iOS) return;
            string projectPath = PBXProject.GetPBXProjectPath(path);
            var project = new PBXProject(); project.ReadFromFile(projectPath);
            project.AddFrameworkToProject(project.GetUnityFrameworkTargetGuid(),"PhotosUI.framework",false);
            project.WriteToFile(projectPath);
        }
    }
}
#endif
