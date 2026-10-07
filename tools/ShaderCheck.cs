// Copied into a temporary editor project by verify-shader.ps1; never part of the phone app.
using System;
using UnityEditor;
using UnityEditor.Rendering;
using UnityEngine;

public static class ShaderCheck
{
    public static void Run()
    {
        var shader=AssetDatabase.LoadAssetAtPath<Shader>("Assets/SurfacePattern.shader");
        if (shader == null) throw new Exception("Surface shader was not imported.");
        var material=new Material(shader);
        for (int pass=0; pass<material.passCount; pass++) ShaderUtil.CompilePass(material,pass,true);
        int errors=0;
        foreach (var message in ShaderUtil.GetShaderMessages(shader))
        {
            if (message.severity==ShaderCompilerMessageSeverity.Error) { Debug.LogError(message.message); errors++; }
        }
        UnityEngine.Object.DestroyImmediate(material);
        Debug.Log("Shader verification: editor=" + Application.unityVersion + ", errors=" + errors);
        if (errors>0) throw new Exception("Shader compilation failed.");
        EditorApplication.Exit(0);
    }
}
