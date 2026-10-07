using System;
using System.IO;
using System.Linq;
using Microsoft.CodeAnalysis;
using Microsoft.CodeAnalysis.CSharp;

public static class SyntaxCheck
{
    public static int Main(string[] args)
    {
        int errors=0,files=0;
        var combinations=new[] {
            new[] { "UNITY_EDITOR", "UNITY_6000_0_OR_NEWER", "UNITY_6000_3_OR_NEWER" },
            new[] { "UNITY_ANDROID", "UNITY_6000_0_OR_NEWER", "UNITY_6000_3_OR_NEWER" },
            new[] { "UNITY_IOS", "UNITY_6000_0_OR_NEWER", "UNITY_6000_3_OR_NEWER" }
        };
        foreach (string path in Directory.GetFiles(args[0],"*.cs",SearchOption.AllDirectories))
        {
            files++;
            foreach (var symbols in combinations)
            {
                var tree=CSharpSyntaxTree.ParseText(File.ReadAllText(path),new CSharpParseOptions(LanguageVersion.Latest,preprocessorSymbols:symbols),path);
                foreach (var error in tree.GetDiagnostics().Where(d => d.Severity==DiagnosticSeverity.Error))
                { Console.Error.WriteLine(error); errors++; }
            }
        }
        Console.WriteLine((errors==0 ? "PASS" : "FAIL") + ": syntax of " + files + " C# files in editor, Android, and iOS configurations");
        Console.WriteLine("Syntax validation does not establish Unity 6 API compatibility or successful device builds.");
        return errors==0 ? 0 : 1;
    }
}
