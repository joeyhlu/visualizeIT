using System;
using VisualizeIt.Tests;

public static class CoreTestRunner
{
    public static int Main()
    {
        try { Console.WriteLine("PASS: " + CoreVerification.Run() + " geometry and tracking checks"); return 0; }
        catch (Exception error) { Console.Error.WriteLine("FAIL: " + error.Message); return 1; }
    }
}
