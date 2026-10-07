using NUnit.Framework;

namespace VisualizeIt.Tests
{
    public sealed class CoreTests
    {
        [Test]
        public void GeometryAndTrackingInvariants() => Assert.Greater(CoreVerification.Run(),1000);
    }
}
