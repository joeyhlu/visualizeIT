using System;
using System.Collections.Generic;
using NUnit.Framework;
using VisualizeIt.Core;

namespace VisualizeIt.Tests
{
    public sealed class MobileTrackingResultGateTests
    {
        [Test]
        public void MaximumAgeMustBeFiniteAndPositiveSeconds()
        {
            double[] invalid = { double.NaN, double.PositiveInfinity, double.NegativeInfinity, 0, -0.25 };
            for (int i = 0; i < invalid.Length; i++)
                Assert.Throws<ArgumentOutOfRangeException>(() => new MobileTrackingResultGate(invalid[i]));
        }

        [Test]
        public void AcceptsExactFreshPoseAndCopiesPoseAcrossEveryBoundary()
        {
            var gate = NewGate(0.25);
            var suppliedPose = RotatedTranslatedPose();
            var result = Result(12, 40.0, pose: suppliedPose);
            suppliedPose[3] = 99;
            gate.Select("session-A", "asset-7");

            MobileTrackingGateDecision decision = Update(gate, result, 12, 40.0, 40.25);

            Assert.That(decision.IsRenderable, Is.True);
            Assert.That(decision.Reason, Is.EqualTo(MobileTrackingGateReason.None));
            Assert.That(decision.HasPose, Is.True);
            double[] decisionCopy = decision.CopyCvCameraFromGltfObjectPoseMetres();
            Assert.That(decisionCopy[3], Is.EqualTo(0.25).Within(1e-12));
            decisionCopy[3] = -100;
            double[] resultCopy = result.CopyCvCameraFromGltfObjectPoseMetres();
            resultCopy[3] = -200;
            Assert.That(gate.Current.CopyCvCameraFromGltfObjectPoseMetres()[3], Is.EqualTo(0.25).Within(1e-12));
        }

        [TestCase(MobileMaskAvailability.Unavailable, MobilePoseTrackingState.Tracking, MobileTrackingRenderState.Visible, MobileTrackingGateReason.MaskUnavailable)]
        [TestCase(MobileMaskAvailability.Available, MobilePoseTrackingState.Initializing, MobileTrackingRenderState.Visible, MobileTrackingGateReason.PoseNotTracking)]
        [TestCase(MobileMaskAvailability.Available, MobilePoseTrackingState.Recovering, MobileTrackingRenderState.Visible, MobileTrackingGateReason.PoseNotTracking)]
        [TestCase(MobileMaskAvailability.Available, MobilePoseTrackingState.Limited, MobileTrackingRenderState.Visible, MobileTrackingGateReason.PoseNotTracking)]
        [TestCase(MobileMaskAvailability.Available, MobilePoseTrackingState.Lost, MobileTrackingRenderState.Visible, MobileTrackingGateReason.PoseNotTracking)]
        [TestCase(MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking, MobileTrackingRenderState.Suppressed, MobileTrackingGateReason.RenderSuppressed)]
        public void EachDeclaredStateIndependentlySuppressesAndClearsPreviousPose(
            MobileMaskAvailability mask, MobilePoseTrackingState poseState,
            MobileTrackingRenderState renderState, MobileTrackingGateReason expectedReason)
        {
            var gate = NewGate(1.0);
            gate.Select("session-A", "asset-7");
            Assert.That(Update(gate, Result(0, 1.0), 0, 1.0, 1.0).IsRenderable, Is.True);

            var suppressed = Result(1, 1.1, mask: mask, poseState: poseState, renderState: renderState);
            MobileTrackingGateDecision decision = Update(gate, suppressed, 1, 1.1, 1.1);

            Assert.That(decision.IsRenderable, Is.False);
            Assert.That(decision.HasPose, Is.False);
            Assert.That(decision.Reason, Is.EqualTo(expectedReason));
            Assert.That(gate.Current.IsRenderable, Is.False);
            Assert.That(gate.Current.CopyCvCameraFromGltfObjectPoseMetres(), Is.Null);

            MobileTrackingGateDecision laterTick = gate.Observe(Observe(2, 1.2));
            Assert.That(laterTick.IsRenderable, Is.False);
            Assert.That(laterTick.HasPose, Is.False);
        }

        [Test]
        public void UsesSecondsAndAcceptsTheInclusiveAgeBoundary()
        {
            var gate = NewGate(0.25);
            gate.Select("session-A", "asset-7");
            MobileTrackingGateDecision boundary = Update(gate, Result(0, 1.0), 0, 1.0, 1.25);
            Assert.That(boundary.IsRenderable, Is.True);

            MobileTrackingGateDecision old = Update(gate, Result(1, 2.0), 1, 2.0, 2.250001);
            Assert.That(old.IsRenderable, Is.False);
            Assert.That(old.Reason, Is.EqualTo(MobileTrackingGateReason.StaleResult));
        }

        [Test]
        public void RejectsFutureResultAndObservationBeforeItsSourceFrame()
        {
            var futureGate = NewGate(1.0);
            futureGate.Select("session-A", "asset-7");
            MobileTrackingGateDecision future = Update(futureGate, Result(0, 5.0), 0, 5.0, 4.999);
            Assert.That(future.Reason, Is.EqualTo(MobileTrackingGateReason.ObservationBeforeCapture));

            var frameGate = NewGate(1.0);
            frameGate.Select("session-A", "asset-7");
            var olderObservation = new MobileTrackingObservation("session-A", 9, 5.0);
            MobileTrackingGateDecision ordered = frameGate.Update(Result(10, 5.0), Capture(10, 5.0), olderObservation);
            Assert.That(ordered.Reason, Is.EqualTo(MobileTrackingGateReason.ObservationPrecedesSourceFrame));
        }

        [Test]
        public void RejectsDuplicateAndOutOfOrderSourceFramesAndConsumesFailures()
        {
            var gate = NewGate(2.0);
            gate.Select("session-A", "asset-7");
            Assert.That(UpdateAt(gate, Result(4, 1.0), 4, 1.0, 4, 1.0).IsRenderable, Is.True);

            Assert.That(UpdateAt(gate, Result(3, 1.1), 3, 1.1, 4, 1.1).Reason,
                Is.EqualTo(MobileTrackingGateReason.OutOfOrderSourceFrameId));
            Assert.That(UpdateAt(gate, Result(5, 0.9), 5, 0.9, 5, 1.1).Reason,
                Is.EqualTo(MobileTrackingGateReason.OutOfOrderCaptureTimestamp));
            Assert.That(UpdateAt(gate, Result(5, 1.1), 5, 1.1, 5, 1.1).Reason,
                Is.EqualTo(MobileTrackingGateReason.DuplicateSourceFrameId));
            Assert.That(UpdateAt(gate, Result(6, 1.0), 6, 1.0, 6, 1.1).IsRenderable, Is.True,
                "Distinct source frames may share a monotonic timestamp; timestamps may not decrease.");
        }

        [Test]
        public void ResultMustMatchItsOriginalCapturePacketAndFailedPacketCannotBeRetried()
        {
            var gate = NewGate(1.0);
            gate.Select("session-A", "asset-7");
            var capture = Capture(8, 3.0);
            var mismatched = Result(7, 3.0);
            var observation = Observe(8, 3.0);

            Assert.That(gate.Update(mismatched, capture, observation).Reason,
                Is.EqualTo(MobileTrackingGateReason.ResultCaptureIdentityMismatch));
            Assert.That(gate.Update(Result(8, 3.0), capture, observation).Reason,
                Is.EqualTo(MobileTrackingGateReason.DuplicateSourceFrameId));

            var timestampGate = NewGate(1.0);
            timestampGate.Select("session-A", "asset-7");
            var timestampMismatch = Result(9, 3.000001);
            var timestampCapture = Capture(9, 3.0);
            var timestampObservation = Observe(9, 3.0);
            Assert.That(timestampGate.Update(timestampMismatch, timestampCapture, timestampObservation).Reason,
                Is.EqualTo(MobileTrackingGateReason.ResultCaptureIdentityMismatch));
            Assert.That(timestampGate.Update(Result(9, 3.0), timestampCapture, timestampObservation).Reason,
                Is.EqualTo(MobileTrackingGateReason.DuplicateSourceFrameId));
        }

        [Test]
        public void OldSessionAndAssetResultsSuppressWithoutPoisoningNewSelectionCursor()
        {
            var gate = NewGate(1.0);
            gate.Select("session-B", "asset-new");
            Assert.That(gate.Observe(Observe(80, 100.0, "session-A")).Reason,
                Is.EqualTo(MobileTrackingGateReason.SessionMismatch));
            var oldSession = new MobileTrackingResult("session-A", "asset-old", 50, 90.0,
                MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                MobileTrackingRenderState.Visible, IdentityPose());
            Assert.That(gate.Update(oldSession, Capture(50, 90.0, "session-A"),
                Observe(50, 90.0, "session-A")).Reason, Is.EqualTo(MobileTrackingGateReason.SessionMismatch));

            var oldAsset = new MobileTrackingResult("session-B", "asset-old", 50, 90.0,
                MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                MobileTrackingRenderState.Visible, IdentityPose());
            Assert.That(gate.Update(oldAsset, Capture(50, 90.0, "session-B"),
                Observe(50, 90.0, "session-B")).Reason, Is.EqualTo(MobileTrackingGateReason.TargetAssetMismatch));

            Assert.That(Update(gate, Result(0, 0.0, "session-B", "asset-new"), 0, 0.0, 0.0, "session-B").IsRenderable, Is.True);
        }

        [Test]
        public void RejectsInvalidFrameIdsAndTimesAndDoesNotAllowRetryOfInvalidPacket()
        {
            var gate = NewGate(1.0);
            gate.Select("session-A", "asset-7");
            Assert.That(UpdateAt(gate, Result(-1, 1.0), -1, 1.0, 0, 1.0).Reason,
                Is.EqualTo(MobileTrackingGateReason.InvalidSourceFrameId));
            Assert.That(UpdateAt(gate, Result(0, double.NaN), 0, double.NaN, 0, 1.0).Reason,
                Is.EqualTo(MobileTrackingGateReason.NonFiniteCaptureTimestamp));
            Assert.That(UpdateAt(gate, Result(0, 0.0), 0, 0.0, 0, 1.0).Reason,
                Is.EqualTo(MobileTrackingGateReason.DuplicateSourceFrameId));
            Assert.That(UpdateAt(gate, Result(1, -1.0), 1, -1.0, 1, 1.0).Reason,
                Is.EqualTo(MobileTrackingGateReason.NegativeCaptureTimestamp));
            Assert.That(UpdateAt(gate, Result(2, 0.0), 2, 0.0, 2, 1.0).IsRenderable, Is.True);
        }

        [Test]
        public void RejectsMalformedOrImproperPosesAndMissingTrackingPose()
        {
            var invalidPoses = new List<double[]>();
            invalidPoses.Add(new double[15]);
            double[] nonFinite = IdentityPose();
            nonFinite[3] = double.NaN;
            invalidPoses.Add(nonFinite);
            double[] nonAffine = IdentityPose();
            nonAffine[12] = 0.01;
            invalidPoses.Add(nonAffine);
            double[] scaled = IdentityPose();
            scaled[0] = 1.1;
            invalidPoses.Add(scaled);
            double[] reflected = IdentityPose();
            reflected[0] = -1;
            invalidPoses.Add(reflected);

            for (int i = 0; i < invalidPoses.Count; i++)
            {
                var gate = NewGate(1.0);
                gate.Select("session-A", "asset-7");
                MobileTrackingGateDecision invalid = Update(gate, Result(0, 1.0, pose: invalidPoses[i]), 0, 1.0, 1.0);
                Assert.That(invalid.Reason, Is.EqualTo(MobileTrackingGateReason.InvalidPose), "case " + i);
                Assert.That(invalid.HasPose, Is.False);
            }

            var missingGate = NewGate(1.0);
            missingGate.Select("session-A", "asset-7");
            MobileTrackingGateDecision missing = Update(missingGate,
                ResultWithoutPose(0, 1.0, MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                    MobileTrackingRenderState.Visible), 0, 1.0, 1.0);
            Assert.That(missing.Reason, Is.EqualTo(MobileTrackingGateReason.MissingPose));
        }

        [Test]
        public void InvalidObservationClearsCurrentPoseAndReturnsSpecificReason()
        {
            var gate = NewGate(0.1);
            gate.Select("session-A", "asset-7");
            Assert.That(Update(gate, Result(0, 2.0), 0, 2.0, 2.0).IsRenderable, Is.True);

            MobileTrackingGateDecision stale = Update(gate, Result(1, 2.25), 1, 2.25, 2.351);

            Assert.That(stale.Reason, Is.EqualTo(MobileTrackingGateReason.StaleResult));
            Assert.That(gate.Current.IsRenderable, Is.False);
            Assert.That(gate.Current.HasPose, Is.False);
        }

        [Test]
        public void ResetClearsSelectionResultAndFrameCursor()
        {
            var gate = NewGate(1.0);
            gate.Select("session-A", "asset-7");
            Assert.That(Update(gate, Result(20, 2.0), 20, 2.0, 2.0).IsRenderable, Is.True);
            Assert.That(gate.Observe(Observe(25, 5.0)).Reason,
                Is.EqualTo(MobileTrackingGateReason.StaleResult));
            gate.Reset();
            Assert.That(gate.IsSelected, Is.False);
            Assert.That(gate.Current.IsRenderable, Is.False);
            Assert.That(gate.Current.HasAcceptedCapture, Is.False);
            Assert.That(gate.Current.Reason, Is.EqualTo(MobileTrackingGateReason.NotSelected));
            Assert.That(Update(gate, Result(0, 0.0), 0, 0.0, 0.0).Reason,
                Is.EqualTo(MobileTrackingGateReason.NotSelected));

            gate.Select("session-B", "asset-7");
            Assert.That(Update(gate, Result(0, 0.0, "session-B", "asset-7"), 0, 0.0, 0.0, "session-B").IsRenderable, Is.True);
        }

        [Test]
        public void ObserveExpiresStalledResultAtInclusiveBoundaryAndNeverResurrectsIt()
        {
            var gate = NewGate(0.25);
            gate.Select("session-A", "asset-7");
            MobileTrackingGateDecision accepted = UpdateAt(gate, Result(4, 10.0), 4, 10.0, 10, 10.0);
            Assert.That(accepted.IsRenderable, Is.True);
            Assert.That(accepted.HasAcceptedCapture, Is.True);
            Assert.That(accepted.AcceptedCaptureFrameId, Is.EqualTo(4));
            Assert.That(accepted.AcceptedCaptureTimestampSeconds, Is.EqualTo(10.0));

            MobileTrackingGateDecision firstTick = gate.Observe(Observe(10, 10.1));
            MobileTrackingGateDecision boundary = gate.Observe(Observe(10, 10.25));
            Assert.That(firstTick.IsRenderable, Is.True);
            Assert.That(boundary.IsRenderable, Is.True);
            Assert.That(boundary.AcceptedCaptureFrameId, Is.EqualTo(4));
            Assert.That(boundary.AcceptedCaptureTimestampSeconds, Is.EqualTo(10.0));

            MobileTrackingGateDecision expired = gate.Observe(Observe(10, 10.250001));
            Assert.That(expired.IsRenderable, Is.False);
            Assert.That(expired.HasPose, Is.False);
            Assert.That(expired.Reason, Is.EqualTo(MobileTrackingGateReason.StaleResult));
            Assert.That(expired.AcceptedCaptureTimestampSeconds, Is.EqualTo(10.0));

            MobileTrackingGateDecision laterTick = gate.Observe(Observe(11, 10.3));
            Assert.That(laterTick.IsRenderable, Is.False);
            Assert.That(laterTick.HasPose, Is.False);
            Assert.That(laterTick.Reason, Is.EqualTo(MobileTrackingGateReason.StaleResult));

            MobileTrackingGateDecision restored = UpdateAt(gate, Result(5, 10.3), 5, 10.3, 11, 10.3);
            Assert.That(restored.IsRenderable, Is.True);
            Assert.That(restored.AcceptedCaptureFrameId, Is.EqualTo(5));
            Assert.That(restored.AcceptedCaptureTimestampSeconds, Is.EqualTo(10.3));
        }

        [Test]
        public void ObservationFrameFrontierIsSharedAcrossUpdateAndObserve()
        {
            var gate = NewGate(2.0);
            gate.Select("session-A", "asset-7");
            Assert.That(UpdateAt(gate, Result(4, 1.0), 4, 1.0, 10, 1.2).IsRenderable, Is.True);

            MobileTrackingGateDecision backwardsObserve = gate.Observe(Observe(9, 1.3));
            Assert.That(backwardsObserve.Reason, Is.EqualTo(MobileTrackingGateReason.OutOfOrderObservationFrameId));
            Assert.That(backwardsObserve.HasPose, Is.False);

            MobileTrackingGateDecision tiedObserve = UpdateAt(gate, Result(5, 1.1), 5, 1.1, 10, 1.3);
            Assert.That(tiedObserve.IsRenderable, Is.True,
                "The rejected observation did not rewind or advance the frame frontier; ties are valid.");

            MobileTrackingGateDecision backwardsUpdate = UpdateAt(gate, Result(6, 1.2), 6, 1.2, 9, 1.4);
            Assert.That(backwardsUpdate.Reason, Is.EqualTo(MobileTrackingGateReason.OutOfOrderObservationFrameId));
            Assert.That(backwardsUpdate.HasPose, Is.False);
            Assert.That(UpdateAt(gate, Result(7, 1.2), 7, 1.2, 10, 1.4).IsRenderable, Is.True);
        }

        [Test]
        public void ObservationTimeFrontierIsSharedAcrossUpdateAndObserve()
        {
            var gate = NewGate(2.0);
            gate.Select("session-A", "asset-7");
            Assert.That(UpdateAt(gate, Result(0, 1.0), 0, 1.0, 5, 1.2).IsRenderable, Is.True);

            MobileTrackingGateDecision backwardsObserve = gate.Observe(Observe(6, 1.1));
            Assert.That(backwardsObserve.Reason, Is.EqualTo(MobileTrackingGateReason.OutOfOrderObservationTime));
            Assert.That(backwardsObserve.HasPose, Is.False);

            MobileTrackingGateDecision tiedTime = UpdateAt(gate, Result(1, 1.1), 1, 1.1, 6, 1.2);
            Assert.That(tiedTime.IsRenderable, Is.True,
                "A rejected observation did not move the time frontier; ties are valid.");

            MobileTrackingGateDecision backwardsUpdate = UpdateAt(gate, Result(2, 1.2), 2, 1.2, 7, 1.15);
            Assert.That(backwardsUpdate.Reason, Is.EqualTo(MobileTrackingGateReason.OutOfOrderObservationTime));
            Assert.That(backwardsUpdate.HasPose, Is.False);

            MobileTrackingGateDecision observationTie = gate.Observe(Observe(7, 1.2));
            Assert.That(observationTie.IsRenderable, Is.False,
                "Observe advances a suppressed gate but cannot resurrect its pose.");
            Assert.That(UpdateAt(gate, Result(3, 1.2), 3, 1.2, 7, 1.2).IsRenderable, Is.True);
        }

        [Test]
        public void StaleObservationFrontierRejectsRewoundCallbackThenAllowsNewResult()
        {
            var gate = NewGate(0.1);
            gate.Select("session-A", "asset-7");
            Assert.That(UpdateAt(gate, Result(4, 1.0), 4, 1.0, 10, 1.0).IsRenderable, Is.True);
            MobileTrackingGateDecision stale = gate.Observe(Observe(12, 1.2));
            Assert.That(stale.Reason, Is.EqualTo(MobileTrackingGateReason.StaleResult));

            MobileTrackingGateDecision rewound = UpdateAt(gate, Result(5, 1.1), 5, 1.1, 12, 1.1);
            Assert.That(rewound.Reason, Is.EqualTo(MobileTrackingGateReason.OutOfOrderObservationTime));
            Assert.That(rewound.HasPose, Is.False);

            MobileTrackingGateDecision recovered = UpdateAt(gate, Result(6, 1.2), 6, 1.2, 13, 1.21);
            Assert.That(recovered.IsRenderable, Is.True);
        }

        [Test]
        public void InvalidObservationsSuppressWithoutMovingObservationFrontiers()
        {
            var gate = NewGate(1.0);
            gate.Select("session-A", "asset-7");
            Assert.That(UpdateAt(gate, Result(0, 1.0), 0, 1.0, 5, 1.0).IsRenderable, Is.True);

            MobileTrackingGateDecision nonFinite = gate.Observe(Observe(6, double.NaN));
            Assert.That(nonFinite.Reason, Is.EqualTo(MobileTrackingGateReason.NonFiniteObservationTime));
            Assert.That(nonFinite.HasPose, Is.False);
            MobileTrackingGateDecision negativeFrame = gate.Observe(Observe(-1, 1.1));
            Assert.That(negativeFrame.Reason, Is.EqualTo(MobileTrackingGateReason.InvalidObservationFrameId));
            MobileTrackingGateDecision negativeTime = gate.Observe(Observe(6, -1.0));
            Assert.That(negativeTime.Reason, Is.EqualTo(MobileTrackingGateReason.NegativeObservationTime));

            MobileTrackingGateDecision validButSuppressed = gate.Observe(Observe(6, 1.1));
            Assert.That(validButSuppressed.IsRenderable, Is.False,
                "A valid observation advances frontiers but cannot restore the pose cleared by invalid input.");
            Assert.That(UpdateAt(gate, Result(1, 1.1), 1, 1.1, 6, 1.1).IsRenderable, Is.True);
        }

        [Test]
        public void SelectClearsAcceptedPoseAndObservationClocksForANewEpoch()
        {
            var gate = NewGate(0.5);
            gate.Select("session-A", "asset-7");
            Assert.That(UpdateAt(gate, Result(40, 50.0), 40, 50.0, 100, 50.0).IsRenderable, Is.True);

            gate.Select("session-B", "asset-new");
            Assert.That(gate.Current.IsRenderable, Is.False);
            Assert.That(gate.Current.HasAcceptedCapture, Is.False);
            Assert.That(UpdateAt(gate, Result(0, 0.0, "session-B", "asset-new"), 0, 0.0, 0, 0.0, "session-B").IsRenderable, Is.True);
        }

        private static MobileTrackingResultGate NewGate(double maxAgeSeconds)
        {
            return new MobileTrackingResultGate(maxAgeSeconds);
        }

        private static MobileTrackingGateDecision Update(MobileTrackingResultGate gate,
            MobileTrackingResult result, long captureFrameId, double captureTime, double now,
            string sessionId = "session-A")
        {
            return gate.Update(result, Capture(captureFrameId, captureTime, sessionId),
                Observe(captureFrameId, now, sessionId));
        }

        private static MobileTrackingGateDecision UpdateAt(MobileTrackingResultGate gate,
            MobileTrackingResult result, long captureFrameId, double captureTime,
            long observationFrameId, double now, string sessionId = "session-A")
        {
            return gate.Update(result, Capture(captureFrameId, captureTime, sessionId),
                Observe(observationFrameId, now, sessionId));
        }

        private static MobileTrackingResult Result(long frameId, double timestamp,
            string sessionId = "session-A", string targetAssetId = "asset-7",
            MobileMaskAvailability mask = MobileMaskAvailability.Available,
            MobilePoseTrackingState poseState = MobilePoseTrackingState.Tracking,
            MobileTrackingRenderState renderState = MobileTrackingRenderState.Visible,
            double[] pose = null)
        {
            return new MobileTrackingResult(sessionId, targetAssetId, frameId, timestamp,
                mask, poseState, renderState, pose ?? IdentityPose());
        }

        private static MobileTrackingResult ResultWithoutPose(long frameId, double timestamp,
            MobileMaskAvailability mask, MobilePoseTrackingState poseState,
            MobileTrackingRenderState renderState)
        {
            return new MobileTrackingResult("session-A", "asset-7", frameId, timestamp,
                mask, poseState, renderState, null);
        }

        private static MobileCaptureIdentity Capture(long frameId, double timestamp, string sessionId = "session-A")
        {
            return new MobileCaptureIdentity(sessionId, frameId, timestamp);
        }

        private static MobileTrackingObservation Observe(long frameId, double now, string sessionId = "session-A")
        {
            return new MobileTrackingObservation(sessionId, frameId, now);
        }

        private static double[] IdentityPose()
        {
            return new double[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        }

        private static double[] RotatedTranslatedPose()
        {
            return new double[] { 0, -1, 0, 0.25, 1, 0, 0, -0.1, 0, 0, 1, 1.2, 0, 0, 0, 1 };
        }
    }
}

