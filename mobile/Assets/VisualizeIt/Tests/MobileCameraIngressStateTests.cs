using System;
using System.Threading;
using NUnit.Framework;
using VisualizeIt.Core;

namespace VisualizeIt.Tests
{
    /// <summary>
    /// Independent state-machine checks for the pure-System camera ingress seam. These fixtures
    /// exercise synthetic metadata only; they do not establish a provider clock, image basis,
    /// camera pose, tracker accuracy, or phone readiness.
    /// </summary>
    public sealed class MobileCameraIngressStateTests
    {
        private const string SessionA = "session-A";
        private const string SessionB = "session-B";
        private const string TargetA = "target-A";
        private const string TargetB = "target-B";
        private const double GateAgeSeconds = 5.0;

        private sealed class Fixture
        {
            public readonly MobileTrackingResultGate Gate;
            public readonly MobileCameraIngressState State;
            public readonly string SessionId;
            public readonly string TargetAssetId;

            public Fixture(int maxRgb24Bytes, string sessionId, string targetAssetId)
            {
                Gate = new MobileTrackingResultGate(GateAgeSeconds);
                State = new MobileCameraIngressState(Gate, maxRgb24Bytes);
                SessionId = sessionId;
                TargetAssetId = targetAssetId;
                State.SetSessionTracking(true);
                MobileCameraIngressReason reason;
                Assert.That(State.BeginEpoch(sessionId, targetAssetId, out reason), Is.True);
                Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.None));
            }
        }

        [TestCase(true)]
        [TestCase(false)]
        public void BackwardRenderClockRetiresSelectedAndDiagnosticEpochs(bool selectedTarget)
        {
            Fixture fixture = NewFixture(selectedTarget ? TargetA : null);
            MobileCameraIngressState state = fixture.State;
            long generation = state.Generation;

            state.RenderTick(10.0);
            Assert.That(state.HasEpoch, Is.True);
            state.RenderTick(9.0);

            AssertRetired(state, MobileCameraIngressReason.ObservationTimestampRegressed, generation);
            Assert.That(state.HasPendingPacket, Is.False);
            Assert.That(state.HasInFlightPacket, Is.False);
            Assert.That(fixture.Gate.IsSelected, Is.False);
        }

        [TestCase(true)]
        [TestCase(false)]
        public void InvalidFirstRenderClockRetiresSelectedAndDiagnosticEpochs(bool selectedTarget)
        {
            double[] invalidTimes =
            {
                double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.01
            };

            for (int i = 0; i < invalidTimes.Length; i++)
            {
                Fixture fixture = NewFixture(selectedTarget ? TargetA : null);
                long generation = fixture.State.Generation;
                fixture.State.RenderTick(invalidTimes[i]);
                AssertRetired(fixture.State, MobileCameraIngressReason.InvalidObservationTimestamp, generation);
                Assert.That(fixture.Gate.IsSelected, Is.False, "case " + i);
            }
        }

        [Test]
        public void RenderClockSharesReceiptFrontierAndRetiresReceiptThatPredatesRender()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.5, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            long generation = fixture.State.Generation;

            fixture.State.RenderTick(9.0);

            AssertRetired(fixture.State, MobileCameraIngressReason.ObservationTimestampRegressed, generation);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
            Assert.That(fixture.State.HasPendingPacket, Is.False);
        }

        [Test]
        public void ProviderReceiptPredatingRenderFrontierRetiresEpoch()
        {
            Fixture fixture = NewFixture();
            fixture.State.RenderTick(10.0);
            long generation = fixture.State.Generation;
            MobileCameraIngressReason reason;

            MobileCameraIngressOfferStatus status = Offer(fixture, 1, 1000.0, 9.0, 8.5, true, out reason);

            Assert.That(status, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.ReceiptTimestampRegressed));
            AssertRetired(fixture.State, MobileCameraIngressReason.ReceiptTimestampRegressed, generation);
        }

        [Test]
        public void ReceiptAndRenderTiesAndMappedCaptureTiesRemainLegal()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            Assert.That(Offer(fixture, 2, 1001.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.ReplacedPending));
            Assert.That(fixture.State.Generation, Is.EqualTo(1));
            Assert.That(fixture.State.HasEpoch, Is.True);
            Assert.That(fixture.State.LatestFrameId, Is.EqualTo(1));
            Assert.That(fixture.State.DroppedPendingPacketCount, Is.EqualTo(1));

            fixture.State.RenderTick(10.0);
            fixture.State.RenderTick(10.0);

            Assert.That(fixture.State.HasEpoch, Is.True);
            Assert.That(fixture.State.Generation, Is.EqualTo(1));
        }

        [Test]
        public void RenderTicksWithoutNewCameraCallbacksStillAdvanceGateFreshnessAtLatestFrameFrontier()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.8, false, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(Offer(fixture, 2, 1001.0, 10.2, 10.0, false, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(fixture.State.LatestFrameId, Is.EqualTo(1));

            // A direct synthetic gate seed makes coordinator observation forwarding visible.
            // It does not model a result accepted by ingress, whose camera-pose guard stays shut.
            MobileCaptureIdentity capture = new MobileCaptureIdentity(SessionA, 0, 10.0);
            MobileTrackingResult seeded = Result(SessionA, TargetA, 0, 10.0,
                MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                MobileTrackingRenderState.Visible, IdentityPose());
            Assert.That(fixture.Gate.Update(seeded, capture,
                new MobileTrackingObservation(SessionA, 1, 10.2)).IsRenderable, Is.True);

            MobileTrackingGateDecision fresh = fixture.State.RenderTick(10.5);
            Assert.That(fresh.IsRenderable, Is.True);
            Assert.That(fresh.SourceFrameId, Is.EqualTo(0));
            MobileTrackingGateDecision expired = fixture.State.RenderTick(15.0001);

            Assert.That(expired.IsRenderable, Is.False);
            Assert.That(expired.HasPose, Is.False);
            Assert.That(expired.Reason, Is.EqualTo(MobileTrackingGateReason.StaleResult));
            Assert.That(fixture.State.HasEpoch, Is.True);
            Assert.That(fixture.State.LatestFrameId, Is.EqualTo(1));
        }

        [Test]
        public void FutureMappedCaptureFailsClosedBeforePacketCreation()
        {
            Fixture fixture = NewFixture();
            long generation = fixture.State.Generation;
            MobileCameraIngressReason reason;

            MobileCameraIngressOfferStatus status = Offer(fixture, 1, 1000.0, 10.0, 10.001, true, out reason);

            Assert.That(status, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.CaptureTimestampAfterReceipt));
            AssertRetired(fixture.State, MobileCameraIngressReason.CaptureTimestampAfterReceipt, generation);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(0));
        }

        [Test]
        public void CaptureMayTieReceiptAndRemainEqualAcrossDistinctProviderImages()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 10.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            Assert.That(Offer(fixture, 2, 1001.0, 11.0, 10.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.ReplacedPending));

            MobileCameraIngressPacket packet = BeginWork(fixture.State);

            Assert.That(packet.FrameId, Is.EqualTo(1));
            Assert.That(packet.ProviderImageTimestampSeconds, Is.EqualTo(1001.0));
            Assert.That(packet.CallbackReceiptTimestampSeconds, Is.EqualTo(11.0));
            Assert.That(packet.CaptureIdentity.CaptureTimestampSeconds, Is.EqualTo(10.0));
            Assert.That(fixture.State.HasEpoch, Is.True);
        }

        [Test]
        public void RawProviderEventAndLocalTimesRemainDistinctAndUnmodified()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            byte[] rgb = Pixels(7, 2, 2);
            long eventTimestamp = 2345000000000L;
            MobileCameraIngressOfferStatus status = fixture.State.OfferFrame(rgb, 2, 2,
                Intrinsics(2, 2), 1000.0, eventTimestamp, 10.0, 9.5, true, out reason);

            Assert.That(status, Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);
            Assert.That(packet.ProviderImageTimestampSeconds, Is.EqualTo(1000.0));
            Assert.That(packet.FrameEventTimestampNanoseconds, Is.EqualTo((long?)eventTimestamp));
            Assert.That(packet.CallbackReceiptTimestampSeconds, Is.EqualTo(10.0));
            Assert.That(packet.CaptureIdentity.CaptureTimestampSeconds, Is.EqualTo(9.5));
            Assert.That(packet.CanDispatchInference, Is.True);
        }

        [Test]
        public void InvalidMappedCapturesAndMappedRegressionRetireAndReleasePendingStorage()
        {
            double[] invalidCaptures = { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.1 };
            for (int i = 0; i < invalidCaptures.Length; i++)
            {
                Fixture fixture = NewFixture();
                long generation = fixture.State.Generation;
                MobileCameraIngressReason reason;
                MobileCameraIngressOfferStatus status = Offer(fixture, 1, 1000.0, 10.0,
                    invalidCaptures[i], true, out reason);
                Assert.That(status, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected), "case " + i);
                Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.InvalidMappedCaptureTimestamp), "case " + i);
                AssertRetired(fixture.State, MobileCameraIngressReason.InvalidMappedCaptureTimestamp, generation);
                Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(0), "case " + i);
            }

            Fixture regressed = NewFixture();
            MobileCameraIngressReason regressionReason;
            Assert.That(Offer(regressed, 1, 1000.0, 10.0, 9.0, true, out regressionReason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            long before = regressed.State.Generation;
            MobileCameraIngressOfferStatus regressionStatus = Offer(regressed, 2, 1001.0, 11.0,
                8.9, true, out regressionReason);
            Assert.That(regressionStatus, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(regressionReason, Is.EqualTo(MobileCameraIngressReason.CaptureTimestampRegressed));
            AssertRetired(regressed.State, MobileCameraIngressReason.CaptureTimestampRegressed, before);
            Assert.That(regressed.State.ReleasedPacketCount, Is.EqualTo(1));
        }

        [Test]
        public void InvalidReceiptTimesAndReceiptRegressionRetireTheEpoch()
        {
            double[] invalidReceipts = { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.1 };
            for (int i = 0; i < invalidReceipts.Length; i++)
            {
                Fixture fixture = NewFixture();
                long generation = fixture.State.Generation;
                MobileCameraIngressReason reason;
                MobileCameraIngressOfferStatus status = Offer(fixture, 1, 1000.0,
                    invalidReceipts[i], 0.0, true, out reason);
                Assert.That(status, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected), "case " + i);
                Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.InvalidReceiptTimestamp), "case " + i);
                AssertRetired(fixture.State, MobileCameraIngressReason.InvalidReceiptTimestamp, generation);
            }

            Fixture regressed = NewFixture();
            regressed.State.RenderTick(10.0);
            long before = regressed.State.Generation;
            MobileCameraIngressReason regressionReason;
            MobileCameraIngressOfferStatus regressionStatus = Offer(regressed, 1, 1000.0,
                9.999, 9.0, true, out regressionReason);
            Assert.That(regressionStatus, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(regressionReason, Is.EqualTo(MobileCameraIngressReason.ReceiptTimestampRegressed));
            AssertRetired(regressed.State, MobileCameraIngressReason.ReceiptTimestampRegressed, before);
        }

        [Test]
        public void ProviderTimestampInvalidityAndStrictRegressionRetireButDuplicateIsDiagnosticOnly()
        {
            double[] invalidProviderTimes = { double.NaN, double.PositiveInfinity, double.NegativeInfinity, -0.1 };
            for (int i = 0; i < invalidProviderTimes.Length; i++)
            {
                Fixture fixture = NewFixture();
                long generation = fixture.State.Generation;
                MobileCameraIngressReason reason;
                MobileCameraIngressOfferStatus status = Offer(fixture, 1, invalidProviderTimes[i],
                    10.0, 9.0, true, out reason);
                Assert.That(status, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected), "case " + i);
                Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.InvalidProviderImageTimestamp), "case " + i);
                AssertRetired(fixture.State, MobileCameraIngressReason.InvalidProviderImageTimestamp, generation);
            }

            Fixture duplicate = NewFixture();
            MobileCameraIngressReason duplicateReason;
            Assert.That(Offer(duplicate, 1, 1000.0, 10.0, 9.0, true, out duplicateReason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressOfferStatus duplicateStatus = Offer(duplicate, 2, 1000.0,
                10.1, 9.0, true, out duplicateReason);
            Assert.That(duplicateStatus, Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(duplicateReason, Is.EqualTo(MobileCameraIngressReason.DuplicateProviderImageTimestamp));
            Assert.That(duplicate.State.HasEpoch, Is.True);
            Assert.That(duplicate.State.Generation, Is.EqualTo(1));
            Assert.That(duplicate.State.LatestFrameId, Is.EqualTo(0));
            Assert.That(duplicate.State.HasPendingPacket, Is.True);
            Assert.That(duplicate.State.ReleasedPacketCount, Is.EqualTo(0));

            Fixture regressed = NewFixture();
            MobileCameraIngressReason regressionReason;
            Assert.That(Offer(regressed, 1, 1000.0, 10.0, 9.0, true, out regressionReason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            long generationBefore = regressed.State.Generation;
            MobileCameraIngressOfferStatus regressionStatus = Offer(regressed, 2, 999.0,
                10.1, 9.0, true, out regressionReason);
            Assert.That(regressionStatus, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(regressionReason, Is.EqualTo(MobileCameraIngressReason.ProviderImageTimestampRegressed));
            AssertRetired(regressed.State, MobileCameraIngressReason.ProviderImageTimestampRegressed, generationBefore);
            Assert.That(regressed.State.ReleasedPacketCount, Is.EqualTo(1));
        }

        [Test]
        public void ClockFailurePrecedesMalformedRgbValidationAndRetiresPendingPacket()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            long generation = fixture.State.Generation;

            MobileCameraIngressOfferStatus status = fixture.State.OfferFrame(new byte[1], 2, 2,
                Intrinsics(2, 2), 1001.0, null, 9.0, 8.0, true, out reason);

            Assert.That(status, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.ReceiptTimestampRegressed));
            AssertRetired(fixture.State, MobileCameraIngressReason.ReceiptTimestampRegressed, generation);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
        }

        [Test]
        public void NoTargetClockAndImageBasisReadinessRemainDiagnosticOnly()
        {
            Fixture noTarget = NewFixture(null);
            MobileCameraIngressReason reason;
            MobileCameraIngressOfferStatus noTargetStatus = Offer(noTarget, 1, 1000.0,
                10.0, 9.0, true, out reason);
            Assert.That(noTargetStatus, Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.NoTarget));
            Assert.That(noTarget.Gate.IsSelected, Is.False);
            Assert.That(noTarget.State.HasPendingPacket, Is.False);
            Assert.That(noTarget.State.LastRenderBlockReason, Is.EqualTo(MobileCameraIngressReason.NoTarget));

            Fixture noClock = NewFixture();
            MobileCameraIngressOfferStatus noClockStatus = Offer(noClock, 1, 1000.0,
                10.0, null, true, out reason);
            Assert.That(noClockStatus, Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.CaptureClockUnmapped));
            Assert.That(noClock.State.HasPendingPacket, Is.False);

            Fixture unverifiedBasis = NewFixture();
            MobileCameraIngressOfferStatus basisStatus = Offer(unverifiedBasis, 1, 1000.0,
                10.0, 9.0, false, out reason);
            Assert.That(basisStatus, Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.ImageBasisUnverified));
            Assert.That(unverifiedBasis.State.HasPendingPacket, Is.False);

            Assert.That(noTarget.State.CanAuthorizeLearnedRendering, Is.False);
            Assert.That(noClock.State.CanAuthorizeLearnedRendering, Is.False);
            Assert.That(unverifiedBasis.State.CanAuthorizeLearnedRendering, Is.False);
        }

        [Test]
        public void IntrinsicsMustMatchImageResolutionAndBeFinitePositive()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            MobileCameraIngressOfferStatus mismatch = fixture.State.OfferFrame(Pixels(1, 2, 2), 2, 2,
                Intrinsics(3, 2), 1000.0, null, 10.0, 9.0, true, out reason);
            Assert.That(mismatch, Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.IntrinsicsResolutionMismatch));
            Assert.That(fixture.State.HasPendingPacket, Is.False);

            Fixture invalid = NewFixture();
            MobileCameraIntrinsics bad = new MobileCameraIntrinsics(
                double.NaN, 10.0, 1.0, 1.0, 2, 2);
            MobileCameraIngressOfferStatus invalidStatus = invalid.State.OfferFrame(Pixels(1, 2, 2), 2, 2,
                bad, 1000.0, null, 10.0, 9.0, true, out reason);
            Assert.That(invalidStatus, Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.InvalidIntrinsics));
            Assert.That(invalid.State.HasPendingPacket, Is.False);
        }

        [Test]
        public void MatchingSyntheticResultStillCannotOpenLearnedRenderingWithoutCameraPose()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 3, 1000.0, 10.0, 9.5, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);
            Assert.That(packet.CanDispatchInference, Is.True);
            Assert.That(packet.HasFrameMatchedCameraPose, Is.False);
            Assert.That(packet.RenderBlockReason,
                Is.EqualTo(MobileCameraIngressReason.FrameMatchedCameraPoseUnverified));
            Assert.That(fixture.State.CanAuthorizeLearnedRendering, Is.False);
            MobileTrackingResult result = Result(packet.SessionId, packet.TargetAssetId,
                packet.FrameId, packet.CaptureIdentity.CaptureTimestampSeconds,
                MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                MobileTrackingRenderState.Visible, IdentityPose());

            Assert.That(fixture.State.CompleteWork(packet, packet.Generation, result),
                Is.EqualTo(MobileCameraIngressReason.None));
            MobileTrackingGateDecision decision = fixture.State.RenderTick(10.0);

            Assert.That(decision.IsRenderable, Is.False);
            Assert.That(decision.HasPose, Is.False);
            Assert.That(decision.Reason, Is.EqualTo(MobileTrackingGateReason.RenderSuppressed));
            Assert.That(fixture.State.LastRenderBlockReason,
                Is.EqualTo(MobileCameraIngressReason.FrameMatchedCameraPoseUnverified));
            Assert.That(packet.IsDisposed, Is.True);
        }

        [Test]
        public void FreshEpochAfterClockRetirementHasFreshFrontiersAndNoRetainedSelectionPose()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            long generation = fixture.State.Generation;
            fixture.State.RenderTick(9.0);
            AssertRetired(fixture.State, MobileCameraIngressReason.ObservationTimestampRegressed, generation);

            fixture.State.SetFocused(false);
            fixture.State.SetFocused(true);
            Assert.That(fixture.State.HasEpoch, Is.False);
            Assert.That(fixture.Gate.IsSelected, Is.False);
            Assert.That(fixture.Gate.Current.HasPose, Is.False);

            MobileCameraIngressReason beginReason;
            Assert.That(fixture.State.BeginEpoch(SessionB, TargetB, out beginReason), Is.True);
            Assert.That(beginReason, Is.EqualTo(MobileCameraIngressReason.None));
            Assert.That(fixture.State.Generation, Is.GreaterThan(generation));
            Assert.That(fixture.State.SessionId, Is.EqualTo(SessionB));
            Assert.That(fixture.State.TargetAssetId, Is.EqualTo(TargetB));
            Assert.That(fixture.Gate.Current.IsRenderable, Is.False);
            Assert.That(fixture.Gate.Current.HasAcceptedCapture, Is.False);

            Assert.That(Offer(fixture, 2, 0.0, 0.0, 0.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket freshPacket = BeginWork(fixture.State);
            Assert.That(freshPacket.FrameId, Is.EqualTo(0));
            Assert.That(freshPacket.SessionId, Is.EqualTo(SessionB));
            Assert.That(freshPacket.ProviderImageTimestampSeconds, Is.EqualTo(0.0));
            Assert.That(freshPacket.CallbackReceiptTimestampSeconds, Is.EqualTo(0.0));
            Assert.That(freshPacket.CaptureIdentity.CaptureTimestampSeconds, Is.EqualTo(0.0));
        }

        [Test]
        public void RetiredReadingWorkerKeepsSoleSlotUntilOwnerReapsObsoleteCompletion()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packetA = BeginWork(fixture.State);
            long generationA = packetA.Generation;

            using (var workerReading = new ManualResetEvent(false))
            using (var allowWorkerStop = new ManualResetEvent(false))
            {
                Exception workerError = null;
                Exception callbackStartError = null;
                MobileCameraIngressReason completionReason = MobileCameraIngressReason.None;
                byte workerReadFirstByte = 0;
                Thread worker = new Thread(delegate()
                {
                    try
                    {
                        byte[] readCopy = packetA.CopyRgb24Hwc();
                        workerReadFirstByte = readCopy[0];
                        workerReading.Set();
                        allowWorkerStop.WaitOne();
                        workerReadFirstByte = readCopy[0];
                        readCopy = null;
                        completionReason = fixture.State.CompleteWork(packetA, generationA,
                            Result(packetA.SessionId, packetA.TargetAssetId, packetA.FrameId,
                                packetA.CaptureIdentity.CaptureTimestampSeconds,
                                MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                                MobileTrackingRenderState.Visible, IdentityPose()));
                        try
                        {
                            MobileCameraIngressPacket ignored;
                            MobileCameraIngressReason ignoredReason;
                            fixture.State.TryBeginNextWork(out ignored, out ignoredReason);
                        }
                        catch (Exception error)
                        {
                            callbackStartError = error;
                        }
                    }
                    catch (Exception error)
                    {
                        workerError = error;
                    }
                    finally
                    {
                        workerReading.Set();
                    }
                });
                worker.IsBackground = true;
                worker.Start();
                try
                {
                    workerReading.WaitOne();
                    Assert.That(workerReadFirstByte, Is.EqualTo((byte)1));
                    Assert.That(Offer(fixture, 2, 1001.0, 10.1, 9.1, true, out reason),
                        Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
                    fixture.State.InvalidateEpoch();
                    Assert.That(fixture.State.HasEpoch, Is.False);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
                    Assert.That(packetA.IsDisposed, Is.False);

                    MobileCameraIngressReason beginReason;
                    Assert.That(fixture.State.BeginEpoch(SessionB, TargetB, out beginReason), Is.True);
                    Assert.That(Offer(fixture, 3, 2000.0, 20.0, 19.0, true, out reason),
                        Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
                    MobileCameraIngressPacket blocked;
                    MobileCameraIngressReason blockedReason;
                    Assert.That(fixture.State.TryBeginNextWork(out blocked, out blockedReason), Is.False);
                    Assert.That(blockedReason, Is.EqualTo(MobileCameraIngressReason.WorkerBusy));
                    Assert.That(fixture.State.HasPendingPacket, Is.True);

                    allowWorkerStop.Set();
                    worker.Join();
                    Assert.That(workerError, Is.Null);
                    Assert.That(completionReason, Is.EqualTo(MobileCameraIngressReason.StaleWorkerCompletion));
                    Assert.That(callbackStartError, Is.TypeOf<InvalidOperationException>());
                    Assert.That(workerReadFirstByte, Is.EqualTo((byte)1));
                    Assert.That(fixture.State.HasCompletion, Is.True);
                    Assert.That(fixture.State.HasInFlightPacket, Is.True);
                    Assert.That(packetA.IsDisposed, Is.False);

                    Assert.That(fixture.State.TryBeginNextWork(out blocked, out blockedReason), Is.False);
                    Assert.That(blockedReason, Is.EqualTo(MobileCameraIngressReason.WorkerBusy));
                    MobileTrackingGateDecision beforeReap = fixture.Gate.Current;
                    Assert.That(beforeReap.SessionId, Is.EqualTo(SessionB));
                    Assert.That(beforeReap.TargetAssetId, Is.EqualTo(TargetB));

                    fixture.State.RenderTick(20.1);

                    Assert.That(packetA.IsDisposed, Is.True);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(2));
                    Assert.That(fixture.State.StaleCompletionCount, Is.EqualTo(1));
                    Assert.That(fixture.State.HasCompletion, Is.False);
                    Assert.That(fixture.State.HasInFlightPacket, Is.False);
                    Assert.That(fixture.State.HasPendingPacket, Is.True);
                    Assert.That(fixture.Gate.Current.SessionId, Is.EqualTo(SessionB));
                    Assert.That(fixture.Gate.Current.Reason, Is.EqualTo(MobileTrackingGateReason.NoResult));
                    Assert.That(fixture.Gate.Current.SourceFrameId, Is.EqualTo(-1));

                    MobileCameraIngressPacket packetC = BeginWork(fixture.State);
                    Assert.That(packetC.SessionId, Is.EqualTo(SessionB));
                    Assert.That(packetC.CopyRgb24Hwc()[0], Is.EqualTo((byte)3));
                    Assert.That(fixture.State.CompleteWork(packetC, packetC.Generation, null),
                        Is.EqualTo(MobileCameraIngressReason.None));
                    fixture.State.RenderTick(20.2);
                    Assert.That(packetC.IsDisposed, Is.True);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(3));
                }
                finally
                {
                    allowWorkerStop.Set();
                    if (worker.IsAlive) worker.Join();
                }
            }
        }

        [Test]
        public void InvalidRenderClockReleasesPendingButRetainsStillReadingWorkerUntilCompletionReap()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packetA = BeginWork(fixture.State);

            using (var workerReading = new ManualResetEvent(false))
            using (var allowWorkerStop = new ManualResetEvent(false))
            {
                Exception workerError = null;
                MobileCameraIngressReason completionReason = MobileCameraIngressReason.None;
                Thread worker = new Thread(delegate()
                {
                    try
                    {
                        byte[] readCopy = packetA.CopyRgb24Hwc();
                        workerReading.Set();
                        allowWorkerStop.WaitOne();
                        if (readCopy[0] != 1) throw new InvalidOperationException("Worker copy changed while reading.");
                        readCopy = null;
                        completionReason = fixture.State.CompleteWork(packetA, packetA.Generation, null);
                    }
                    catch (Exception error)
                    {
                        workerError = error;
                    }
                    finally
                    {
                        workerReading.Set();
                    }
                });
                worker.IsBackground = true;
                worker.Start();
                try
                {
                    workerReading.WaitOne();
                    Assert.That(Offer(fixture, 2, 1001.0, 10.1, 9.1, true, out reason),
                        Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
                    long generation = fixture.State.Generation;
                    fixture.State.RenderTick(9.0);
                    AssertRetired(fixture.State, MobileCameraIngressReason.ObservationTimestampRegressed, generation);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
                    Assert.That(packetA.IsDisposed, Is.False);
                    Assert.That(fixture.State.HasInFlightPacket, Is.True);
                    Assert.That(fixture.State.HasPendingPacket, Is.False);

                    allowWorkerStop.Set();
                    worker.Join();
                    Assert.That(workerError, Is.Null);
                    Assert.That(completionReason, Is.EqualTo(MobileCameraIngressReason.StaleWorkerCompletion));
                    Assert.That(fixture.State.HasCompletion, Is.True);
                    Assert.That(packetA.IsDisposed, Is.False);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));

                    fixture.State.RenderTick(10.0);
                    Assert.That(packetA.IsDisposed, Is.True);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(2));
                    Assert.That(fixture.State.HasInFlightPacket, Is.False);
                    Assert.That(fixture.State.HasCompletion, Is.False);
                }
                finally
                {
                    allowWorkerStop.Set();
                    if (worker.IsAlive) worker.Join();
                }
            }
        }

        [Test]
        public void CurrentCompletionWaitsForOwnerRenderTickToReleasePacket()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);
            Exception workerError = null;
            MobileCameraIngressReason completionReason = MobileCameraIngressReason.UnexpectedWorkerCompletion;
            Thread worker = new Thread(delegate()
            {
                try
                {
                    completionReason = fixture.State.CompleteWork(packet, packet.Generation, null);
                }
                catch (Exception error)
                {
                    workerError = error;
                }
            });
            worker.IsBackground = true;
            worker.Start();
            worker.Join();

            Assert.That(workerError, Is.Null);
            Assert.That(completionReason, Is.EqualTo(MobileCameraIngressReason.None));
            Assert.That(fixture.State.HasCompletion, Is.True);
            Assert.That(fixture.State.HasInFlightPacket, Is.True);
            Assert.That(packet.IsDisposed, Is.False);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(0));
            MobileCameraIngressPacket blocked;
            MobileCameraIngressReason blockedReason;
            Assert.That(fixture.State.TryBeginNextWork(out blocked, out blockedReason), Is.False);
            Assert.That(blockedReason, Is.EqualTo(MobileCameraIngressReason.WorkerBusy));

            fixture.State.RenderTick(10.0);

            Assert.That(fixture.State.HasCompletion, Is.False);
            Assert.That(fixture.State.HasInFlightPacket, Is.False);
            Assert.That(packet.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
            Assert.That(fixture.Gate.Current.Reason, Is.EqualTo(MobileTrackingGateReason.MissingResult));
        }

        [Test]
        public void DuplicateAndUnexpectedCompletionsCannotReleaseOrReplaceActualWorkerCompletion()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packetA = BeginWork(fixture.State);
            Assert.That(fixture.State.CompleteWork(packetA, packetA.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.None));
            MobileTrackingResult duplicateResult = Result(SessionB, TargetB, packetA.FrameId,
                packetA.CaptureIdentity.CaptureTimestampSeconds, MobileMaskAvailability.Available,
                MobilePoseTrackingState.Tracking, MobileTrackingRenderState.Visible, IdentityPose());
            Assert.That(fixture.State.CompleteWork(packetA, packetA.Generation, duplicateResult),
                Is.EqualTo(MobileCameraIngressReason.DuplicateWorkerCompletion));
            Assert.That(fixture.State.HasCompletion, Is.True);
            Assert.That(packetA.IsDisposed, Is.False);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(0));
            fixture.State.RenderTick(10.0);
            Assert.That(fixture.Gate.Current.Reason, Is.EqualTo(MobileTrackingGateReason.MissingResult));
            Assert.That(packetA.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));

            Assert.That(Offer(fixture, 2, 1001.0, 10.1, 9.1, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packetB = BeginWork(fixture.State);
            Assert.That(fixture.State.CompleteWork(packetA, packetB.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.UnexpectedWorkerCompletion));
            Assert.That(fixture.State.CompleteWork(packetB, packetB.Generation - 1, null),
                Is.EqualTo(MobileCameraIngressReason.UnexpectedWorkerCompletion));
            Assert.That(fixture.State.HasInFlightPacket, Is.True);
            Assert.That(packetB.IsDisposed, Is.False);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));

            Assert.That(fixture.State.CompleteWork(packetB, packetB.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.None));
            fixture.State.RenderTick(10.2);
            Assert.That(packetB.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(2));
        }

        [Test]
        public void RepeatedInvalidationAndInvalidationAfterQueuedCompletionReleaseExactlyOnce()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);
            Assert.That(fixture.State.CompleteWork(packet, packet.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.None));
            Assert.That(fixture.State.HasCompletion, Is.True);

            fixture.State.InvalidateEpoch();
            long retiredGeneration = fixture.State.Generation;
            Assert.That(fixture.State.HasEpoch, Is.False);
            Assert.That(fixture.State.HasCompletion, Is.False);
            Assert.That(fixture.State.HasInFlightPacket, Is.False);
            Assert.That(packet.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
            fixture.State.InvalidateEpoch();
            Assert.That(fixture.State.Generation, Is.EqualTo(retiredGeneration));
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
            Assert.That(fixture.State.CompleteWork(packet, packet.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.UnexpectedWorkerCompletion));
            fixture.State.RenderTick(20.0);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
        }

        [Test]
        public void TerminalDisposeReleasesPendingImmediatelyAndReadingWorkerOnShutdownCallback()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);

            using (var workerReading = new ManualResetEvent(false))
            using (var allowWorkerStop = new ManualResetEvent(false))
            {
                Exception workerError = null;
                MobileCameraIngressReason completionReason = MobileCameraIngressReason.None;
                byte readValueAfterDispose = 0;
                Thread worker = new Thread(delegate()
                {
                    try
                    {
                        byte[] readCopy = packet.CopyRgb24Hwc();
                        workerReading.Set();
                        allowWorkerStop.WaitOne();
                        readValueAfterDispose = readCopy[0];
                        readCopy = null;
                        completionReason = fixture.State.CompleteWork(packet, packet.Generation, null);
                    }
                    catch (Exception error)
                    {
                        workerError = error;
                    }
                    finally
                    {
                        workerReading.Set();
                    }
                });
                worker.IsBackground = true;
                worker.Start();
                try
                {
                    workerReading.WaitOne();
                    Assert.That(Offer(fixture, 2, 1001.0, 10.1, 9.1, true, out reason),
                        Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
                    fixture.State.Dispose();
                    Assert.That(fixture.State.HasEpoch, Is.False);
                    Assert.That(fixture.State.HasPendingPacket, Is.False);
                    Assert.That(fixture.State.HasInFlightPacket, Is.True);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
                    Assert.That(packet.IsDisposed, Is.False);
                    Assert.That(fixture.Gate.IsSelected, Is.False);

                    allowWorkerStop.Set();
                    worker.Join();
                    Assert.That(workerError, Is.Null);
                    Assert.That(readValueAfterDispose, Is.EqualTo((byte)1));
                    Assert.That(completionReason, Is.EqualTo(MobileCameraIngressReason.StaleWorkerCompletion));
                    Assert.That(packet.IsDisposed, Is.True);
                    Assert.That(fixture.State.HasInFlightPacket, Is.False);
                    Assert.That(fixture.State.HasCompletion, Is.False);
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(2));

                    fixture.State.Dispose();
                    Assert.That(fixture.State.CompleteWork(packet, packet.Generation, null),
                        Is.EqualTo(MobileCameraIngressReason.UnexpectedWorkerCompletion));
                    MobileCameraIngressPacket ignored;
                    MobileCameraIngressReason ignoredReason;
                    Assert.That(fixture.State.TryBeginNextWork(out ignored, out ignoredReason), Is.False);
                    Assert.That(ignoredReason, Is.EqualTo(MobileCameraIngressReason.Disposed));
                    Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(2));
                }
                finally
                {
                    allowWorkerStop.Set();
                    if (worker.IsAlive) worker.Join();
                }
            }
        }

        [Test]
        public void DisposeAfterQueuedCompletionNeedsNoFutureRenderAndDoesNotDoubleRelease()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);
            Assert.That(fixture.State.CompleteWork(packet, packet.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.None));
            Assert.That(fixture.State.HasCompletion, Is.True);
            Assert.That(packet.IsDisposed, Is.False);

            fixture.State.Dispose();

            Assert.That(fixture.State.HasCompletion, Is.False);
            Assert.That(fixture.State.HasInFlightPacket, Is.False);
            Assert.That(packet.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
            fixture.State.Dispose();
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
            Assert.That(fixture.State.CompleteWork(packet, packet.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.UnexpectedWorkerCompletion));
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
        }

        [Test]
        public void OwnerOperationsRejectForeignThreadBeforeMutationWhileCompletionIsWorkerSafe()
        {
            Fixture fixture = NewFixture();
            long generation = fixture.State.Generation;
            Exception[] errors = new Exception[7];
            Thread foreign = new Thread(delegate()
            {
                MobileCameraIngressReason reason;
                MobileCameraIngressPacket packet;
                try { fixture.State.SetFocused(false); } catch (Exception error) { errors[0] = error; }
                try { fixture.State.BeginEpoch(SessionB, TargetB, out reason); } catch (Exception error) { errors[1] = error; }
                try { fixture.State.OfferFrame(Pixels(1, 2, 2), 2, 2, Intrinsics(2, 2),
                    1000.0, null, 10.0, 9.0, true, out reason); } catch (Exception error) { errors[2] = error; }
                try { fixture.State.TryBeginNextWork(out packet, out reason); } catch (Exception error) { errors[3] = error; }
                try { fixture.State.RenderTick(10.0); } catch (Exception error) { errors[4] = error; }
                try { fixture.State.InvalidateEpoch(); } catch (Exception error) { errors[5] = error; }
                try { fixture.State.Dispose(); } catch (Exception error) { errors[6] = error; }
            });
            foreign.IsBackground = true;
            foreign.Start();
            foreign.Join();

            for (int i = 0; i < errors.Length; i++)
                Assert.That(errors[i], Is.TypeOf<InvalidOperationException>(), "operation " + i);
            Assert.That(fixture.State.Generation, Is.EqualTo(generation));
            Assert.That(fixture.State.IsActive, Is.True);
            Assert.That(fixture.State.HasEpoch, Is.True);
            Assert.That(fixture.State.SessionId, Is.EqualTo(SessionA));
            Assert.That(fixture.State.TargetAssetId, Is.EqualTo(TargetA));
            Assert.That(fixture.State.HasPendingPacket, Is.False);
            Assert.That(fixture.State.HasInFlightPacket, Is.False);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(0));
        }

        [Test]
        public void FocusPauseEnableAndTrackingFlagsRequireFreshEpochAfterRecovery()
        {
            Fixture fixture = NewFixture();
            fixture.State.SetFocused(false);
            fixture.State.SetPaused(true);
            Assert.That(fixture.State.HasEpoch, Is.False);
            fixture.State.SetFocused(true);
            Assert.That(fixture.State.IsActive, Is.False);
            fixture.State.SetPaused(false);
            Assert.That(fixture.State.IsActive, Is.False);
            Assert.That(fixture.State.HasEpoch, Is.False);

            MobileCameraIngressReason reason;
            Assert.That(fixture.State.BeginEpoch(SessionB, TargetA, out reason), Is.True);
            Assert.That(fixture.State.IsActive, Is.True);

            fixture.State.SetComponentEnabled(false);
            fixture.State.SetComponentEnabled(true);
            Assert.That(fixture.State.IsActive, Is.False);
            Assert.That(fixture.State.HasEpoch, Is.False);
            fixture.State.SetSessionTracking(false);
            fixture.State.SetSessionTracking(true);
            Assert.That(fixture.State.HasEpoch, Is.False);
            Assert.That(fixture.State.IsActive, Is.False);

            Assert.That(fixture.State.BeginEpoch("session-C", TargetA, out reason), Is.True);
            Assert.That(fixture.State.IsActive, Is.True);
            Assert.That(fixture.State.SessionId, Is.EqualTo("session-C"));
        }

        [Test]
        public void PendingReplacementUsesLatestPixelsAndDefensiveCopiesWithExactlyOnceRelease()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            byte[] producerA = Pixels(1, 2, 2);
            Assert.That(fixture.State.OfferFrame(producerA, 2, 2, Intrinsics(2, 2),
                1000.0, null, 10.0, 9.0, true, out reason), Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            producerA[0] = 101;
            MobileCameraIngressPacket packetA = BeginWork(fixture.State);
            byte[] returnedCopy = packetA.CopyRgb24Hwc();
            Assert.That(returnedCopy[0], Is.EqualTo((byte)1));
            returnedCopy[0] = 102;
            Assert.That(packetA.CopyRgb24Hwc()[0], Is.EqualTo((byte)1));

            byte[] producerB = Pixels(2, 2, 2);
            Assert.That(fixture.State.OfferFrame(producerB, 2, 2, Intrinsics(2, 2),
                1001.0, null, 10.1, 9.1, true, out reason), Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            producerB[0] = 103;
            Assert.That(Offer(fixture, 3, 1002.0, 10.2, 9.2, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.ReplacedPending));
            Assert.That(Offer(fixture, 4, 1003.0, 10.3, 9.3, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.ReplacedPending));
            byte[] producerE = Pixels(5, 2, 2);
            Assert.That(fixture.State.OfferFrame(producerE, 2, 2, Intrinsics(2, 2),
                1004.0, null, 10.4, 9.4, true, out reason), Is.EqualTo(MobileCameraIngressOfferStatus.ReplacedPending));
            producerE[0] = 104;

            Assert.That(fixture.State.DroppedPendingPacketCount, Is.EqualTo(3));
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(3));
            Assert.That(packetA.IsDisposed, Is.False);
            MobileCameraIngressPacket blocked;
            MobileCameraIngressReason blockedReason;
            Assert.That(fixture.State.TryBeginNextWork(out blocked, out blockedReason), Is.False);
            Assert.That(blockedReason, Is.EqualTo(MobileCameraIngressReason.WorkerBusy));

            Assert.That(fixture.State.CompleteWork(packetA, packetA.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.None));
            fixture.State.RenderTick(10.5);
            Assert.That(packetA.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(4));

            MobileCameraIngressPacket packetE = BeginWork(fixture.State);
            Assert.That(packetE.CopyRgb24Hwc()[0], Is.EqualTo((byte)5));
            Assert.That(fixture.State.CompleteWork(packetE, packetE.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.None));
            fixture.State.RenderTick(10.6);
            Assert.That(packetE.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(5));
            Assert.That(fixture.State.DroppedPendingPacketCount, Is.EqualTo(3));
            Assert.Throws<ObjectDisposedException>(() => packetE.CopyRgb24Hwc());
        }

        [Test]
        public void ExactByteCapIsAcceptedAndCapPlusOneAndOverflowAreRejectedWithoutDisturbingOwnership()
        {
            Fixture exact = NewFixture(12);
            MobileCameraIngressReason reason;
            Assert.That(exact.State.OfferFrame(Pixels(1, 2, 2), 2, 2, Intrinsics(2, 2),
                1000.0, null, 10.0, 9.0, true, out reason), Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            Assert.That(BeginWork(exact.State).PixelByteCount, Is.EqualTo(12));

            Fixture bounded = NewFixture(17);
            Assert.That(Offer(bounded, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket inFlight = BeginWork(bounded.State);
            Assert.That(Offer(bounded, 2, 1001.0, 10.1, 9.1, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            long latestBeforeRejected = bounded.State.LatestFrameId;
            MobileCameraIngressOfferStatus capPlusOne = bounded.State.OfferFrame(
                Pixels(8, 2, 3), 2, 3, Intrinsics(2, 3), 1002.0, null, 10.2, 9.2, true, out reason);
            Assert.That(capPlusOne, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.Rgb24PacketTooLarge));
            Assert.That(bounded.State.LatestFrameId, Is.EqualTo(latestBeforeRejected));
            Assert.That(bounded.State.ReleasedPacketCount, Is.EqualTo(0));
            Assert.That(inFlight.IsDisposed, Is.False);
            Assert.That(bounded.State.HasPendingPacket, Is.True);

            MobileCameraIngressOfferStatus overflow = bounded.State.OfferFrame(new byte[0],
                int.MaxValue, 2, Intrinsics(int.MaxValue, 2), 1002.0, null, 10.2, 9.2, true, out reason);
            Assert.That(overflow, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.Rgb24SizeOverflow));
            Assert.That(bounded.State.LatestFrameId, Is.EqualTo(latestBeforeRejected));
            Assert.That(bounded.State.ReleasedPacketCount, Is.EqualTo(0));
            Assert.That(inFlight.IsDisposed, Is.False);
            Assert.That(bounded.State.HasPendingPacket, Is.True);
        }

        [Test]
        public void InvalidDimensionsAndLengthAreRejectedWithoutReplacingPendingPacket()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            long latest = fixture.State.LatestFrameId;

            MobileCameraIngressOfferStatus zeroDimension = fixture.State.OfferFrame(
                new byte[0], 0, 2, Intrinsics(0, 2), 1001.0, null, 10.1, 9.1, true, out reason);
            Assert.That(zeroDimension, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.InvalidImageDimensions));
            MobileCameraIngressOfferStatus badLength = fixture.State.OfferFrame(
                new byte[11], 2, 2, Intrinsics(2, 2), 1002.0, null, 10.2, 9.2, true, out reason);
            Assert.That(badLength, Is.EqualTo(MobileCameraIngressOfferStatus.Rejected));
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.Rgb24LengthMismatch));

            Assert.That(fixture.State.LatestFrameId, Is.EqualTo(latest));
            Assert.That(fixture.State.HasPendingPacket, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(0));
            MobileCameraIngressPacket pending = BeginWork(fixture.State);
            Assert.That(pending.CopyRgb24Hwc()[0], Is.EqualTo((byte)1));
        }

        [Test]
        public void ForeignSessionTargetAndEmptyIdentityAreRejectedBeforeGateMutation()
        {
            string[] resultSessions = { SessionB, SessionA, SessionB, null, SessionA, string.Empty };
            string[] resultTargets = { TargetA, TargetB, TargetB, TargetA, null, string.Empty };

            for (int i = 0; i < resultSessions.Length; i++)
            {
                Fixture fixture = NewFixture();
                EstablishSuppressedMatchingBaseline(fixture);
                MobileTrackingGateDecision baseline = fixture.Gate.Current;
                Assert.That(baseline.Reason, Is.EqualTo(MobileTrackingGateReason.RenderSuppressed));
                Assert.That(baseline.SourceFrameId, Is.EqualTo(0));
                Assert.That(baseline.HasAcceptedCapture, Is.False);

                MobileCameraIngressReason reason;
                Assert.That(Offer(fixture, (byte)(10 + i), 1001.0, 11.0, 9.0, true, out reason),
                    Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
                MobileCameraIngressPacket packet = BeginWork(fixture.State);
                MobileTrackingResult foreign = Result(resultSessions[i], resultTargets[i], packet.FrameId,
                    packet.CaptureIdentity.CaptureTimestampSeconds, MobileMaskAvailability.Available,
                    MobilePoseTrackingState.Tracking, MobileTrackingRenderState.Visible, IdentityPose());
                Assert.That(fixture.State.CompleteWork(packet, packet.Generation, foreign),
                    Is.EqualTo(MobileCameraIngressReason.None));

                fixture.State.RenderTick(11.1);

                Assert.That(fixture.State.LastReason, Is.EqualTo(MobileCameraIngressReason.ForeignResultIdentity), "case " + i);
                AssertSameDecision(baseline, fixture.Gate.Current, "case " + i);
                Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(2));
                Assert.That(packet.IsDisposed, Is.True);
            }
        }

        [Test]
        public void OldGenerationCompletionCannotChangeFreshEpochGateBaseline()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket oldPacket = BeginWork(fixture.State);
            fixture.State.InvalidateEpoch();
            Assert.That(oldPacket.IsDisposed, Is.False);
            MobileCameraIngressReason beginReason;
            Assert.That(fixture.State.BeginEpoch(SessionB, TargetB, out beginReason), Is.True);
            Assert.That(Offer(fixture, 2, 2000.0, 20.0, 19.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileTrackingGateDecision freshBaseline = fixture.Gate.Current;
            Assert.That(freshBaseline.SessionId, Is.EqualTo(SessionB));
            Assert.That(freshBaseline.Reason, Is.EqualTo(MobileTrackingGateReason.NoResult));

            Assert.That(fixture.State.CompleteWork(oldPacket, oldPacket.Generation,
                Result(SessionA, TargetA, oldPacket.FrameId,
                    oldPacket.CaptureIdentity.CaptureTimestampSeconds,
                    MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                    MobileTrackingRenderState.Visible, IdentityPose())),
                Is.EqualTo(MobileCameraIngressReason.StaleWorkerCompletion));
            fixture.State.RenderTick(20.1);

            Assert.That(oldPacket.IsDisposed, Is.True);
            Assert.That(fixture.State.StaleCompletionCount, Is.EqualTo(1));
            Assert.That(fixture.State.HasPendingPacket, Is.True);
            AssertSameDecision(freshBaseline, fixture.Gate.Current, "old generation");
            Assert.That(fixture.State.LastReason, Is.EqualTo(MobileCameraIngressReason.StaleWorkerCompletion));
        }

        [Test]
        public void MatchingResultUsesOriginalMappedIdentityAndWrongFrameOrTimeUsesGateFailure()
        {
            Fixture validFixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(validFixture, 1, 1000.0, 10.0, 9.5, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket validPacket = BeginWork(validFixture.State);
            MobileTrackingResult validResult = Result(SessionA, TargetA, validPacket.FrameId, 9.5,
                MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                MobileTrackingRenderState.Visible, IdentityPose());
            Assert.That(validFixture.State.CompleteWork(validPacket, validPacket.Generation, validResult),
                Is.EqualTo(MobileCameraIngressReason.None));
            validFixture.State.RenderTick(10.0);
            Assert.That(validFixture.Gate.Current.Reason, Is.EqualTo(MobileTrackingGateReason.RenderSuppressed));
            Assert.That(validFixture.Gate.Current.SourceFrameId, Is.EqualTo(validPacket.FrameId));
            Assert.That(validFixture.State.LastReason, Is.EqualTo(MobileCameraIngressReason.None));
            Assert.That(validPacket.CaptureIdentity.CaptureTimestampSeconds, Is.EqualTo(9.5));

            for (int variant = 0; variant < 2; variant++)
            {
                Fixture mismatchFixture = NewFixture();
                Assert.That(Offer(mismatchFixture, 1, 1000.0, 10.0, 9.5, true, out reason),
                    Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
                MobileCameraIngressPacket packet = BeginWork(mismatchFixture.State);
                long resultFrame = variant == 0 ? packet.FrameId + 1 : packet.FrameId;
                double resultTime = variant == 0 ? packet.CaptureIdentity.CaptureTimestampSeconds
                    : packet.CaptureIdentity.CaptureTimestampSeconds + 0.001;
                MobileTrackingResult mismatch = Result(SessionA, TargetA, resultFrame, resultTime,
                    MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                    MobileTrackingRenderState.Visible, IdentityPose());
                Assert.That(mismatchFixture.State.CompleteWork(packet, packet.Generation, mismatch),
                    Is.EqualTo(MobileCameraIngressReason.None));
                mismatchFixture.State.RenderTick(10.0);
                Assert.That(mismatchFixture.Gate.Current.Reason,
                    Is.EqualTo(MobileTrackingGateReason.ResultCaptureIdentityMismatch), "variant " + variant);
                Assert.That(mismatchFixture.State.LastReason, Is.EqualTo(MobileCameraIngressReason.None));
                Assert.That(packet.IsDisposed, Is.True);
                Assert.That(mismatchFixture.State.ReleasedPacketCount, Is.EqualTo(1));
            }
        }

        [Test]
        public void OrdinaryCurrentSelectionFailuresClearPreviouslyAcceptedGatePose()
        {
            MobileTrackingGateReason[] expectedReasons =
            {
                MobileTrackingGateReason.MissingResult,
                MobileTrackingGateReason.MaskUnavailable,
                MobileTrackingGateReason.PoseNotTracking,
                MobileTrackingGateReason.RenderSuppressed
            };

            for (int i = 0; i < expectedReasons.Length; i++)
            {
                Fixture fixture = NewFixture();
                MobileCameraIngressReason reason;
                Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.8, false, out reason),
                    Is.EqualTo(MobileCameraIngressOfferStatus.DiagnosticOnly));
                SeedGatePose(fixture, 0, 9.8, 10.0);
                Assert.That(fixture.Gate.Current.IsRenderable, Is.True, "seed " + i);

                Assert.That(Offer(fixture, 2, 1001.0, 10.1, 9.9, true, out reason),
                    Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
                MobileCameraIngressPacket packet = BeginWork(fixture.State);
                MobileTrackingResult failedResult = null;
                if (i == 1)
                    failedResult = Result(SessionA, TargetA, packet.FrameId,
                        packet.CaptureIdentity.CaptureTimestampSeconds,
                        MobileMaskAvailability.Unavailable, MobilePoseTrackingState.Tracking,
                        MobileTrackingRenderState.Visible, IdentityPose());
                else if (i == 2)
                    failedResult = Result(SessionA, TargetA, packet.FrameId,
                        packet.CaptureIdentity.CaptureTimestampSeconds,
                        MobileMaskAvailability.Available, MobilePoseTrackingState.Lost,
                        MobileTrackingRenderState.Visible, IdentityPose());
                else if (i == 3)
                    failedResult = Result(SessionA, TargetA, packet.FrameId,
                        packet.CaptureIdentity.CaptureTimestampSeconds,
                        MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                        MobileTrackingRenderState.Suppressed, IdentityPose());

                Assert.That(fixture.State.CompleteWork(packet, packet.Generation, failedResult),
                    Is.EqualTo(MobileCameraIngressReason.None));
                fixture.State.RenderTick(10.1);

                Assert.That(fixture.Gate.Current.IsRenderable, Is.False, "case " + i);
                Assert.That(fixture.Gate.Current.HasPose, Is.False, "case " + i);
                Assert.That(fixture.Gate.Current.Reason, Is.EqualTo(expectedReasons[i]), "case " + i);
                Assert.That(packet.IsDisposed, Is.True, "case " + i);
                Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1), "case " + i);
            }
        }

        [Test]
        public void PacketPixelsAreReleasedExactlyOnceOnInvalidationAndConsumerCopyIsIndependent()
        {
            Fixture fixture = NewFixture();
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);
            byte[] copy = packet.CopyRgb24Hwc();
            copy[0] = 77;
            Assert.That(packet.CopyRgb24Hwc()[0], Is.EqualTo((byte)1));

            fixture.State.InvalidateEpoch();
            Assert.That(packet.IsDisposed, Is.False);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(0));
            Assert.That(fixture.State.CompleteWork(packet, packet.Generation, null),
                Is.EqualTo(MobileCameraIngressReason.StaleWorkerCompletion));
            fixture.State.RenderTick(20.0);

            Assert.That(packet.IsDisposed, Is.True);
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
            Assert.Throws<ObjectDisposedException>(() => packet.CopyRgb24Hwc());
            fixture.State.InvalidateEpoch();
            Assert.That(fixture.State.ReleasedPacketCount, Is.EqualTo(1));
        }

        private static Fixture NewFixture(string targetAssetId = TargetA)
        {
            return NewFixture(24, SessionA, targetAssetId);
        }

        private static Fixture NewFixture(int maxRgb24Bytes)
        {
            return NewFixture(maxRgb24Bytes, SessionA, TargetA);
        }

        private static Fixture NewFixture(int maxRgb24Bytes, string sessionId, string targetAssetId)
        {
            return new Fixture(maxRgb24Bytes, sessionId, targetAssetId);
        }

        private static MobileCameraIngressOfferStatus Offer(Fixture fixture, byte marker,
            double providerTimestamp, double receiptTimestamp, double? mappedCaptureTimestamp,
            bool imageBasisVerified, out MobileCameraIngressReason reason)
        {
            return fixture.State.OfferFrame(Pixels(marker, 2, 2), 2, 2, Intrinsics(2, 2),
                providerTimestamp, 2345000000000L, receiptTimestamp, mappedCaptureTimestamp,
                imageBasisVerified, out reason);
        }

        private static MobileCameraIngressPacket BeginWork(MobileCameraIngressState state)
        {
            MobileCameraIngressPacket packet;
            MobileCameraIngressReason reason;
            Assert.That(state.TryBeginNextWork(out packet, out reason), Is.True);
            Assert.That(reason, Is.EqualTo(MobileCameraIngressReason.None));
            Assert.That(packet, Is.Not.Null);
            return packet;
        }

        private static void EstablishSuppressedMatchingBaseline(Fixture fixture)
        {
            MobileCameraIngressReason reason;
            Assert.That(Offer(fixture, 1, 1000.0, 10.0, 9.0, true, out reason),
                Is.EqualTo(MobileCameraIngressOfferStatus.Queued));
            MobileCameraIngressPacket packet = BeginWork(fixture.State);
            MobileTrackingResult result = Result(packet.SessionId, packet.TargetAssetId,
                packet.FrameId, packet.CaptureIdentity.CaptureTimestampSeconds,
                MobileMaskAvailability.Available, MobilePoseTrackingState.Tracking,
                MobileTrackingRenderState.Visible, IdentityPose());
            Assert.That(fixture.State.CompleteWork(packet, packet.Generation, result),
                Is.EqualTo(MobileCameraIngressReason.None));
            fixture.State.RenderTick(10.0);
        }

        private static void SeedGatePose(Fixture fixture, long frameId, double captureTime, double now)
        {
            MobileCaptureIdentity capture = new MobileCaptureIdentity(fixture.SessionId, frameId, captureTime);
            MobileTrackingResult result = Result(fixture.SessionId, fixture.TargetAssetId,
                frameId, captureTime, MobileMaskAvailability.Available,
                MobilePoseTrackingState.Tracking, MobileTrackingRenderState.Visible, IdentityPose());
            MobileTrackingGateDecision seeded = fixture.Gate.Update(result, capture,
                new MobileTrackingObservation(fixture.SessionId, frameId, now));
            Assert.That(seeded.IsRenderable, Is.True);
        }

        private static void AssertRetired(MobileCameraIngressState state,
            MobileCameraIngressReason expectedReason, long priorGeneration)
        {
            Assert.That(state.HasEpoch, Is.False);
            Assert.That(state.IsActive, Is.False);
            Assert.That(state.Generation, Is.GreaterThan(priorGeneration));
            Assert.That(state.SessionId, Is.Null);
            Assert.That(state.TargetAssetId, Is.Null);
            Assert.That(state.LastReason, Is.EqualTo(expectedReason));
        }

        private static void AssertSameDecision(MobileTrackingGateDecision expected,
            MobileTrackingGateDecision actual, string context)
        {
            Assert.That(actual.IsRenderable, Is.EqualTo(expected.IsRenderable), context + " renderability");
            Assert.That(actual.Reason, Is.EqualTo(expected.Reason), context + " reason");
            Assert.That(actual.SessionId, Is.EqualTo(expected.SessionId), context + " session");
            Assert.That(actual.TargetAssetId, Is.EqualTo(expected.TargetAssetId), context + " target");
            Assert.That(actual.SourceFrameId, Is.EqualTo(expected.SourceFrameId), context + " source frame");
            Assert.That(actual.HasAcceptedCapture, Is.EqualTo(expected.HasAcceptedCapture), context + " accepted flag");
            Assert.That(actual.AcceptedCaptureFrameId, Is.EqualTo(expected.AcceptedCaptureFrameId), context + " accepted frame");
            Assert.That(actual.AcceptedCaptureTimestampSeconds, Is.EqualTo(expected.AcceptedCaptureTimestampSeconds),
                context + " accepted time");
            Assert.That(actual.HasPose, Is.EqualTo(expected.HasPose), context + " pose");
        }

        private static MobileTrackingResult Result(string sessionId, string targetAssetId,
            long frameId, double captureTimestamp, MobileMaskAvailability mask,
            MobilePoseTrackingState poseState, MobileTrackingRenderState renderState, double[] pose)
        {
            return new MobileTrackingResult(sessionId, targetAssetId, frameId, captureTimestamp,
                mask, poseState, renderState, pose);
        }

        private static MobileCameraIntrinsics Intrinsics(int width, int height)
        {
            return new MobileCameraIntrinsics(100.0, 101.0, width / 2.0, height / 2.0, width, height);
        }

        private static byte[] Pixels(byte marker, int width, int height)
        {
            byte[] pixels = new byte[checked(checked(width * height) * 3)];
            for (int i = 0; i < pixels.Length; i++) pixels[i] = marker;
            return pixels;
        }

        private static double[] IdentityPose()
        {
            return new double[]
            {
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
            };
        }
    }
}
