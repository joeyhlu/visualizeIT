using System;

namespace VisualizeIt.Core
{
    /// <summary>Independent availability of the selected object's observed mask.</summary>
    public enum MobileMaskAvailability
    {
        Unavailable = 0,
        Available = 1
    }

    /// <summary>Pose state reported by the tracker; only Tracking may drive an overlay.</summary>
    public enum MobilePoseTrackingState
    {
        Initializing = 0,
        Recovering = 1,
        Tracking = 2,
        Limited = 3,
        Lost = 4
    }

    /// <summary>Independent render decision reported by the tracker.</summary>
    public enum MobileTrackingRenderState
    {
        Suppressed = 0,
        Visible = 1
    }

    /// <summary>Identity and monotonic timestamp of the exact captured frame passed to a tracker.</summary>
    public sealed class MobileCaptureIdentity
    {
        public string SessionId { get; private set; }
        public long FrameId { get; private set; }
        public double CaptureTimestampSeconds { get; private set; }

        public MobileCaptureIdentity(string sessionId, long frameId, double captureTimestampSeconds)
        {
            SessionId = sessionId;
            FrameId = frameId;
            CaptureTimestampSeconds = captureTimestampSeconds;
        }
    }

    /// <summary>Current camera-frame observation used to judge result ordering and freshness.</summary>
    public sealed class MobileTrackingObservation
    {
        public string SessionId { get; private set; }
        public long FrameId { get; private set; }
        public double MonotonicNowSeconds { get; private set; }

        public MobileTrackingObservation(string sessionId, long frameId, double monotonicNowSeconds)
        {
            SessionId = sessionId;
            FrameId = frameId;
            MonotonicNowSeconds = monotonicNowSeconds;
        }
    }

    /// <summary>
    /// Immutable tracker output. Pose is optional and, when supplied, is a row-major 4x4
    /// cvCameraFromGltfObject transform with translation in metres. It stays in that source
    /// basis; this type does not convert it to Unity coordinates or estimate mask/pose confidence.
    /// </summary>
    public sealed class MobileTrackingResult
    {
        private readonly double[] cvCameraFromGltfObjectPoseMetres;

        public string SessionId { get; private set; }
        public string TargetAssetId { get; private set; }
        public long SourceFrameId { get; private set; }
        public double CaptureTimestampSeconds { get; private set; }
        public MobileMaskAvailability MaskAvailability { get; private set; }
        public MobilePoseTrackingState PoseState { get; private set; }
        public MobileTrackingRenderState RenderState { get; private set; }
        public bool HasPose { get { return cvCameraFromGltfObjectPoseMetres != null; } }

        public MobileTrackingResult(string sessionId, string targetAssetId, long sourceFrameId,
            double captureTimestampSeconds, MobileMaskAvailability maskAvailability,
            MobilePoseTrackingState poseState, MobileTrackingRenderState renderState,
            double[] cvCameraFromGltfObjectPoseMetres)
        {
            SessionId = sessionId;
            TargetAssetId = targetAssetId;
            SourceFrameId = sourceFrameId;
            CaptureTimestampSeconds = captureTimestampSeconds;
            MaskAvailability = maskAvailability;
            PoseState = poseState;
            RenderState = renderState;
            this.cvCameraFromGltfObjectPoseMetres = Copy(cvCameraFromGltfObjectPoseMetres);
        }

        public double[] CopyCvCameraFromGltfObjectPoseMetres()
        {
            return Copy(cvCameraFromGltfObjectPoseMetres);
        }

        private static double[] Copy(double[] values)
        {
            return values == null ? null : (double[])values.Clone();
        }
    }

    public enum MobileTrackingGateReason
    {
        None = 0,
        NotSelected,
        NoResult,
        MissingResult,
        MissingCaptureIdentity,
        MissingObservation,
        SessionMismatch,
        TargetAssetMismatch,
        InvalidSourceFrameId,
        DuplicateSourceFrameId,
        OutOfOrderSourceFrameId,
        NonFiniteCaptureTimestamp,
        NegativeCaptureTimestamp,
        OutOfOrderCaptureTimestamp,
        InvalidResultFrameId,
        NonFiniteResultTimestamp,
        NegativeResultTimestamp,
        ResultCaptureIdentityMismatch,
        InvalidObservationFrameId,
        ObservationPrecedesSourceFrame,
        OutOfOrderObservationFrameId,
        NonFiniteObservationTime,
        NegativeObservationTime,
        OutOfOrderObservationTime,
        ObservationBeforeCapture,
        StaleResult,
        UnknownMaskAvailability,
        MaskUnavailable,
        UnknownPoseState,
        PoseNotTracking,
        UnknownRenderState,
        RenderSuppressed,
        MissingPose,
        InvalidPose
    }

    /// <summary>
    /// Snapshot of the latest decision. Current is historical: time passing does not update it.
    /// Call Observe on every camera/render tick, including ticks without a tracker result, and use
    /// its returned decision. Suppressed decisions never carry a render pose.
    /// </summary>
    public sealed class MobileTrackingGateDecision
    {
        private readonly double[] cvCameraFromGltfObjectPoseMetres;

        public bool IsRenderable { get; private set; }
        public MobileTrackingGateReason Reason { get; private set; }
        public string SessionId { get; private set; }
        public string TargetAssetId { get; private set; }
        public long SourceFrameId { get; private set; }
        // This pair is diagnostic history for the last accepted pose, even when this decision is suppressed.
        public long? AcceptedCaptureFrameId { get; private set; }
        public double? AcceptedCaptureTimestampSeconds { get; private set; }
        public bool HasAcceptedCapture
        {
            get { return AcceptedCaptureFrameId.HasValue && AcceptedCaptureTimestampSeconds.HasValue; }
        }
        public bool HasPose { get { return cvCameraFromGltfObjectPoseMetres != null; } }

        internal MobileTrackingGateDecision(bool isRenderable, MobileTrackingGateReason reason,
            string sessionId, string targetAssetId, long sourceFrameId,
            long? acceptedCaptureFrameId, double? acceptedCaptureTimestampSeconds,
            double[] cvCameraFromGltfObjectPoseMetres)
        {
            IsRenderable = isRenderable;
            Reason = reason;
            SessionId = sessionId;
            TargetAssetId = targetAssetId;
            SourceFrameId = sourceFrameId;
            AcceptedCaptureFrameId = acceptedCaptureFrameId;
            AcceptedCaptureTimestampSeconds = acceptedCaptureTimestampSeconds;
            this.cvCameraFromGltfObjectPoseMetres = cvCameraFromGltfObjectPoseMetres == null
                ? null : (double[])cvCameraFromGltfObjectPoseMetres.Clone();
        }

        public double[] CopyCvCameraFromGltfObjectPoseMetres()
        {
            return cvCameraFromGltfObjectPoseMetres == null
                ? null : (double[])cvCameraFromGltfObjectPoseMetres.Clone();
        }
    }

    /// <summary>
    /// Accepts only a fresh result for the exact captured frame, selected session and asset.
    /// Each invalid update clears the current render pose. Pose stays in cvCameraFromGltfObject
    /// source coordinates; callers must use the documented native basis converter and the matching
    /// capture's camera pose before positioning a Unity mesh.
    /// </summary>
    public sealed class MobileTrackingResultGate
    {
        private const double AffineTolerance = 1e-6;
        private const double RotationTolerance = 1e-5;

        private readonly double maxAgeSeconds;
        private string selectedSessionId;
        private string selectedTargetAssetId;
        private bool hasLastSourceFrame;
        private long lastSourceFrameId;
        private bool hasLastCaptureTimestamp;
        private double lastCaptureTimestampSeconds;
        private bool hasLastObservation;
        private long lastObservationFrameId;
        private double lastObservationTimeSeconds;
        private long? lastAcceptedCaptureFrameId;
        private double? lastAcceptedCaptureTimestampSeconds;
        private MobileTrackingGateDecision current;

        public double MaxAgeSeconds { get { return maxAgeSeconds; } }
        public MobileTrackingGateDecision Current { get { return current; } }
        public bool IsSelected { get { return selectedSessionId != null; } }

        public MobileTrackingResultGate(double maxAgeSeconds)
        {
            if (!Finite(maxAgeSeconds) || maxAgeSeconds <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxAgeSeconds), "Maximum result age must be finite and greater than zero seconds.");
            this.maxAgeSeconds = maxAgeSeconds;
            Reset();
        }

        /// <summary>Starts a selection epoch and clears every result and frame/time cursor.</summary>
        public void Select(string sessionId, string targetAssetId)
        {
            if (string.IsNullOrEmpty(sessionId)) throw new ArgumentException("A session ID is required.", nameof(sessionId));
            if (string.IsNullOrEmpty(targetAssetId)) throw new ArgumentException("A target asset ID is required.", nameof(targetAssetId));
            selectedSessionId = sessionId;
            selectedTargetAssetId = targetAssetId;
            ClearEpoch();
        }

        /// <summary>Clears the selection, current decision, accepted capture, and all cursors.</summary>
        public void Reset()
        {
            selectedSessionId = null;
            selectedTargetAssetId = null;
            ClearEpoch();
        }

        /// <summary>
        /// Advance the observation frontier and expire an accepted pose without requiring a new
        /// tracker result. Call this on every camera/render tick, including model stalls. Frame and
        /// time ties are allowed. An observation after suppression cannot restore the pose; a new
        /// valid Update is required.
        /// </summary>
        public MobileTrackingGateDecision Observe(MobileTrackingObservation observation)
        {
            if (!IsSelected) return current = Suppressed(MobileTrackingGateReason.NotSelected, -1);
            if (observation == null) return current = Suppressed(MobileTrackingGateReason.MissingObservation, CurrentSourceFrameId());
            if (!Same(selectedSessionId, observation.SessionId))
                return current = Suppressed(MobileTrackingGateReason.SessionMismatch, CurrentSourceFrameId());

            long minimumFrameId = current != null && current.IsRenderable
                ? current.SourceFrameId : -1;
            MobileTrackingGateReason observationReason = AdvanceObservation(observation, minimumFrameId);
            if (observationReason != MobileTrackingGateReason.None)
                return current = Suppressed(observationReason, CurrentSourceFrameId());

            if (current == null || !current.IsRenderable) return current;
            if (!current.AcceptedCaptureTimestampSeconds.HasValue || !current.AcceptedCaptureFrameId.HasValue)
                return current = Suppressed(MobileTrackingGateReason.MissingCaptureIdentity, current.SourceFrameId);

            double captureTimestamp = current.AcceptedCaptureTimestampSeconds.Value;
            if (observation.MonotonicNowSeconds < captureTimestamp)
                return current = Suppressed(MobileTrackingGateReason.ObservationBeforeCapture, current.SourceFrameId);
            if (observation.MonotonicNowSeconds - captureTimestamp > maxAgeSeconds)
                return current = Suppressed(MobileTrackingGateReason.StaleResult, current.SourceFrameId);

            current = new MobileTrackingGateDecision(true, MobileTrackingGateReason.None,
                selectedSessionId, selectedTargetAssetId, current.SourceFrameId,
                current.AcceptedCaptureFrameId, current.AcceptedCaptureTimestampSeconds,
                current.CopyCvCameraFromGltfObjectPoseMetres());
            return current;
        }

        /// <summary>
        /// Evaluate a result against its exact source packet and current observation. Session/asset
        /// identity is checked before any selection-epoch cursor changes. A matching source frame
        /// is consumed before later validation, including failed results. Frame IDs strictly increase;
        /// capture and observation timestamps may tie but may not decrease.
        /// </summary>
        public MobileTrackingGateDecision Update(MobileTrackingResult result,
            MobileCaptureIdentity sourceCapture, MobileTrackingObservation observation)
        {
            if (!IsSelected) return current = Suppressed(MobileTrackingGateReason.NotSelected, -1);

            if (result != null && !Same(selectedSessionId, result.SessionId))
                return current = Suppressed(MobileTrackingGateReason.SessionMismatch, sourceCapture != null ? sourceCapture.FrameId : result.SourceFrameId);
            if (result != null && !Same(selectedTargetAssetId, result.TargetAssetId))
                return current = Suppressed(MobileTrackingGateReason.TargetAssetMismatch, sourceCapture != null ? sourceCapture.FrameId : result.SourceFrameId);
            if (sourceCapture != null && !Same(selectedSessionId, sourceCapture.SessionId))
                return current = Suppressed(MobileTrackingGateReason.SessionMismatch, sourceCapture.FrameId);
            if (observation != null && !Same(selectedSessionId, observation.SessionId))
                return current = Suppressed(MobileTrackingGateReason.SessionMismatch, sourceCapture != null ? sourceCapture.FrameId : -1);

            long candidateFrameId = sourceCapture != null
                ? sourceCapture.FrameId : (result != null ? result.SourceFrameId : -1);
            MobileTrackingGateReason sourceFrameReason = sourceCapture == null
                ? MobileTrackingGateReason.None : ConsumeSourceFrameId(sourceCapture.FrameId);

            long minimumObservationFrameId = sourceCapture != null ? sourceCapture.FrameId : -1;
            MobileTrackingGateReason observationReason = observation == null
                ? MobileTrackingGateReason.MissingObservation
                : AdvanceObservation(observation, minimumObservationFrameId);

            MobileTrackingGateReason captureTimestampReason = MobileTrackingGateReason.None;
            if (sourceCapture != null && sourceFrameReason == MobileTrackingGateReason.None)
                captureTimestampReason = AdvanceCaptureTimestamp(sourceCapture.CaptureTimestampSeconds);

            // A valid observation advances even when the source/result later fails. Invalid observers
            // do not move their own frame/time frontier; the selected source frame remains consumed.
            if (observationReason != MobileTrackingGateReason.None)
                return current = Suppressed(observationReason, candidateFrameId);
            if (sourceFrameReason != MobileTrackingGateReason.None)
                return current = Suppressed(sourceFrameReason, candidateFrameId);
            if (captureTimestampReason != MobileTrackingGateReason.None)
                return current = Suppressed(captureTimestampReason, candidateFrameId);
            if (sourceCapture == null)
                return current = Suppressed(MobileTrackingGateReason.MissingCaptureIdentity, candidateFrameId);
            if (result == null)
                return current = Suppressed(MobileTrackingGateReason.MissingResult, candidateFrameId);

            if (result.SourceFrameId < 0)
                return current = Suppressed(MobileTrackingGateReason.InvalidResultFrameId, candidateFrameId);
            if (!Finite(result.CaptureTimestampSeconds))
                return current = Suppressed(MobileTrackingGateReason.NonFiniteResultTimestamp, candidateFrameId);
            if (result.CaptureTimestampSeconds < 0)
                return current = Suppressed(MobileTrackingGateReason.NegativeResultTimestamp, candidateFrameId);
            if (result.SourceFrameId != sourceCapture.FrameId
                || result.CaptureTimestampSeconds != sourceCapture.CaptureTimestampSeconds)
                return current = Suppressed(MobileTrackingGateReason.ResultCaptureIdentityMismatch, candidateFrameId);

            if (observation.MonotonicNowSeconds < sourceCapture.CaptureTimestampSeconds)
                return current = Suppressed(MobileTrackingGateReason.ObservationBeforeCapture, candidateFrameId);
            if (observation.MonotonicNowSeconds - sourceCapture.CaptureTimestampSeconds > maxAgeSeconds)
                return current = Suppressed(MobileTrackingGateReason.StaleResult, candidateFrameId);

            if (result.MaskAvailability != MobileMaskAvailability.Available
                && result.MaskAvailability != MobileMaskAvailability.Unavailable)
                return current = Suppressed(MobileTrackingGateReason.UnknownMaskAvailability, candidateFrameId);
            if (result.PoseState != MobilePoseTrackingState.Initializing
                && result.PoseState != MobilePoseTrackingState.Recovering
                && result.PoseState != MobilePoseTrackingState.Tracking
                && result.PoseState != MobilePoseTrackingState.Limited
                && result.PoseState != MobilePoseTrackingState.Lost)
                return current = Suppressed(MobileTrackingGateReason.UnknownPoseState, candidateFrameId);
            if (result.RenderState != MobileTrackingRenderState.Suppressed
                && result.RenderState != MobileTrackingRenderState.Visible)
                return current = Suppressed(MobileTrackingGateReason.UnknownRenderState, candidateFrameId);

            double[] pose = result.CopyCvCameraFromGltfObjectPoseMetres();
            if (pose != null && !IsValidPose(pose))
                return current = Suppressed(MobileTrackingGateReason.InvalidPose, candidateFrameId);
            if (result.MaskAvailability != MobileMaskAvailability.Available)
                return current = Suppressed(MobileTrackingGateReason.MaskUnavailable, candidateFrameId);
            if (result.PoseState != MobilePoseTrackingState.Tracking)
                return current = Suppressed(MobileTrackingGateReason.PoseNotTracking, candidateFrameId);
            if (result.RenderState != MobileTrackingRenderState.Visible)
                return current = Suppressed(MobileTrackingGateReason.RenderSuppressed, candidateFrameId);
            if (pose == null)
                return current = Suppressed(MobileTrackingGateReason.MissingPose, candidateFrameId);

            lastAcceptedCaptureFrameId = sourceCapture.FrameId;
            lastAcceptedCaptureTimestampSeconds = sourceCapture.CaptureTimestampSeconds;
            current = new MobileTrackingGateDecision(true, MobileTrackingGateReason.None,
                selectedSessionId, selectedTargetAssetId, candidateFrameId,
                lastAcceptedCaptureFrameId, lastAcceptedCaptureTimestampSeconds, pose);
            return current;
        }

        private MobileTrackingGateReason ConsumeSourceFrameId(long frameId)
        {
            if (frameId < 0) return MobileTrackingGateReason.InvalidSourceFrameId;
            if (hasLastSourceFrame && frameId <= lastSourceFrameId)
                return frameId == lastSourceFrameId
                    ? MobileTrackingGateReason.DuplicateSourceFrameId
                    : MobileTrackingGateReason.OutOfOrderSourceFrameId;
            hasLastSourceFrame = true;
            lastSourceFrameId = frameId;
            return MobileTrackingGateReason.None;
        }

        private MobileTrackingGateReason AdvanceCaptureTimestamp(double timestampSeconds)
        {
            if (!Finite(timestampSeconds)) return MobileTrackingGateReason.NonFiniteCaptureTimestamp;
            if (timestampSeconds < 0) return MobileTrackingGateReason.NegativeCaptureTimestamp;
            if (hasLastCaptureTimestamp && timestampSeconds < lastCaptureTimestampSeconds)
                return MobileTrackingGateReason.OutOfOrderCaptureTimestamp;
            hasLastCaptureTimestamp = true;
            lastCaptureTimestampSeconds = timestampSeconds;
            return MobileTrackingGateReason.None;
        }

        private MobileTrackingGateReason AdvanceObservation(MobileTrackingObservation observation,
            long minimumFrameId)
        {
            if (observation == null) return MobileTrackingGateReason.MissingObservation;
            if (observation.FrameId < 0) return MobileTrackingGateReason.InvalidObservationFrameId;
            if (minimumFrameId >= 0 && observation.FrameId < minimumFrameId)
                return MobileTrackingGateReason.ObservationPrecedesSourceFrame;
            if (!Finite(observation.MonotonicNowSeconds))
                return MobileTrackingGateReason.NonFiniteObservationTime;
            if (observation.MonotonicNowSeconds < 0)
                return MobileTrackingGateReason.NegativeObservationTime;
            if (hasLastObservation && observation.FrameId < lastObservationFrameId)
                return MobileTrackingGateReason.OutOfOrderObservationFrameId;
            if (hasLastObservation && observation.MonotonicNowSeconds < lastObservationTimeSeconds)
                return MobileTrackingGateReason.OutOfOrderObservationTime;

            hasLastObservation = true;
            lastObservationFrameId = observation.FrameId;
            lastObservationTimeSeconds = observation.MonotonicNowSeconds;
            return MobileTrackingGateReason.None;
        }

        private void ClearEpoch()
        {
            hasLastSourceFrame = false;
            lastSourceFrameId = 0;
            hasLastCaptureTimestamp = false;
            lastCaptureTimestampSeconds = 0;
            hasLastObservation = false;
            lastObservationFrameId = 0;
            lastObservationTimeSeconds = 0;
            lastAcceptedCaptureFrameId = null;
            lastAcceptedCaptureTimestampSeconds = null;
            current = Suppressed(IsSelected ? MobileTrackingGateReason.NoResult : MobileTrackingGateReason.NotSelected, -1);
        }

        private long CurrentSourceFrameId()
        {
            return current == null ? -1 : current.SourceFrameId;
        }

        private MobileTrackingGateDecision Suppressed(MobileTrackingGateReason reason, long sourceFrameId)
        {
            return new MobileTrackingGateDecision(false, reason, selectedSessionId,
                selectedTargetAssetId, sourceFrameId, lastAcceptedCaptureFrameId,
                lastAcceptedCaptureTimestampSeconds, null);
        }

        private static bool IsValidPose(double[] matrix)
        {
            if (matrix.Length != 16) return false;
            for (int i = 0; i < matrix.Length; i++)
                if (!Finite(matrix[i])) return false;

            if (Math.Abs(matrix[12]) > AffineTolerance
                || Math.Abs(matrix[13]) > AffineTolerance
                || Math.Abs(matrix[14]) > AffineTolerance
                || Math.Abs(matrix[15] - 1) > AffineTolerance)
                return false;

            for (int rowA = 0; rowA < 3; rowA++)
            {
                for (int rowB = 0; rowB < 3; rowB++)
                {
                    double dot = 0;
                    for (int column = 0; column < 3; column++)
                        dot += matrix[rowA * 4 + column] * matrix[rowB * 4 + column];
                    double expected = rowA == rowB ? 1 : 0;
                    if (Math.Abs(dot - expected) > RotationTolerance) return false;
                }
            }

            double determinant =
                matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
                - matrix[1] * (matrix[4] * matrix[10] - matrix[6] * matrix[8])
                + matrix[2] * (matrix[4] * matrix[9] - matrix[5] * matrix[8]);
            return Math.Abs(determinant - 1) <= RotationTolerance;
        }

        private static bool Same(string left, string right)
        {
            return string.Equals(left, right, StringComparison.Ordinal);
        }

        private static bool Finite(double value)
        {
            return !double.IsNaN(value) && !double.IsInfinity(value);
        }
    }
}
