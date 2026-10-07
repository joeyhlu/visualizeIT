using System;
using System.Threading;

namespace VisualizeIt.Core
{
    public enum MobileCameraIngressReason
    {
        None = 0,
        NotActive,
        Disposed,
        InvalidSessionId,
        FocusLost,
        Paused,
        ComponentDisabled,
        SessionNotTracking,
        NoTarget,
        CaptureClockUnmapped,
        ImageBasisUnverified,
        FrameMatchedCameraPoseUnverified,
        InvalidImageDimensions,
        Rgb24SizeOverflow,
        Rgb24PacketTooLarge,
        Rgb24LengthMismatch,
        InvalidProviderImageTimestamp,
        DuplicateProviderImageTimestamp,
        ProviderImageTimestampRegressed,
        InvalidReceiptTimestamp,
        ReceiptTimestampRegressed,
        InvalidObservationTimestamp,
        ObservationTimestampRegressed,
        InvalidMappedCaptureTimestamp,
        CaptureTimestampRegressed,
        CaptureTimestampAfterReceipt,
        IntrinsicsUnavailable,
        InvalidIntrinsics,
        IntrinsicsResolutionMismatch,
        WorkerBusy,
        NoPendingPacket,
        StalePacketGeneration,
        UnexpectedWorkerCompletion,
        DuplicateWorkerCompletion,
        StaleWorkerCompletion,
        InvalidPacketIdentity,
        ForeignResultIdentity,
        FrameOrdinalOverflow
    }

    public enum MobileCameraIngressOfferStatus
    {
        Rejected = 0,
        DiagnosticOnly,
        Queued,
        ReplacedPending
    }

    public enum MobileCameraImageTransformation
    {
        None = 0
    }

    /// <summary>Camera intrinsics in pixel units at their own declared image resolution.</summary>
    public struct MobileCameraIntrinsics
    {
        public double FocalLengthX { get; private set; }
        public double FocalLengthY { get; private set; }
        public double PrincipalPointX { get; private set; }
        public double PrincipalPointY { get; private set; }
        public int ResolutionWidth { get; private set; }
        public int ResolutionHeight { get; private set; }

        public MobileCameraIntrinsics(double focalLengthX, double focalLengthY,
            double principalPointX, double principalPointY, int resolutionWidth, int resolutionHeight)
        {
            FocalLengthX = focalLengthX;
            FocalLengthY = focalLengthY;
            PrincipalPointX = principalPointX;
            PrincipalPointY = principalPointY;
            ResolutionWidth = resolutionWidth;
            ResolutionHeight = resolutionHeight;
        }
    }

    /// <summary>
    /// Owned native-resolution RGB24 HWC transport packet. It exposes only defensive copies of
    /// its pixels. The ingress state owns packet disposal after accepting a packet. The producer's
    /// converted input remains caller-owned while this packet clone is made; CopyRgb24Hwc returns
    /// another caller-owned allocation that the consumer must release. Those allocations are
    /// outside the coordinator's one-in-flight plus one-pending packet bound.
    /// </summary>
    public sealed class MobileCameraIngressPacket
    {
        private readonly object bufferLock = new object();
        private byte[] rgb24Hwc;
        private bool disposed;

        public long Generation { get; private set; }
        public string SessionId { get; private set; }
        public string TargetAssetId { get; private set; }
        public long FrameId { get; private set; }
        public double ProviderImageTimestampSeconds { get; private set; }
        public long? FrameEventTimestampNanoseconds { get; private set; }
        public double CallbackReceiptTimestampSeconds { get; private set; }
        public MobileCaptureIdentity CaptureIdentity { get; private set; }
        public int Width { get; private set; }
        public int Height { get; private set; }
        public int InputRectX { get { return 0; } }
        public int InputRectY { get { return 0; } }
        public int InputRectWidth { get { return Width; } }
        public int InputRectHeight { get { return Height; } }
        public MobileCameraImageTransformation ImageTransformation { get { return MobileCameraImageTransformation.None; } }
        public int RowStrideBytes { get; private set; }
        public MobileCameraIntrinsics Intrinsics { get; private set; }
        public bool ImageBasisVerified { get; private set; }
        public bool CanDispatchInference
        {
            get { return !string.IsNullOrEmpty(TargetAssetId) && CaptureIdentity != null && ImageBasisVerified; }
        }
        public bool HasFrameMatchedCameraPose { get { return false; } }
        public MobileCameraIngressReason RenderBlockReason
        {
            get { return MobileCameraIngressReason.FrameMatchedCameraPoseUnverified; }
        }
        public bool IsDisposed
        {
            get { lock (bufferLock) return disposed; }
        }
        public int PixelByteCount
        {
            get { return checked(Width * Height * 3); }
        }

        internal MobileCameraIngressPacket(long generation, string sessionId, string targetAssetId,
            long frameId, double providerImageTimestampSeconds, long? frameEventTimestampNanoseconds,
            double callbackReceiptTimestampSeconds, MobileCaptureIdentity captureIdentity,
            int width, int height, MobileCameraIntrinsics intrinsics, bool imageBasisVerified,
            byte[] sourceRgb24Hwc)
        {
            Generation = generation;
            SessionId = sessionId;
            TargetAssetId = targetAssetId;
            FrameId = frameId;
            ProviderImageTimestampSeconds = providerImageTimestampSeconds;
            FrameEventTimestampNanoseconds = frameEventTimestampNanoseconds;
            CallbackReceiptTimestampSeconds = callbackReceiptTimestampSeconds;
            CaptureIdentity = captureIdentity;
            Width = width;
            Height = height;
            RowStrideBytes = checked(width * 3);
            Intrinsics = intrinsics;
            ImageBasisVerified = imageBasisVerified;
            rgb24Hwc = (byte[])sourceRgb24Hwc.Clone();
        }

        /// <summary>Returns an independently owned copy; callers must release their copy when done.</summary>
        public byte[] CopyRgb24Hwc()
        {
            lock (bufferLock)
            {
                if (disposed || rgb24Hwc == null) throw new ObjectDisposedException(nameof(MobileCameraIngressPacket));
                return (byte[])rgb24Hwc.Clone();
            }
        }

        internal bool DisposeOwnedPixels()
        {
            lock (bufferLock)
            {
                if (disposed) return false;
                disposed = true;
                if (rgb24Hwc != null) Array.Clear(rgb24Hwc, 0, rgb24Hwc.Length);
                rgb24Hwc = null;
                return true;
            }
        }
    }

    /// <summary>
    /// Pure-System transport ownership and lifecycle coordinator. Construct and call all owner
    /// operations on one owner thread; only CompleteWork may be called by a worker callback.
    /// It admits bounded metadata-ready packets and calls the result gate on render ticks. No packet
    /// carries a frame-matched camera pose, so learned rendering stays shut.
    /// </summary>
    public sealed class MobileCameraIngressState : IDisposable
    {
        private readonly object stateLock = new object();
        private readonly MobileTrackingResultGate resultGate;
        private readonly int maxRgb24Bytes;
        private readonly int ownerThreadId;
        private bool focused = true;
        private bool paused;
        private bool componentEnabled = true;
        private bool sessionTracking;
        private bool epochActive;
        private bool disposed;
        private string sessionId;
        private string targetAssetId;
        private long generation;
        private long nextFrameId;
        private long latestFrameId = -1;
        private bool hasProviderTimestamp;
        private double lastProviderTimestampSeconds;
        private bool hasLocalMonotonicFrontier;
        private double lastLocalMonotonicSeconds;
        private bool hasMappedCaptureTimestamp;
        private double lastMappedCaptureTimestampSeconds;
        private MobileCameraIngressPacket pendingPacket;
        private MobileCameraIngressPacket inFlightPacket;
        private long inFlightGeneration;
        private bool hasCompletion;
        private MobileTrackingResult completedResult;
        private long completionGeneration;
        private long droppedPendingPacketCount;
        private long releasedPacketCount;
        private long staleCompletionCount;
        private long diagnosticOnlyFrameCount;
        private MobileCameraIngressReason lastReason;
        private MobileCameraIngressReason lastRenderBlockReason = MobileCameraIngressReason.NotActive;

        public MobileCameraIngressState(MobileTrackingResultGate resultGate, int maxRgb24Bytes)
        {
            if (resultGate == null) throw new ArgumentNullException(nameof(resultGate));
            if (maxRgb24Bytes <= 0) throw new ArgumentOutOfRangeException(nameof(maxRgb24Bytes));
            this.resultGate = resultGate;
            this.maxRgb24Bytes = maxRgb24Bytes;
            ownerThreadId = Thread.CurrentThread.ManagedThreadId;
        }

        public int MaxRgb24Bytes { get { return maxRgb24Bytes; } }
        public long Generation { get { lock (stateLock) return generation; } }
        public string SessionId { get { lock (stateLock) return sessionId; } }
        public string TargetAssetId { get { lock (stateLock) return targetAssetId; } }
        public long LatestFrameId { get { lock (stateLock) return latestFrameId < 0 ? 0 : latestFrameId; } }
        public bool HasEpoch { get { lock (stateLock) return epochActive; } }
        public bool IsActive { get { lock (stateLock) return epochActive && RuntimeAvailable(); } }
        public bool HasPendingPacket { get { lock (stateLock) return pendingPacket != null; } }
        public bool HasInFlightPacket { get { lock (stateLock) return inFlightPacket != null; } }
        public bool HasCompletion { get { lock (stateLock) return hasCompletion; } }
        public long DroppedPendingPacketCount { get { lock (stateLock) return droppedPendingPacketCount; } }
        public long ReleasedPacketCount { get { lock (stateLock) return releasedPacketCount; } }
        public long StaleCompletionCount { get { lock (stateLock) return staleCompletionCount; } }
        public long DiagnosticOnlyFrameCount { get { lock (stateLock) return diagnosticOnlyFrameCount; } }
        public MobileCameraIngressReason LastReason { get { lock (stateLock) return lastReason; } }
        public MobileCameraIngressReason LastRenderBlockReason { get { lock (stateLock) return lastRenderBlockReason; } }
        public bool CanAuthorizeLearnedRendering { get { return false; } }

        /// <summary>
        /// Opens a new app-owned epoch. A null/empty target allows transport diagnostics only.
        /// The caller must use a fresh session ID after any lifecycle or clock invalidation.
        /// </summary>
        public bool BeginEpoch(string newSessionId, string selectedTargetAssetId,
            out MobileCameraIngressReason reason)
        {
            EnsureOwnerThread();
            lock (stateLock)
            {
                if (disposed) return Fail(MobileCameraIngressReason.Disposed, out reason);
                if (string.IsNullOrEmpty(newSessionId))
                {
                    if (epochActive || pendingPacket != null || hasCompletion)
                        RetireEpoch(true, MobileCameraIngressReason.InvalidSessionId);
                    return Fail(MobileCameraIngressReason.InvalidSessionId, out reason);
                }
                if (!RuntimeAvailable()) return Fail(MobileCameraIngressReason.NotActive, out reason);

                RetireEpoch(true, MobileCameraIngressReason.None);
                sessionId = newSessionId;
                targetAssetId = string.IsNullOrEmpty(selectedTargetAssetId) ? null : selectedTargetAssetId;
                epochActive = true;
                nextFrameId = 0;
                latestFrameId = -1;
                ResetClockFrontiers();
                if (targetAssetId == null) resultGate.Reset();
                else resultGate.Select(sessionId, targetAssetId);
                lastReason = MobileCameraIngressReason.None;
                lastRenderBlockReason = targetAssetId == null
                    ? MobileCameraIngressReason.NoTarget
                    : MobileCameraIngressReason.FrameMatchedCameraPoseUnverified;
                reason = MobileCameraIngressReason.None;
                return true;
            }
        }

        public void SetFocused(bool value) { SetRuntimeCondition(ref focused, value, MobileCameraIngressReason.FocusLost); }
        public void SetPaused(bool value) { SetRuntimeCondition(ref paused, value, MobileCameraIngressReason.Paused); }
        public void SetComponentEnabled(bool value) { SetRuntimeCondition(ref componentEnabled, value, MobileCameraIngressReason.ComponentDisabled); }
        public void SetSessionTracking(bool value) { SetRuntimeCondition(ref sessionTracking, value, MobileCameraIngressReason.SessionNotTracking); }

        /// <summary>
        /// Accepts one converted full-resolution RGB24/HWC image. mappedCaptureTimestampSeconds
        /// must be null until a reviewed provider-to-Observe clock mapper exists; it is never copied
        /// from ProviderImageTimestampSeconds or CallbackReceiptTimestampSeconds implicitly.
        /// Callback receipt and RenderTick values share one owner-validated monotonic frontier; a
        /// supplied mapped capture instant must not be later than this callback's receipt time.
        /// </summary>
        public MobileCameraIngressOfferStatus OfferFrame(byte[] rgb24Hwc, int width, int height,
            MobileCameraIntrinsics? intrinsics, double providerImageTimestampSeconds,
            long? frameEventTimestampNanoseconds, double callbackReceiptTimestampSeconds,
            double? mappedCaptureTimestampSeconds, bool imageBasisVerified,
            out MobileCameraIngressReason reason)
        {
            EnsureOwnerThread();
            lock (stateLock)
            {
                if (disposed) return FailOffer(MobileCameraIngressOfferStatus.Rejected, MobileCameraIngressReason.Disposed, out reason);
                if (!epochActive || !RuntimeAvailable())
                    return FailOffer(MobileCameraIngressOfferStatus.Rejected, MobileCameraIngressReason.NotActive, out reason);

                if (!Finite(providerImageTimestampSeconds) || providerImageTimestampSeconds < 0)
                    return InvalidateOnClockFailure(MobileCameraIngressReason.InvalidProviderImageTimestamp, out reason);
                if (!Finite(callbackReceiptTimestampSeconds) || callbackReceiptTimestampSeconds < 0)
                    return InvalidateOnClockFailure(MobileCameraIngressReason.InvalidReceiptTimestamp, out reason);
                if (hasLocalMonotonicFrontier && callbackReceiptTimestampSeconds < lastLocalMonotonicSeconds)
                    return InvalidateOnClockFailure(MobileCameraIngressReason.ReceiptTimestampRegressed, out reason);
                if (hasProviderTimestamp && providerImageTimestampSeconds < lastProviderTimestampSeconds)
                    return InvalidateOnClockFailure(MobileCameraIngressReason.ProviderImageTimestampRegressed, out reason);

                if (mappedCaptureTimestampSeconds.HasValue)
                {
                    double mapped = mappedCaptureTimestampSeconds.Value;
                    if (!Finite(mapped) || mapped < 0)
                        return InvalidateOnClockFailure(MobileCameraIngressReason.InvalidMappedCaptureTimestamp, out reason);
                    if (mapped > callbackReceiptTimestampSeconds)
                        return InvalidateOnClockFailure(MobileCameraIngressReason.CaptureTimestampAfterReceipt, out reason);
                    if (hasMappedCaptureTimestamp && mapped < lastMappedCaptureTimestampSeconds)
                        return InvalidateOnClockFailure(MobileCameraIngressReason.CaptureTimestampRegressed, out reason);
                }

                if (hasProviderTimestamp && providerImageTimestampSeconds == lastProviderTimestampSeconds)
                {
                    AdvanceLocalMonotonicFrontier(callbackReceiptTimestampSeconds);
                    return FailOffer(MobileCameraIngressOfferStatus.DiagnosticOnly,
                        MobileCameraIngressReason.DuplicateProviderImageTimestamp, out reason);
                }

                AdvanceLocalMonotonicFrontier(callbackReceiptTimestampSeconds);

                int expectedBytes;
                MobileCameraIngressReason sizeReason = ValidateRgb24Size(width, height, rgb24Hwc, out expectedBytes);
                if (sizeReason != MobileCameraIngressReason.None)
                    return FailOffer(MobileCameraIngressOfferStatus.Rejected, sizeReason, out reason);

                if (nextFrameId == long.MaxValue)
                    return InvalidateOnClockFailure(MobileCameraIngressReason.FrameOrdinalOverflow, out reason);

                long frameId = nextFrameId++;
                latestFrameId = frameId;
                hasProviderTimestamp = true;
                lastProviderTimestampSeconds = providerImageTimestampSeconds;
                if (mappedCaptureTimestampSeconds.HasValue)
                {
                    hasMappedCaptureTimestamp = true;
                    lastMappedCaptureTimestampSeconds = mappedCaptureTimestampSeconds.Value;
                }

                if (!intrinsics.HasValue)
                    return FailOffer(MobileCameraIngressOfferStatus.DiagnosticOnly,
                        MobileCameraIngressReason.IntrinsicsUnavailable, out reason);
                MobileCameraIngressReason intrinsicsReason = ValidateIntrinsics(intrinsics.Value, width, height);
                if (intrinsicsReason != MobileCameraIngressReason.None)
                    return FailOffer(MobileCameraIngressOfferStatus.DiagnosticOnly, intrinsicsReason, out reason);
                if (targetAssetId == null)
                    return FailOffer(MobileCameraIngressOfferStatus.DiagnosticOnly, MobileCameraIngressReason.NoTarget, out reason);
                if (!mappedCaptureTimestampSeconds.HasValue)
                    return FailOffer(MobileCameraIngressOfferStatus.DiagnosticOnly,
                        MobileCameraIngressReason.CaptureClockUnmapped, out reason);
                if (!imageBasisVerified)
                    return FailOffer(MobileCameraIngressOfferStatus.DiagnosticOnly,
                        MobileCameraIngressReason.ImageBasisUnverified, out reason);

                var captureIdentity = new MobileCaptureIdentity(sessionId, frameId,
                    mappedCaptureTimestampSeconds.Value);
                bool replaced = pendingPacket != null;
                if (replaced)
                {
                    // Dispose the ingress-owned pending buffer before cloning its replacement.
                    Release(pendingPacket);
                    pendingPacket = null;
                    droppedPendingPacketCount++;
                }
                var packet = new MobileCameraIngressPacket(generation, sessionId, targetAssetId,
                    frameId, providerImageTimestampSeconds, frameEventTimestampNanoseconds,
                    callbackReceiptTimestampSeconds, captureIdentity, width, height,
                    intrinsics.Value, imageBasisVerified, rgb24Hwc);
                pendingPacket = packet;
                lastReason = MobileCameraIngressReason.None;
                lastRenderBlockReason = MobileCameraIngressReason.FrameMatchedCameraPoseUnverified;
                reason = MobileCameraIngressReason.None;
                return replaced ? MobileCameraIngressOfferStatus.ReplacedPending
                    : MobileCameraIngressOfferStatus.Queued;
            }
        }

        /// <summary>Transfers the pending packet to the sole worker slot when no old worker remains.</summary>
        public bool TryBeginNextWork(out MobileCameraIngressPacket packet,
            out MobileCameraIngressReason reason)
        {
            EnsureOwnerThread();
            lock (stateLock)
            {
                packet = null;
                if (disposed) return Fail(MobileCameraIngressReason.Disposed, out reason);
                if (!epochActive || !RuntimeAvailable()) return Fail(MobileCameraIngressReason.NotActive, out reason);
                if (inFlightPacket != null || hasCompletion) return Fail(MobileCameraIngressReason.WorkerBusy, out reason);
                if (pendingPacket == null) return Fail(MobileCameraIngressReason.NoPendingPacket, out reason);

                if (pendingPacket.Generation != generation)
                {
                    Release(pendingPacket);
                    pendingPacket = null;
                    return Fail(MobileCameraIngressReason.StalePacketGeneration, out reason);
                }

                inFlightPacket = pendingPacket;
                pendingPacket = null;
                inFlightGeneration = inFlightPacket.Generation;
                packet = inFlightPacket;
                reason = MobileCameraIngressReason.None;
                lastReason = reason;
                return true;
            }
        }

        /// <summary>
        /// Worker-completion callback seam. It only publishes one immutable terminal result; the
        /// owner thread retains the in-flight slot and reaps it on RenderTick. After Dispose, this
        /// callback may release the finished worker's packet as the terminal-shutdown exception.
        /// </summary>
        public MobileCameraIngressReason CompleteWork(MobileCameraIngressPacket packet,
            long workerGeneration, MobileTrackingResult result)
        {
            lock (stateLock)
            {
                if (packet == null || inFlightPacket == null
                    || !object.ReferenceEquals(packet, inFlightPacket)
                    || workerGeneration != inFlightGeneration)
                    return lastReason = MobileCameraIngressReason.UnexpectedWorkerCompletion;
                if (hasCompletion)
                    return lastReason = MobileCameraIngressReason.DuplicateWorkerCompletion;

                if (disposed)
                {
                    // Shutdown has no future owner/render tick. This callback runs after the worker
                    // stopped reading, so it may perform the one terminal-shutdown release.
                    Release(inFlightPacket);
                    inFlightPacket = null;
                    staleCompletionCount++;
                    return lastReason = MobileCameraIngressReason.StaleWorkerCompletion;
                }

                completedResult = result;
                completionGeneration = workerGeneration;
                hasCompletion = true;
                if (!epochActive || workerGeneration != generation || !RuntimeAvailable())
                    return lastReason = MobileCameraIngressReason.StaleWorkerCompletion;
                return lastReason = MobileCameraIngressReason.None;
            }
        }

        /// <summary>
        /// Render-thread seam. Drains at most one completed result, then calls the existing gate's
        /// Observe exactly once with this tick's observation, even if no camera/result callback ran.
        /// </summary>
        public MobileTrackingGateDecision RenderTick(double monotonicNowSeconds)
        {
            EnsureOwnerThread();
            lock (stateLock)
            {
                var observation = new MobileTrackingObservation(sessionId ?? string.Empty,
                    latestFrameId < 0 ? 0 : latestFrameId, monotonicNowSeconds);
                MobileCameraIngressReason clockReason = ValidateOwnerMonotonicTimestamp(monotonicNowSeconds);
                if (clockReason != MobileCameraIngressReason.None)
                {
                    RetireEpoch(epochActive || pendingPacket != null || hasCompletion,
                        clockReason);
                    return resultGate.Observe(observation);
                }
                AdvanceLocalMonotonicFrontier(monotonicNowSeconds);

                if (hasCompletion)
                {
                    MobileCameraIngressPacket completedPacket = inFlightPacket;
                    MobileTrackingResult result = completedResult;
                    long resultGeneration = completionGeneration;
                    hasCompletion = false;
                    completedResult = null;
                    completionGeneration = 0;

                    try
                    {
                        if (completedPacket != null && epochActive && RuntimeAvailable()
                            && resultGeneration == generation && completedPacket.Generation == generation)
                        {
                            if (!PacketIdentityMatchesSelection(completedPacket))
                            {
                                lastReason = MobileCameraIngressReason.InvalidPacketIdentity;
                            }
                            else if (result != null
                                && (!Same(result.SessionId, sessionId)
                                    || !Same(result.TargetAssetId, targetAssetId)))
                            {
                                // A foreign result must not clear or advance the current selection.
                                lastReason = MobileCameraIngressReason.ForeignResultIdentity;
                            }
                            else
                            {
                                MobileTrackingResult gatedResult = result;
                                if (gatedResult != null && !completedPacket.HasFrameMatchedCameraPose)
                                {
                                    gatedResult = new MobileTrackingResult(gatedResult.SessionId,
                                        gatedResult.TargetAssetId, gatedResult.SourceFrameId,
                                        gatedResult.CaptureTimestampSeconds, gatedResult.MaskAvailability,
                                        gatedResult.PoseState, MobileTrackingRenderState.Suppressed, null);
                                    lastRenderBlockReason = MobileCameraIngressReason.FrameMatchedCameraPoseUnverified;
                                }
                                // Preserve the exact packet identity. The gate still validates
                                // result frame/time equality and consumes ordinary failed results.
                                resultGate.Update(gatedResult, completedPacket.CaptureIdentity, observation);
                            }
                        }
                        else
                        {
                            staleCompletionCount++;
                            lastReason = MobileCameraIngressReason.StaleWorkerCompletion;
                        }
                    }
                    finally
                    {
                        if (completedPacket != null) Release(completedPacket);
                        if (object.ReferenceEquals(inFlightPacket, completedPacket)) inFlightPacket = null;
                    }
                }

                return resultGate.Observe(observation);
            }
        }

        public void InvalidateEpoch()
        {
            EnsureOwnerThread();
            lock (stateLock)
            {
                if (disposed) return;
                RetireEpoch(epochActive || pendingPacket != null || hasCompletion,
                    MobileCameraIngressReason.NotActive);
            }
        }

        public void Dispose()
        {
            EnsureOwnerThread();
            lock (stateLock)
            {
                if (disposed) return;
                disposed = true;
                RetireEpoch(epochActive || pendingPacket != null || hasCompletion,
                    MobileCameraIngressReason.Disposed);
            }
        }

        private void SetRuntimeCondition(ref bool field, bool value, MobileCameraIngressReason reason)
        {
            EnsureOwnerThread();
            lock (stateLock)
            {
                if (disposed || field == value) return;
                bool wasAvailable = RuntimeAvailable();
                field = value;
                bool isAvailable = RuntimeAvailable();
                if (wasAvailable && !isAvailable)
                    RetireEpoch(epochActive || pendingPacket != null || hasCompletion, reason);
            }
        }

        private bool RuntimeAvailable()
        {
            return focused && !paused && componentEnabled && sessionTracking;
        }

        private MobileCameraIngressOfferStatus InvalidateOnClockFailure(
            MobileCameraIngressReason failure, out MobileCameraIngressReason reason)
        {
            RetireEpoch(epochActive || pendingPacket != null || hasCompletion, failure);
            reason = failure;
            return MobileCameraIngressOfferStatus.Rejected;
        }

        private void RetireEpoch(bool advanceGeneration, MobileCameraIngressReason reason)
        {
            if (advanceGeneration) generation = checked(generation + 1);
            if (pendingPacket != null)
            {
                Release(pendingPacket);
                pendingPacket = null;
            }
            if (hasCompletion)
            {
                if (inFlightPacket != null) Release(inFlightPacket);
                inFlightPacket = null;
                hasCompletion = false;
                completedResult = null;
                completionGeneration = 0;
            }

            // An unfinished worker retains the only in-flight slot until CompleteWork reaps it.
            resultGate.Reset();
            epochActive = false;
            sessionId = null;
            targetAssetId = null;
            nextFrameId = 0;
            latestFrameId = -1;
            ResetClockFrontiers();
            lastReason = reason;
            lastRenderBlockReason = reason;
        }

        private void ResetClockFrontiers()
        {
            hasProviderTimestamp = false;
            lastProviderTimestampSeconds = 0;
            hasLocalMonotonicFrontier = false;
            lastLocalMonotonicSeconds = 0;
            hasMappedCaptureTimestamp = false;
            lastMappedCaptureTimestampSeconds = 0;
        }

        private MobileCameraIngressReason ValidateOwnerMonotonicTimestamp(double timestampSeconds)
        {
            if (!Finite(timestampSeconds) || timestampSeconds < 0)
                return MobileCameraIngressReason.InvalidObservationTimestamp;
            if (hasLocalMonotonicFrontier && timestampSeconds < lastLocalMonotonicSeconds)
                return MobileCameraIngressReason.ObservationTimestampRegressed;
            return MobileCameraIngressReason.None;
        }

        private void AdvanceLocalMonotonicFrontier(double timestampSeconds)
        {
            hasLocalMonotonicFrontier = true;
            lastLocalMonotonicSeconds = timestampSeconds;
        }

        private bool PacketIdentityMatchesSelection(MobileCameraIngressPacket packet)
        {
            MobileCaptureIdentity capture = packet.CaptureIdentity;
            return targetAssetId != null
                && Same(packet.SessionId, sessionId)
                && Same(packet.TargetAssetId, targetAssetId)
                && capture != null
                && Same(capture.SessionId, packet.SessionId)
                && capture.FrameId == packet.FrameId;
        }

        private void EnsureOwnerThread()
        {
            if (Thread.CurrentThread.ManagedThreadId != ownerThreadId)
                throw new InvalidOperationException("Mobile camera ingress owner operations must run on the construction thread.");
        }

        private void Release(MobileCameraIngressPacket packet)
        {
            if (packet != null && packet.DisposeOwnedPixels()) releasedPacketCount++;
        }

        private bool Fail(MobileCameraIngressReason failure, out MobileCameraIngressReason reason)
        {
            lastReason = failure;
            reason = failure;
            return false;
        }

        private MobileCameraIngressOfferStatus FailOffer(MobileCameraIngressOfferStatus status,
            MobileCameraIngressReason failure, out MobileCameraIngressReason reason)
        {
            lastReason = failure;
            reason = failure;
            if (status == MobileCameraIngressOfferStatus.DiagnosticOnly)
            {
                diagnosticOnlyFrameCount++;
                lastRenderBlockReason = failure;
            }
            return status;
        }

        private MobileCameraIngressReason ValidateRgb24Size(int width, int height,
            byte[] rgb24Hwc, out int expectedBytes)
        {
            expectedBytes = 0;
            if (width <= 0 || height <= 0) return MobileCameraIngressReason.InvalidImageDimensions;
            try
            {
                expectedBytes = checked(checked(width * height) * 3);
            }
            catch (OverflowException)
            {
                return MobileCameraIngressReason.Rgb24SizeOverflow;
            }
            if (expectedBytes > maxRgb24Bytes) return MobileCameraIngressReason.Rgb24PacketTooLarge;
            if (rgb24Hwc == null || rgb24Hwc.Length != expectedBytes)
                return MobileCameraIngressReason.Rgb24LengthMismatch;
            return MobileCameraIngressReason.None;
        }

        private static MobileCameraIngressReason ValidateIntrinsics(
            MobileCameraIntrinsics intrinsics, int width, int height)
        {
            if (intrinsics.ResolutionWidth <= 0 || intrinsics.ResolutionHeight <= 0
                || !Finite(intrinsics.FocalLengthX) || !Finite(intrinsics.FocalLengthY)
                || !Finite(intrinsics.PrincipalPointX) || !Finite(intrinsics.PrincipalPointY)
                || intrinsics.FocalLengthX <= 0 || intrinsics.FocalLengthY <= 0)
                return MobileCameraIngressReason.InvalidIntrinsics;
            if (intrinsics.ResolutionWidth != width || intrinsics.ResolutionHeight != height)
                return MobileCameraIngressReason.IntrinsicsResolutionMismatch;
            return MobileCameraIngressReason.None;
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
