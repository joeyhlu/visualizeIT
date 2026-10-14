"""Initialized RGB rigid tracker. No evaluation annotations are accepted here."""
from dataclasses import dataclass
import numpy as np
from .vision import cv2


def _exact_unique_indices(rows):
    """Return first indices for exact numeric rows, retaining input order."""
    if not len(rows):
        return np.empty(0, dtype=np.int64)
    _, indices = np.unique(rows, axis=0, return_index=True)
    return np.sort(indices)


def _canonicalize_correspondences(pixels, points):
    """Collapse exact pairs and withhold ambiguous source or pixel identities.

    The returned proposal count is the number of distinct source anchors before
    collision exclusions. Coordinates are compared exactly; nearby anchors or
    observations remain separate.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if (pixels.ndim != 2 or pixels.shape[1:] != (2,) or
            points.shape != (len(pixels), 3) or
            not np.isfinite(pixels).all() or not np.isfinite(points).all()):
        raise ValueError('Correspondences must be finite, paired Nx2/Nx3 arrays.')
    if not len(points):
        return (np.empty((0, 2), dtype=np.float64),
                np.empty((0, 3), dtype=np.float64), 0, 0, 0, 0)

    pairs = np.column_stack([points, pixels])
    unique_pair_indices = _exact_unique_indices(pairs)
    pairs = pairs[unique_pair_indices]
    pair_points = points[unique_pair_indices]
    pair_pixels = pixels[unique_pair_indices]
    proposal_count = len(np.unique(pair_points, axis=0))

    # One frozen source identity cannot contribute several observed pixels to
    # the same pose fit. Keep the proposal in the denominator, but withhold all
    # of its alternatives from support.
    _, point_inverse, point_counts = np.unique(
        pair_points, axis=0, return_inverse=True, return_counts=True)
    ambiguous = np.zeros(len(pair_points), dtype=bool)
    for group_index, group_count in enumerate(point_counts):
        if group_count > 1:
            ambiguous[point_inverse == group_index] = True
    ambiguous_anchor_count = int(np.sum(point_counts > 1))

    _, pixel_inverse, pixel_counts = np.unique(
        pair_pixels, axis=0, return_inverse=True, return_counts=True)
    collision = np.zeros(len(pair_points), dtype=bool)
    for group_index, group_count in enumerate(pixel_counts):
        if group_count <= 1:
            continue
        rows = np.flatnonzero(pixel_inverse == group_index)
        if len(np.unique(pair_points[rows], axis=0)) > 1:
            collision[rows] = True
    excluded = ambiguous | collision
    return (pair_pixels[~excluded], pair_points[~excluded], proposal_count,
            len(pixels)-len(unique_pair_indices), ambiguous_anchor_count,
            int(np.sum(collision & ~ambiguous)))


def _support_nonplanarity(points):
    """Return (passes, rank ratio) for the established nonplanarity gate."""
    if len(points) < 3:
        return False, 0.
    spread = np.linalg.svd(points-points.mean(axis=0), compute_uv=False)
    ratio = float(spread[2]/max(spread[0], 1e-15))
    return bool(spread[1] >= 1e-6 and ratio >= .001), ratio


def _native_pixels(pixels, pixel_to_native):
    if pixel_to_native is None:
        return np.asarray(pixels, dtype=np.float64).copy()
    transform = np.asarray(pixel_to_native, dtype=np.float64)
    if (transform.shape != (3, 3) or not np.isfinite(transform).all() or
            not np.allclose(transform[2], [0., 0., 1.], atol=1e-10, rtol=0) or
            not np.allclose(transform[0, 1], 0., atol=1e-10, rtol=0) or
            not np.allclose(transform[1, 0], 0., atol=1e-10, rtol=0) or
            transform[0, 0] <= 0. or transform[1, 1] <= 0. or
            abs(float(np.linalg.det(transform))) <= 1e-12):
        raise ValueError('Pixel-to-native transform must describe positive crop/resize geometry.')
    homogeneous = np.column_stack([pixels, np.ones(len(pixels))])@transform.T
    if not np.isfinite(homogeneous).all() or (np.abs(homogeneous[:, 2]) <= 1e-12).any():
        raise ValueError('Pixel-to-native transform produced invalid coordinates.')
    return homogeneous[:, :2]/homogeneous[:, 2, None]


def reconcile_flow_fresh(flow_pixels, flow_points, fresh_pixels, fresh_points,
                         k_native, frame_id, solver, pixel_to_native=None):
    """Fuse flow and descriptor observations using an independent pose witness.

    Inputs are kept separate until exact source-anchor ambiguity and native-pixel
    conflict checks are complete. ``solver`` is a side-effect-free pose solver
    such as ``RigidTracker.solve``; its result is used only as private witness
    evidence. The returned proposal count preserves the pre-exclusion denominator.
    """
    empty_pixels = np.empty((0, 2), dtype=np.float64)
    empty_points = np.empty((0, 3), dtype=np.float64)
    diagnostics = dict(
        proposedAnchors=0, flowFreshConflicts=0, witnessState='not_needed',
        witnessSupport=0, selectedFlow=0, selectedFresh=0,
        excludedAmbiguousAnchors=0, excludedUnsupportedAnchors=0,
        excludedCollidingAnchors=0, duplicatePairsCollapsed=0,
        fusionFailure=None,
    )

    def checked_source(raw_pixels, raw_points):
        source_pixels = np.asarray(raw_pixels, dtype=np.float64)
        source_points = np.asarray(raw_points, dtype=np.float64)
        if (source_pixels.ndim != 2 or source_pixels.shape[1:] != (2,) or
                source_points.shape != (len(source_pixels), 3) or
                not np.isfinite(source_pixels).all() or not np.isfinite(source_points).all()):
            raise ValueError('Flow and fresh correspondences must be finite and paired.')
        source_pixels = _native_pixels(source_pixels, pixel_to_native)
        return source_pixels, source_points

    try:
        flow_pixels, flow_points = checked_source(flow_pixels, flow_points)
        fresh_pixels, fresh_points = checked_source(fresh_pixels, fresh_points)
        k_native = np.asarray(k_native, dtype=np.float64)
        if k_native.shape != (3, 3) or not np.isfinite(k_native).all():
            raise ValueError('Native intrinsics must be a finite 3x3 matrix.')
    except (TypeError, ValueError, np.linalg.LinAlgError):
        diagnostics['fusionFailure'] = 'invalid_flow_fresh_correspondences'
        return empty_pixels, empty_points, 0, diagnostics

    def source_map(source_pixels, source_points):
        # Collapse only exact repeated pairs. A source anchor with two different
        # observations is ambiguous and omitted from both witness and final set.
        by_anchor = {}
        seen_pairs = set()
        duplicate_count = 0
        for pixel, point in zip(source_pixels, source_points):
            point_key = tuple(float(value) for value in point)
            pair_key = (point_key, tuple(float(value) for value in pixel))
            if pair_key in seen_pairs:
                duplicate_count += 1
                continue
            seen_pairs.add(pair_key)
            by_anchor.setdefault(point_key, []).append((pixel.copy(), point.copy()))
        ambiguous = {key for key, rows in by_anchor.items() if len(rows) > 1}
        mapping = {key: rows[0] for key, rows in by_anchor.items() if key not in ambiguous}
        return mapping, ambiguous, duplicate_count

    # Detect exact observation collisions before any same-anchor alternative is
    # selected or withheld as a flow/fresh conflict. Otherwise an unused fresh
    # row can launder a pixel shared by a different frozen identity out of the
    # independent witness. Compare already-converted native UV exactly.
    native_pixel_anchors = {}
    proposed_anchor_keys = set()
    for source_pixels, source_points in ((flow_pixels, flow_points),
                                         (fresh_pixels, fresh_points)):
        for pixel, point in zip(source_pixels, source_points):
            pixel_key = tuple(float(value) for value in pixel)
            anchor_key = tuple(float(value) for value in point)
            native_pixel_anchors.setdefault(pixel_key, set()).add(anchor_key)
            proposed_anchor_keys.add(anchor_key)
    colliding_keys = set()
    for anchor_keys in native_pixel_anchors.values():
        if len(anchor_keys) > 1:
            colliding_keys.update(anchor_keys)

    flow_map, flow_ambiguous, flow_duplicates = source_map(flow_pixels, flow_points)
    fresh_map, fresh_ambiguous, fresh_duplicates = source_map(fresh_pixels, fresh_points)
    all_anchor_keys = set(flow_map) | set(fresh_map) | flow_ambiguous | fresh_ambiguous
    anchor_order = list(dict.fromkeys(
        list(flow_map)+list(fresh_map)+list(flow_ambiguous)+list(fresh_ambiguous)))
    proposal_count = len(proposed_anchor_keys)
    # Keep source-map bookkeeping as a consistency guard if the representation
    # changes; every proposed frozen identity must survive into either a map or
    # its same-source ambiguity set.
    if all_anchor_keys != proposed_anchor_keys:
        diagnostics['fusionFailure'] = 'invalid_flow_fresh_identity_groups'
        return empty_pixels, empty_points, proposal_count, diagnostics
    diagnostics['proposedAnchors'] = proposal_count
    diagnostics['duplicatePairsCollapsed'] = flow_duplicates+fresh_duplicates
    ambiguous_keys = flow_ambiguous | fresh_ambiguous
    diagnostics['excludedAmbiguousAnchors'] = len(ambiguous_keys)
    collision_anchors = set(colliding_keys)
    diagnostics['excludedCollidingAnchors'] = len(collision_anchors)

    witness_rows = []
    selected_rows = []
    conflicts = []
    for key in anchor_order:
        if key in ambiguous_keys or key in colliding_keys:
            continue
        flow = flow_map.get(key)
        fresh = fresh_map.get(key)
        if flow is not None and fresh is not None:
            disagreement = float(np.linalg.norm(flow[0]-fresh[0]))
            if not np.isfinite(disagreement):
                diagnostics['excludedUnsupportedAnchors'] += 1
                continue
            if disagreement > 3.:
                conflicts.append((key, flow, fresh))
                continue
            # Existing flow-first behavior is retained for nonconflicting anchors.
            witness_rows.append(flow)
            selected_rows.append(('flow', flow))
        elif flow is not None:
            witness_rows.append(flow)
            selected_rows.append(('flow', flow))
        elif fresh is not None:
            witness_rows.append(fresh)
            selected_rows.append(('fresh', fresh))

    diagnostics['flowFreshConflicts'] = len(conflicts)
    def rows_to_arrays(rows):
        if not rows:
            return empty_pixels.copy(), empty_points.copy()
        return (np.asarray([row[0] for row in rows], dtype=np.float64).reshape(-1, 2),
                np.asarray([row[1] for row in rows], dtype=np.float64).reshape(-1, 3))

    def collision_filter(rows):
        pixels, points = rows_to_arrays(rows)
        kept_pixels, kept_points, _, duplicate_count, _, collision_count = _canonicalize_correspondences(
            pixels, points)
        if collision_count:
            # Record identities so an anchor removed in both passes is counted once.
            before_keys = [tuple(float(value) for value in point) for point in points]
            kept_keys = {tuple(float(value) for value in point) for point in kept_points}
            for key in before_keys:
                if key not in kept_keys:
                    collision_anchors.add(key)
        diagnostics['duplicatePairsCollapsed'] += duplicate_count
        return kept_pixels, kept_points

    witness_pixels, witness_points = collision_filter(witness_rows)
    if conflicts:
        witness = solver(witness_pixels, witness_points, k_native, frame_id,
                         proposal_count=proposal_count)
        if isinstance(witness, tuple):
            witness_result = witness[0]
            diagnostics['witnessState'] = witness_result.state
            diagnostics['witnessSupport'] = int(witness_result.statistics.get('inliers', 0))
            try:
                witness_rvec = cv2.Rodrigues(witness_result.rotation)[0]
                conflict_choices = []
                for key, flow, fresh in conflicts:
                    point = flow[1].reshape(1, 3)
                    projected = cv2.projectPoints(point, witness_rvec,
                                                   witness_result.translation,
                                                   k_native, None)[0].reshape(2)
                    flow_error = float(np.linalg.norm(projected-flow[0]))
                    fresh_error = float(np.linalg.norm(projected-fresh[0]))
                    flow_ok = np.isfinite(flow_error) and flow_error <= 3.
                    fresh_ok = np.isfinite(fresh_error) and fresh_error <= 3.
                    if flow_ok and fresh_ok:
                        if fresh_error < flow_error:
                            conflict_choices.append(('fresh', fresh))
                        else:
                            conflict_choices.append(('flow', flow))
                    elif flow_ok:
                        conflict_choices.append(('flow', flow))
                    elif fresh_ok:
                        conflict_choices.append(('fresh', fresh))
                    else:
                        diagnostics['excludedUnsupportedAnchors'] += 1
                selected_rows.extend(conflict_choices)
            except (cv2.error, TypeError, ValueError, IndexError):
                diagnostics['witnessState'] = 'invalid_witness_pose'
                diagnostics['excludedUnsupportedAnchors'] = len(conflicts)
        else:
            diagnostics['witnessState'] = witness.state
            diagnostics['witnessSupport'] = int(witness.statistics.get('inliers', 0))
            diagnostics['excludedUnsupportedAnchors'] += len(conflicts)

    final_pixels, final_points = collision_filter([row for _, row in selected_rows])
    final_keys = {tuple(float(value) for value in point) for point in final_points}
    selected_source = {tuple(float(value) for value in row[1]): source
                       for source, row in selected_rows}
    diagnostics['selectedFlow'] = sum(selected_source.get(key) == 'flow' for key in final_keys)
    diagnostics['selectedFresh'] = sum(selected_source.get(key) == 'fresh' for key in final_keys)
    diagnostics['reconciledAnchors'] = len(final_points)
    diagnostics['excludedCollidingAnchors'] = len(collision_anchors)
    return final_pixels, final_points, proposal_count, diagnostics


@dataclass
class TrackingResult:
    frame_id: int
    state: str
    reason: str | None
    rotation: np.ndarray | None
    translation: np.ndarray | None
    statistics: dict

    def manifest(self):
        matrix = None
        if self.rotation is not None:
            matrix = np.eye(4); matrix[:3, :3] = self.rotation; matrix[:3, 3] = self.translation
        return dict(schemaVersion=1, frameId=self.frame_id, state=self.state, failureReason=self.reason,
                    cvCameraFromSourceObject=matrix.tolist() if matrix is not None else None,
                    units='metres', objectBasis='Original right-handed BOP source model axes', **self.statistics)


class RigidTracker:
    def __init__(self, reference_points, descriptors, model_positions, previous_gray=None,
                 previous_pixels=None, previous_points=None):
        self.reference = np.asarray(reference_points, dtype=np.float64).copy()
        self.descriptors = np.asarray(descriptors, dtype=np.uint8).copy()
        self.model = np.asarray(model_positions, dtype=np.float64).copy()
        if (self.reference.ndim != 2 or self.reference.shape[1:] != (3,) or
                self.descriptors.ndim != 2 or self.descriptors.shape[1:] != (32,) or
                len(self.reference) != len(self.descriptors) or len(self.reference) < 2):
            raise ValueError('ORB reference descriptors and 3D points must agree.')
        if (self.model.ndim != 2 or self.model.shape[1:] != (3,) or len(self.model) < 4 or
                not np.isfinite(self.reference).all() or not np.isfinite(self.model).all()):
            raise ValueError('Reference points and model positions must be finite Nx3 arrays.')
        # Onboarding is the fixed identity source for this tracker. Keep a private,
        # read-only snapshot so caller mutation cannot silently change recovery.
        self.reference.setflags(write=False)
        self.descriptors.setflags(write=False)
        self.model.setflags(write=False)
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.previous_gray = previous_gray
        self.previous_pixels = previous_pixels
        self.previous_points = previous_points
        self.lost = previous_pixels is None
        self.iteration = 0

    def match(self, gray):
        try:
            features, descriptors = self.orb.detectAndCompute(gray, None)
        except cv2.error:
            return np.empty((0, 2)), np.empty((0, 3))
        if descriptors is None or len(descriptors) < 2 or len(self.descriptors) < 2:
            return np.empty((0, 2)), np.empty((0, 3))
        try:
            forward = self.matcher.knnMatch(self.descriptors, descriptors, k=2)
            reverse = self.matcher.match(descriptors, self.descriptors)
        except cv2.error:
            return np.empty((0, 2)), np.empty((0, 3))
        mutual = {match.queryIdx: match.trainIdx for match in reverse}
        pairs = [a for group in forward if len(group) == 2 for a, b in [group]
                 if a.distance < .75*b.distance and mutual.get(a.trainIdx) == a.queryIdx]
        return np.array([features[p.trainIdx].pt for p in pairs]).reshape(-1, 2), self.reference[[p.queryIdx for p in pairs]]

    def flow(self, gray, pixel_to_native=None):
        if self.previous_gray is None or self.previous_pixels is None or not len(self.previous_pixels):
            return np.empty((0, 2)), np.empty((0, 3))
        if self.previous_points is None or len(self.previous_points) != len(self.previous_pixels):
            return np.empty((0, 2)), np.empty((0, 3))
        original = np.asarray(self.previous_pixels, dtype=np.float32).reshape(-1, 1, 2)
        expected = len(original)
        try:
            current, status, _ = cv2.calcOpticalFlowPyrLK(self.previous_gray, gray, original, None, winSize=(21, 21), maxLevel=3)
            if current is None or status is None: return np.empty((0, 2)), np.empty((0, 3))
            # Do not pass malformed forward output into the reverse LK call. Apart
            # from being unsafe for OpenCV, this makes the failure path deterministic
            # for truncated status/pixel arrays.
            current = np.asarray(current)
            status = np.asarray(status).reshape(-1)
            if current.size != 2*expected or status.size != expected:
                return np.empty((0, 2)), np.empty((0, 3))
            current = current.reshape(-1, 1, 2)
            backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous_gray, current, None, winSize=(21, 21), maxLevel=3)
            if backward is None or reverse_status is None:
                return np.empty((0, 2)), np.empty((0, 3))
            backward = np.asarray(backward)
            reverse_status = np.asarray(reverse_status).reshape(-1)
            if backward.size != 2*expected or reverse_status.size != expected:
                return np.empty((0, 2)), np.empty((0, 3))
        except cv2.error:
            return np.empty((0, 2)), np.empty((0, 3))
        except (TypeError, ValueError, IndexError):
            return np.empty((0, 2)), np.empty((0, 3))
        round_trip = backward.reshape(-1, 2)-original.reshape(-1, 2)
        if pixel_to_native is not None:
            transform = np.asarray(pixel_to_native, dtype=np.float64)
            if transform.shape != (3, 3) or not np.isfinite(transform).all():
                return np.empty((0, 2)), np.empty((0, 3))
            round_trip = round_trip@transform[:2, :2].T
        good = (status == 1)&(reverse_status == 1)&(np.linalg.norm(round_trip, axis=1) <= 1)
        points = current.reshape(-1, 2)
        good &= (points[:, 0] >= 0)&(points[:, 1] >= 0)&(points[:, 0] < gray.shape[1])&(points[:, 1] < gray.shape[0])
        return points[good], np.asarray(self.previous_points)[good]

    def solve(self, pixels, points, k, frame_id, *, proposal_count=None):
        stats = dict(correspondences=0, proposedAnchors=0, inliers=0, inlierRatio=0.,
                     medianReprojectionErrorPixels=None, maximumReprojectionErrorPixels=None,
                     p95ReprojectionErrorPixels=None, acceptedNonplanarityRatio=None,
                     featureCoverage=0., trackingConfidence=None,
                     duplicatePairsCollapsed=0, excludedAmbiguousAnchors=0,
                     excludedCollidingAnchors=0)

        def failure(reason, state='lost'):
            return TrackingResult(frame_id, state, reason, None, None, dict(stats))

        try:
            pixels = np.asarray(pixels, dtype=np.float64)
            points = np.asarray(points, dtype=np.float64)
            k = np.asarray(k, dtype=np.float64)
        except (TypeError, ValueError, OverflowError):
            return failure('invalid_correspondences_or_calibration')
        if (pixels.ndim != 2 or pixels.shape[1:] != (2,) or
                points.shape != (len(pixels), 3) or k.shape != (3, 3) or
                not np.isfinite(pixels).all() or not np.isfinite(points).all() or
                not np.isfinite(k).all()):
            return failure('invalid_correspondences_or_calibration')
        if (k[0, 0] <= 0 or k[1, 1] <= 0 or
                not np.allclose(k[2], [0, 0, 1], atol=1e-9, rtol=0)):
            return failure('invalid_correspondences_or_calibration')

        try:
            pixels, points, distinct_anchor_count, duplicate_count, ambiguous_count, collision_count = (
                _canonicalize_correspondences(pixels, points))
        except ValueError:
            return failure('invalid_correspondences_or_calibration')
        if proposal_count is None:
            denominator = distinct_anchor_count
        elif (not isinstance(proposal_count, (int, np.integer)) or
              isinstance(proposal_count, (bool, np.bool_)) or
              int(proposal_count) < distinct_anchor_count):
            return failure('invalid_proposal_count')
        else:
            denominator = int(proposal_count)
        stats.update(correspondences=denominator, proposedAnchors=denominator,
                     duplicatePairsCollapsed=duplicate_count,
                     excludedAmbiguousAnchors=ambiguous_count,
                     excludedCollidingAnchors=collision_count)

        def update_support(support_indices, support_points):
            count = len(support_indices)
            stats['inliers'] = count
            stats['inlierRatio'] = float(count/denominator) if denominator else 0.
            stats['acceptedNonplanarityRatio'] = _support_nonplanarity(support_points)[1]

        if len(points) < 12:
            return failure('insufficient_correspondences')
        if denominator < 12:
            return failure('insufficient_correspondences')
        proposal_nonplanar, _ = _support_nonplanarity(points)
        if not proposal_nonplanar:
            return failure('planar_or_underconstrained_correspondences', 'limited')

        cv2.setRNGSeed(17)
        try:
            ransac = cv2.solvePnPRansac(
                points, pixels, k, None, iterationsCount=150, reprojectionError=3,
                confidence=.99, flags=cv2.SOLVEPNP_EPNP)
            if not isinstance(ransac, tuple) or len(ransac) != 4:
                return failure('invalid_pnp_result')
            success, rvec, translation, raw_inliers = ransac
            if not success or raw_inliers is None:
                return failure('pnp_failed')
            raw_inliers = np.asarray(raw_inliers)
            if raw_inliers.ndim == 2 and raw_inliers.shape[1:] == (1,):
                raw_inliers = raw_inliers[:, 0]
            if (raw_inliers.ndim != 1 or not np.issubdtype(raw_inliers.dtype, np.integer) or
                    len(raw_inliers) == 0 or (raw_inliers < 0).any() or
                    (raw_inliers >= len(points)).any() or
                    len(np.unique(raw_inliers)) != len(raw_inliers)):
                return failure('invalid_pnp_inliers')
            support = raw_inliers.astype(np.int64, copy=True)
            support_points = points[support]
            update_support(support, support_points)
            if len(support) < 12 or stats['inlierRatio'] < .5:
                return failure('insufficient_inliers')
            support_nonplanar, support_rank_ratio = _support_nonplanarity(support_points)
            stats['acceptedNonplanarityRatio'] = support_rank_ratio
            if not support_nonplanar:
                return failure('planar_or_underconstrained_inliers', 'limited')

            def valid_vectors(candidate_rvec, candidate_translation):
                candidate_rvec = np.asarray(candidate_rvec, dtype=np.float64)
                candidate_translation = np.asarray(candidate_translation, dtype=np.float64)
                return (candidate_rvec.size == 3 and candidate_translation.size == 3 and
                        np.isfinite(candidate_rvec).all() and
                        np.isfinite(candidate_translation).all())

            if not valid_vectors(rvec, translation):
                return failure('invalid_refinement_pose')
            rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
            translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
            stable = False
            final_error = None
            for refinement_call in range(3):
                refined = cv2.solvePnPRefineLM(
                    points[support], pixels[support], k, None, rvec, translation)
                if not isinstance(refined, tuple) or len(refined) != 2:
                    return failure('invalid_refinement_pose')
                rvec, translation = refined
                if not valid_vectors(rvec, translation):
                    return failure('invalid_refinement_pose')
                rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
                translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
                rotation = cv2.Rodrigues(rvec)[0]
                translation_vector = translation.reshape(3)
                if (not np.isfinite(rotation).all() or
                        not np.allclose(rotation@rotation.T, np.eye(3), atol=1e-5, rtol=0) or
                        abs(float(np.linalg.det(rotation))-1.) > 1e-5):
                    return failure('invalid_or_improper_rotation')
                projected = cv2.projectPoints(points[support], rvec, translation, k, None)[0].reshape(-1, 2)
                errors = np.linalg.norm(projected-pixels[support], axis=1)
                keep = np.isfinite(errors) & (errors <= 3.)
                if bool(keep.all()):
                    stable = True
                    final_error = errors
                    break

                support = support[keep]
                support_points = points[support]
                update_support(support, support_points)
                support_nonplanar, support_rank_ratio = _support_nonplanarity(support_points)
                stats['acceptedNonplanarityRatio'] = support_rank_ratio
                if refinement_call == 2:
                    return failure('refinement_support_unstable')
                if len(support) < 12 or stats['inlierRatio'] < .5:
                    return failure('insufficient_inliers')
                if not support_nonplanar:
                    return failure('planar_or_underconstrained_inliers', 'limited')

            if not stable or final_error is None:
                return failure('refinement_support_unstable')
            update_support(support, points[support])
            support_nonplanar, support_rank_ratio = _support_nonplanarity(points[support])
            stats['acceptedNonplanarityRatio'] = support_rank_ratio
            if (len(support) < 12 or stats['inlierRatio'] < .5 or not support_nonplanar):
                return failure('insufficient_final_support', 'limited')
            if not np.isfinite(final_error).all() or (final_error > 3.).any():
                return failure('refinement_support_unstable')
            stats['medianReprojectionErrorPixels'] = float(np.median(final_error))
            stats['maximumReprojectionErrorPixels'] = float(np.max(final_error))
            stats['p95ReprojectionErrorPixels'] = float(np.percentile(final_error, 95))
            model_camera = self.model@rotation.T+translation_vector
            if not np.isfinite(model_camera).all() or (model_camera[:, 2] <= .03).any():
                return failure('invalid_or_behind_camera_pose')
            model_projected = cv2.projectPoints(self.model, rvec, translation, k, None)[0].reshape(-1, 2)
            if not np.isfinite(model_projected).all():
                return failure('invalid_or_behind_camera_pose')
            area = np.prod(np.maximum(np.ptp(model_projected, axis=0), 1))
            stats['featureCoverage'] = float(np.prod(np.ptp(pixels[support], axis=0))/area)
            if not np.isfinite(stats['featureCoverage']) or stats['featureCoverage'] < .2:
                return failure('insufficient_feature_coverage', 'limited')
            stats['trackingConfidence'] = 'thresholds_met_not_a_probability'
            return (TrackingResult(frame_id, 'tracking', None, rotation, translation_vector, dict(stats)),
                    pixels[support].copy(), points[support].copy())
        except cv2.error:
            return failure('opencv_pose_error')
        except (TypeError, ValueError, IndexError, np.linalg.LinAlgError, OverflowError):
            return failure('invalid_pose_solver_output')

    def reset_temporal(self):
        """Clear all frame-local tracking state while retaining frozen onboarding data."""
        self.previous_gray = None
        self.previous_pixels = None
        self.previous_points = None
        self.lost = True

    def update(self, rgb, k, frame_id):
        gray = rgb if rgb.ndim == 2 else cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        flow_pixels, flow_points = self.flow(gray) if not self.lost else (np.empty((0, 2)), np.empty((0, 3)))
        fresh_pixels, fresh_points = np.empty((0, 2)), np.empty((0, 3))
        if self.iteration%5 == 0 or self.lost or len(flow_points) < 12:
            fresh_pixels, fresh_points = self.match(gray)
        pixels, points, proposals, fusion = reconcile_flow_fresh(
            flow_pixels, flow_points, fresh_pixels, fresh_points,
            k, frame_id, self.solve)
        if fusion['fusionFailure'] is not None:
            result = TrackingResult(frame_id, 'lost', fusion['fusionFailure'], None, None, dict(fusion))
        else:
            result = self.solve(pixels, points, k, frame_id, proposal_count=proposals)
            result_object = result[0] if isinstance(result, tuple) else result
            result_object.statistics.update(fusion)
        if isinstance(result, tuple):
            result, self.previous_pixels, self.previous_points = result
            self.lost = False
            self.previous_gray = gray
        else:
            # A failed/limited solve invalidates temporal correspondences. The next
            # update must search the immutable onboarding references again.
            self.reset_temporal()
        self.iteration += 1
        return result
