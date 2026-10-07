"""Metric chart candidates and explicit seam/overlap promotion safeguards."""
import numpy as np
from .model import Mesh
from .scan_mapping import mapping_quality, seam_quality


def boundaries(faces, chart, metric):
    records = {}
    for face, vertices in enumerate(faces):
        for i, j in ((0, 1), (1, 2), (2, 0)):
            a, b = int(vertices[i]), int(vertices[j])
            xy = metric[face, [i, j]]
            if a > b:
                a, b, xy = b, a, xy[::-1]
            records.setdefault((a, b), []).append((face, int(chart[face, i]), xy.copy()))
    return [(edge, *pair) for edge, pair in records.items() if len(pair) == 2 and pair[0][1] != pair[1][1]]


def correct_scale(mesh, metric, chart):
    faces = mesh.triangles
    p, u = mesh.positions[faces], metric[faces]
    edges = np.stack([p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]], axis=2)
    ue = np.stack([u[:, 1]-u[:, 0], u[:, 2]-u[:, 0]], axis=2)
    area = np.linalg.norm(np.cross(edges[:, :, 0], edges[:, :, 1]), axis=1)/2
    det = np.linalg.det(ue)
    valid = (area > 1e-14)&(np.abs(det) > 1e-20)
    density = np.ones(len(faces))
    density[valid] = np.sqrt(np.prod(np.linalg.svd(edges[valid]@np.linalg.inv(ue[valid]), compute_uv=False), axis=1))
    output = metric.copy()
    for identifier in np.unique(chart):
        region = (chart[faces[:, 0]] == identifier)&valid
        samples = density[region]
        if not len(samples):
            continue
        weights = area[region]
        order = np.argsort(samples)
        factor = np.interp(.5, np.cumsum(weights[order])/weights.sum(), samples[order])
        mask = chart == identifier
        centre = metric[mask].mean(axis=0)
        output[mask] = centre+(metric[mask]-centre)*factor
    return output


def align_charts(mesh, source_faces, chart_ids, metric, visibility, repeating, tile=.04, aspect=1, rotation_degrees=0):
    """Weighted chart orientation synchronisation, then integer-aware translation LS."""
    ids = np.unique(chart_ids)
    chart = np.searchsorted(ids, chart_ids)
    count = len(ids)
    corner_chart = chart[mesh.triangles]
    pairs = boundaries(source_faces, corner_chart, metric[mesh.triangles])
    areas = np.linalg.norm(np.cross(mesh.positions[mesh.triangles[:, 1]]-mesh.positions[mesh.triangles[:, 0]],
                                   mesh.positions[mesh.triangles[:, 2]]-mesh.positions[mesh.triangles[:, 0]]), axis=1)/2
    root = int(np.argmax(np.bincount(corner_chart[:, 0], weights=areas, minlength=count)))
    graph = np.zeros((count, count), dtype=complex)
    equations, endpoints, weights = [], [], []
    for edge, left, right in pairs:
        face_a, ca, uv_a = left; face_b, cb, uv_b = right
        da, db = uv_a[1]-uv_a[0], uv_b[1]-uv_b[0]
        indices = [int(np.flatnonzero(source_faces[face_a] == vertex)[0]) for vertex in edge]
        length = np.linalg.norm(mesh.positions[mesh.triangles[face_a, indices[1]]]-mesh.positions[mesh.triangles[face_a, indices[0]]])
        weight = max(length, 1e-8)*(1+(visibility[face_a]+visibility[face_b])/2)
        angle = np.angle(complex(*da))-np.angle(complex(*db))
        graph[ca, cb] += weight*np.exp(-1j*angle)
        graph[cb, ca] += weight*np.exp(1j*angle)
        row = np.zeros(count); row[ca] = 1; row[cb] = -1
        equations.append(row); endpoints.append((ca, cb, uv_a, uv_b)); weights.append(weight)
    if not equations:
        return metric.copy()
    degree = np.maximum(np.abs(graph).sum(axis=1), 1e-15)
    normalized = graph/np.sqrt(degree[:, None]*degree[None, :])
    _, vectors = np.linalg.eigh(normalized)
    angles = np.angle(vectors[:, -1]); angles -= angles[root]
    rotated = np.empty_like(metric)
    matrices = []
    for identifier, angle in enumerate(angles):
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        matrices.append(rotation)
        mask = chart == identifier
        rotated[mask] = metric[mask]@rotation.T
    a = np.array(equations)
    weights = np.asarray(weights)
    differences = np.array([(uv_b@matrices[cb].T-uv_a@matrices[ca].T).mean(axis=0)
                            for ca, cb, uv_a, uv_b in endpoints])
    free = np.arange(count) != root
    weighted = a[:, free]*np.sqrt(weights[:, None])
    normal = weighted.T@weighted+np.eye(free.sum())*1e-12
    translations = np.zeros((count, 2))
    angle = np.radians(rotation_degrees)
    pattern_rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    period = np.array([tile,tile*aspect])
    best, best_error = rotated.copy(), np.inf
    for _ in range(10):
        target = differences.copy()
        if repeating:
            target -= (period*np.round(((target-a@translations)@pattern_rotation.T)/period))@pattern_rotation
        translations[free] = np.linalg.solve(normal, weighted.T@(target*np.sqrt(weights[:, None])))
        residual = a@translations-differences
        if repeating:
            phase = (residual@pattern_rotation.T)/period
            residual = (period*(phase-np.round(phase)))@pattern_rotation
        error = float(np.sum(weights*np.sum(residual**2, axis=1)))
        if error < best_error:
            best, best_error = rotated+translations[chart], error
    return best


def design_uv(metric, repeating, tile=.04, aspect=1, rotation_degrees=0):
    angle = np.radians(rotation_degrees)
    rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    metric = metric@rotation.T
    if repeating:
        return metric/np.array([tile, tile*aspect])
    low, high = metric.min(axis=0), metric.max(axis=0)
    extent = high-low
    width = max(extent[0], extent[1]/aspect, 1e-8)
    return (metric-(low+high)/2)/np.array([width, width*aspect])+.5


def diagnostic_color(uv, repeating):
    """Asymmetric coloured directional pattern; no checker symmetry hiding seams."""
    xy = np.mod(uv, 1) if repeating else np.clip(uv, 0, 1)
    color = np.column_stack([xy[:, 0], xy[:, 1], .2+.6*(xy[:, 0] > xy[:, 1])])
    # Narrow stripes distinguish rotation and reflected mappings.
    stripe = (np.floor(xy[:, 0]*11)%2)*.15
    return np.clip(color-stripe[:, None], 0, 1)


def seam_design_quality(mesh, source_faces, chart, metric, repeating, image=None, tile=.04, rotation_degrees=0):
    pairs = boundaries(source_faces, chart[mesh.triangles], metric[mesh.triangles])
    aspect = image.shape[0]/image.shape[1] if image is not None else 1
    uv = design_uv(metric, repeating, tile=tile, aspect=aspect, rotation_degrees=rotation_degrees)
    mismatch, orientation, weights, faces = [], [], [], set()
    for edge, a, b in pairs:
        fa, _, _ = a; fb, _, _ = b
        fractions = np.linspace(.001, .999, 17)
        samples = []
        directions = []
        for face in (fa, fb):
            indices = [int(np.flatnonzero(source_faces[face] == v)[0]) for v in edge]
            endpoints = uv[mesh.triangles[face, indices]]
            samples.append((1-fractions[:, None])*endpoints[0]+fractions[:, None]*endpoints[1])
            directions.append(endpoints[1]-endpoints[0])
        mismatch.append(float(np.linalg.norm(sample_design(samples[0], repeating, image)-sample_design(samples[1], repeating, image), axis=1).mean()/np.sqrt(3)))
        denominator = max(np.linalg.norm(directions[0])*np.linalg.norm(directions[1]), 1e-20)
        orientation.append(float(np.degrees(np.arccos(np.clip(directions[0].dot(directions[1])/denominator, -1, 1)))))
        p = mesh.positions[mesh.triangles[fa]]
        indices = [int(np.flatnonzero(source_faces[fa] == v)[0]) for v in edge]
        weights.append(float(np.linalg.norm(p[indices[1]]-p[indices[0]])))
        faces.update((fa, fb))
    from .scan_mapping import weighted_percentile
    w = np.array(weights)
    return dict(meanSeamColorError=float(np.average(mismatch, weights=w)) if w.sum() else 0,
                p95SeamColorError=weighted_percentile(np.array(mismatch), w, .95) if w.sum() else 0,
                meanBoundaryDirectionDifferenceDegrees=float(np.average(orientation, weights=w)) if w.sum() else 0,
                seamFaces=sorted(faces))


def polygon_intersection(a, b):
    polygon = list(a)
    signed = np.cross(b[1]-b[0], b[2]-b[0])
    direction = 1 if signed >= 0 else -1
    for index in range(3):
        origin, end = b[index], b[(index+1)%3]
        edge = end-origin
        output = []
        if not polygon:
            return 0
        for i, point in enumerate(polygon):
            previous = polygon[i-1]
            d1 = direction*np.cross(edge, point-origin)
            d0 = direction*np.cross(edge, previous-origin)
            if (d1 > 1e-12) != (d0 > 1e-12) and abs(d1-d0) > 1e-20:
                output.append(previous+(point-previous)*d0/(d0-d1))
            if d1 > 1e-12:
                output.append(point)
        polygon = output
    if len(polygon) < 3:
        return 0
    p = np.array(polygon)
    return abs(np.sum(p[:, 0]*np.roll(p[:, 1], -1)-p[:, 1]*np.roll(p[:, 0], -1)))/2


def has_overlaps(metric, triangles):
    """Spatial hash + separating-axis tests; touching borders are permitted."""
    extent = np.maximum(np.ptp(metric, axis=0), 1e-10)
    normalized = (metric-metric.min(axis=0))/extent
    cells = {}
    polys = normalized[triangles]
    low, high = polys.min(axis=1), polys.max(axis=1)
    for index, polygon in enumerate(polys):
        candidates = set()
        lo, hi = np.floor(low[index]*64).astype(int), np.floor(high[index]*64).astype(int)
        for x in range(lo[0], hi[0]+1):
            for y in range(lo[1], hi[1]+1):
                candidates.update(cells.get((x, y), ()))
        if candidates:
            other = np.array(list(candidates), dtype=int)
            intersects = np.all(high[index] > low[other]+1e-12, axis=1)&np.all(high[other] > low[index]+1e-12, axis=1)
            other = other[intersects]
            if len(other):
                # Separating-axis theorem, batched across nearby convex triangles.
                a_edges = np.roll(polygon, -1, axis=0)-polygon
                b_edges = np.roll(polys[other], -1, axis=1)-polys[other]
                axes = np.concatenate([np.broadcast_to(a_edges, b_edges.shape), b_edges], axis=1)[..., ::-1].copy()
                axes[..., 0] *= -1
                pa = np.einsum('vd,nkd->nkv', polygon, axes)
                pb = np.einsum('nvd,nkd->nkv', polys[other], axes)
                eps = np.linalg.norm(axes, axis=2)*1e-10
                separated = ((pa.max(axis=2) <= pb.min(axis=2)+eps)|(pb.max(axis=2) <= pa.min(axis=2)+eps)).any(axis=1)
                if (~separated).any():
                    return True
        for x in range(lo[0], hi[0]+1):
            for y in range(lo[1], hi[1]+1):
                cells.setdefault((x, y), []).append(index)
    return False


def sample_design(uv, repeating, image=None):
    if image is None: return diagnostic_color(uv, repeating)
    uv = np.mod(uv, 1) if repeating else np.clip(uv, 0, 1)
    height, width = image.shape[:2]
    x = uv[:, 0]*(width-1); y = (1-uv[:, 1])*(height-1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0+1, width-1), np.minimum(y0+1, height-1)
    dx, dy = (x-x0)[:, None], (y-y0)[:, None]
    return ((1-dx)*(1-dy)*image[y0,x0]+dx*(1-dy)*image[y0,x1]+(1-dx)*dy*image[y1,x0]+dx*dy*image[y1,x1])/255


def candidates(mesh, details, visibility, repeating, image=None, tile=.04, rotation_degrees=0):
    metric = details['metricUV']
    chart = details['chartIds']
    source = details['sourceFaces']
    corrected = correct_scale(mesh, metric, chart)
    aspect = image.shape[0]/image.shape[1] if image is not None else 1
    alternatives = [('baseline', metric), ('scale_corrected', corrected),
                    ('seam_aligned', align_charts(mesh, source, chart, corrected, visibility, repeating,tile,aspect,rotation_degrees))]
    reports = []
    base_sign = np.sign(np.linalg.det(np.stack([metric[mesh.triangles[:, 1]]-metric[mesh.triangles[:, 0]],
                                              metric[mesh.triangles[:, 2]]-metric[mesh.triangles[:, 0]]], axis=2)))
    for name, uv in alternatives:
        quality, errors, trusted = mapping_quality(mesh.positions, mesh.triangles, uv)
        quality.update(seam_quality(mesh.positions, source, uv[mesh.triangles], chart[mesh.triangles], tile,aspect,rotation_degrees))
        quality.update(seam_design_quality(mesh, source, chart, uv, repeating, image,tile,rotation_degrees))
        signs = np.sign(np.linalg.det(np.stack([uv[mesh.triangles[:, 1]]-uv[mesh.triangles[:, 0]],
                                               uv[mesh.triangles[:, 2]]-uv[mesh.triangles[:, 0]]], axis=2)))
        quality['invertedFaces'] = int(((base_sign*signs) < 0).sum())
        quality['artworkOverlap'] = bool(has_overlaps(uv, mesh.triangles)) if not repeating else None
        quality['candidate'] = name
        quality['rejectionReasons'] = []
        if reports:
            base = reports[0][1]
            reasons = quality['rejectionReasons']
            if quality['invalidSurfaceAreaFraction'] > base['invalidSurfaceAreaFraction']+1e-10: reasons.append('increased_invalid_area')
            if quality['invertedFaces']: reasons.append('inverted_uv')
            if quality['artworkOverlap']: reasons.append('overlapping_artwork')
            if quality['areaWeightedP95ScaleError'] > base['areaWeightedP95ScaleError']+.005: reasons.append('scale_regression')
            if quality['areaWithinProvisionalMappingLimits'] < base['areaWithinProvisionalMappingLimits']-.01: reasons.append('coverage_regression')
        else:
            if quality['artworkOverlap']: quality['rejectionReasons'].append('overlapping_artwork')
        reports.append((uv, quality, errors, trusted))
    eligible = [i for i, (_, q, _, _) in enumerate(reports) if not q['rejectionReasons']]
    chosen = min(eligible, key=lambda i: reports[i][1]['meanSeamColorError']) if eligible else 0
    baseline = reports[0][1]['meanSeamColorError']
    improvement = 1-reports[chosen][1]['meanSeamColorError']/baseline if baseline > 1e-12 else 0
    for i, (_, q, _, _) in enumerate(reports):
        q['selected'] = i == chosen
        q['provisionalImprovementGatePassed'] = i == chosen and not q['rejectionReasons'] and improvement >= .3
        q['meanSeamColorImprovement'] = improvement if i == chosen else None
    return reports, chosen
