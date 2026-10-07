"""UV parameterization and independent area-weighted distortion/seam measurements."""
from pathlib import Path
import sys
import numpy as np
from .model import Mesh


def weighted_percentile(values, weights, percentile):
    if len(values) == 0 or weights.sum() <= 0:
        return None
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])/weights.sum()
    return float(np.interp(percentile, cumulative, values[order]))


def mapping_quality(positions, triangles, uv_metres):
    p = np.asarray(positions)[triangles]
    uv = np.asarray(uv_metres)[triangles]
    edges = np.stack([p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]], axis=2)
    uv_edges = np.stack([uv[:, 1]-uv[:, 0], uv[:, 2]-uv[:, 0]], axis=2)
    area = np.linalg.norm(np.cross(edges[:, :, 0], edges[:, :, 1]), axis=1)/2
    determinant = np.linalg.det(uv_edges)
    uv_scale = np.max(np.sum(uv_edges*uv_edges, axis=1), axis=1)
    valid_geometry = area > 1e-14
    valid_uv = np.abs(determinant) > np.maximum(uv_scale, 1e-30)*1e-12
    valid = valid_geometry & valid_uv
    scale_error = np.full(len(triangles), np.inf)
    anisotropy = np.full(len(triangles), np.inf)
    if valid.any():
        jacobian = edges[valid]@np.linalg.inv(uv_edges[valid])
        singular = np.linalg.svd(jacobian, compute_uv=False)
        scale_error[valid] = np.max(np.abs(singular-1), axis=1)
        anisotropy[valid] = singular[:, 0]/np.maximum(singular[:, 1], 1e-30)
    trustworthy = valid & (scale_error <= .02) & (anisotropy <= 1.05)
    total = area.sum()
    if total <= 0:
        raise ValueError('Mesh has no measurable area.')
    report = dict(triangles=len(triangles), invalidTriangles=int((~valid).sum()), surfaceAreaMetres2=float(total),
                  degenerateGeometryTriangles=int((~valid_geometry).sum()),
                  collapsedUVTriangles=int((valid_geometry & ~valid_uv).sum()),
                  invalidSurfaceAreaFraction=float(area[~valid].sum()/total),
                  quantilePopulation='Valid triangles, weighted by physical surface area; invalid area reported separately',
                  areaWeightedMedianScaleError=weighted_percentile(scale_error[valid], area[valid], .5),
                  areaWeightedP95ScaleError=weighted_percentile(scale_error[valid], area[valid], .95),
                  areaWeightedP95Anisotropy=weighted_percentile(anisotropy[valid], area[valid], .95),
                  areaWithinProvisionalMappingLimits=float(area[trustworthy].sum()/total),
                  provisionalLimits={'maximumScaleError': .02, 'maximumAnisotropy': 1.05},
                  qualityGatePassed=False, physicalValidationPassed=False)
    return report, scale_error, trustworthy


def unwrap(scan, name, details=False):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'.cache'/'vision'))
    import xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(scan.positions.astype(np.float32), scan.triangles.astype(np.uint32), scan.normals.astype(np.float32))
    pack = xatlas.PackOptions()
    pack.texels_per_unit = 8192
    pack.padding = 2
    pack.rotate_charts = False
    pack.rotate_charts_to_axis = False
    atlas.generate(pack_options=pack)
    mapping, faces, uv = atlas[0]
    source_faces = mapping[faces]
    if not np.array_equal(source_faces, scan.triangles):
        raise ValueError('Parameterization changed source face correspondence.')
    density = float(atlas.texels_per_unit)
    if not np.isfinite(density) or density <= 0:
        raise ValueError('Atlas did not provide metric texel density.')
    metric = uv*np.array([atlas.width, atlas.height])/density
    _, chart = atlas.get_mesh_vertex_assignment(0)
    mesh = Mesh(name, scan.positions[mapping], scan.normals[mapping], metric/.04, faces)
    quality, error, trusted = mapping_quality(mesh.positions, mesh.triangles, metric)
    quality.update(charts=int(atlas.chart_count), atlasSize=[int(atlas.width), int(atlas.height)],
                   texelsPerMetre=density, tileWidthMetres=.04, chartBoundaryContinuityGuaranteed=False)
    seams = seam_quality(scan.positions, source_faces, metric[faces], np.asarray(chart)[faces], .04)
    quality.update(seams)
    if details:
        return mesh, quality, error, trusted, dict(sourceVertices=mapping, sourceFaces=source_faces,
                                                  chartIds=np.asarray(chart), metricUV=metric)
    return mesh, quality, error, trusted


def seam_quality(positions, source_faces, corner_uv, corner_chart, tile_width, aspect=1, rotation_degrees=0):
    edges, coordinates, charts = [], [], []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        edge = source_faces[:, [i, j]]
        xy = corner_uv[:, [i, j]]
        swap = edge[:, 0] > edge[:, 1]
        edge[swap] = edge[swap][:, ::-1]
        xy[swap] = xy[swap][:, ::-1]
        edges.append(edge); coordinates.append(xy); charts.append(corner_chart[:, i])
    edge = np.concatenate(edges)
    coordinate = np.concatenate(coordinates)
    chart = np.concatenate(charts)
    unique, inverse, counts = np.unique(edge, axis=0, return_inverse=True, return_counts=True)
    order = np.argsort(inverse, kind='stable')
    offsets = np.concatenate([[0], np.cumsum(counts)])
    phase, lengths = [], []
    seam_edges = 0
    for index in np.flatnonzero(counts == 2):
        a, b = order[offsets[index]:offsets[index+1]]
        if chart[a] == chart[b]:
            continue
        seam_edges += 1
        fractions = np.linspace(0, 1, 7)
        difference = coordinate[a]-coordinate[b]
        angle = np.radians(rotation_degrees)
        rotation = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
        jump = (((1-fractions[:, None])*difference[0]+fractions[:, None]*difference[1])@rotation.T)/np.array([tile_width,tile_width*aspect])
        wrapped = jump-np.round(jump)
        mismatch = float(np.mean(np.linalg.norm(wrapped, axis=1)))
        phase.append(mismatch)
        lengths.append(float(np.linalg.norm(positions[unique[index, 1]]-positions[unique[index, 0]])))
    return dict(boundaryEdges=int((counts == 1).sum()), nonManifoldEdges=int((counts > 2).sum()),
                chartSeamEdges=seam_edges, chartSeamLengthMetres=sum(lengths),
                lengthWeightedSeamPhaseMismatch=weighted_percentile(np.asarray(phase), np.asarray(lengths), .5) if lengths else 0,
                seamMetric='Median by seam length of mean wrapped UV jump along each edge, in texture tiles; zero is ideal')
