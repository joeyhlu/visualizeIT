"""Pure, injected semantic gates for saved capture FoundPose banks.

The module deliberately imports no model, tensor, renderer, or optional numeric
runtime. Callers supply NumPy plus the pinned nearest-centroid, TF-IDF replay,
and model-only retrieval adapters after loading a hash-bound private artifact.
"""
import math

_TEMPLATE_COUNT = 798
_WORD_COUNT = 2048
_FEATURE_LIMIT = 319200
_FEATURE_DIM = 256
_RAW_FEATURE_DIM = 384
_PCA_COMPONENTS = 256
_CHUNK_ROWS = 8192
_VERTEX_TOLERANCE_M = 0.002
_VERTEX_RELATIVE_TOLERANCE = 0.0001
_IDF_RTOL = 2e-5
_IDF_ATOL = 2e-6
_DESC_RTOL = 2e-5
_DESC_ATOL = 2e-6
_PCA_ORTHOGONAL_TOLERANCE = 0.002
_PCA_VARIANCE_RTOL = 0.002
_PCA_VARIANCE_ATOL = 1e-6
_PCA_RATIO_RTOL = 0.002
_PCA_RATIO_ATOL = 1e-6

_REQUIRED_SOURCE_PIN_PATHS = (
    "bench/quality_assets.py",
    "bench/quality_runner.py",
    "bench/quality_capture_bank_semantics.py",
    "bench/quality_foundpose.py",
    "bench/quality_neighbors.py",
    ".cache/model-quality/sources/gotrack/utils/cluster_util.py",
    ".cache/model-quality/sources/gotrack/utils/pca_util.py",
    ".cache/model-quality/sources/gotrack/utils/template_util.py",
)
_REQUIRED_CHECKS = (
    "state_keys",
    "asset_binding",
    "template_count",
    "tensor_layouts",
    "template_coverage",
    "finite_features_centroids_vertices",
    "pca_contract",
    "pca_orthonormality",
    "pca_covariance",
    "vertex_bounds",
    "idf_top1_replay",
    "descriptor_top3_replay",
    "descriptor_rows_finite_nonzero",
    "retrieval_probe_finite_deterministic",
)
_SEMANTIC_RECEIPT_KEYS = frozenset({
    "status", "validator_source_sha256", "artifact_sha256", "artifact_byte_count",
    "source_pins", "checks", "summary",
})
_SUMMARY_KEYS = frozenset({
    "artifact_sha256", "artifact_byte_count", "feature_rows", "template_count",
    "visual_word_count", "pca_sample_count", "pca_max_orthogonality_error",
    "pca_variance_max_abs_error", "vertex_min_m", "vertex_max_m",
    "vertex_bound_tolerance_m", "idf_unused_word_count",
    "descriptor_max_abs_error", "retrieval_probe_rows", "retrieval_top_n",
})


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _exact_dict(value, keys, name):
    _require(type(value) is dict and set(value) == set(keys),
             name + " has an unexpected field set")
    return value


def _sha256(value, name):
    _require(type(value) is str and len(value) == 64 and value == value.upper() and
             all(character in "0123456789ABCDEF" for character in value),
             name + " must be an uppercase SHA-256 digest")
    return value


def _positive_int(value, name, maximum=None):
    _require(type(value) is int and value > 0 and
             (maximum is None or value <= maximum),
             name + " must be a bounded positive integer")
    return value


def _finite_number(value, name, *, nonnegative=False):
    _require(type(value) in (int, float) and math.isfinite(float(value)),
             name + " must be a finite number")
    result = float(value)
    if nonnegative:
        _require(result >= 0.0, name + " must be nonnegative")
    return result


def _finite_json_tree(value, name, depth=0):
    _require(depth <= 12, name + " is nested too deeply")
    if value is None or type(value) in (bool, str):
        if type(value) is str:
            _require(len(value) <= 4096, name + " contains an oversized string")
        return
    if type(value) is int:
        _require(-(2 ** 53) <= value <= 2 ** 53, name + " integer exceeds the exact JSON range")
        return
    if type(value) is float:
        _require(math.isfinite(value), name + " contains a non-finite number")
        return
    if type(value) is list:
        _require(len(value) <= 4096, name + " list exceeds its bound")
        for index, child in enumerate(value):
            _finite_json_tree(child, name + "[" + str(index) + "]", depth + 1)
        return
    if type(value) is dict:
        _require(len(value) <= 4096 and all(type(key) is str for key in value),
                 name + " object has invalid keys or exceeds its bound")
        for key, child in value.items():
            _finite_json_tree(child, name + "." + key, depth + 1)
        return
    raise ValueError(name + " contains a non-JSON value")


def _as_array(value, numpy, name):
    try:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        result = numpy.asarray(value)
    except Exception as exc:
        raise ValueError(name + " cannot be viewed as a bounded numeric array") from exc
    return result


def _shape(value, expected, name):
    actual = tuple(int(dimension) for dimension in getattr(value, "shape", ()))
    _require(actual == expected, name + " has an unexpected shape")
    return actual


def _dtype(value, expected, name):
    actual = str(getattr(value, "dtype", "")).lower()
    _require(actual.endswith(expected), name + " has an unexpected dtype")


def _all_finite(array, numpy, name, *, chunk_rows=_CHUNK_ROWS):
    rows = int(array.shape[0]) if getattr(array, "ndim", 0) else 1
    for start in range(0, rows, chunk_rows):
        piece = array[start:start + chunk_rows] if getattr(array, "ndim", 0) else array
        _require(bool(numpy.isfinite(piece).all()), name + " contains NaN or infinity")


def _source_pins(value):
    _require(hasattr(value, "items"), "source_pins must be a mapping")
    value = dict(value)
    _require(type(value) is dict and set(value) == set(_REQUIRED_SOURCE_PIN_PATHS),
             "source_pins must contain exactly the required producer and upstream pins")
    result = {}
    for path in _REQUIRED_SOURCE_PIN_PATHS:
        result[path] = _sha256(value[path], "source_pins[" + path + "]")
    return result


def _asset_bounds(asset_receipt, numpy):
    document = getattr(asset_receipt, "document", None)
    _require(type(document) is dict or hasattr(document, "get"),
             "asset_receipt must expose its verified immutable document")
    bounds = document.get("post_node_bounds_m")
    _require(hasattr(bounds, "keys") and hasattr(bounds, "get"),
             "post_node_bounds_m must be a mapping")
    bounds = dict(bounds)
    _exact_dict(bounds, {"min", "max", "extents"}, "post_node_bounds_m")
    vectors = {}
    for name in ("min", "max", "extents"):
        raw = bounds[name]
        _require(type(raw) in (list, tuple) and len(raw) == 3,
                 "post_node_bounds_m." + name + " must contain three coordinates")
        vectors[name] = numpy.asarray([_finite_number(item, "post_node_bounds_m." + name)
                                       for item in raw], dtype=numpy.float64)
    _require(bool((vectors["extents"] > 0.0).all()),
             "post_node_bounds_m.extents must be positive")
    _require(bool((numpy.abs((vectors["max"] - vectors["min"]) - vectors["extents"]) <=
                   (1e-5 + 1e-4 * numpy.abs(vectors["extents"]))).all()),
             "post_node_bounds_m max/min/extents disagree")
    return vectors


def _pca_arrays(projector, numpy, feature_rows):
    fitted = getattr(projector, "pca", None)
    _require(fitted is not None and getattr(projector, "whiten", None) is False and
             getattr(projector, "n_components", None) == _PCA_COMPONENTS and
             getattr(fitted, "whiten", None) is False and
             getattr(fitted, "n_components_", None) == _PCA_COMPONENTS and
             getattr(fitted, "n_features_in_", None) == _RAW_FEATURE_DIM,
             "projector is not the fitted non-whitened PCA256-from-384 contract")
    expected_samples = min(feature_rows, 100000)
    _require(type(getattr(fitted, "n_samples_", None)) is int and
             fitted.n_samples_ == expected_samples and expected_samples > 256,
             "fitted PCA sample count does not match the capped feature population")
    expected_shapes = {
        "components_": (_PCA_COMPONENTS, _RAW_FEATURE_DIM),
        "mean_": (_RAW_FEATURE_DIM,),
        "explained_variance_": (_PCA_COMPONENTS,),
        "explained_variance_ratio_": (_PCA_COMPONENTS,),
        "singular_values_": (_PCA_COMPONENTS,),
    }
    arrays = {}
    for name, shape in expected_shapes.items():
        arrays[name] = _as_array(getattr(fitted, name, None), numpy, "PCA " + name)
        _shape(arrays[name], shape, "PCA " + name)
        _require(numpy.issubdtype(arrays[name].dtype, numpy.floating),
                 "PCA " + name + " must have a real floating dtype")
        _all_finite(arrays[name], numpy, "PCA " + name)
    noise = _as_array(getattr(fitted, "noise_variance_", None), numpy,
                      "PCA noise_variance_")
    _require(tuple(noise.shape) == () and numpy.issubdtype(noise.dtype, numpy.floating),
             "PCA noise_variance_ must be a real floating scalar")
    _all_finite(noise, numpy, "PCA noise_variance_")
    noise_value = float(noise.item())
    _require(noise_value >= 0.0, "PCA noise variance must be nonnegative")
    return arrays, noise_value, expected_samples


def _check_execution_receipt(document, expected_bindings):
    top_keys = {
        "schema_version", "kind", "status", "resource", "asset", "inputs",
        "worker", "artifact", "parent_execution", "join_cleanup",
        "semantic_validation",
    }
    _exact_dict(document, top_keys, "bank execution receipt")
    _require(type(document["schema_version"]) is int and document["schema_version"] == 1 and
             document["kind"] == "quality-capture-foundpose-bank-execution-v1" and
             document["status"] == "accepted",
             "bank execution receipt schema or parent status is unsupported")

    expected_keys = {"resource", "asset", "inputs", "artifact"}
    _exact_dict(expected_bindings, expected_keys, "expected receipt bindings")
    for name in ("resource", "asset", "inputs", "artifact"):
        _require(document[name] == expected_bindings[name],
                 "bank execution receipt " + name + " binding differs from the current issued resource")

    worker = _exact_dict(document["worker"], {
        "pending_sha256", "pending_byte_count", "execution_id", "worker_status", "worker_outcome",
    }, "bank execution worker")
    _sha256(worker["pending_sha256"], "worker.pending_sha256")
    _positive_int(worker["pending_byte_count"], "worker.pending_byte_count", 16 * 1024 * 1024)
    _require(type(worker["execution_id"]) is str and 1 <= len(worker["execution_id"]) <= 128 and
             worker["worker_status"] == "pending" and
             worker["worker_outcome"] == "measured_success",
             "worker evidence must retain pending status and measured-success outcome")

    artifact = document["artifact"]
    _exact_dict(artifact, {"sha256", "byte_count"}, "bank execution artifact")
    _sha256(artifact["sha256"], "artifact.sha256")
    _positive_int(artifact["byte_count"], "artifact.byte_count", 512 * 1024 * 1024)

    parent = _exact_dict(document["parent_execution"], {
        "pid", "exit_code", "process_reaped", "timed_out", "actual_execution",
        "source_closure_stable", "resources_stable", "checkpoint_stable", "inputs_stable",
    }, "bank parent execution")
    _positive_int(parent["pid"], "parent_execution.pid")
    _require(type(parent["exit_code"]) is int and parent["exit_code"] == 0 and
             parent["process_reaped"] is True and parent["timed_out"] is False and
             parent["actual_execution"] is True and
             parent["source_closure_stable"] is True and parent["resources_stable"] is True and
             parent["checkpoint_stable"] is True and parent["inputs_stable"] is True,
             "parent execution did not prove stable successful child completion")

    join = _exact_dict(document["join_cleanup"], {
        "logical_attempted", "logical_returned", "completed_pairs", "native_attempted",
        "native_returned", "native_verified", "joined_templates",
        "ordered_high_low_join_verified", "renderer_close_attempted", "renderer_closed",
        "metadata_cleared", "primary_error", "close_error", "metadata_error",
    }, "bank event join and cleanup")
    for name in ("logical_attempted", "logical_returned", "completed_pairs", "joined_templates"):
        _require(type(join[name]) is int and join[name] == _TEMPLATE_COUNT,
                 "bank event join must prove all 798 templates: " + name)
    for name in ("native_attempted", "native_returned", "native_verified"):
        _require(type(join[name]) is int and join[name] == 2 * _TEMPLATE_COUNT,
                 "bank event join must prove all 1596 native calls: " + name)
    _require(join["ordered_high_low_join_verified"] is True and
             join["renderer_close_attempted"] is True and join["renderer_closed"] is True and
             join["metadata_cleared"] is True and join["primary_error"] is None and
             join["close_error"] is None and join["metadata_error"] is None,
             "bank event order or renderer cleanup is not accepted")

    semantic = _exact_dict(document["semantic_validation"], _SEMANTIC_RECEIPT_KEYS,
                           "bank semantic validation")
    _require(semantic["status"] == "accepted",
             "bank semantic validation is not accepted")
    _sha256(semantic["artifact_sha256"], "semantic_validation.artifact_sha256")
    _require(semantic["artifact_sha256"] == artifact["sha256"],
             "semantic receipt does not bind the bank artifact digest")
    _positive_int(semantic["artifact_byte_count"], "semantic_validation.artifact_byte_count",
                  512 * 1024 * 1024)
    _require(semantic["artifact_byte_count"] == artifact["byte_count"],
             "semantic receipt does not bind the bank artifact byte count")
    pins = _source_pins(semantic["source_pins"])
    _require(semantic["validator_source_sha256"] == pins["bench/quality_capture_bank_semantics.py"],
             "semantic receipt validator pin differs from the pinned semantic module")
    _sha256(semantic["validator_source_sha256"], "semantic_validation.validator_source_sha256")
    source_closure = document["resource"].get("source_closure")
    _require(type(source_closure) is dict, "resource source_closure must be an object")
    for path, digest in pins.items():
        _require(source_closure.get(path) == digest,
                 "semantic source pin differs from the issued producer closure: " + path)

    checks = _exact_dict(semantic["checks"], _REQUIRED_CHECKS, "semantic validation gates")
    _require(all(value is True for value in checks.values()),
             "accepted semantic receipt must carry every fixed gate as true")
    summary = _exact_dict(semantic["summary"], _SUMMARY_KEYS, "semantic validation summary")
    _finite_json_tree(summary, "semantic_validation.summary")
    for name in ("feature_rows", "template_count", "visual_word_count", "pca_sample_count",
                 "idf_unused_word_count", "retrieval_probe_rows", "retrieval_top_n",
                 "artifact_byte_count"):
        _require(type(summary[name]) is int, "semantic summary " + name + " must be an integer")
    for name in ("pca_max_orthogonality_error", "pca_variance_max_abs_error",
                 "vertex_bound_tolerance_m", "descriptor_max_abs_error"):
        _finite_number(summary[name], "semantic summary " + name, nonnegative=True)
    _require(type(summary["vertex_min_m"]) is list and len(summary["vertex_min_m"]) == 3 and
             type(summary["vertex_max_m"]) is list and len(summary["vertex_max_m"]) == 3,
             "semantic vertex bounds summary must contain three-coordinate vectors")
    for coordinate in summary["vertex_min_m"] + summary["vertex_max_m"]:
        _finite_number(coordinate, "semantic vertex bound coordinate")
    _require(summary["artifact_sha256"] == artifact["sha256"] and
             summary["artifact_byte_count"] == artifact["byte_count"] and
             summary["feature_rows"] <= _FEATURE_LIMIT and summary["feature_rows"] > 0 and
             summary["template_count"] == _TEMPLATE_COUNT and
             summary["visual_word_count"] == _WORD_COUNT and
             summary["pca_sample_count"] == min(summary["feature_rows"], 100000) and
             summary["pca_sample_count"] > 256 and
             0 <= summary["idf_unused_word_count"] <= _WORD_COUNT and
             summary["retrieval_probe_rows"] == min(32, summary["feature_rows"]) and
             summary["retrieval_top_n"] == 10 and
             summary["pca_max_orthogonality_error"] <= _PCA_ORTHOGONAL_TOLERANCE and
             summary["vertex_bound_tolerance_m"] <= 0.01 and
             summary["descriptor_max_abs_error"] <= 0.001 and
             summary["pca_variance_max_abs_error"] <= 1e12 and
             all(abs(value) <= 1000.0 for value in summary["vertex_min_m"] + summary["vertex_max_m"]),
             "semantic validation summary exceeds the fixed numeric contract")
    return True


def validate_saved_foundpose_bank(
        state, *, asset_receipt, nearest_centroid, recompute_tfidf, retrieval_probe,
        source_pins, artifact_sha256, artifact_byte_count, numpy, chunk_rows=_CHUNK_ROWS):
    """Replay bounded saved-bank math using injected pinned numeric adapters.

    nearest_centroid(centroids, k=1, metric="l2") returns a callable which
    accepts feature chunks and returns the exact IsolatedKNN.search pair
    (squared_l2_distances, word_ids). recompute_tfidf receives full feature,
    top-one word, template-id, centroid arrays and fixed upstream keyword
    arguments, and returns (descriptors, idfs) from template_util's exact
    k=3/hard-assignment implementation. retrieval_probe wraps the pinned
    template_util tfidf_matching path and returns (template_ids, scores) for
    the supplied model-only query rows; it receives identical fixed options.
    """
    _require(type(state) is dict and set(state) == {
        "vertices", "feat_vectors", "feat_to_template_ids", "centroids", "descs",
        "idfs", "projector", "template_count", "asset_sha256",
    }, "saved bank state has an unexpected exact key set")
    expected_asset_sha = _sha256(getattr(asset_receipt, "asset_sha256", None),
                                 "verified asset receipt digest")
    _require(type(state["asset_sha256"]) is str and
             state["asset_sha256"].upper() == expected_asset_sha,
             "saved bank asset digest differs from the authenticated GLB")
    _require(type(state["template_count"]) is int and
             state["template_count"] == _TEMPLATE_COUNT,
             "saved bank must declare exactly 798 templates")

    pins = _source_pins(source_pins)
    artifact_sha256 = _sha256(artifact_sha256, "artifact_sha256")
    artifact_byte_count = _positive_int(artifact_byte_count, "artifact_byte_count",
                                        512 * 1024 * 1024)
    _require(type(chunk_rows) is int and 1 <= chunk_rows <= 65536,
             "chunk_rows must be within the bounded validation limit")
    _require(all(callable(value) for value in
                 (nearest_centroid, recompute_tfidf, retrieval_probe)),
             "pinned nearest-centroid, TF-IDF, and retrieval adapters are required")

    vertices = _as_array(state["vertices"], numpy, "vertices")
    features = _as_array(state["feat_vectors"], numpy, "feat_vectors")
    template_ids = _as_array(state["feat_to_template_ids"], numpy, "feat_to_template_ids")
    centroids = _as_array(state["centroids"], numpy, "centroids")
    descriptors = _as_array(state["descs"], numpy, "descs")
    idfs = _as_array(state["idfs"], numpy, "idfs")
    _require(getattr(vertices, "ndim", 0) == 2 and vertices.shape[1] == 3 and
             1 <= int(vertices.shape[0]) <= _FEATURE_LIMIT,
             "vertices must be a bounded nonempty Nx3 array")
    rows = int(vertices.shape[0])
    _shape(features, (rows, _FEATURE_DIM), "feat_vectors")
    _shape(template_ids, (rows,), "feat_to_template_ids")
    _shape(centroids, (_WORD_COUNT, _FEATURE_DIM), "centroids")
    _shape(descriptors, (_TEMPLATE_COUNT, _WORD_COUNT), "descs")
    _shape(idfs, (_WORD_COUNT,), "idfs")
    for name, value, expected in (
            ("vertices", vertices, "float32"), ("feat_vectors", features, "float32"),
            ("feat_to_template_ids", template_ids, "int64"),
            ("centroids", centroids, "float32"), ("descs", descriptors, "float32"),
            ("idfs", idfs, "float32")):
        _dtype(value, expected, name)
    _require(numpy.issubdtype(template_ids.dtype, numpy.integer),
             "feat_to_template_ids must be integer-valued")
    for name, array in (("vertices", vertices), ("feat_vectors", features),
                        ("centroids", centroids), ("descs", descriptors)):
        _all_finite(array, numpy, name, chunk_rows=chunk_rows)
    _require(not bool(numpy.isnan(idfs).any()) and not bool(numpy.isneginf(idfs).any()),
             "saved IDF values may only be finite or positive infinity")
    _require(bool((template_ids >= 0).all()) and
             bool((template_ids < _TEMPLATE_COUNT).all()),
             "template IDs fall outside the exact 0..797 range")
    represented = numpy.unique(template_ids)
    _require(int(represented.size) == _TEMPLATE_COUNT and
             bool((represented == numpy.arange(_TEMPLATE_COUNT, dtype=represented.dtype)).all()),
             "saved bank does not represent every one of the 798 templates")

    pca_arrays, noise_variance, pca_samples = _pca_arrays(
        state["projector"], numpy, rows)
    _require(pca_samples == min(rows, 100000),
             "PCA fitted sample count does not match min(feature_rows,100000)")
    components = pca_arrays["components_"].astype(numpy.float64, copy=False)
    gram = numpy.matmul(components, components.T)
    orthogonality_error = float(numpy.max(numpy.abs(
        gram - numpy.eye(_PCA_COMPONENTS, dtype=numpy.float64))))
    _require(math.isfinite(orthogonality_error) and
             orthogonality_error <= _PCA_ORTHOGONAL_TOLERANCE,
             "PCA component rows are not near-orthonormal")
    variance = pca_arrays["explained_variance_"].astype(numpy.float64, copy=False)
    ratios = pca_arrays["explained_variance_ratio_"].astype(numpy.float64, copy=False)
    singular = pca_arrays["singular_values_"].astype(numpy.float64, copy=False)
    _require(bool((variance > 0.0).all()) and
             bool((singular > 0.0).all()) and bool((ratios >= 0.0).all()) and
             bool((ratios <= 1.0 + _PCA_RATIO_ATOL).all()) and
             float(numpy.sum(ratios)) <= 1.0 + 0.002,
             "PCA covariance spectrum or explained-variance ratios are invalid")
    expected_variance = numpy.square(singular) / float(pca_samples - 1)
    variance_error = float(numpy.max(numpy.abs(expected_variance - variance)))
    variance_close = numpy.allclose(expected_variance, variance,
                                    rtol=_PCA_VARIANCE_RTOL,
                                    atol=_PCA_VARIANCE_ATOL)
    _require(bool(variance_close) and math.isfinite(variance_error),
             "PCA singular values disagree with covariance at the saved sample count")
    residual_rank = min(_RAW_FEATURE_DIM, pca_samples) - _PCA_COMPONENTS
    total_variance = float(numpy.sum(variance) + noise_variance * residual_rank)
    _require(math.isfinite(total_variance) and total_variance > 0.0,
             "PCA total covariance is not positive and finite")
    expected_ratios = variance / total_variance
    _require(bool(numpy.allclose(expected_ratios, ratios,
                                 rtol=_PCA_RATIO_RTOL, atol=_PCA_RATIO_ATOL)),
             "PCA explained-variance ratios disagree with fitted covariance")

    bound = _asset_bounds(asset_receipt, numpy)
    # Saved feature vertices are model-space millimetres; convert once for the
    # authenticated post-node bounds, whose coordinates are in metres.
    vertices_m = vertices.astype(numpy.float64, copy=False) * 0.001
    vertex_min = numpy.min(vertices_m, axis=0)
    vertex_max = numpy.max(vertices_m, axis=0)
    tolerance_m = _VERTEX_TOLERANCE_M + _VERTEX_RELATIVE_TOLERANCE * float(
        numpy.max(bound["extents"]))
    _require(bool((vertex_min >= bound["min"] - tolerance_m).all()) and
             bool((vertex_max <= bound["max"] + tolerance_m).all()),
             "saved model-space millimetre vertices exceed authenticated metric asset bounds")

    top1_search = nearest_centroid(centroids, k=1, metric="l2")
    _require(callable(top1_search),
             "nearest_centroid must return a fitted bounded chunk search callable")
    assigned_words = numpy.empty((rows,), dtype=numpy.int64)
    for start in range(0, rows, chunk_rows):
        end = min(rows, start + chunk_rows)
        distances, ids = top1_search(features[start:end])
        distances = _as_array(distances, numpy, "nearest-centroid distances")
        ids = _as_array(ids, numpy, "nearest-centroid IDs")
        _require(tuple(distances.shape) == (end - start, 1) and
                 tuple(ids.shape) == (end - start, 1),
                 "nearest-centroid top-one search returned an unexpected shape")
        _require(numpy.issubdtype(ids.dtype, numpy.integer) and
                 bool(numpy.isfinite(distances).all()) and bool((distances >= 0.0).all()) and
                 bool((ids >= 0).all()) and bool((ids < _WORD_COUNT).all()),
                 "nearest-centroid top-one assignments are invalid")
        assigned_words[start:end] = ids[:, 0].astype(numpy.int64, copy=False)

    pair_codes = template_ids.astype(numpy.int64, copy=False) * _WORD_COUNT + assigned_words
    distinct_pairs = numpy.unique(pair_codes)
    occurrence = numpy.bincount(
        (distinct_pairs % _WORD_COUNT).astype(numpy.int64, copy=False),
        minlength=_WORD_COUNT)
    used_words = occurrence > 0
    expected_idfs = numpy.empty((_WORD_COUNT,), dtype=numpy.float32)
    expected_idfs[used_words] = numpy.log(
        numpy.float32(_TEMPLATE_COUNT) / occurrence[used_words].astype(numpy.float32))
    expected_idfs[~used_words] = numpy.float32(numpy.inf)
    _require(bool(numpy.isposinf(idfs[~used_words]).all()) and
             bool(numpy.isfinite(idfs[used_words]).all()) and
             bool(numpy.allclose(idfs[used_words], expected_idfs[used_words],
                                 rtol=_IDF_RTOL, atol=_IDF_ATOL)),
             "saved IDF values do not match distinct-template top-one occurrence counts")

    replayed_descriptors, replayed_idfs = recompute_tfidf(
        features, assigned_words, template_ids, centroids,
        num_templates=_TEMPLATE_COUNT, tfidf_knn_k=3,
        tfidf_soft_assign=False, tfidf_soft_sigma_squared=10.0)
    replayed_descriptors = _as_array(replayed_descriptors, numpy, "replayed descriptors")
    replayed_idfs = _as_array(replayed_idfs, numpy, "replayed IDFs")
    _shape(replayed_descriptors, (_TEMPLATE_COUNT, _WORD_COUNT), "replayed descriptors")
    _shape(replayed_idfs, (_WORD_COUNT,), "replayed IDFs")
    _require(bool(numpy.isfinite(replayed_descriptors).all()),
             "pinned k=3 hard-assignment descriptor replay is non-finite")
    replayed_norms = numpy.linalg.norm(
        replayed_descriptors.astype(numpy.float64, copy=False), axis=1)
    _require(bool((replayed_norms > 0.0).all()),
             "pinned replay produced an empty template descriptor")
    _require(bool(numpy.isposinf(replayed_idfs[~used_words]).all()) and
             bool(numpy.allclose(replayed_idfs[used_words], expected_idfs[used_words],
                                 rtol=_IDF_RTOL, atol=_IDF_ATOL)),
             "pinned template_util replay uses different IDF semantics")
    descriptor_error = float(numpy.max(numpy.abs(
        replayed_descriptors.astype(numpy.float64, copy=False) -
        descriptors.astype(numpy.float64, copy=False))))
    _require(math.isfinite(descriptor_error) and
             bool(numpy.allclose(replayed_descriptors, descriptors,
                                 rtol=_DESC_RTOL, atol=_DESC_ATOL)),
             "saved descriptors differ from pinned k=3 hard-assignment TF-IDF replay")
    descriptor_norms = numpy.linalg.norm(descriptors.astype(numpy.float64, copy=False), axis=1)
    _require(bool(numpy.isfinite(descriptor_norms).all()) and
             bool((descriptor_norms > 0.0).all()),
             "saved TF-IDF descriptor rows must be finite and nonzero")

    probe_rows = min(32, rows)
    query = features[:probe_rows]
    first_ids, first_scores = retrieval_probe(
        query, centroids, descriptors, idfs, knn_k=3, soft_assign=False,
        sigma_squared=10.0, top_n=10)
    second_ids, second_scores = retrieval_probe(
        query, centroids, descriptors, idfs, knn_k=3, soft_assign=False,
        sigma_squared=10.0, top_n=10)
    first_ids = _as_array(first_ids, numpy, "retrieval template IDs")
    first_scores = _as_array(first_scores, numpy, "retrieval scores")
    second_ids = _as_array(second_ids, numpy, "repeat retrieval template IDs")
    second_scores = _as_array(second_scores, numpy, "repeat retrieval scores")
    expected_retrieval_shape = (probe_rows, 10)
    _shape(first_ids, expected_retrieval_shape, "retrieval template IDs")
    _shape(first_scores, expected_retrieval_shape, "retrieval scores")
    _shape(second_ids, expected_retrieval_shape, "repeat retrieval template IDs")
    _shape(second_scores, expected_retrieval_shape, "repeat retrieval scores")
    _require(numpy.issubdtype(first_ids.dtype, numpy.integer) and
             bool((first_ids >= 0).all()) and bool((first_ids < _TEMPLATE_COUNT).all()) and
             bool(numpy.isfinite(first_scores).all()) and
             all(int(numpy.unique(row).size) == 10 for row in first_ids) and
             bool(numpy.all(first_scores[:, :-1] >= first_scores[:, 1:])) and
             bool(numpy.array_equal(first_ids, second_ids)) and
             bool(numpy.array_equal(first_scores, second_scores)),
             "model-only retrieval must be finite, ordered, and deterministic")

    checks = {name: True for name in _REQUIRED_CHECKS}
    summary = {
        "artifact_sha256": artifact_sha256,
        "artifact_byte_count": artifact_byte_count,
        "feature_rows": rows,
        "template_count": _TEMPLATE_COUNT,
        "visual_word_count": _WORD_COUNT,
        "pca_sample_count": pca_samples,
        "pca_max_orthogonality_error": orthogonality_error,
        "pca_variance_max_abs_error": variance_error,
        "vertex_min_m": [float(value) for value in vertex_min],
        "vertex_max_m": [float(value) for value in vertex_max],
        "vertex_bound_tolerance_m": float(tolerance_m),
        "idf_unused_word_count": int((~used_words).sum()),
        "descriptor_max_abs_error": descriptor_error,
        "retrieval_probe_rows": probe_rows,
        "retrieval_top_n": 10,
    }
    result = {
        "status": "accepted",
        "validator_source_sha256": pins["bench/quality_capture_bank_semantics.py"],
        "artifact_sha256": artifact_sha256,
        "artifact_byte_count": artifact_byte_count,
        "source_pins": pins,
        "checks": checks,
        "summary": summary,
    }
    _finite_json_tree(result, "semantic validation result")
    return result


def validate_execution_receipt(document, *, expected_bindings):
    """Validate a parent-owned sealed execution receipt against issued inputs."""
    _check_execution_receipt(document, expected_bindings)
    return True
