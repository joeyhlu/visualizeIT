"""Join actual frozen-feature receipts with a separately authored label inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import uuid

from .quality_adaptation_data import SCHEMA, CONVENTION, canonical_hash, load_dataset, read_json, sha256
from .quality_adaptation_runtime import LOCK, atomic_stream


def assemble(root: Path, inventory_path: Path, output: Path) -> dict:
    """Paths in the inventory are relative to the extraction directory.

    The inventory is an operator-authored split/label record; this command does
    not manufacture supervision, choose a split, or relabel failed examples.
    """
    if output.exists() or output.resolve().parent != root.resolve():
        raise ValueError("manifest output must be a new file in the extraction directory")
    recipe = read_json(root / "recipe.json")
    index = read_json(root / "index.json")
    inventory = read_json(inventory_path)
    if (set(inventory) != {"schema", "label_origin", "rows"}
            or inventory["schema"] != "correspondence_label_inventory_v1"
            or inventory["label_origin"] not in ("independent_annotations", "synthetic_renderer_training_only")):
        raise ValueError("invalid independently authored label inventory")
    identity = canonical_hash(recipe)
    if index.get("recipe_sha256") != identity:
        raise ValueError("extraction recipe/index/source identity mismatch")
    from .quality_adaptation_smoke import validate_frozen_feature_source
    validate_frozen_feature_source(recipe)
    extracted = {r["id"]: r for r in index["rows"]}
    if len(extracted) != len(index["rows"]):
        raise ValueError("duplicate extraction ids")
    rows = []
    inventory_ids = set()
    for source in inventory["rows"]:
        row = dict(source)
        inventory_ids.add(row["id"])
        if row["split"] != "test" and row["status"] == "available":
            feature = extracted.get(row["id"])
            if feature is None or feature["status"] != "available":
                raise ValueError("available label row lacks successful extraction; preserve it as unavailable")
            if "features" in row:
                raise ValueError("training features are bound by extraction receipt, not inventory overrides")
            if feature.get("query_rgb_sha256") != row.get("query_rgb_sha256") or not feature.get("query_rgb_sha256"):
                raise ValueError("inventory query RGB hash differs from extractor index")
            row["features"] = feature["features"]
        rows.append(row)
    if set(extracted) - inventory_ids:
        raise ValueError("inventory omits extracted rows/failures")
    doc = {"schema": SCHEMA, "conventions": CONVENTION,
           "provenance": {"feature_recipe_sha256": identity, "source_lock_sha256": sha256(LOCK),
                          "annotation_inputs": [], "label_origin": inventory["label_origin"]}, "rows": rows}
    encoded = (json.dumps(doc, indent=2, allow_nan=False) + "\n").encode()
    # Validate a temporary manifest at the same root before publishing output.
    temporary = root / (".adaptation-manifest-" + uuid.uuid4().hex + ".json")
    try:
        with temporary.open("xb") as f:
            f.write(encoded)
        load_dataset(temporary)
        with atomic_stream(root, output, max_file_bytes=len(encoded), max_output_bytes=8 * 1024**3) as f:
            f.write(encoded)
    finally:
        temporary.unlink(missing_ok=True)
    return {"manifest": str(output), "sha256": sha256(output), "rows": len(rows)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(assemble(args.extraction, args.labels, args.output), indent=2))


if __name__ == "__main__":
    main()
