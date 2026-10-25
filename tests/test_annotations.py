import json

import numpy as np
import pytest

from vishc_ppd.data.annotations import AnnotationError, parse_region_id, polygons_to_mask, write_labelme_json, mask_to_polygons


def _ann(shapes, h=40, w=30):
    return {"version": "5.0.2", "flags": {}, "shapes": shapes, "imagePath": "x.jpg", "imageData": None,
            "imageHeight": h, "imageWidth": w}


def _poly(label, pts):
    return {"label": label, "points": pts, "group_id": None, "shape_type": "polygon", "flags": {}}


def test_parse_strict_accepts_only_P1_to_P8():
    assert parse_region_id("P3") == 3
    for bad in ("p3", "P4e", "P9", "line", "", None):
        with pytest.raises(AnnotationError):
            parse_region_id(bad)


def test_parse_non_canonical_variants_when_not_strict():
    assert parse_region_id("p3", strict=False) == 3
    assert parse_region_id("P4e", strict=False) == 4
    assert parse_region_id("truc ban chan", strict=False) is None


def test_smaller_region_wins_on_overlap_regardless_of_order():
    big = _poly("P8", [[0, 0], [29, 0], [29, 39], [0, 39]])          # whole image
    small = _poly("P1", [[5, 5], [15, 5], [15, 15], [5, 15]])        # inside it
    m1 = polygons_to_mask(_ann([big, small]))
    m2 = polygons_to_mask(_ann([small, big]))
    assert np.array_equal(m1, m2)
    assert m1[10, 10] == 1 and m1[30, 20] == 8


def test_self_intersecting_polygon_is_repaired_deterministically():
    bowtie = _poly("P2", [[0, 0], [20, 20], [20, 0], [0, 20]])       # crosses itself
    mask = polygons_to_mask(_ann([bowtie]))
    assert mask.max() == 2
    # both triangles of the bow-tie are filled (make_valid -> multipolygon)
    assert mask[10, 5] == 2 and mask[10, 15] == 2      # left and right triangles


def test_absent_region_yields_no_pixels():
    mask = polygons_to_mask(_ann([_poly("P1", [[0, 0], [5, 0], [5, 5], [0, 5]])]))
    assert set(np.unique(mask)) == {0, 1}


def test_duplicate_label_rejected_in_strict_mode():
    sq = [[0, 0], [5, 0], [5, 5], [0, 5]]
    with pytest.raises(AnnotationError):
        polygons_to_mask(_ann([_poly("P1", sq), _poly("P1", sq)]))


def test_points_outside_image_are_clipped():
    mask = polygons_to_mask(_ann([_poly("P5", [[-10, -10], [50, -10], [50, 60], [-10, 60]])]))
    assert mask.shape == (40, 30) and mask.min() == 5


def test_json_roundtrip(tmp_path):
    mask = np.zeros((40, 30), np.uint8)
    mask[5:20, 5:25] = 3
    out = tmp_path / "pred.json"
    write_labelme_json(out, "pred.jpg", mask)
    d = json.loads(out.read_text())
    assert d["imageData"] is None and d["imagePath"] == "pred.jpg"
    assert [s["label"] for s in d["shapes"]] == ["P3"]
    back = polygons_to_mask(d)
    inter = np.logical_and(back == 3, mask == 3).sum()
    union = np.logical_or(back == 3, mask == 3).sum()
    assert inter / union > 0.9
    assert mask_to_polygons(mask, 3)
