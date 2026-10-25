"""
Data utilities.

``annotations`` and ``splits`` depend only on numpy / OpenCV / shapely so that
``validate_dataset.py`` can run without PyTorch; the torch-based
``PlantarDataset`` and transforms are imported lazily on first access.
"""

from .annotations import (
    AnnotationError,
    load_labelme,
    load_mask,
    mask_to_polygons,
    parse_region_id,
    polygons_to_mask,
    write_labelme_json,
)
from .splits import (
    IMAGES_DIRNAME,
    METADATA_FILENAME,
    SPLIT_FILENAME,
    Sample,
    collect_from_dataset,
    collect_from_folder,
    read_metadata,
    read_split_table,
    subject_id_from_stem,
    subject_level_split,
    summarize,
)

_LAZY = {"PlantarDataset", "build_eval_transform", "build_train_transform", "read_image_rgb"}


def __getattr__(name):
    if name in _LAZY:
        from . import dataset as _ds

        return getattr(_ds, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AnnotationError", "load_labelme", "load_mask", "mask_to_polygons", "parse_region_id", "polygons_to_mask",
    "write_labelme_json", "PlantarDataset", "build_eval_transform", "build_train_transform", "read_image_rgb",
    "IMAGES_DIRNAME", "METADATA_FILENAME", "SPLIT_FILENAME", "Sample", "collect_from_dataset", "collect_from_folder",
    "read_metadata", "read_split_table", "subject_id_from_stem", "subject_level_split", "summarize",
]
