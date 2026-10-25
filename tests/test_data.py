import torch

from vishc_ppd.data import PlantarDataset, collect_from_folder, subject_id_from_stem, subject_level_split


def test_subject_id_parsing():
    assert subject_id_from_stem("S0005_freegait_L") == "S0005"
    assert subject_id_from_stem("S0003_treadmill_R_trial2") == "S0003"
    assert subject_id_from_stem("something_else") == "something_else"


def test_dataset_and_split(sample_dir):
    samples = collect_from_folder(sample_dir, require_json=True)
    assert len(samples) >= 4
    tr, va = subject_level_split(samples, 0.3, seed=0)
    assert not ({s.subject_id for s in tr} & {s.subject_id for s in va})
    ds = PlantarDataset(samples, (128, 64), train=True)
    x, y, image_id = ds[0]
    assert x.shape == (3, 128, 64) and y.shape == (128, 64) and y.dtype == torch.long
    assert 0 <= int(y.min()) and int(y.max()) <= 8
    assert image_id == samples[0].image_id
