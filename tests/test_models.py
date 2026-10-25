import pytest
import torch

from vishc_ppd.models import MODEL_NAMES, get_model

ALL_MODELS = ["deeplabv3_r50", "unet", "segformer_b2", "swin_unet", "dinov2_vitb14"]


@pytest.mark.parametrize("name", ALL_MODELS)
def test_forward_shape_and_interface(name):
    torch.manual_seed(0)
    model = get_model(name, num_classes=9, pretrained=False, input_hw=(256, 128)).eval()
    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        out = model(x)
    assert out["out"].shape == (2, 9, 256, 128)
    n_bb = sum(p.numel() for p in model.backbone_parameters())
    n_head = sum(p.numel() for p in model.head_parameters())
    assert n_bb > 0 and n_head > 0 and n_bb + n_head == model.count_parameters()


def test_registry_contains_exactly_the_benchmark_models():
    assert set(MODEL_NAMES) == set(ALL_MODELS)


def test_dino_handles_non_multiple_of_patch():
    model = get_model("dinov2_vitb14", pretrained=False).eval()   # patch 14
    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 128))["out"]           # 256/14 is not an integer
    assert out.shape == (1, 9, 256, 128)
