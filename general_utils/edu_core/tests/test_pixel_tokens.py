"""Content/order tests distinguish reversible patchification from embedding."""
import pytest
import torch
from edu_core import patchify, unpatchify, tubelet_patchify, SelfAttentionBlock, PatchEmbed, TubeletEmbed


def test_rectangular_multichannel_patch_order_and_inverse():
    image = torch.arange(48).reshape(1, 2, 4, 6)
    patches = patchify(image, 2)
    assert torch.equal(patches[0, 0], torch.tensor([0, 24, 1, 25, 6, 30, 7, 31]))
    assert torch.equal(patches[0, 1], patches[0, 0] + 2)
    assert torch.equal(unpatchify(patches, 2, 2, grid=(2, 3)), image)
    with pytest.raises(ValueError):
        unpatchify(patches, 2, 2)


def test_tubelet_time_grid_and_local_channel_order():
    video = torch.arange(64).reshape(1, 2, 4, 2, 4)
    patches = tubelet_patchify(video, 2, 2)
    assert torch.equal(patches[0, 0], torch.tensor([0, 32, 1, 33, 4, 36, 5, 37, 8, 40, 9, 41, 12, 44, 13, 45]))
    assert torch.equal(patches[0, 1], patches[0, 0] + 2)
    assert torch.equal(patches[0, 2], patches[0, 0] + 16)


def test_square_default_and_pixel_gradients():
    image = torch.randn(2, 3, 4, 4, requires_grad=True)
    restored = unpatchify(patchify(image, 2), 2)
    assert torch.equal(restored, image)
    restored.sum().backward()
    assert torch.equal(image.grad, torch.ones_like(image))


@pytest.mark.parametrize("operation", [lambda: patchify(torch.zeros(1, 3, 3, 4), 2),
                                        lambda: tubelet_patchify(torch.zeros(1, 3, 3, 4, 4), 2, 2),
                                        lambda: unpatchify(torch.zeros(1, 4, 12), 2, 3, grid=(1, 3))])
def test_invalid_grid_is_rejected(operation):
    with pytest.raises(ValueError):
        operation()


def test_encoder_block_preserves_residual_and_has_no_cross_parameters():
    block = SelfAttentionBlock(4, 2)
    assert not any("cross" in name for name, _ in block.named_parameters())
    with torch.no_grad():
        for parameter in block.parameters():
            parameter.zero_()
    x = torch.randn(2, 3, 4)
    assert torch.equal(block(x), x)


def test_pixel_tokens_match_convolution_when_weight_axes_are_aligned():
    """Verify the documented local axis order against an independent Conv path."""
    torch.manual_seed(3)
    image = torch.randn(2, 3, 4, 6)
    embed = PatchEmbed(3, 5, 2)
    weight = embed.proj.weight.permute(0, 2, 3, 1).reshape(5, -1)
    linear = torch.nn.functional.linear(patchify(image, 2), weight, embed.proj.bias)
    torch.testing.assert_close(linear, embed(image)[0])
    video = torch.randn(2, 3, 4, 4, 6)
    embed3d = TubeletEmbed(3, 5, 2, 2)
    weight3d = embed3d.proj.weight.permute(0, 2, 3, 4, 1).reshape(5, -1)
    linear3d = torch.nn.functional.linear(tubelet_patchify(video, 2, 2), weight3d, embed3d.proj.bias)
    torch.testing.assert_close(linear3d, embed3d(video)[0])
