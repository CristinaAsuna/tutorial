"""Local pretrained components through the LLaVA projector/packing path. No implicit download."""
import argparse
import torch
from edu_core.backbones import load_backbone
from backbone import build_with_backbone


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path")
    parser.add_argument("--implementation", choices=("reference", "practice"), default="reference")
    args = parser.parse_args()
    torch.manual_seed(22); torch.set_num_threads(2)
    backend = load_backbone("smolvlm", model_path=args.model_path, frozen=True)
    model = build_with_backbone(backend, args.implementation)
    model.set_training_stage("pretrain_projector")
    ids = torch.tensor([[1, -200, 8, 9, 10, 2]])
    labels = torch.tensor([[-100, -100, -100, -100, 10, 2]])
    before = model.projector.layers[0].weight.detach().clone()
    try:
        result = model(ids, torch.rand(1, 3, 32, 32), labels=labels)
        result["loss"].backward()
        assert model.projector.layers[0].weight.grad.abs().sum() > 0
        assert all(p.grad is None for p in model.vision_encoder.parameters())
        assert all(p.grad is None for p in model.llm.parameters())
        torch.optim.SGD(model.projector.parameters(), lr=.001).step()
        assert not torch.equal(before, model.projector.layers[0].weight)
        # Check disjoint ownership: unfreezing language cannot accidentally unfreeze vision.
        model.set_training_stage("instruction_tuning")
        assert all(not p.requires_grad for p in model.vision_encoder.parameters())
        assert all(p.requires_grad for p in model.llm.parameters())
        print(f"LLaVA component smoke PASS: loss={result['loss'].item():.4f}; projector updated, backbone gradients frozen")
    except NotImplementedError as error:
        print(f"未完成练习：{error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
