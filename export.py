# export.py
import argparse
import os
import torch
from networks.Unet import UNet3D
from networks.transunet import TransUNet3D


def _load_weights_safe(model: torch.nn.Module, ckpt_path: str) -> None:
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict) and any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)


def export_to_onnx(model: torch.nn.Module, input_shape: tuple, onnx_file_path: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dummy_input = torch.randn(*input_shape, device=device)

    out_dir = os.path.dirname(onnx_file_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    print(f"ONNX file path: {onnx_file_path}")
    print(f"Device: {device}")
    print(f"Input shape (B, C, D, H, W): {tuple(dummy_input.shape)}")
    print("Starting ONNX export...")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_file_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=16,
        export_params=True,
        keep_initializers_as_inputs=True,
    )
    print(f"ONNX model exported to {onnx_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export UNet3D or TransUNet3D to ONNX format.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the ONNX file.")
    parser.add_argument("--onnx_file_name", type=str, default="unet_fp32.onnx", help="Name of the ONNX file.")
    parser.add_argument("--roi_width", type=int, default=96, help="ROI width.")
    parser.add_argument("--roi_height", type=int, default=96, help="ROI height.")
    parser.add_argument("--roi_depth", type=int, default=96, help="ROI depth.")
    parser.add_argument("--num_classes", type=int, default=14, help="Number of output classes.")
    parser.add_argument("--net", type=str, default="unet", choices=["unet", "transunet"], help="Net to export.")
    parser.add_argument("--pretrained_weights_path", type=str, default=None, help="Path to pretrained weights.")

    # TransUNet hyperparameters (kept consistent with train.py)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--mlp_dim", type=int, default=3072)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    onnx_file_path = os.path.join(args.output_dir, args.onnx_file_name)

    # PyTorch/Conv3d convention: (B, C, D, H, W)
    input_shape = (1, 1, args.roi_depth, args.roi_height, args.roi_width)

    if args.net == "unet":
        model = UNet3D(n_channels=1, n_classes=args.num_classes, base_features=16)
        if args.pretrained_weights_path:
            _load_weights_safe(model, args.pretrained_weights_path)
            print("Successfully loaded pretrained weights for UNet3D")

    else:  # transunet
        model = TransUNet3D(
            img_size=(args.roi_width, args.roi_height, args.roi_depth),  # matches train.py
            in_channels=1,
            num_classes=args.num_classes,
            base_channels=args.base_channels,
            embed_dim=args.embed_dim,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            dropout=args.dropout,
        )
        if args.pretrained_weights_path:
            _load_weights_safe(model, args.pretrained_weights_path)
            print("Successfully loaded pretrained weights for TransUNet3D")

    export_to_onnx(model, input_shape, onnx_file_path)
