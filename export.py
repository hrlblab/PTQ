# export.py
import argparse
import torch
import os
from networks.Unet import UNet3D
from networks.transunet import TransUNet3D


def export_to_onnx(model, input_shape, onnx_file_path):
    """Export the PyTorch model to ONNX format."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dummy_input = torch.randn(*input_shape).to(device)

    output_dir = os.path.dirname(onnx_file_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"ONNX file path: {onnx_file_path}")
    print(f"Device: {device}")
    print(f"Input shape: {dummy_input.shape}")
    print("Starting ONNX export...")

    try:
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
    except Exception as e:
        print(f"Error during ONNX export: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export UNet3D or TransUNet3D to ONNX format.")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the ONNX file.")
    parser.add_argument("--onnx_file_name", type=str, default="unet_fp32.onnx", help="Name of the ONNX file.")
    parser.add_argument("--roi_width", type=int, default=96, help="ROI width for the input.")
    parser.add_argument("--roi_height", type=int, default=96, help="ROI height for the input.")
    parser.add_argument("--roi_depth", type=int, default=96, help="ROI depth for the input.")
    parser.add_argument("--num_classes", type=int, default=14, help="Number of output classes.")
    parser.add_argument("--net", type=str, default="unet", choices=["unet", "transunet"], help="Net to export.")
    parser.add_argument("--pretrained_weights_path", type=str, default=None, help="Path to pretrained weights.")

    # TransUNet hyperparameters
    parser.add_argument("--base_channels", type=int, default=16, help="Base channels for TransUNet.")
    parser.add_argument("--embed_dim", type=int, default=768, help="Embedding dimension for Transformer.")
    parser.add_argument("--mlp_dim", type=int, default=3072, help="MLP hidden dimension in Transformer.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of attention heads.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of Transformer encoder layers.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate in Transformer.")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    onnx_file_path = os.path.join(args.output_dir, args.onnx_file_name)
    input_shape = (1, 1, args.roi_depth, args.roi_height, args.roi_width)

    if args.net == "unet":
        model = UNet3D(n_channels=1, n_classes=args.num_classes, base_features=16)
        if args.pretrained_weights_path:
            state_dict = torch.load(args.pretrained_weights_path, map_location="cpu")
            model.load_state_dict(state_dict["model_state_dict"])
            print("Successfully loaded pretrained weights for UNet3D")

    elif args.net == "transunet":
        model = TransUNet3D(
            img_size=(args.roi_width, args.roi_height, args.roi_depth),
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
            state_dict = torch.load(args.pretrained_weights_path, map_location="cpu")
            if "model_state_dict" in state_dict:
                model.load_state_dict(state_dict["model_state_dict"])
            else:
                model.load_state_dict(state_dict)
            print("Successfully loaded pretrained weights for TransUNet3D")

    export_to_onnx(model, input_shape, onnx_file_path)
