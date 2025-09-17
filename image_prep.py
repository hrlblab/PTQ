# image_prep.py
import argparse
import numpy as np
import torch

from utils import get_calibration_data_loader


def main():
    parser = argparse.ArgumentParser(description="Prepare calibration tensors for PTQ.")
    parser.add_argument("--calibration_data_size", type=int, default=20, help="Number of images to save.")
    parser.add_argument("--fp16", action="store_true", help="Save tensors in float16.")
    parser.add_argument("--output_path", type=str, default="calib.npy", help="Output .npy file path.")
    parser.add_argument("--data_path", type=str, default="/path/to/images", help="Images directory.")
    parser.add_argument("--label_path", type=str, default="/path/to/labels", help="Labels directory (optional).")
    parser.add_argument("--batch_size", type=int, default=1, help="DataLoader batch size.")
    parser.add_argument("--roi_width", type=int, default=96)
    parser.add_argument("--roi_height", type=int, default=96)
    parser.add_argument("--roi_depth", type=int, default=96)
    # Keep preprocess params aligned with training/eval
    parser.add_argument("--a_min", type=float, default=-175.0)
    parser.add_argument("--a_max", type=float, default=250.0)
    parser.add_argument("--b_min", type=float, default=0.0)
    parser.add_argument("--b_max", type=float, default=1.0)
    args = parser.parse_args()

    data_loader = get_calibration_data_loader(
        data_path=args.data_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        roi_depth=args.roi_depth,
        roi_height=args.roi_height,
        roi_width=args.roi_width,
        a_min=args.a_min,
        a_max=args.a_max,
        b_min=args.b_min,
        b_max=args.b_max,
        num_samples_per_image=1,
    )

    tensors = []
    total = 0
    limit = int(args.calibration_data_size)

    for batch in data_loader:
        img = batch["image"]
        if args.fp16:
            img = img.half()
        # Ensure tensor on CPU and detached
        tensors.append(img.detach().cpu().numpy())
        total += img.shape[0]
        if total >= limit:
            break

    if not tensors:
        raise RuntimeError("No images found for calibration. Check data_path/label_path and transforms.")

    calib = np.concatenate(tensors, axis=0)[:limit]
    np.save(args.output_path, calib)
    print(f"Saved calibration tensor: {calib.shape} -> {args.output_path}")


if __name__ == "__main__":
    main()
