# MedPTQ — Usage Tutorial

This page walks you through two ways to run MedPTQ:

1) **Quick evaluation** of prebuilt ONNX models (INT8 vs FP32)  
2) **Full PTQ pipeline** on your own data: train → export → calibrate → quantize → deploy

> For environment setup and dependencies, see: [`Installation Guide`](documents/INSTALL.md)

---

## 1. Quick Evaluation (deploy prebuilt ONNX)

Evaluate **INT8** and compute Dice on BTCV:
```bash
python deploy.py \
  --onnx_path unet_int8.onnx \
  --data_path ./BTCV/data/test/images \
  --label_path ./BTCV/data/test/labels \
  --compute_dice
```

Compare with the **FP32** counterpart:
```bash
python deploy.py \
  --onnx_path unet_fp32.onnx \
  --data_path ./BTCV/data/test/images \
  --label_path ./BTCV/data/test/labels \
  --compute_dice
```

---

## 2. Full MedPTQ Pipeline 

### 2.1 Train (example: U-Net on BTCV)

```bash
python train.py \
  --model unet3d \
  --data_path ./BTCV/data/train/images \
  --label_path ./BTCV/data/train/labels \
  --val_data_path ./BTCV/data/test/images \
  --val_label_path ./BTCV/data/test/labels \
  --output_dir ./output_unet \
  --batch_size 2
```

> **Notes**
> - Default ROI is `96×96×96`. Customize with `--roi_width/--roi_height/--roi_depth`.
> - For **TransUNet**, each ROI dimension must be divisible by `--patch_size`.

(Optional) Train **TransUNet**:
```bash
python train.py \
  --model transunet \
  --data_path ./BTCV/data/train/images \
  --label_path ./BTCV/data/train/labels \
  --val_data_path ./BTCV/data/test/images \
  --val_label_path ./BTCV/data/test/labels \
  --output_dir ./output_transunet \
  --batch_size 2 \
  --patch_size 16 \
  --embed_dim 768 \
  --mlp_dim 3072 \
  --num_heads 12 \
  --depth 12
```

---

### 2.2 Export PyTorch → ONNX

Export the trained **U-Net** checkpoint:
```bash
python export.py \
  --onnx_file_name unet_fp32.onnx \
  --net unet \
  --pretrained_weights_path ./checkpoints/unet_checkpoint.pth
```

Export the trained **TransUNet** checkpoint:
```bash
python export.py \
  --onnx_file_name transunet_fp32.onnx \
  --net transunet \
  --pretrained_weights_path ./checkpoints/transunet_checkpoint.pth \
  --patch_size 16 \
  --embed_dim 768 \
  --mlp_dim 3072 \
  --num_heads 12 \
  --num_layers 12 \
  --base_channels 16
```

> Keep the hyperparameters consistent with training (e.g., `--patch_size`, `--embed_dim`).

---

### 2.3 Prepare Calibration Data

```bash
python image_prep.py \
  --output_path calib_btcv.npy \
  --data_path ./BTCV/data/test/images \
  --label_path ./BTCV/data/test/labels
```

> **Tip:** `image_prep.py` shares preprocessing with training.  

---

### 2.4 Run INT8 Quantization (ONNX → INT8 ONNX)

Quantize **U-Net**:
```bash
python -m modelopt.onnx.quantization \
  --onnx_path unet_fp32.onnx \
  --quantize_mode int8 \
  --calibration_data calib_btcv.npy \
  --calibration_method max \
  --output_path unet_int8.onnx \
  --high_precision_dtype fp32
```

Quantize **TransUNet**:
```bash
python -m modelopt.onnx.quantization \
  --onnx_path transunet_fp32.onnx \
  --quantize_mode int8 \
  --calibration_data calib_btcv.npy \
  --calibration_method max \
  --output_path transunet_int8.onnx \
  --high_precision_dtype fp32
```

> Common calibration methods: `max`, `entropy`, `mse`.  
> If you observe saturation or quality drop, try another `--calibration_method` and ensure the calibration set is representative.

---

### 2.5 Deploy INT8

```bash
python deploy.py \
  --onnx_path unet_int8.onnx \
  --data_path ./BTCV/data/test/images \
  --label_path ./BTCV/data/test/labels \
  --compute_dice
```

or for TransUNet:
```bash
python deploy.py \
  --onnx_path transunet_int8.onnx \
  --data_path ./BTCV/data/test/images \
  --label_path ./BTCV/data/test/labels \
  --compute_dice
```

---

## Dataset Layout (BTCV example)

```
BTCV/
└── data/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

---

> [!CAUTION]
>
> - **Shape mismatch**  
>   Check `--roi_width/--roi_height/--roi_depth`. For TransUNet, each ROI dim must be divisible by `--patch_size`.
>
> - **ONNX export fails**  
>   Try a different opset (default is 16 in `export.py`) and verify any custom ops.
>
> - **Quality drop after INT8**  
>   Use a representative calibration set (20–100 volumes is a good start), and try `entropy` or `mse`.
>
> - **Runtime differences**  
>   TensorRT/GPU/driver versions can slightly change latency; ensure drivers and CUDA/TensorRT are aligned with your environment.
