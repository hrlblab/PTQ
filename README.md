<div align="center">
  <img src="documents/fig_ptq.png" alt="MedPTQ overview" width=70%>
</div>

<h2><p align="center">
  MedPTQ: A Practical Toolkit for Real Post-Training Quantization<br>
  in 3D Medical Image Segmentation
</p></h2>

<div align="center">

[![MedPTQ Models](https://img.shields.io/badge/MedPTQ-Models-009F76.svg)](#medptq-models)
[![MedPTQ Paper](https://img.shields.io/badge/MedPTQ-Paper-009F76.svg)](https://arxiv.org/abs/2501.17343)
[![GitHub Repo stars](https://img.shields.io/github/stars/hrlblab/PTQ?style=social)](https://github.com/hrlblab/PTQ/stargazers)


</div>

We introduce **MedPTQ**, an open-source toolkit for real post-training quantization (PTQ) that implements true 8-bit (INT8) inference on state-of-the-art (SOTA) 3D medical segmentation models

## News

- **[2025-09-17]** 🔥 New — INT8 quantized **U-Net** and **TransUNet** have been released (see [MedPTQ Models](#medptq-models)).

## Method
<div align="center">
  <img src="documents/fig_workflow.png" alt="MedPTQ overview" width=80%>
</div>

**Overview of MedPTQ.** The top row illustrates the original FP32 pipeline, where both activation $X$ and weight $W$ are in full precision and pass through Conv–BN–ReLU sequentially. The middle row shows the simulated quantization stage: `QuantizeLinear` and `DequantizeLinear` nodes are inserted after both activations and weights to simulate INT8 quantization semantics, while the model still executes in FP32. The bottom row demonstrates the real INT8 TensorRT engine, where TensorRT fuses FP32 weights with their associated `QuantizeLinear` into INT8 weights, and merges activation `DequantizeLinear`, weight `DequantizeLinear` convolution, BN, and ReLU into a single fused convolution block. This fusion enables optimized INT8 convolution kernels, reducing memory traffic and improving efficiency while preserving accuracy.

## MedPTQ Models

<table>
  <tr>
    <th>Model</th>
    <th>Download</th>
    <th>Dataset</th>
  </tr>

  <tr>
    <td>U-Net</td>
    <td>
      <a href="https://www.dropbox.com/scl/fi/2ym99l4gaf6umow9c6z57/unet_fp32.onnx?rlkey=42j06jicadpaw8qfe4gx1liik&st=cga8r0gw&dl=1">
        <img alt="FP32 ONNX" src="https://img.shields.io/badge/FP32.onnx-E89E33.svg">
      </a><br>
      <a href="https://www.dropbox.com/scl/fi/ux2jp2sd8t2g74y190v2a/unet_int8.onnx?rlkey=x6gl2yyd1xa0moc73r22i99yn&st=j84pmxkb&dl=1">
        <img alt="INT8 ONNX" src="https://img.shields.io/badge/INT8.onnx-009F76.svg">
      </a>
    </td>
    <td rowspan="2"><a href="https://www.synapse.org/Synapse:syn3193805/wiki/89480">BTCV</a></td>
  </tr>

  <tr>
    <td>TransUNet</td>
    <td>
      <a href="https://www.dropbox.com/scl/fi/i47fjndx3mmgdx0sseds2/transunet_fp32.onnx?rlkey=sv9hocvxiae4zr8cnwrv16fen&st=r0obt4in&dl=1">
        <img alt="FP32 ONNX" src="https://img.shields.io/badge/FP32.onnx-E89E33.svg">
      </a><br>
      <a href="https://www.dropbox.com/scl/fi/qcfk8tl0gehy3dilkml4v/transunet_int8.onnx?rlkey=k87xj51wrw6vevouq1a4fggcp&st=683ydc2c&dl=1">
        <img alt="INT8 ONNX" src="https://img.shields.io/badge/INT8.onnx-009F76.svg">
      </a>
    </td>
  </tr>
</table>

<!-- Progress -->
> [!NOTE]
> **Release progress:**
> 🟩🟩⬜⬜⬜⬜⬜  2/7  
> Released: U-Net, TransUNet • Coming soon: UNesT, VISTA3D, SegResNet, SwinUNETR, nnU-Net

## Getting Started

- [Installation Guide](documents/INSTALL.md)
- [Usage Tutorial](documents/USAGE.md)



## Performance

#### BTCV (N = 20, C = 13)
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th colspan="2">Model Size (MB)</th>
      <th colspan="2">Latency (ms)</th>
      <th colspan="2">mDSC</th>
    </tr>
    <tr>
      <th></th>
      <th>FP32</th><th>INT8</th>
      <th>FP32</th><th>INT8</th>
      <th>FP32</th><th>INT8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>U-Net</td>
      <td align="right">23.11</td><td align="right">6.61</td>
      <td align="right">2.62</td><td align="right">1.05</td>
      <td align="right">0.822</td><td align="right">0.822</td>
    </tr>
    <tr>
      <td>TransUNet</td>
      <td align="right">351.85</td><td align="right">91.90</td>
      <td align="right">4.09</td><td align="right">1.74</td>
      <td align="right">0.816</td><td align="right">0.816</td>
    </tr>
  </tbody>
</table>

#### Whole Brain Segmentation (N = 50, C = 133)
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th colspan="2">Model Size (MB)</th>
      <th colspan="2">Latency (ms)</th>
      <th colspan="2">mDSC</th>
    </tr>
    <tr>
      <th></th>
      <th>FP32</th><th>INT8</th>
      <th>FP32</th><th>INT8</th>
      <th>FP32</th><th>INT8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>UNesT</td>
      <td align="right">349.41</td><td align="right">96.72</td>
      <td align="right">5.59</td><td align="right">2.72</td>
      <td align="right">0.702</td><td align="right">0.701</td>
    </tr>
  </tbody>
</table>

#### TotalSegmentator V2 (N = 200, C = 104)
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th colspan="2">Model Size (MB)</th>
      <th colspan="2">Latency (ms)</th>
      <th colspan="2">mDSC</th>
    </tr>
    <tr>
      <th></th>
      <th>FP32</th><th>INT8</th>
      <th>FP32</th><th>INT8</th>
      <th>FP32</th><th>INT8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>nnU-Net</td>
      <td align="right">107.84</td><td align="right">33.97</td>
      <td align="right">2.99</td><td align="right">1.25</td>
      <td align="right">0.901</td><td align="right">0.895</td>
    </tr>
    <tr>
      <td>SwinUNETR</td>
      <td align="right">247.96</td><td align="right">70.18</td>
      <td align="right">9.85</td><td align="right">3.59</td>
      <td align="right">0.878</td><td align="right">0.877</td>
    </tr>
    <tr>
      <td>SegResNet</td>
      <td align="right">170.44</td><td align="right">50.29</td>
      <td align="right">5.14</td><td align="right">2.06</td>
      <td align="right">0.882</td><td align="right">0.879</td>
    </tr>
    <tr>
      <td>VISTA3D</td>
      <td align="right">264.57</td><td align="right">71.18</td>
      <td align="right">4.59</td><td align="right">1.93</td>
      <td align="right">0.893</td><td align="right">0.891</td>
    </tr>
  </tbody>
</table>

**Quantization results of SOTA medical segmentation models.** We evaluate MedPTQ on seven models (U-Net, TransUNet, UNesT, nnU-Net, SwinUNETR, SegResNet, VISTA3D) across three datasets with different numbers of samples (N) and classes (C): BTCV (N = 20, C = 13), Whole Brain Segmentation (N = 50, C = 133), and TotalSegmentator V2 (N = 200, C = 104). All models are compiled to TensorRT for both FP32 and INT8; we report **Model Size (MB)**, **Latency (ms)**, and **mDSC**. Compared with FP32, INT8 consistently compresses model size by **3.17×–3.83×** and reduces latency by **2.06×–2.74×**, while maintaining accuracy (absolute ΔmDSC ≤ 0.006).




## Acknowledgments

This research was supported by NIH R01DK135597 (Huo), DoD HT9425-23-1-0003 (HCY), NSF 2434229 (Huo), and KPMP Glue Grant. This work was also supported by Vanderbilt Seed Success Grant, Vanderbilt Discovery Grant, and VISE Seed Grant. This project was supported by The Leona M. and Harry B. Helmsley Charitable Trust grant G-1903-03793 and G-2103-05128. This research was also supported by NIH grants R01EB033385, R01DK132338, REB017230, R01MH125931, and NSF 2040462. We extend gratitude to NVIDIA for their support by means of the NVIDIA hardware grant. This work was also supported by NSF NAIRR Pilot Award NAIRR240055.