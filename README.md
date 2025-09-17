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
<p><strong>Release progress:</strong> 2/7</p>
<div style="height:10px;background:#e5e7eb;border-radius:6px;overflow:hidden;max-width:420px;">
  <div style="width:28.6%;height:100%;background:#009F76;"></div>
</div>
<p><sub>Released: U-Net, TransUNet • Coming soon: UNesT, VISTA3D, SegResNet, SwinUNETR, nnU-Net</sub></p>


## Acknowledgments

This research was supported by NIH R01DK135597 (Huo), DoD HT9425-23-1-0003 (HCY), NSF 2434229 (Huo), and KPMP Glue Grant. This work was also supported by Vanderbilt Seed Success Grant, Vanderbilt Discovery Grant, and VISE Seed Grant. This project was supported by The Leona M. and Harry B. Helmsley Charitable Trust grant G-1903-03793 and G-2103-05128. This research was also supported by NIH grants R01EB033385, R01DK132338, REB017230, R01MH125931, and NSF 2040462. We extend gratitude to NVIDIA for their support by means of the NVIDIA hardware grant. This work was also supported by NSF NAIRR Pilot Award NAIRR240055.