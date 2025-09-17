# Installation
### 0. Download TensorRT
---
```bash
wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/10.3.0/tars/TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
tar -xvzf TensorRT-10.3.0.26.Linux.x86_64-gnu.cuda-12.5.tar.gz
```
##### Define Variables
```bash
TENSORRT_PATH=$(pwd)/TensorRT-10.3.0.26
```
```
echo "export TENSORRT_HOME=${TENSORRT_PATH}" >> ~/.bashrc
echo "export PATH=\$TENSORRT_HOME/bin:\$PATH" >> ~/.bashrc
echo "export LD_LIBRARY_PATH=\$TENSORRT_HOME/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc
```
```
source ~/.bashrc
```

### 1. Conda Environment Setup
---
```
conda create -n ptq python=3.9
conda activate ptq
```
##### Install TensorRT wheel
```
cd TensorRT-10.3.0.26/python/
pip install ./tensorrt-10.3.0-cp39-none-linux_x86_64.whl
cd ../..
```
##### Install all required package
```
cd PTQ
pip install -r requirements.txt
```
