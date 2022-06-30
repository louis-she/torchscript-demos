# torchscript-demos

A brief of TorchScript by MNIST and YOLOv5.

## Requirements

Any x86 arch CPU and UNIX like system should work.

For training in Python

* Python==3.7+
* PyTorch==1.8.1
* MineTorch==0.6.12

For inference in C++

* cmake
* LibTorch
* OpenCV

## Installation

This guide will cover the part of the LibTorch and OpenCV installation and assume other things are already installed. Everything will installed within directory of the repo so uninstallation will be the same as removing the whole directory.

1. Clone this repo.

```bash
git clone https://github.com/louis-she/torchscript-demos.git
cd torchscript-demos
```

2. Install OpenCV

```bash
# In repo base dir
git clone --branch 3.4 --depth 1 https://github.com/opencv/opencv.git
mkdir opencv/build && cd opencv/build
cmake ..
make -j 4
```

3. Download LibTorch

```bash
# In repo base dir
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

4. Clone YOLOv5(Optional)

> Required if you want to try the YOLOv5 example.

```bash
git clone https://github.com/ultralytics/yolov5
```

## MNIST

**1. Training**

```bash
# In repo base dir
python3 train.py
```

Use `ctrl + C` to stop training process, the training log and graphs can be found at `./alchemistic_directory/baseline`.

**2. Build binary**

```bash
# In repo base dir

# Dump TorchScript Module file `./jit_module.pth`
python3 jit_extract.py

# Build C++ binary `./build/mnist`
mkdir build && cd build
cmake .. -DTARGET=mnist
make
```

**3. Make inference with the binary**

```bash
# In repo base dir
./build/mnist jit_module.pth six.jpg

# Output: The number is: 6
```

## YOLOv5

**1. Build binary**

```bash
# In repo base dir

# Build C++ binary `./build/mnist`
mkdir build && cd build
cmake .. -DTARGET=yolov5
make
```

**2. Export TorchScript module**

```
# In YOLOv5 base dir
python export.py --weights yolov5s.pt --img 640 --batch 1

# Then copy the yolov5s.torchscript to base dir of this repo
```

**3. Make inference with the binary**

```bash
# In base dir

./build/yolov5 yolov5s.torchscript bus.jpg
```
