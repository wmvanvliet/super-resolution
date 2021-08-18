# Super-resolution using nVidia-VPF

Depends on nVidia's [Video Processing Framework](https://github.com/NVIDIA/VideoProcessingFramework) with python bindings, and on [PyTorch](https://pytorch.org).

First, download the model weights:

```bash
cd weights
bash download_weights.sh
```

Then, run the superresolution on a video file (an example is included in the repo):

```bash
python superresolution.py example.mp4 example-scaled.mp4
```
