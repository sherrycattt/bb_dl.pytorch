# Barzilai-Borwein-based Adaptive Learning Rate for Deep Learning
PyTorch implementation of BB learning rate proposed by the following paper:
[Barzilai-Borwein-based Adaptive Learning Rate for Deep Learning](https://doi.org/10.1016/j.patrec.2019.08.029).
- A Barzilai–Borwein-based method for adaptive learning rate of training DNNs.
- The method is highly insensitive to initial learning rate which greatly reduces computational effort.
- The method has its advantage over others on learning speed and generalization performance.
- Convergence guarantee of the method for training DNNs.

## Files

- bb_dl.py: the source code for BB learning rate
- demo.py: an example showing how to use BB for training NNs

## Usage

You can use BB just like any other PyTorch optimizers.

```python3
optimizer = BB(model.parameters(), lr=1e-1, steps=400, beta=0.01, max_lr=10.0, min_lr=1e-1)
```

## Dependencies

- python==3.6
- torch==1.2.0
- torchvision==0.2.1

Other versions might also work.

## Citation
If you use BB for your research, please cite:
```text
@Article{liang2019bb_dl,
  Title                    = {Barzilai-Borwein-Based Adaptive Learning Rate for Deep Learning},
  Author                   = {Liang, Jinxiu and Xu, Yong and Bao, Chenglong and Quan, Yuhui and Ji, Hui},
  Journal                  = {Pattern Recognition Letters},
  Year                     = {2019},
  Pages                    = {197 - 203},
  Volume                   = {128},
}
```