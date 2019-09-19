# Barzilai-Borwein-based Adaptive Learning Rate for Deep Learning
PyTorch implementation of BB learning rate proposed by the following paper:
[Barzilai-Borwein-based Adaptive Learning Rate for Deep Learning](http://www.sciencedirect.com/science/article/pii/S0167865519302429).

## Files

- bb_dl.py: the source code for BB learning rate
- demo.py: an example showing how to use BB for training NNs

## Usage

You can use BB just like any other PyTorch optimizers.

```python3
optimizer = BB(model.parameters(), lr=1e-1, steps=200, beta=0.005, max_lr=10.0, min_lr=1e-1)
```

## Dependencies

- python==3.6
- torch==1.2.0
- torchvision==0.2.1

Other versions might also work.

## Citation
If you use BB for your research, please cite:
```text
@article{liang2019BB,
  title = {Barzilai-Borwein-Based Adaptive Learning Rate for Deep Learning},
  issn = {0167-8655},
  url = {http://www.sciencedirect.com/science/article/pii/S0167865519302429},
  doi = {10/gf7gbd},
  journaltitle = {Pattern Recognition Letters},
  date = {2019-08-30},
  author = {Liang, Jinxiu and Xu, Yong and Bao, Chenglong and Quan, Yuhui and Ji, Hui},
}
```

## License
[MIT](./LICENSE)