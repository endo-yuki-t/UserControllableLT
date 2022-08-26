# User-Controllable Latent Transformer for StyleGAN Image Layout Editing
  <!--a href="https://arxiv.org/abs/2103.14877"><img src="https://img.shields.io/badge/arXiv-2103.14877-b31b1b.svg"></a-->
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
<p align="center">
<img src="docs/teaser.jpg" width="800px"/>
</p>

This repository contains our implementation of the following paper:

Yuki Endo: "User-Controllable Latent Transformer for StyleGAN Image Layout Editing," Computer Graphpics Forum (Pacific Graphics 2022) [[Project](http://www.cgg.cs.tsukuba.ac.jp/~endo/projects/UserControllableLT)] [[PDF (preprint)]()]

## Prerequisites  
1. Python 3.8
2. PyTorch 1.9.0
3. Flask
4. Others (see env.yml)

## Preparation
Download and decompress <a href="https://drive.google.com/file/d/1lBL_J-uROvqZ0BYu9gmEcMCNyaPo9cBY/view?usp=sharing">our pre-trained models</a>.

## Inference with our pre-trained models
<img src="docs/thumb.gif" width="150px"/><img src="docs/car.gif" width="150px"/><img src="docs/church.gif" width="150px"/><img src="docs/ffhq.gif" width="150px"/><img src="docs/anime.gif" width="150px"/><br>
We provide an interactive interface based on Flask. This interface can be locally launched with
```
python interface/flask_app.py --checkpoint_path=pretrained_models/latent_transformer/cat.pt
```
The interface can be accessed via http://localhost:8000/.

## Training
The latent transformer can be trained with
```
python scripts/train.py --exp_dir=results --stylegan_weights=pretrained_models/stylegan2-cat-config-f.pt
```

## Citation
Please cite our paper if you find the code useful:
```
@Article{endoPG2022,
Title = {User-Controllable Latent Transformer for StyleGAN Image Layout Editing},
Author = {Yuki Endo},
Journal = {Computer Graphics Forum},
volume = {},
number = {},
pages = {},
doi = {},
Year = {2022}
}
```

## Acknowledgements
This code heavily borrows from the [pixel2style2pixel](https://github.com/eladrich/pixel2style2pixel) and [expansion](https://github.com/gengshan-y/expansion) repositories. 
