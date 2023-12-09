# Mixture of Sparse Adapters

This repository contains the official PyTorch implementation for MoSA.

![MoSA_demo](figure/demo.gif)

## Environment setup

See `env_setup.sh`

## Datasets preperation

- Fine-Grained Visual Classification (FGVC): The datasets can be downloaded following the official links. We split the training data if the public validation set is not available. The splitted dataset can be found here: [Dropbox](https://cornell.box.com/v/vptfgvcsplits), [Google Drive](https://drive.google.com/drive/folders/1mnvxTkYxmOr2W9QjcgS64UBpoJ4UmKaM?usp=sharing).

  - [CUB200 2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)

  - [NABirds](http://info.allaboutbirds.org/nabirds/)

  - [Oxford Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/)

  - [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/main.html)

  - [Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)

- Visual Task Adaptation Benchmark (VTAB): See [`VTAB_SETUP.md`](https://github.com/KMnP/vpt/blob/main/VTAB_SETUP.md) for detailed instructions and tips.

- General Image Classification Datasets (GICD): The datasets will be automatically downloaded when you run an experiment using them via `MoSA`.

## Pre-trained weights preperation

Download and place the pre-trained Transformer-based backbones to `MODEL.MODEL_ROOT`.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Pre-trained Backbone</th>
<th valign="bottom">Link</th>
<th valign="bottom">md5sum</th>
<!-- TABLE BODY -->
<tr><td align="left">ViT-B/16</td>
<td align="center"><a href="https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz">link</a></td>
<td align="center"><tt>d9715d</tt></td>
</tr>
<tr><td align="left">ViT-L/16</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/moco-v3/vit-b-300ep/linear-vit-b-300ep.pth.tar">link</a></td>
<td align="center"><tt>8f39ce</tt></td>
</tr>
<tr><td align="left">Swin-B</td>
<td align="center"><a href="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth">link</a></td>
<td align="center"><tt>bf9cc1</tt></td>
</tr>
</tbody></table>

## Training

To fine-tune a pre-trained ViT model via MoSA on FGVC-cub, you can run:

```bash
bash scripts/mosa/vit/FGVC/cub.sh
```

## License

The majority of MoSA is licensed under the CC-BY-NC 4.0 license (see [LICENSE](https://github.com/KMnP/vpt/blob/main/LICENSE) for details).
