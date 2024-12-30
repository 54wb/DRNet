<div align="center"> 

<h1>✨DRNet✨</h1> 

[![Typing SVG](https://readme-typing-svg.herokuapp.com?font=Fira+Code&size=18&duration=2000&pause=200&center=true&vCenter=true&multiline=true&random=false&width=500&height=90&lines=Learning+Discriminative+Representation;for+Fine-Grained+Object+Detection;+in+Remote+Sensing+Images)](https://git.io/typing-svg)

</div>

## Introduction

This is the official implementation of DRNet, which is implemented on [mmrotate](https://github.com/open-mmlab/mmrotate)

## Results and models

FAIR1M-1.0 12epochs: [Download]()   

FAIR1M-1.0 12epochs & Multi-Scale & Rotation Augmentation: [Download]()    

FAIR1M-1.0
|                           Backbone                            |  lr schd  | ms | rr | Batch Size |                                   mAP                                    |                                                               Download                                                               |    
| :--------------------------------------------------------: | :---: | :---: | :-----: | :--------: | :--------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | 
| R50-FPN | 1x | - | - | 2 |  40.87  |                [model](https://drive.google.com/file/d/1h85bHRoqsUMPkjABNGP8I14Pbdxus8wX/view?usp=drive_link)                 | 
| R50-FPN | 1x | √ | √ | 2 |  45.82  | [model](https://drive.google.com/file/d/1YGIDXwD5ydH4U13KzFz0FggoPtOso5TE/view?usp=drive_link) |





## Installation

Please refer to [install.md](docs/zh_cn/install.md) for installation and dataset preparation.

## Get Started

### How to use mmrotate

If you want to train or test a oriented model, please refer to [get_started.md](docs/zh_cn/get_started.md).

### How to Start DRNet

#### Dataset Prepare

Because the annotation format of FAIR1M is XML, it first needs to be converted to DOTA format

```bash
python tools/data/fair1m/fair_to_dota.py {your_fair_dir} {dota_save_dir}
```

Next, split images to the right size.

```bash
python tools/data/dota/split/img_split.py tools/data/dota/split/split_configs/ss_train.json
```
```bash
python tools/data/dota/split/img_split.py tools/data/dota/split/split_configs/ss_test.json
```

#### Train

```bash
python tools/train.py configs/drnet/drnet_r50_fpn_1x_fair1m_le90.py
```

#### Test

```bash
python tools/test.py configs/drnet/drnet_r50_fpn_1x_fair1m_le90.py {ckpt_dir} --format-only --eval-options submission_dir={save_results_dir} nprco=1
```

Convert the result format for online submission testing

```bash
python tools/data/fair1m/dota_to_fair.py {save_results_dir} {online_sub_dir} {you_fair_images_dir}
```


## License

This project is released under the [Apache 2.0 license](LICENSE).
