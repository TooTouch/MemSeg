# MemSeg
Unofficial Re-implementation for [MemSeg: A semi-supervised method for image surface defect detection using differences and commonalities](https://arxiv.org/abs/2205.00908)

# Environments

- Docker image: nvcr.io/nvidia/pytorch:20.12-py3

```
einops==0.5.0
timm==0.5.4
wandb==0.12.17
omegaconf
imgaug==0.4.0
```


# Process

## 1. Anomaly Simulation Strategy 

- [notebook](https://github.com/TooTouch/MemSeg/blob/main/%5Bexample%5D%20anomaly_simulation_strategy.ipynb)
- Describable Textures Dataset(DTD) [ [download](https://www.google.com/search?q=dtd+texture+dataset&rlz=1C5CHFA_enKR999KR999&oq=dtd+texture+dataset&aqs=chrome..69i57j69i60.2253j0j7&sourceid=chrome&ie=UTF-8) ]

<p align='center'>
    <img width='700' src='https://user-images.githubusercontent.com/37654013/198960273-ba763f40-6b30-42e3-ab2c-a8e632df63e9.png'>
</p>

## 2. Model Process 

- [notebook](https://github.com/TooTouch/MemSeg/blob/main/%5Bexample%5D%20model%20overview.ipynb)

<p align='center'>
    <img width='1500' src='https://user-images.githubusercontent.com/37654013/198960086-fdbf39df-f680-4510-b94b-48341836f960.png'>
</p>


# Run

```bash
python main.py --yaml_config ./configs.yaml DATASET.target capsule
```

## Demo

```
voila "[demo] model inference.ipynb" --port ${port} --Voila.ip ${ip}
```

![](https://github.com/TooTouch/MemSeg/blob/main/assets/memseg.gif)

# Results

TBD

| target     |   AUROC-image |   AUROC-pixel |   AUPRO-pixel |
|:-----------|--------------:|--------------:|--------------:|
| leather    |        100    |         97.29 |         96.14 |
| pill       |         93.67 |         92.47 |         84.14 |
| carpet     |         97.87 |         96.55 |         90.74 |
| hazelnut   |         99.79 |         93.92 |         92    |
| tile       |        100    |         98.79 |         97.09 |
| cable      |         81.22 |         67.08 |         52.64 |
| transistor |         95.04 |         72.34 |         68.8  |
| zipper     |         98.74 |         88.33 |         75.87 |
| metal_nut  |         99.8  |         75.91 |         86.55 |
| grid       |         99.25 |         95.42 |         89.53 |
| bottle     |        100    |         95.78 |         90.53 |
| capsule    |         85.08 |         88.17 |         75.95 |
| wood       |        100    |         94.79 |         88.61 |
| **Average**    |         96.19 |         88.99 |         83.74 |

# Citation

```
@article{DBLP:journals/corr/abs-2205-00908,
  author    = {Minghui Yang and
               Peng Wu and
               Jing Liu and
               Hui Feng},
  title     = {MemSeg: {A} semi-supervised method for image surface defect detection
               using differences and commonalities},
  journal   = {CoRR},
  volume    = {abs/2205.00908},
  year      = {2022},
  url       = {https://doi.org/10.48550/arXiv.2205.00908},
  doi       = {10.48550/arXiv.2205.00908},
  eprinttype = {arXiv},
  eprint    = {2205.00908},
  timestamp = {Tue, 03 May 2022 15:52:06 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2205-00908.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
