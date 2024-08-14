[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2408.00498)

# <p align=center>`How Effective are Self-Supervised Models for Contact Identification in Videos`</p>

> **Abstract:** *The exploration of video content via Self-Supervised Learning (SSL) models has unveiled a dynamic field of study, emphasizing both the complex challenges and unique opportunities inherent in this area. Despite the growing body of research, the ability of SSL models to detect physical contacts in videos remains largely unexplored, particularly the effectiveness of methods such as downstream supervision with linear probing or full fine-tuning. This work aims to bridge this gap by employing eight different convolutional neural networks (CNNs) based video SSL models to identify instances of physical contact within video sequences specifically. The Something-Something v2 (SSv2) and Epic-Kitchen (EK-100) datasets were chosen for evaluating these approaches due to the promising results on UCF101 and HMDB51, coupled with their limited prior assessment on SSv2 and EK-100. Additionally, these datasets feature diverse environments and scenarios, essential for testing the robustness and accuracy of video-based models. This approach not only examines the effectiveness of each model in recognizing physical contacts but also explores the performance in the action recognition downstream task. By doing so, valuable insights into the adaptability of SSL models in interpreting complex, dynamic visual information are contributed.*

<p style='text-align: justify;'>
This repository contains a collection of state-of-the-art CNN based self-supervised learning in video approaches evaluated in our research work.
 </p>

### Evaluated Video Self-Supervised Learning methods

Below are the video self-supervised methods  that we evaluate for [SSv2](https://developer.qualcomm.com/software/ai-datasets/something-something) and [Epic Kitchens](https://epic-kitchens.github.io/2024) Datasets.

| Model | URL |
|-------|-----|
| MoCo| https://github.com/tinapan-pt/VideoMoCo |
| VideoMoCo | https://github.com/tinapan-pt/VideoMoCo |
| Pretext-Contrast | https://github.com/BestJuly/Pretext-Contrastive-Learning  |
| RSPNet | https://github.com/PeihaoChen/RSPNet |
| AVID-CMA | https://github.com/facebookresearch/AVID-CMA |
| CtP | https://github.com/microsoft/CtP |
| TCLR | https://github.com/DAVEISHAN/TCLR |
| GDT | https://github.com/facebookresearch/GDT |


If you find our work useful, please consider citing our paper:
```
@misc{gunawardhana2024effectiveselfsupervisedmodelscontact,
    title = {How Effective are Self-Supervised Models for Contact Identification in Videos}, 
    author = {Malitha Gunawardhana and Limalka Sadith and Liel David and Daniel Harari and Muhammad Haris Khan},
    year = {2024},
    eprint = {2408.00498},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url = {https://arxiv.org/abs/2408.00498}, 
}
