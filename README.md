[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2408.00498)

# <p align=center>`How Effective are Self-Supervised Models for Contact Identification in Videos`</p>
[Malitha Gunawardhana*](https://malitha123.github.io/malitha/), [Limalka Sadith](https://www.linkedin.com/in/limalka-sadith/1000/), [Liel David](https://www.linkedin.com/in/liel-david-0bb41244/), [Daniel Harari](https://scholar.google.com/citations?hl=en&user=xwdcDjUAAAAJ), [Muhammad Haris Khan](https://m-haris-khan.com/)

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


### Evalaution
- [SSv2](https://github.com/Malitha123/Model_Eval/tree/main/Eval_SSv2_Dataset)
- [EK-100](https://github.com/Malitha123/Model_Eval/tree/main/Eval_Epic_Kitchens_Dataset)

### Acknowledgments
This research was supported by the joint grant P007 from [Mohamed Bin Zayed University of Artificial Intelligence](https://mbzuai.ac.ae/)  and the [Weizmann Institute of Science](https://www.weizmann.ac.il/pages/). The authors would like to express their sincere gratitude for this generous support, which made the study possible.

### Citing
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
