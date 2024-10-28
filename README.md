# OTTER: Effortless Label Distribution Adaptation of Zero-shot Models
[Changho Shin](https://ch-shin.github.io/), [Jitian Zhao](https://jzhao326.github.io/), [Sonia Cromp](https://socromp.github.io/), [Harit Vishwakarma](https://harit7.github.io/), [Frederic Sala](https://pages.cs.wisc.edu/~fredsala/)

Advances in Neural Information Processing Systems 37 (NeurIPS 2024).

Paper Link: [https://arxiv.org/abs/2404.08461](https://arxiv.org/abs/2404.08461)

<div align="center">
<img src="https://github.com/user-attachments/assets/7c4bc7b0-f4ea-4eae-84d1-b81b5d308642" alt="otter_with_buckets" width="600">
</div>

## Abstract
Popular zero-shot models suffer due to artifacts inherited from pretraining. One particularly detrimental issue, caused by unbalanced web-scale pretraining data, is *mismatched label distribution*. Existing approaches that seek to repair the label distribution are not suitable in zero-shot settings, as they have mismatching  requirements, such as needing access to labeled downstream task data or knowledge of the true label balance in the pretraining distribution. We sidestep these challenges and introduce a simple and lightweight approach to adjust pretrained model predictions via optimal transport. Our technique requires only an *estimate* of the label distribution of a downstream task. Theoretically, we characterize the improvement produced by our procedure under certain mild conditions and provide bounds on the error caused by misspecification. Empirically, we validate our method in a wide array of zero-shot image and text classification tasks, improving accuracy by 4.8\% and 15.9\% on average, and beating baselines like prior matching---often by significant margins---in 17 out of 21 datasets. 


## Installation

We recommend you create a conda environment as follows.

```
conda env create -f environment.yml
```

and activate it with

```
conda activate otter
```
