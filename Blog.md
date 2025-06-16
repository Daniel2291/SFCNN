---

title: Reproducing “Learning Steerable Filters for Rotation Equivariant CNNs” (Weiler et al., 2018)
tags: [machine learning, reproduction, steerable filters, CNNs, e2cnn,  SFCNN]
---
![Title Image](https://raw.githubusercontent.com/Daniel2291/SFCNN/main/images/titlepic.png)



# Reproducing “Learning Steerable Filters for Rotation Equivariant CNNs” (Weiler et al., 2018)

**Authors:** Daniel Stefanov, Sohail Faizan  TU Delft
**Course:** DSAIT4205 Fundamental Research in Machine and Deep Learning TU Delft
**Date:** June 2025

---

| Name          | Email                            | StudentID | Task                                              |
|---------------|----------------------------------|-----------|---------------------------------------------------|
| Daniel Stefanov| dvstefanov@tudelft.nl  | 6171257   | Atomic filters / Hyperparameter test/ Augmentation strategy |
| Sohail Faizan  |    |    |  |


##  Introduction

Reproducibility is becoming a cornerstone of reliable machine learning research. Being able to re-run experiments and obtain similar results gives confidence that scientific claims are sound and not just coincidental or dependent on specific conditions (random seed). For instance in "Unreproducible Rresearch is Reproducible", Bouthillier introduces inferential reproducibility: a finding is reproducible if it can be drawn from a different experimental setup.
In this project, we focus on reproducing the results of the paper Learning Steerable Filters for Rotation Equivariant CNNs by Weiler et al. (CVPR 2018). Its main achievement is the introduction of a CNN architecture that is equivariant to rotations by design, meaning the network’s outputs transform predictably when the inputs are rotated. Such equivariance is highly desirable for image tasks where objects can appear in any orientation. For example, in medical or astronomical images, there may be no single dominant orientation for features of interest.

**Why this paper?**
Steerable filters (the core idea of the paper) offer a fascinating approach to incorporate rotational symmetry into deep networks. By reproducing this work, we aim to verify the authors’ claims and learn how enforcing symmetry constraints can lead to improved generalization. In what follows, we provide an overview of the SFCNN paper, explain the concept of steerable filters, summarize the proposed architecture, and then describe my reproduction effort and findings.


---

##  Paper overview

**Problem:** Standard CNNs succeed in learning translation-invariant features (nature of convolutions), but they are not inherently rotation-equivariant. In practice, this means a conventional CNN must learn or be trained (via data augmentation) to handle rotated images, often by learning many duplicates. This redundancy increases the model’s complexity and needs far more data. The paper by Weiler et al. addresses this limitation by building a CNN that is equivariant to rotations by architecture, eliminating the need to learn the same pattern at multiple angles.


##  Understanding Steerable Filters

Steerability means that a filter can be “turned” to different orientations in a predictable, continuous manner. In essence, if a filter is steerable, you do not need to explicitly store multiple rotated copies of that filter,  you can get any rotated version as a linear combination of some fixed basis (atomic) filters. Formally, a filter $\Psi(x)$ (think of it as a function over $\mathbb{R}^2$, mapping image coordinates to a response) is rotationally steerable if for every rotation angle $\theta$, the rotated filter $\rho_\theta[\Psi]$ can be expressed in terms of a fixed set of basis filters ${\psi_q}$.
In SFCNN, the chosen basis filters are circular harmonics. A circular harmonic can be thought of as a wave-like pattern on a circle, mathematically written in polar coordinates $(r,\phi)$ as:
![image](https://hackmd.io/_uploads/H1n_Naamlx.png)

Each such $\psi_{jk}$ consists of a radial part $\tau_j(r)$ (for example, one might use Gaussian radial profiles) and an angular part $e^{i k \phi}$ which introduces an oscillation with integer frequency $k$ around the circle. A script for plotting the atomic filters is not present in the provided author repository. Therefore circular harmonics are replicated just using the expression describing them. 

<figure style="text-align: center;">
  <img src="https://raw.githubusercontent.com/Daniel2291/SFCNN/main/images/atomic_filters.png" alt="Learned steerable filters" height="500">
  <figcaption><em>Figure 1 (or Fig.2 in paper):</em> Illustration of the circular harmonics (atomic filters).</figcaption>
</figure>



---

## Reproduction: 

### Tools & Libraries

- computer: Lenovo LOQ15 NVIDIA RTX4050 CUDA 12.4
- `e2cnn_experiments`:[Github Repo](https://github.com/QUVA-Lab/e2cnn_experiments)
- `PyTorch 3.9`: Deep learning framework  





### Hyperparameter testing:
- `hhyperparam_test_plot.py` - file for experiment
- `hyperparam_test_plot.py` -  same but X axis is sample size

Authors method for presenting SFCNN performance is by evaluating the test error versus number of sampled filter orientations (N) or capital lambda in the paper, for different training subsets from mnist-rot dataset. The number of sampled orientations (e.g., 4, 6, 8, 12...) determines how finely the steerable filter is applied in rotation space. Therefore it controls the resolution and directonal sensitivity in the network. It is not a direct hyperparameter , but its not learned during training, affects model capacity, influences performance. For reproducing the experiment, `e2cnn_experiments` library is well studied. Authors provide `mnist_bench_single.sh` which performs a single train and test for set a set of parameters, but sample size is not one of them. Therefore, in the dataset loader function, sample limiting option is implemented. Total runtime is 3h for training and testing.

<figure style="text-align: center;">
  <img src="https://raw.githubusercontent.com/Daniel2291/SFCNN/main/images/Figure_4LEFT.png" alt="Learned steerable filters" height="400">
  <figcaption><em>Figure 2 (or Fig.4 LEFT in paper):</em> Test error versus number of sampled filter orientations (N) for different training subsets from mnist-rot.</figcaption>
</figure>

### Augmentation testing:
- `mnist_bench_single.sh` - using the 

- to be written


##  Conclusion



---

## 🔗 References


1. Weiler, M. et al. (2018). *Learning Steerable Filters for Rotation Equivariant CNNs*. [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Weiler_Learning_Steerable_Filters_CVPR_2018_paper.html)  
2. e2cnn Library: [https://github.com/QUVA-Lab/e2cnn](https://github.com/QUVA-Lab/e2cnn)  

---

## 📂 Code & Poster

- 🔗 GitHub Repository: [https://github.com/Daniel2291/SFCNN](https://github.com/Daniel2291/SFCNN)  
-  Poster PDF: 
[https://github.com/Daniel2291/SFCNN/blob/main/images/SFCNNPoster.pdf] 


---

##  Acknowledgments

Thanks to the authors for responding to my inquiries, and to the creators of `e2cnn` for maintaining an excellent library. Thanks to DeNoize, a start-up in YesDelft where I (Daniel) am an intern and used the work Laptop for SFCNN Reproduction.
