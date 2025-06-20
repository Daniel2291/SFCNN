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
| Sohail Faizan  | sohailfaizan@student.tudelft.nl   | 6273866   | ISBI Challenge Recreation |


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

### ISBI-2012 EM Segmentation Challenge:
The original paper uses the ISBI-2012 EM Segmentation challenge as a benchmark. It uses a three-stage pipeline that takes a raw image and produces a final instance segmentation. Some changes have been made from the original to simplify package dependencies and required training resources.

#### Architecture:

##### Stage 1: Boundary Prediction with a Steerable U-Net
The model follows a U-Net-like encoder-decoder architecture (highly effective for biomedical image segmentation).
SFCNN: Instead of standard convolutions, the network is built from custom SteerableCNNLayer modules. These layers implement the central idea of the reference paper.

##### Stage 2: Superpixel Generation (Watershed Management)
The raw boundary predictions from the network are used to generate an initial, oversegmented image where the image is divided into many small, regular regions called "superpixels."
This function implements the Distance Transform Watershed algorithm. The boundary probability map is inverted to create a "foreground" map. The distance_transform_edt function from scikit-image is used to find pixels that are furthest from any predicted boundary. These points act as robust seeds for the center of potential cells. The watershed algorithm from scikit-image is applied, using the seeds to flood the foreground map, creating the final superpixel segmentation.

##### Stage 3: Final Segmentation (Graph-Based Merging)
Here is the first major deviation from the paper's implementation. From the paper and ref 27 it is inferred that it uses the Nifty package for the Multi-cut function. Unfortunately finding and compiling the package is a big problem.
Instead the implementation uses  Hierarchical Agglomerative Clustering (merge_hierarchical) from scikit-image. This method is more powerful than a simple cut_threshold because it iteratively merges the weakest links first. However, it is still a "greedy" algorithm and not a true global optimization like Multicut. Thus the performance is expected to be noticibly lower. 

#### Deviations from the paper's architecture:

- The biggest deviation is the bypassing of using the Multicut, which is expected to create a non trivial loss in performance.

- The paper mentions using an "elastic net penalty," which is a combination of L1 and L2 regularization. Implementation uses Weight Decay in the Adam optimizer. This is equivalent to standard L2 regularization.

- The paper mentions the use of elastic deformations, a powerful augmentation technique that applies non-rigid warping to images, making the model more robust to shape variations. The implementation Uses RandomAffine transformations. It is an effective and standard form of augmentation (rotation, scaling, translation) but does not include the more complex elastic deformations.

- The authors likely used a much larger network (more channels (24 vs 16), more orientations, larger kernels) and more epochs.

#### Benchmark:
The original ISBI-2012 challenge's evaluation set was private. The predictions were sent to the challenge hosts to get results. Since the challenge has been discontibued, the dataset and the algorithm for scoring has been made public. 
The original script is in Java, and has been converted to python.

#### Results:
We achieve 
Average Dice Score:        0.8766
Average Jaccard Index (IoU): 0.7807
For our general testing. Comparable numbers are not available for the paper.
For the FoM of the paper:

| Score Type     | Achieved  | Reference Paper [1] |          |
|---------------|------------|-------------------------|------|
| V_rand        | 0.9819     | 0.98792   | (higher is better) |
| V_info        | 1.0716     | 0.99183   | (lower is better)  |

Detailed results of our model:

| Score Type     | Achieved  |           |
|---------------|------------|-----------|
| V_rand        | 0.9819     |  (higher is better) |
| V_info_split  | 0.2285     |  (lower is better)  |
| V_info_merge  | 0.8431     |  (lower is better)  |

(V_rand is just 1- Rand error)

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
