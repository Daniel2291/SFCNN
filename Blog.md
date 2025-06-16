---

title: Reproducing “Learning Steerable Filters for Rotation Equivariant CNNs” (Weiler et al., 2018)
tags: [machine learning, reproduction, steerable filters, CNNs, e2cnn,  SFCNN]
---
![titlepic](https://hackmd.io/_uploads/H1nHfapQxl.png)

# Reproducing “Learning Steerable Filters for Rotation Equivariant CNNs” (Weiler et al., 2018)

**Authors:** Daniel Stefanov, Sohail Faizan  TU Delft
**Course:** DSAIT4205 Fundamental Research in Machine and Deep Learning TU Delft
**Date:** June 2025

---

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
Each such $\psi_{jk}$ consists of a radial part $\tau_j(r)$ (for example, one might use Gaussian radial profiles) and an angular part $e^{i k \phi}$ which introduces an oscillation with integer frequency $k$ around the circle. A script for plotting the atomic filters is not present in the provided author repository, but such is replicated as no libraries are required, just the expression describing them. 
![filters2](https://hackmd.io/_uploads/rJouHapQgx.png)






## 🛠️ Methodology

### Workflow

```mermaid
graph LR
  A[Paper Analysis] --> B[Manual Filter Design]
  B --> C[Use of e2cnn Library]
  C --> D[Custom Training & Visualization]
```

### Tools & Libraries

- `e2cnn`: Group-equivariant operations  
- `PyTorch`: Deep learning framework  
- `Matplotlib`: Visualization  
- `NumPy`, `SciPy`: Numerical ops  

---

## 🧪 Results

### 🔍 Recreated Filters (Figure 2)

Side-by-side visualization of my learned steerable filters vs. the original figure from the paper:

![YourFilters](filters_reproduced.png)  
*My implementation (left), Paper's filters (right)*

### 📈 Reproduced Figure 4 Results

#### Left: Accuracy vs. Number of Group Elements

![AccuracyPlot](fig4_left.png)

#### Right: Accuracy vs. Data Fraction

![DataEfficiencyPlot](fig4_right.png)

---

## 🚧 Challenges

- The 2017 code sent by the authors was based on deprecated versions of PyTorch and Theano.  
- Lack of code required a **deep understanding of representation theory** and implementation details.  
- Mapping the theoretical basis of steerable filters to `e2cnn`'s API took time and care.  

---

## 🔎 Observations

- Filter structures matched well visually with the original.  
- Reproduced performance curves followed similar trends to the paper, though absolute values differed slightly due to training details.  
- Using `e2cnn` greatly simplified the implementation of group convolutions.  

---

## ✅ Conclusion

This project demonstrated that **faithful reproduction** is possible with effort, even without code.  
It also highlighted the importance of **open, maintainable implementations** in ML research.

---

## 🔗 References

1. Weiler, M., & Cesa, G. (2019). *General E(2)-Equivariant Steerable CNNs*. [arXiv:1911.08251](https://arxiv.org/abs/1911.08251)  
2. Weiler, M. et al. (2018). *Learning Steerable Filters for Rotation Equivariant CNNs*. [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Weiler_Learning_Steerable_Filters_CVPR_2018_paper.html)  
3. e2cnn Library: [https://github.com/QUVA-Lab/e2cnn](https://github.com/QUVA-Lab/e2cnn)  

---

## 📂 Code & Poster

- 🔗 GitHub Repository: [https://github.com/yourusername/sfcnn-reproduction](https://github.com/yourusername/sfcnn-reproduction)  
- 🖼 Poster PDF: [link-to-poster.pdf](#)

---

## 🙏 Acknowledgments

Thanks to the authors for responding to my inquiries, and to the creators of `e2cnn` for maintaining an excellent library.
