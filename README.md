# Awesome-Few-Shot-Generation-Papers
*Note:* We provide a comprehensive survey on various image synthesis tasks under limited data, to help the community track the evolution of this field.

> **Image Synthesis under Limited Data: A Survey and Taxonomy** <br>
> Mengping Yang, Zhe Wang*<br>
> *https://arxiv.org/abs/xxxx.xxxx*<br>
> (* denotes corresponding author)


## Overview

This repository collects the studies on image synthesis under limited data.
According to the problem definition, requirements, training, and testing schemes, image synthesis under limited data can be roughly categoried into four groups: namely
[Data-efficient generative models](#data-efficient-generative-models),
[Few-shot generative adaptation](#few-shot-generative-adaptation),
[Few-shot image generation](#few-shot-image-generaion),
and [One-shot image generation](#one-shot-image-generation).


Overall, this paper collection is organized as follows. *If you find some work is missing, feel free to raise an issue or create a pull request. We appreciate contributions in any form.*

- [Data-efficient generative models](#data-efficient-generative-models)
  - [Augmentation-based approaches](#augmentation-based-approaches)
  - [Architecture-variants](#architecture-variants)
  - [Regularization-based approaches](#regularization)
  - [Off-the-shelf Models](#off-the-shelf)
- [Few-shot generative adaptation](#few-shot-generative-adaptation)
  - [Fine-tuning](#fine-tuning)
  - [Extra-branches](#extra-branches)
  - [Model Regularization](#model-regularization)
  - [Kernel modulation](#kernel-modulation)
- [Few-shot image generation](#few-shot-image-generaion)
  - [Optimization-based](#optimization-based)
  - [Transformation-based](#transformation-based)
  - [Fusion-based](#fusion-based)
  - [Inversion-based](#inversion-based)
  - [Diffusion-based](#diffusion-based)
- [One-shot image generation](#one-shot-image-generation)
  - [GAN-based](#gan-based)
  - [Diffusion-based](#diffusion-based-1)
  - [Non-parametric-based](#non-parametric-based)


## Data-efficient generative models
**Problem Definition:** Data efficient generative models refer to the scenario where a generative model is trained on a limited amount of training data, such as 100 images from 100-shot datasets[DiffAug] or 1316 images from the MetFace[StyleGAN-ADA], to produce diverse and plausible images that follow the given distribution. However, several issues such as model overfitting and memorization are prone to occur when training a generative model on limited samples $D$ from scratch.  Additionally, the imbalance between the discrete limited training samples and continuous latent distribution might lead to unstable training. Therefore, data efficient generative models are expected to have satisfactory data efficiency to capture the given distribution. We categorize the studies on data-efficient generative models according to the methodology concepts/motivations.

### Augmentation-based approaches
- DiffAug: Differentiable augmentation for data-efficient gan training <br>
  [NeurIPS 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- ADA: Training generative adversarial networks with limited data <br>
  [NeurIPS 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- APA: Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data <br>
  [NeurIPS 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- ContraD: Training gans with stronger augmentations via contrastive discriminator <br>
  [ICLR 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- On data augmentation for gan training <br>
  [TIP 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Masked generative adversarial networks are data-efficient generation learners <br>
  [NeurIPS 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Adaptive Feature Interpolation for Low-Shot Image Generation <br>
  [ECCV 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Regularization-based approaches
- Regularizing generative adversarial networks under limited data <br>
  [CVPR 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Consistency Regularization for Generative Adversarial Networks <br>
  [ICLR 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Improved consistency regularization for gans <br>
  [AAAI 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- ProtoGAN: Towards high diversity and fidelity image synthesis under limited data <br>
  [Information Sciences 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Feature statistics mixing regularization for generative adversarial networks <br>
  [CVPR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Data-efficient instance generation from instance discrimination <br>
  [NeurIPS 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- DigGAN: Discriminator gradIent Gap Regularization for GAN Training with Limited Data <br>
  [NeurIPS 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- FakeCLR: Exploring contrastive learning for solving latent discontinuity in data-efficient GANs <br>
  [ECCV 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot image generation with mixup-based distance learning <br>
  [ECCV 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Improving GAN Training via Feature Space Shrinkage <br>
  [CVPR 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Architecture-variants
- Towards faster and stabilized gan training for high-fidelity few-shot image synthesis <br>
  [ICLR 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Prototype memory and attention mechanisms for few shot image generation <br>
  [ICLR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- FreGAN: Exploiting Frequency Components for Training GANs under Limited Data <br>
  [NeurIPS 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- AutoInfo GAN: Toward a better image synthesis GAN framework for high-fidelity few-shot datasets via NAS and contrastive learning <br>
  [KBS 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- GenCo: generative co-training for generative adversarial networks with limited data <br>
  [AAAI 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Data-efficient gan training beyond (just) augmentations: A lottery ticket perspective <br>
  [NeurIPS 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Improving gans with a dynamic discriminator <br>
  [NeurIPS 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration <br>
  [CVPR 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Off-the-shelf Models
- Projected gans converge faster <br>
  [NeurIPS 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Ensembling off-the-shelf models for gan training <br>
  [CVPR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- KD-DLGAN: Data Limited Image Generation via Knowledge Distillation <br>
  [CVPR 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Data instance prior (disp) in generative adversarial networks <br>
  [WACV 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Stylegan-xl: Scaling stylegan to large diverse datasets <br>
  [SIGGRAPH 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

## Few-shot generative adaptation
**Problem Definition:**
Akin to transfer learning, the goal of few-shot generative adaptation is to transfer the knowledge of pre-trained generative models from large-scale source domains (*e.g.*, FFHQ) to target domains with limited data (*e.g.*, 10-shot images of baby faces) in a fast and efficient manner. Ideally, the adapted generative model should 1) inherent the attributes of the source generative models that are invariant to the distribution shift, such as the overall structure, synthesis diversity, and semantic variances of generated images, and 2) capture the internal distribution of the target domain to synthesize novel samples following the target distribution. However, the limited amount of training data available for adaptation may cause the model to overfit, leading to model degradation. Additionally, when the domain gaps between the source domain and the target domain are significant, negative transfer may occur, resulting in unrealistic generation. Furthermore, inappropriate knowledge transfer may also lead to a deterioration in synthesis performance.

### Fine-tuning
- Transferring gans: generating images from limited data <br>
  [ECCV 2018](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Image generation from small datasets via batch statistics adaptation <br>
  [ICCV 2019](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- On leveraging pretrained GANs for generation with limited data <br>
  [ICML 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot Image Generation with Elastic Weight Consolidation <br>
  [NeurIPS 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs <br>
  [CVPR WorkShop 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot adaptation of generative adversarial networks <br>
  [ArXiv 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Fine-tuning Diffusion Models with Limited Data <br>
  [NeurIPS WorkShop 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Extra-branches
- Minegan: effective knowledge transfer from gans to target domains with few images <br>
  [CVPR 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- One-shot generative domain adaptation <br>
  [ICCV 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Domain Re-Modulation for Few-Shot Generative Domain Adaptation <br>
  [ArXiv 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot adaptation of generative adversarial networks <br>
  [ArXiv 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot image generation with diffusion models <br>
  [ArXiv 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Model Regularization
- Few-shot image generation via cross-domain correspondence <br>
  [CVPR 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few shot generative model adaption via relaxed spatial structural alignment <br>
  [CVPR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Dynamic Weighted Semantic Correspondence for Few-Shot Image Generative Adaptation <br>
  [ACM MM 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- A closer look at few-shot image generation <br>
  [CVPR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Towards diverse and faithful one-shot adaption of generative adversarial networks <br>
  [NeurIPS 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Generalized One-shot Domain Adaptation of Generative Adversarial Networks <br>
  [NeurIPS 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few Shot Generative Domain Adaptation via Inference-Stage Latent Learning in GANs <br>
  [NeurIPS Workshop 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- WeditGAN: Few-shot Image Generation via Latent Space Relocation <br>
  [ArXiv 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Diffusion guided domain adaptation of image generators <br>
  [ArXiv 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Kernel modulation
- Few-shot image generation via adaptation-aware kernel modulation <br>
  [NeurIPS 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Exploring incompatible knowledge transfer in few-shot image generation <br>
  [CVPR 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

## Few-shot image generation
**Problem Definition:**Following the prior philosophy of few-shot learning, few-shot image generation is formulated to synthesize diverse and photorealistic images for a new category given $K$ input images from the same category. The model is trained in an episodic task-by-task manner, wherein each $N$-way-$K$-shot image generation task is defined by $K$ input images from each of the $N$ classes. The training and testing phases of few-shot image generation involve splitting the dataset into two disjoint subsets: seen classes $\mathbb{C}_s$ and unseen classes $\mathbb{C}_u$. During training, a considerable number of $N$-way-$K$-shot image generation tasks from $\mathbb{C}_s$ is randomly sampled, with the aim of encouraging the model to acquire the ability to generate novel samples. In the testing phase, the model is expected to generalize this ability to generate new images for $\mathbb{C}_u$, based on only a few samples from each class. Few-shot image generation is known to suffer from catastrophic forgetting, whereby the model forgets previous knowledge and focuses excessively on new tasks, thus impairing its ability to generalize to unseen classes.

### Optimization-based
- FIGR: Few-shot image generation with reptile <br>
  [ArXiv 2019](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- DAWSON: A domain adaptive few shot generation framework <br>
  [ArXiv 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Fast adaptive meta-learning for few-shot image generation <br>
  [TMM 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot image generation based on contrastive meta-learning generative adversarial network <br>
  [The Visual Computer 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Transformation-based
- Data augmentation generative adversarial networks <br>
  [ArXiv 2017](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Delta{GAN}: Towards diverse few-shot image generation with sample-specific delta <br>
  [ECCV 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few Shot Image Generation via Implicit Autoencoding of Support Sets <br>
  [NeurIPS Workshop 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot Image Generation Using Discrete Content Representation <br>
  [ACM MM 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Learning To Memorize Feature Hallucination for One-Shot Image Generation <br>
  [CVPR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Fusion-based
- Matchinggan: Matching-Based Few-Shot Image Generation <br>
  [ICME 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Few-shot generative modelling with generative matching networks <br>
  [AISTATS 2018](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- F2GAN: Fusing-and-Filling GAN for Few-shot Image Generation <br>
  [ACM MM 2020](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Lofgan: Fusing local representations for few-shot image generation <br>
  [ICCV 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- WaveGAN: Frequency-Aware GAN for High-Fidelity Few-Shot Image Generation <br>
  [ECCV 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- AMMGAN: adaptive multi-scale modulation generative adversarial network for few-shot image generation <br>
  [Applied Intelligence 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Inversion-based
- Attribute Group Editing for Reliable Few-shot Image Generation <br>
  [CVPR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Stable Attribute Group Editing for Reliable Few-shot Image Generation <br>
  [ArXiv 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- The Euclidean Space is Evil: Hyperbolic Attribute Editing for Few-shot Image Generation <br>
  [ArXiv 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Where Is My Spot? Few-Shot Image Generation via Latent Subspace Optimization <br>
  [CVPR 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)


### Diffusion-based
- Few-shot diffusion models <br>
  [ArXiv 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)


## One-shot image synthesis
**Problem Definition:**One-shot image generation refers to the task of training a generative model to produce novel and diverse images using only a single reference image, without the use of any pre-trained generative models for knowledge transfer. This task is of significant importance as it demonstrates the potential application of generative models in practical domains where collecting large-scale training samples is not feasible. Consequently, the model is expected to capture the internal distribution of the training image and generate diverse images that share the same internal distribution as the reference image, which is an extremely challenging task. Intuitively, synthesizing images directly from only one image presents a risk of low-variation generation, as the model may simply replicate the given sample. However, existing methods address this issue by modeling the internal statistics of patches within the training image, allowing the model to capture the information of the target distribution.

### GAN-based
- Singan: Learning a generative model from a single natural image <br>
  [ICCV 2019](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- SA-SinGAN: self-attention for single-image generation adversarial networks <br>
  [Machine Vision and Applications 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- ExSinGAN: Learning an explainable generative model from a single image <br>
  [ArXiv 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Recurrent SinGAN: Towards scale-agnostic single image GANs <br>
  [Proceedings of the 2021 5th International Conference on Electronic Information Technology and Computer Engineering 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Improved techniques for training single-image gans <br>
  [WACV 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- One-shot gan: Learning to generate samples from single images and videos <br>
  [CVPR 2021](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Petsgan: Rethinking priors for single image generation <br>
  [AAAI 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- TcGAN: Semantic-Aware and Structure-Preserved GANs with Individual Vision Transformer for Fast Arbitrary One-Shot Image Generation <br>
  [ArXiv 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Diffusion-based
- Sindiffusion: Learning a diffusion model from a single natural image <br>
  [ArXiv 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Sinddm: A single image denoising diffusion model <br>
  [ICML 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- SinFusion: Training Diffusion Models on a Single Image or Video <br>
  [ICML 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

### Non-parametric-based
- Drop the gan: In defense of patches nearest neighbors as single image generative models <br>
  [CVPR 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- A Patch-Based Algorithm for Diverse and High Fidelity Single Image Generation <br>
  [ICIP 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Generating natural images with direct patch distributions matching <br>
  [ECCV 2022](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- Exploring incompatible knowledge transfer in few-shot image generation <br>
  [CVPR 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)

## Other related awesome surveys

- Exploring incompatible knowledge transfer in few-shot image generation <br>
  [CVPR 2023](https://arxiv.org/pdf/2006.10738) / [Code](https://github.com/mit-han-lab/data-efficient-gans)


*We would continuously update the relevant papers and resources, stay tuned!*