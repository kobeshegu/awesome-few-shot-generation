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
  [NeurIPS 2020](https://arxiv.org/abs/2006.06676) / [Code](https://github.com/NVlabs/stylegan2-ada)
- APA: Deceive D: Adaptive Pseudo Augmentation for GAN Training with Limited Data <br>
  [NeurIPS 2021](https://arxiv.org/abs/2111.06849) / [Code](https://github.com/EndlessSora/DeceiveD)
- ContraD: Training gans with stronger augmentations via contrastive discriminator <br>
  [ICLR 2021](https://arxiv.org/abs/2103.09742) / [Code](https://github.com/jh-jeong/ContraD)
- On data augmentation for gan training <br>
  [TIP 2021](https://ieeexplore.ieee.org/document/9319516) / [Code](https://github.com/tntrung/dag-gans)
- Masked generative adversarial networks are data-efficient generation learners <br>
  [NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/0efcb1885b8534109f95ca82a5319d25-Abstract-Conference.html) / [Code]()
- Adaptive Feature Interpolation for Low-Shot Image Generation <br>
  [ECCV 2022](https://arxiv.org/abs/2112.02450) / [Code](https://github.com/dzld00/Adaptive-Feature-Interpolation-for-Low-Shot-Image-Generation)

### Regularization-based approaches
- Regularizing generative adversarial networks under limited data <br>
  [CVPR 2021](https://arxiv.org/abs/2104.03310) / [Code](https://github.com/google/lecam-gan)
- Consistency Regularization for Generative Adversarial Networks <br>
  [ICLR 2020](https://arxiv.org/abs/1910.12027) / [Code](https://github.com/google/compare_gan)
- Improved consistency regularization for gans <br>
  [AAAI 2021](https://arxiv.org/abs/2002.04724) / [Code](https://github.com/google/compare_gan)
- ProtoGAN: Towards high diversity and fidelity image synthesis under limited data <br>
  [Information Sciences 2023](https://www.sciencedirect.com/science/article/pii/S0020025523003389) / [Code](https://github.com/kobeshegu/ProtoGAN)
- Feature statistics mixing regularization for generative adversarial networks <br>
  [CVPR 2022](https://arxiv.org/abs/2112.04120) / [Code](https://github.com/naver-ai/FSMR)
- Data-efficient instance generation from instance discrimination <br>
  [NeurIPS 2021](https://arxiv.org/abs/2106.04566) / [Code](https://github.com/genforce/insgen)
- DigGAN: Discriminator gradIent Gap Regularization for GAN Training with Limited Data <br>
  [NeurIPS 2022](https://arxiv.org/abs/2211.14694) / [Code](https://github.com/AilsaF/DigGAN)
- FakeCLR: Exploring contrastive learning for solving latent discontinuity in data-efficient GANs <br>
  [ECCV 2022](https://arxiv.org/abs/2207.08630) / [Code](https://github.com/iceli1007/FakeCLR)
- Few-shot image generation with mixup-based distance learning <br>
  [ECCV 2022](https://arxiv.org/abs/2111.11672) / [Code](https://github.com/reyllama/mixdl)
- Improving GAN Training via Feature Space Shrinkage <br>
  [CVPR 2023](https://arxiv.org/abs/2303.01559) / [Code](https://github.com/WentianZhang-ML/AdaptiveMix)

### Architecture-variants
- Towards faster and stabilized gan training for high-fidelity few-shot image synthesis <br>
  [ICLR 2021](https://arxiv.org/abs/2101.04775) / [Code](https://github.com/odegeasslbc/FastGAN-pytorch)
- Prototype memory and attention mechanisms for few shot image generation <br>
  [ICLR 2022](https://openreview.net/forum?id=lY0-7bj0Vfz) / [Code](https://github.com/Crazy-Jack/MoCA_release)
- FreGAN: Exploiting Frequency Components for Training GANs under Limited Data <br>
  [NeurIPS 2022](https://arxiv.org/abs/2210.05461) / [Code](https://github.com/kobeshegu/FreGAN_NeurIPS2022)
- AutoInfo GAN: Toward a better image synthesis GAN framework for high-fidelity few-shot datasets via NAS and contrastive learning <br>
  [KBS 2023](https://www.sciencedirect.com/science/article/pii/S0950705123005075) / [Code](https://github.com/shijiachen1/AutoInfoGAN)
- GenCo: generative co-training for generative adversarial networks with limited data <br>
  [AAAI 2022](https://arxiv.org/abs/2110.01254) / [Code](https://github.com/jxhuang0508/GenCo)
- Data-efficient gan training beyond (just) augmentations: A lottery ticket perspective <br>
  [NeurIPS 2021](https://arxiv.org/abs/2103.00397) / [Code](https://github.com/VITA-Group/Ultra-Data-Efficient-GAN-Training)
- Improving gans with a dynamic discriminator <br>
  [NeurIPS 2022](https://arxiv.org/abs/2209.09897v1) / [Code](https://github.com/genforce/dynamicd)
- Re-GAN: Data-Efficient GANs Training via Architectural Reconfiguration <br>
  [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Saxena_Re-GAN_Data-Efficient_GANs_Training_via_Architectural_Reconfiguration_CVPR_2023_paper.html) / [Code](https://github.com/IntellicentAI-Lab/Re-GAN)

### Off-the-shelf Models
- Projected gans converge faster <br>
  [NeurIPS 2021](https://arxiv.org/abs/2111.01007v1) / [Code](https://github.com/autonomousvision/projected-gan)
- Ensembling off-the-shelf models for gan training <br>
  [CVPR 2022](https://arxiv.org/abs/2112.09130) / [Code](https://github.com/nupurkmr9/vision-aided-gan)
- KD-DLGAN: Data Limited Image Generation via Knowledge Distillation <br>
  [CVPR 2023](https://arxiv.org/abs/2303.17158) / [Code](https://github.com/cuikaiwen18/KD_DLGAN)
- Data instance prior (disp) in generative adversarial networks <br>
  [WACV 2022](https://arxiv.org/abs/2012.04256) / [Code]()
- Stylegan-xl: Scaling stylegan to large diverse datasets <br>
  [SIGGRAPH 2022](https://arxiv.org/abs/2202.00273) / [Code](https://github.com/autonomousvision/stylegan-xl)

## Few-shot generative adaptation
**Problem Definition:**
Akin to transfer learning, the goal of few-shot generative adaptation is to transfer the knowledge of pre-trained generative models from large-scale source domains (*e.g.*, FFHQ) to target domains with limited data (*e.g.*, 10-shot images of baby faces) in a fast and efficient manner. Ideally, the adapted generative model should 1) inherent the attributes of the source generative models that are invariant to the distribution shift, such as the overall structure, synthesis diversity, and semantic variances of generated images, and 2) capture the internal distribution of the target domain to synthesize novel samples following the target distribution. However, the limited amount of training data available for adaptation may cause the model to overfit, leading to model degradation. Additionally, when the domain gaps between the source domain and the target domain are significant, negative transfer may occur, resulting in unrealistic generation. Furthermore, inappropriate knowledge transfer may also lead to a deterioration in synthesis performance.

### Fine-tuning
- Transferring gans: generating images from limited data <br>
  [ECCV 2018](https://arxiv.org/abs/1805.01677) / [Code](https://github.com/yaxingwang/Transferring-GANs)
- Image generation from small datasets via batch statistics adaptation <br>
  [ICCV 2019](https://arxiv.org/abs/1904.01774) / [Code](https://github.com/nogu-atsu/small-dataset-image-generation)
- On leveraging pretrained GANs for generation with limited data <br>
  [ICML 2020](https://arxiv.org/abs/2002.11810) / [Code](https://github.com/MiaoyunZhao/GANTransferLimitedData)
- Few-shot Image Generation with Elastic Weight Consolidation <br>
  [NeurIPS 2020](https://arxiv.org/abs/2012.02780) / [Code](https://yijunmaverick.github.io/publications/ewc/)
- Freeze the Discriminator: a Simple Baseline for Fine-Tuning GANs <br>
  [CVPR WorkShop 2020](https://arxiv.org/abs/2002.10964) / [Code](https://github.com/sangwoomo/FreezeD)
- Few-shot adaptation of generative adversarial networks <br>
  [ArXiv 2020](https://arxiv.org/abs/2010.11943) / [Code](https://github.com/e-271/few-shot-gan)
- Fine-tuning Diffusion Models with Limited Data <br>
  [NeurIPS WorkShop 2020](https://openreview.net/forum?id=0J6afk9DqrR)

### Extra-branches
- Minegan: effective knowledge transfer from gans to target domains with few images <br>
  [CVPR 2020](https://arxiv.org/abs/1912.05270) / [Code](https://github.com/yaxingwang/MineGAN)
- One-shot generative domain adaptation <br>
  [ICCV 2023](https://arxiv.org/abs/2111.09876) / [Code](https://github.com/genforce/genda)
- Domain Re-Modulation for Few-Shot Generative Domain Adaptation <br>
  [ArXiv 2023](https://arxiv.org/abs/2302.02550) / [Code](https://github.com/wuyi2020/DoRM)
- Few-shot image generation with diffusion models <br>
  [ArXiv 2022](https://arxiv.org/abs/2211.03264)

### Model Regularization
- Few-shot image generation via cross-domain correspondence <br>
  [CVPR 2021](https://arxiv.org/abs/2104.06820) / [Code](https://github.com/utkarshojha/few-shot-gan-adaptation)
- Few shot generative model adaption via relaxed spatial structural alignment <br>
  [CVPR 2022](https://arxiv.org/abs/2203.04121) / [Code](https://github.com/StevenShaw1999/RSSA)
- Dynamic Weighted Semantic Correspondence for Few-Shot Image Generative Adaptation <br>
  [ACM MM 2022](https://dl.acm.org/doi/10.1145/3503161.3548270)
- A closer look at few-shot image generation <br>
  [CVPR 2022](https://arxiv.org/abs/2205.03805) / [Code](https://yunqing-me.github.io/A-Closer-Look-at-FSIG)
- Towards diverse and faithful one-shot adaption of generative adversarial networks <br>
  [NeurIPS 2022](https://arxiv.org/abs/2207.08736) / [Code](https://github.com/YBYBZhang/DiFa)
- Generalized One-shot Domain Adaptation of Generative Adversarial Networks <br>
  [NeurIPS 2022](https://arxiv.org/abs/2209.03665) / [Code](https://github.com/zhangzc21/Generalized-One-shot-GAN-adaptation)
- Few Shot Generative Domain Adaptation via Inference-Stage Latent Learning in GANs <br>
  [NeurIPS Workshop 2022](https://openreview.net/forum?id=yWf4wxAUcDo) / [Code](https://github.com/mit-han-lab/data-efficient-gans)
- WeditGAN: Few-shot Image Generation via Latent Space Relocation <br>
  [ArXiv 2023](https://arxiv.org/abs/2305.06671)
- Diffusion guided domain adaptation of image generators <br>
  [ArXiv 2022](https://arxiv.org/abs/2212.04473) / [Code](https://github.com/KunpengSong/styleganfusion)

### Kernel modulation
- Few-shot image generation via adaptation-aware kernel modulation <br>
  [NeurIPS 2022](https://arxiv.org/abs/2210.16559) / [Code](https://yunqing-me.github.io/AdAM/)
- Exploring incompatible knowledge transfer in few-shot image generation <br>
  [CVPR 2023](https://arxiv.org/abs/2304.07574) / [Code](http://yunqing-me.github.io/RICK)

## Few-shot image generation
**Problem Definition:**Following the prior philosophy of few-shot learning, few-shot image generation is formulated to synthesize diverse and photorealistic images for a new category given $K$ input images from the same category. The model is trained in an episodic task-by-task manner, wherein each $N$-way-$K$-shot image generation task is defined by $K$ input images from each of the $N$ classes. The training and testing phases of few-shot image generation involve splitting the dataset into two disjoint subsets: seen classes $\mathbb{C}_s$ and unseen classes $\mathbb{C}_u$. During training, a considerable number of $N$-way-$K$-shot image generation tasks from $\mathbb{C}_s$ is randomly sampled, with the aim of encouraging the model to acquire the ability to generate novel samples. In the testing phase, the model is expected to generalize this ability to generate new images for $\mathbb{C}_u$, based on only a few samples from each class. Few-shot image generation is known to suffer from catastrophic forgetting, whereby the model forgets previous knowledge and focuses excessively on new tasks, thus impairing its ability to generalize to unseen classes.

### Optimization-based
- FIGR: Few-shot image generation with reptile <br>
  [ArXiv 2019](https://arxiv.org/abs/1901.02199) / [Code](https://github.com/LuEE-C/FIGR)
- DAWSON: A domain adaptive few shot generation framework <br>
  [ArXiv 2020](https://arxiv.org/abs/2001.00576)
- Fast adaptive meta-learning for few-shot image generation <br>
  [TMM 2021](https://ieeexplore.ieee.org/document/9424414) / [Code](https://github.com/phaphuang/FAML)
- Few-shot image generation based on contrastive meta-learning generative adversarial network <br>
  [The Visual Computer 2022](https://link.springer.com/article/10.1007/s00371-022-02566-3)

### Transformation-based
- Data augmentation generative adversarial networks <br>
  [ArXiv 2017](https://arxiv.org/abs/1711.04340) / [Code](https://github.com/AntreasAntoniou/DAGAN)
- Delta{GAN}: Towards diverse few-shot image generation with sample-specific delta <br>
  [ECCV 2022](https://arxiv.org/abs/2207.10271) / [Code](https://github.com/bcmi/DeltaGAN-Few-Shot-Image-Generation)
- Few Shot Image Generation via Implicit Autoencoding of Support Sets <br>
  [NeurIPS Workshop 2021](https://openreview.net/pdf?id=fem00ckyS8t)
- Few-shot Image Generation Using Discrete Content Representation <br>
  [ACM MM 2022](https://arxiv.org/abs/2207.10833)
- Learning To Memorize Feature Hallucination for One-Shot Image Generation <br>
  [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Xie_Learning_To_Memorize_Feature_Hallucination_for_One-Shot_Image_Generation_CVPR_2022_paper.html)

### Fusion-based
- Matchinggan: Matching-Based Few-Shot Image Generation <br>
  [ICME 2020](https://arxiv.org/abs/2003.03497) / [Code](https://github.com/bcmi/MatchingGAN-Few-Shot-Image-Generation)
- Few-shot generative modelling with generative matching networks <br>
  [AISTATS 2018](http://proceedings.mlr.press/v84/bartunov18a/bartunov18a.pdf)
- F2GAN: Fusing-and-Filling GAN for Few-shot Image Generation <br>
  [ACM MM 2020](https://arxiv.org/abs/2008.01999) / [Code](https://github.com/bcmi/F2GAN-Few-Shot-Image-Generation)
- Lofgan: Fusing local representations for few-shot image generation <br>
  [ICCV 2021](https://ieeexplore.ieee.org/document/9710556/) / [Code](https://github.com/edward3862/LoFGAN-pytorch)
- WaveGAN: Frequency-Aware GAN for High-Fidelity Few-Shot Image Generation <br>
  [ECCV 2022](https://arxiv.org/abs/2207.07288) / [Code](https://github.com/kobeshegu/ECCV2022_WaveGAN)
- AMMGAN: adaptive multi-scale modulation generative adversarial network for few-shot image generation <br>
  [Applied Intelligence 2023](https://link.springer.com/article/10.1007/s10489-023-04559-8)

### Inversion-based
- Attribute Group Editing for Reliable Few-shot Image Generation <br>
  [CVPR 2022](https://arxiv.org/abs/2203.08422) / [Code](https://github.com/UniBester/AGE)
- Stable Attribute Group Editing for Reliable Few-shot Image Generation <br>
  [ArXiv 2023](https://arxiv.org/abs/2302.00179) / [Code](https://github.com/UniBester/SAGE)
- The Euclidean Space is Evil: Hyperbolic Attribute Editing for Few-shot Image Generation <br>
  [ArXiv 2022](https://arxiv.org/abs/2211.12347) / [Code](https://yizhang025.github.io/publications/)
- Where Is My Spot? Few-Shot Image Generation via Latent Subspace Optimization <br>
  [CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Zheng_Where_Is_My_Spot_Few-Shot_Image_Generation_via_Latent_Subspace_CVPR_2023_paper.html) / [Code](https://github.com/chansey0529/LSO)


### Diffusion-based
- Few-shot diffusion models <br>
  [ArXiv 2022](https://arxiv.org/abs/2205.15463)

## One-shot image synthesis
**Problem Definition:**One-shot image generation refers to the task of training a generative model to produce novel and diverse images using only a single reference image, without the use of any pre-trained generative models for knowledge transfer. This task is of significant importance as it demonstrates the potential application of generative models in practical domains where collecting large-scale training samples is not feasible. Consequently, the model is expected to capture the internal distribution of the training image and generate diverse images that share the same internal distribution as the reference image, which is an extremely challenging task. Intuitively, synthesizing images directly from only one image presents a risk of low-variation generation, as the model may simply replicate the given sample. However, existing methods address this issue by modeling the internal statistics of patches within the training image, allowing the model to capture the information of the target distribution.

### GAN-based
- Singan: Learning a generative model from a single natural image <br>
  [ICCV 2019](https://arxiv.org/abs/1905.01164) / [Code](https://github.com/FriedRonaldo/SinGAN)
- SA-SinGAN: self-attention for single-image generation adversarial networks <br>
  [Machine Vision and Applications 2021](https://link.springer.com/article/10.1007/s00138-021-01228-z)
- ExSinGAN: Learning an explainable generative model from a single image <br>
  [ArXiv 2021](https://arxiv.org/abs/2105.07350) / [Code](https://github.com/zhangzc21/ExSinGAN)
- Recurrent SinGAN: Towards scale-agnostic single image GANs <br>
  [Proceedings of the 2021 5th International Conference on Electronic Information Technology and Computer Engineering 2021](https://dl.acm.org/doi/10.1145/3501409.3501476)
- Improved techniques for training single-image gans <br>
  [WACV 2021](https://arxiv.org/abs/2003.11512) / [Code](https://github.com/tohinz/ConSinGAN)
- One-shot gan: Learning to generate samples from single images and videos <br>
  [CVPR 2021](https://arxiv.org/abs/2103.13389v1) / [Code](https://github.com/boschresearch/one-shot-synthesis)
- Petsgan: Rethinking priors for single image generation <br>
  [AAAI 2022](https://arxiv.org/abs/2203.01488) / [Code](https://github.com/zhangzc21/PetsGAN)
- TcGAN: Semantic-Aware and Structure-Preserved GANs with Individual Vision Transformer for Fast Arbitrary One-Shot Image Generation <br>
  [ArXiv 2023](https://arxiv.org/abs/2302.08047)

### Diffusion-based
- Sindiffusion: Learning a diffusion model from a single natural image <br>
  [ArXiv 2022](https://arxiv.org/abs/2211.12445) / [Code](https://github.com/WeilunWang/SinDiffusion)
- Sinddm: A single image denoising diffusion model <br>
  [ICML 2023](https://arxiv.org/abs/2211.16582) / [Code](https://matankleiner.github.io/sinddm/)
- SinFusion: Training Diffusion Models on a Single Image or Video <br>
  [ICML 2023](https://arxiv.org/abs/2211.11743) / [Code](https://yanivnik.github.io/sinfusion/)

### Non-parametric-based
- Drop the gan: In defense of patches nearest neighbors as single image generative models <br>
  [CVPR 2022](https://arxiv.org/abs/2103.15545) / [Code](https://github.com/iyttor/GPNN)
- A Patch-Based Algorithm for Diverse and High Fidelity Single Image Generation <br>
  [ICIP 2022](https://ieeexplore.ieee.org/document/9897913)
- Generating natural images with direct patch distributions matching <br>
  [ECCV 2022](https://arxiv.org/abs/2203.11862) / [Code](https://github.com/ariel415el/GPDM)

## Other related awesome surveys

**GAN Inversion:**
- GAN Inversion: A Survey <br>
  [TPAMI 2022](https://arxiv.org/abs/2101.05278) / [Code](https://github.com/weihaox/awesome-gan-inversion)

**3D-aware GANs:**
- Awesome Text-to-Image Diffusion papers <br>
  [Code](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image)
- A Survey on 3D-aware Image Synthesis <br>
  [ArXiv 2022](https://arxiv.org/abs/2210.14267) / [Code](https://github.com/weihaox/awesome-3D-aware-synthesis)

**Diffusion Models:**
- Diffusion Models: A Comprehensive Survey of Methods and Applications <br>
  [ArXiv 2022](https://arxiv.org/abs/2209.00796) / [Code](https://github.com/YangLing0818/Diffusion-Models-Papers-Survey-Taxonomy)
- A Survey on Generative Diffusion Model <br>
  [ArXiv 2022](https://arxiv.org/pdf/2209.02646.pdf) / [Code](https://github.com/chq1155/A-Survey-on-Generative-Diffusion-Model)
- Awesome Video-generation/editing diffusion papers <br>
[Code](https://github.com/yzhang2016/video-generation-survey)
- Awesome Text-to-Image diffusion papers <br>
[Code](https://github.com/Yutong-Zhou-cv/Awesome-Text-to-Image)
- 

*We would continuously update the relevant papers and resources, stay tuned!*