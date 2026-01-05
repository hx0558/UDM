# UDM: [Uncertainty-based Dendritic Model for Multimodal Remote Sensing Data Classification]

Accepted by **IEEE Transactions on Geoscience and Remote Sensing (TGRS)**, 2026.

[[Paper Link](https://ieeexplore.ieee.org/document/11320871)]

🔥 🔥 🔥
## 📖 Abstract
> **Abstract:** *Multimodal remote sensing data is inevitably affected by noise due to atmospheric conditions, sensor limitations, and other factors. However, existing deep learning-based multimodal remote sensing classification (MRSC) methods overlook the impact of these data noise, which produces the uncertainty and decreases classification accuracy. To address this problem, this paper first explores uncertainty-based dendritic model (UDM) for MRSC, which reduces the uncertainty at single-modality feature extraction and multimodal feature fusion stages. At the single-modality feature extraction stage, dendrites, as a novel type of neurons, have been demonstrated with strong feature extraction abilities. Inspired by the dendritic structure, we first design a dendritic-based spatial-channel feature extraction (DSCE) module. Specifically, a dendritic neural layer (DNL) is designed in DSCE. The proposed DNL constructs a multi-branch fusion strategy to enhance the expressive capacity of multimodal remote sensing feature extraction, which achieves the localized subspace computation ability. Furthermore, based on the extracted features by DSCE, a dendritic uncertainty-based feature enhancement (DUFE) module is explored to reduce the uncertainty from the extracted features. DUFE exploits uncertainty estimation to adaptively refine the extracted representations, thereby improving feature robustness and discriminative power for MRSC. At the multimodal feature fusion stage, considering the feature redundancy of the different modalities, a dendritic-based uncertainty-aware fusion (DUAF) module is proposed. DUAF performs feature fusion by dynamically assigning weights based on the estimated uncertainty of each modality, thus enhancing classification performance. Experiments on benchmark datasets demonstrate that the proposed UDM outperforms current state-of-the-art methods based on Transformer, convolutional neural network, and Mamba for MRSC. The code is available at https://github.com/hx0558/UDM.*


![overview](https://github.com/user-attachments/assets/f7afe974-c7a8-4bf8-b4ea-5344d918a953)

**Fig. 1. Illustration of the proposed UDM for multimodal classification. Overall structure is divided into five stages, namely data process, DSCE, DUFE, DUAF, and classification.**

![DNL](https://github.com/user-attachments/assets/faa0b498-72ea-4bda-8ce6-9ea4429b3222)

**Fig. 2. The structure of the proposed DNL. The dendritic structure is comprised of four layers: synaptic layer, dendritic layer, membrance layer, and soma layer. $M$ denotes the number of dendritic branches.**

## Dependencies

1. Python 3.10
2. PyTorch
3. NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

## Installation & Run

```bash
# 1. Clone this repository
git clone https://github.com/hx0558/UDM.git
cd UDM

# 2. Create environment
conda create -n udm python=3.10
conda activate udm

# 3. Install dependencies
pip install -r Requirements.txt
```

## Citation

If you find the code helpful in your research or work, please cite the following paper(s).

```bib
@ARTICLE{11320871,
  author={He, Xin and Han, Xiao and Zhao, Yaqin and Chen, Yushi and Zou, Limin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Uncertainty-based Dendritic Model for Multimodal Remote Sensing Data Classification}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Uncertainty;Remote sensing;Transformers;Computational modeling;Noise;Laser radar;Synthetic aperture radar;Redundancy;Data models;deep learning;multimodal remote sensing data classification},
  doi={10.1109/TGRS.2025.3649778}}

```
