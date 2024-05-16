<div align="center">
  
# InstantMesh V2: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models

![Layout](https://github.com/sarshardorosti/MasterClass/assets/50841748/e16a716a-e05e-46ed-82bd-eb4957d89a39)


</div>
---


Machine Learning Techniques for Enhanced 3D Reconstruction from Multi-Camera Studio Views
Project Overview
This project explores the use of machine learning (ML) for 3D reconstruction of studio sets from three or more camera views. It was conducted as part of the MSc in Artificial Intelligence for Media at Bournemouth University. 
The research investigates advanced methods for converting 2D images into 3D models, emphasizing deep learning technologies like CNNs and diffusion models. This project aims to overcome the challenges by combining techniques to produce high-quality 3D models from multiple camera views, particularly in studio environments. We reviewed over 80 papers from 2014 to 2024 and analyzed 18 models to address challenges in low-resolution images, occlusions, and varying lighting conditions. Practical experiments were conducted using simulated studio camera settings (Canon CJ14ex4.3) in 3D Max and Unreal Engine, evaluating various models for their accuracy, detail, realism, and efficiency.

During the final phases of our assessment, we identified and tested InstantMesh, a solution that effectively addresses the limitations of earlier models. InstantMesh emerged as the optimal choice due to its ability to manage multiviews, a challenge for many previous models that relied on single-image inputs or had limited capacity to handle diverse view angles.
InstantMesh integrates sparse-view large reconstruction models and multi-view diffusion models to ensure rapid processing and significantly enhance the detail and quality of 3D meshes. This dual approach makes InstantMesh faster and more accurate than other methods, ideal for our studio's fast-paced, quality-focused production environment.
Initial experiments showed that the enhanced InstantMesh model excels in edge detection, object recognition, and background removal. The modifications resulted in significant improvements in fidelity and flexibility, allowing for more detailed and customizable 3D reconstructions from multiple camera angles.



# 🚩 **Key Features**

- [x] Multi-View Reconstruction: Leveraging multiple camera views to create detailed and accurate 3D models.
- [x] Deep Learning Integration: Utilizing CNNs, GANs, and diffusion models for high-fidelity 3D reconstructions.
- [x] User Customization: Allowing user selection from generated angles to create customized 3D models.
- [x] Efficiency and Speed: Optimizing processing speed and resource usage for practical applications in virtual studio environments.


![diagram2 copy](https://github.com/sarshardorosti/MasterClass/assets/50841748/765baaca-da96-4483-8216-104a3f06b087)


# ⚙️ Dependencies and Installation

We recommend using `Python>=3.10`, `PyTorch>=2.1.0`, and `CUDA>=12.1`.
```bash
conda create --name instantmesh python=3.10
conda activate instantmesh
pip install -U pip

# Ensure Ninja is installed
conda install Ninja

# Install the correct version of CUDA
conda install cuda -c nvidia/label/cuda-12.1.0

# Install PyTorch and xformers
# You may need to install another xformers version if you use a different PyTorch version
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install xformers==0.0.22.post7

# For Linux users: Install Triton 
pip install triton

# For Windows users: Use the prebuilt version of Triton provided here:
pip install https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/triton-2.0.0-cp310-cp310-win_amd64.whl

# Install other requirements
pip install -r requirements.txt
```

# How to Use

## Download the models

We provide 4 sparse-view reconstruction model variants and a customized Zero123++ UNet for white-background image generation in the [model card](https://huggingface.co/TencentARC/InstantMesh).

Our inference script will download the models automatically. Alternatively, you can manually download the models and put them under the `ckpts/` directory.

By default, we use the `instant-mesh-large` reconstruction model variant.

## Running with command line

To generate 3D meshes from images via command line, simply run:
```bash
python run.py configs/instant-mesh-large.yaml examples/hatsune_miku.png --save_video
```

# :books: Citation

If you find our work useful for your research or applications, please cite using this BibTeX:

```BibTeX
@article{xu2024instantmesh,
  title={InstantMesh: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models},
  author={Xu, Jiale and Cheng, Weihao and Gao, Yiming and Wang, Xintao and Gao, Shenghua and Shan, Ying},
  journal={arXiv preprint arXiv:2404.07191},
  year={2024}
}
```

# 🤗 Acknowledgements

We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
- [Instant3D](https://instant-3d.github.io/)

Thank [@camenduru](https://github.com/camenduru) for implementing [Replicate Demo](https://replicate.com/camenduru/instantmesh) and [Colab Demo](https://colab.research.google.com/github/camenduru/InstantMesh-jupyter/blob/main/InstantMesh_jupyter.ipynb)!  
Thank [@jtydhr88](https://github.com/jtydhr88) for implementing [ComfyUI support](https://github.com/jtydhr88/ComfyUI-InstantMesh)!
