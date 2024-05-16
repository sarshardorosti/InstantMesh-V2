<div align="center">
  
# InstantMesh V2: Efficient 3D Mesh Generation from a Single Image with Sparse-view Large Reconstruction Models

</div>
---


Machine Learning Techniques for Enhanced 3D Reconstruction from Multi-Camera Studio Views
Project Overview
This project explores the use of machine learning (ML) for 3D reconstruction of studio sets from three or more camera views. It was conducted as part of the MSc in Artificial Intelligence for Media at Bournemouth University. 
The research investigates advanced methods for converting 2D images into 3D models, emphasizing deep learning technologies like CNNs and diffusion models. This project aims to overcome the challenges by combining techniques to produce high-quality 3D models from multiple camera views, particularly in studio environments. We reviewed over 80 papers from 2014 to 2024 and analyzed 18 models to address challenges in low-resolution images, occlusions, and varying lighting conditions. Practical experiments were conducted using simulated studio camera settings (Canon CJ14ex4.3) in 3D Max and Unreal Engine, evaluating various models for their accuracy, detail, realism, and efficiency.

During the final phases of our assessment, we identified and tested InstantMesh, a solution that effectively addresses the limitations of earlier models. InstantMesh emerged as the optimal choice due to its ability to manage multiviews—a challenge for many previous models that relied on single-image inputs or had limited capacity to handle diverse view angles. InstantMesh integrates sparse-view large reconstruction models and multi-view diffusion models to ensure rapid processing and significantly enhance the detail and quality of 3D meshes. This dual approach makes InstantMesh faster and more accurate than other methods, making it ideal for our studio's fast-paced, quality-focused production environment. Initial experiments demonstrated that the enhanced InstantMesh model excels in edge detection, object recognition, and background removal, resulting in significant improvements in fidelity and flexibility. This allows for more detailed and customizable 3D reconstructions from multiple camera angles.

Identified Limitations and Proposed Enhancements
Despite its robustness, InstantMesh does exhibit some limitations, particularly regarding the number of input views it can process simultaneously and the degree of user interaction it allows. To further tailor InstantMesh to meet our specific needs and push the boundaries of its capabilities, we propose an innovative modification to the model. This enhancement involves configuring InstantMesh to accept three images instead of one, thereby enriching the input data and potentially enhancing the depth and accuracy of the multiview reconstructions. Furthermore, this modified version will allow users to select six new angles from various generated angles, offering greater flexibility and user control in the final 3D model creation.

Impact on Virtual Productions
We hope to produce a flexible and user-friendly model by putting these modifications into practice that not only satisfies but also surpasses the dynamic needs of Mo-Sys studio projects. This development ensures that we can make accurate predictions about the placement of cameras relative to objects, as well as the positioning of objects within a virtual scene in our virtual studio productions. By resolving the limitations of the studied model, we can ensure more precise and reliable outcomes in our productions.


# 🚩 **Contribution**
We have successfully integrated several advanced features into our project:

- [x] Multi-View Reconstruction: Leveraging multiple camera views to create detailed and accurate 3D models.
- [x] Deep Learning Integration: Utilizing CNNs, GANs, and diffusion models for high-fidelity 3D reconstructions.
- [x] User Customization: Allowing user selection from generated angles to create customized 3D models.
- [x] Efficiency and Speed: Optimizing processing speed and resource usage for practical applications in virtual studio environments.

![diagram2 copy](https://github.com/sarshardorosti/MasterClass/assets/50841748/765baaca-da96-4483-8216-104a3f06b087)

## InstantMesh V2 Architecture
![1111111](https://github.com/sarshardorosti/MasterClass/assets/50841748/6cd9de02-08f5-4570-8f81-415b3ea6c75f)

It is also worth mentioning that with further investigation, we were able to run the InstantMesh code in ComfyUI and in future updates, we plan to connect it to Unreal Engine for real-time output.
<a href="https://github.com/jtydhr88/ComfyUI-InstantMesh"><img src="https://img.shields.io/badge/Demo-ComfyUI-8A2BE2"></a>

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
python run.py configs/instant-mesh-large.yaml examples/ --save_video
```

Please place the required images as input in the examples folder. 

After running the code from the command line, the newly generated angles will be created in the multiviews directory within images.

You will then be prompted to sequentially enter the numbers of 6 desired angles in the command line.

A 3*2 multiview image with a .png extension will be produced in the images folder, and using this image, a 3D mesh model will be created in the meshes folder.


## Repositorymap

copy instant_mesh_base.ckpt to the ckpt folder

copy instant-mesh-base.yaml to the configs folder

```
instantmesh
│
├── app.py
├── run.py
├── train.py
├── assets
├── ckpts
├── configs
├── docker
├── examples
│   ├── 1.jpg
│   ├── 2.jpg
│   └── 3.jpg
├── outputs
│   └── instant-mesh-base
│       ├── images
│       │   └── multiviews
│       ├── meshes
│       └── videos
├── src
└── zero123plus
```

## Resource Challenges and Solutions
During development, we faced significant challenges related to the high resource demands of the code, particularly RAM, CPU, and GPU utilization. Initial tests on a personal laptop (16GB RAM, NVIDIA RTX 3060 6GB GPU, AMD Ryzen 9 6900HX CPU) revealed that the system quickly reached full capacity, halting the process after generating only a few views. Subsequent tests on a more powerful setup at the university (13th Gen Intel Core™ i9-13700X, NVIDIA GeForce RTX 4080 16GB GDDR6 GPU, 64GB DDR5 7800MHz RAM, RedHat Enterprise 64bit) encountered similar issues, necessitating further code modifications.

To optimize the process, we consolidated the generation of initial images into a single 'for' loop, which marginally increased processing speed and significantly reduced RAM and GPU requirements. Additionally, we cleared the pipeline after combining images to further minimize memory usage. Despite modifying mesh resolution, input image size, and generated multiview image dimensions, these changes did not yield a notable impact, leading to a decision to revert to the original settings.

Model Adjustments and Performance
We transitioned from the 'mesh_large' model to 'mesh_base'. Comparative analysis of the outputs derived from 'mesh_large' accessed online showed no significant difference compared to our test image. This indicated that the impact of combining selected angles, added to our code, could approximate the 'large' model results while using the 'base' model. Our experiences throughout the project helped us increase implementation speed and systematically improve quality, enabling additional code modifications for processing on personal systems.

## Code Implementation
Code Initialization and Configuration
A conditional statement was used to determine whether to create a rembg session based on user input (args.no_rembg). This decision allowed for optional removal of backgrounds from input images, providing flexibility in preprocessing based on project requirements. When enabled, a new rembg session was initialized.

Directory and List Setup
Directories and lists were established to manage outputs, images, and tensors. A specific directory was created for saving multiview images, ensuring structured storage for results, crucial for maintaining workflow and preventing data loss.

Image Preprocessing and Multiview Generation
Each input image was processed sequentially, with filenames extracted for later reference. If background removal was activated, the background was removed using the rembg session, and the foreground resized for consistent scaling across all images. This preprocessing ensured uniformity and quality in the generated multiview images.

The preprocessed image was passed through a diffusion pipeline to generate an output image, which was converted into a NumPy array, normalized, and transformed into a PyTorch tensor. The tensor was rearranged to represent six distinct views of the image, simulating a multiview scenario. Each view was saved individually to the designated multiview directory, creating a comprehensive set of perspectives for further analysis and processing.

User Interaction and Image Selection
Post-generation, the multiview images were compiled into a list, and users were prompted to select specific views from the outputs. The selected views were concatenated to form a combined image, saved to the specified output path, enhancing the visual comprehensiveness of the generated content.

Memory Management and Tensor Conversion
To optimize memory usage, the diffusion pipeline was deleted after processing images. The saved multiview images were loaded and converted back into PyTorch tensors, rearranged to maintain the multiview format, and added to the output list, ensuring all processed data was correctly formatted for the next stages of reconstruction.

Enhancements and User Flexibility
We implemented significant modifications to better suit our specific needs. The modified system now allows the input of any number of images, increasing the depth and accuracy of multiview reconstructions. Users can select six new generated angles, enabling the creation of new 3D models from selected views. These changes aim to make the system more versatile and user-friendly, allowing greater customization and improving the overall quality of 3D models, aligning closely with the dynamic and evolving demands of Mo-Sys Studio.


## Describe



![Layout](https://github.com/sarshardorosti/MasterClass/assets/50841748/e16a716a-e05e-46ed-82bd-eb4957d89a39)


Enhancements and Objectives By implementing these changes, we aim to create a versatile and user-friendly model that not only meets but exceeds the dynamic requirements of Mo-Sys studio projects. This next step leverages the strengths of InstantMesh while addressing its minor shortcomings. 




# 🤗 Acknowledgements


We thank the authors of the following projects for their excellent contributions to 3D generative AI!

- [Zero123++](https://github.com/SUDO-AI-3D/zero123plus)
- [OpenLRM](https://github.com/3DTopia/OpenLRM)
- [FlexiCubes](https://github.com/nv-tlabs/FlexiCubes)
- [Instant3D](https://instant-3d.github.io/)
Thank [@bluestyle97](https://github.com/bluestyle97) for implementing https://github.com/TencentARC/InstantMesh
Thank [@camenduru](https://github.com/camenduru) for implementing [Replicate Demo](https://replicate.com/camenduru/instantmesh) and [Colab Demo](https://colab.research.google.com/github/camenduru/InstantMesh-jupyter/blob/main/InstantMesh_jupyter.ipynb)!  
Thank [@jtydhr88](https://github.com/jtydhr88) for implementing [ComfyUI support](https://github.com/jtydhr88/ComfyUI-InstantMesh)!
