# Restormer Model ONNX Export
Clone desired Github Repo. 
> `git clone https://github.com/swz30/Restormer.git` \
> `cd Restormer`

Paste the `restormer_export.py` file in the root of the above repo directory i.e. your-desired-path/Restormer.

1. Install dependencies in your virtual environment (used torch=1.12.1 version here instead)

> `pip install torch torchvision matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm`

> `pip install einops gdown addict future lmdb numpy pyyaml requests scipy tb-nightly yapf lpips`

2. Download the weights and configurations file for MelGAN architecture.
- [Deraining](https://drive.google.com/drive/folders/1ZEDDEVW0UgkpWi-N4Lj_JUoVChGXCu_u)
- [Motion Deblurring](https://drive.google.com/drive/folders/1czMyfRTQDX3j3ErByYeZ1PM4GVLbJeGK)
- [Defocus Deblurring](https://drive.google.com/drive/folders/1bRBG8DG_72AGA6-eRePvChlT5ZO4cwJ4?usp=sharing)
- [Denoising](https://drive.google.com/drive/folders/1Qwsjyny54RZWa7zC4Apg7exixLBo4uF0)

3. Run the export using below command
    > `python restormer_export.py --task Single_Image_Defocus_Deblurring --input_dir './demo/degraded/portrait.jpg' --result_dir './demo/restored/'`

Additional info -
