# CenterNet Model ONNX Export

Clone desired Github Repo. 
> `git clone https://github.com/xingyizhou/CenterNet.git`
> `cd CenterNet`

Paste the `centernet_export.py` file in the root of the above repo directory i.e. <your-desired-path>/CenterNet/.

1. Install dependencies in your virtual environment (used torch=1.12.1 version here instead)

> `pip install -r requirements.txt`

2. Disable `from .DCNv2.dcn_v2 import DCN` from src\lib\models\networks\pose_dla_dcn.py and src\lib\models\networks\resnet_dcn.py

3. Download the weights and configurations file for MelGAN architecture.
- multi_pose_dla_3x.pth
-- https://drive.google.com/file/d/1PO1Ax_GDtjiemEmDVD7oPWwqQkUu28PI/view

4. Run the export using below command
> `python centernet_export.py --torch_model_path <user-path>/multi_pose_dla_3x.pth`

Additional info -
DCNv2 operator is not supported in pytorch and it requires Nvidia GPU for  build running.