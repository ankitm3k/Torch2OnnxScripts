# DeepSpeech Model ONNX Export
Clone desired Github Repo. 
> `git clone https://github.com/SeanNaren/deepspeech.pytorch.git` \
> `cd deepspeech.pytorch`

Paste the `deepspeech_export.py` file in the root of the above repo directory i.e. your-desired-path/deepspeech.pytorch.

1. Install dependencies in your virtual environment (used torch=1.12.1 version here instead)

> `pip install -r requirements.txt`


2. Download the pretrained pytorch model and paste it in your models location.
- [ted_pretrained_v2.pth](https://github.com/SeanNaren/deepspeech.pytorch/releases)

3. Run the export using below command
> `python deepspeech_export.py`

Additional info -

This repo was cited from https://github.com/mlcommons/training.git

DISCLAIMER: The weights used here might not be useful for gauging the model accuracy, this exercise is just to make sure that model gets converted to ONNX format successfully.