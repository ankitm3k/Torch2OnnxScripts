# TransformerXL Model ONNX Export
Clone desired Github Repo. 
> `git clone https://github.com/mozilla/TTS.git`
> `cd TTS`

Paste the `tts_export.py` file in the root of the above repo directory i.e. <your-desired-path>/TTS/.

1. Install dependencies in your virtual environment (used torch=1.12.1 version here instead)

> `pip install TTS`

> `pip install -r requirements.txt`

2. Download the weights and configurations file for MelGAN architecture.
- vocoder_model.pth.tar
-- https://drive.google.com/uc?id=1Ty5DZdOc0F7OTGj9oJThYbL5iVu_2G0K
- config_vocoder.json
-- https://drive.google.com/uc?id=1Rd0R_nRCrbjEdpOwq6XwZAktvugiBvmu

3. Run the export using below command
> `python tts_export.py --config_path <user-path>/vocoder_config.json --torch_model_path <user-path>/vocoder_model.pth.tar`

Additional info -
- MelGAN: [paper](https://arxiv.org/abs/1910.06711)