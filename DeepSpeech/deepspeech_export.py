import argparse

import torch

from deepspeech_pytorch.loader.data_loader import SpectrogramParser
from deepspeech_pytorch.model import DeepSpeech
from deepspeech_pytorch.configs.train_config import SpectConfig, AdamConfig, UniDirectionalConfig

parser = argparse.ArgumentParser(description='DeepSpeech ONNX Export')
parser.add_argument('--save_folder', default='models/', help='Location to save epoch models')
parser.add_argument('--model_path', default='models/ted_pretrained_v2.pth',
                    help='Location to save best validation model')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

spect_parser = SpectrogramParser(
        audio_conf= SpectConfig,
        normalize=True
    )

model = DeepSpeech(
        labels=['_',],
        model_cfg=UniDirectionalConfig,
        optim_cfg=AdamConfig,
        precision=32,
        spect_cfg=SpectConfig)


checkpoint = torch.load(args.model_path,map_location=torch.device('cpu'))

checkpoint['state_dict']['fc.0.module.1.weight'] = torch.rand([1,1024])

model.load_state_dict(checkpoint['state_dict'], strict= False)
model.eval()

audio_path ="./models/preamble.wav"
spect = spect_parser.parse_audio(audio_path).contiguous()
spect = spect.view(1, 1, spect.size(0), spect.size(1))
spect = spect.to(device)
input_sizes = torch.IntTensor([spect.size(3)]).int()

torch.onnx.export(model,(spect, input_sizes),args.save_folder + "deepspeech.onnx",opset_version=13)