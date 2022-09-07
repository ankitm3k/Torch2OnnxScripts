import argparse
import os
import numpy as np
import torch

from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import setup_generator

# prevent GPU use
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# define args
parser = argparse.ArgumentParser()
parser.add_argument('--torch_model_path',
                    type=str,
                    help='Path to target torch model to be converted to ONNX')
parser.add_argument('--config_path',
                    type=str,
                    help='Path to config file of torch model.')

args = parser.parse_args()

# load model config
config_path = args.config_path
c = load_config(config_path)
num_speakers = 0

# init torch model (MelGAN architecture)
model = setup_generator(c)
checkpoint = torch.load(args.torch_model_path,
                        map_location=torch.device('cpu'))
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.remove_weight_norm()
state_dict = model.state_dict()

model.eval()
dummy_input_torch = torch.ones((1, 80, 10))

torch.onnx.export(model, dummy_input_torch,"tts_melgan.onnx", verbose=False)
