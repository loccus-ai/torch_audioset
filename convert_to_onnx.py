# -*- coding: utf-8 -*-
import argparse

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio

import openvino as ov
import onnxruntime

from nnAudio import features

from torch_audioset.data.torch_input_processing import WaveformToInput as TorchTransform
from torch_audioset.params import YAMNetParams
from torch_audioset.yamnet.model import yamnet as torch_yamnet
from torch_audioset.yamnet.model import yamnet_category_metadata


def sf_load_from_int16(fname):
    x, sr = sf.read(fname, dtype='int16', always_2d=True)
    x = x / 2 ** 15
    x = x.T.astype(np.float32)
    return x, sr


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
       
        self.yamnet_model = torch_yamnet(pretrained=False)
        # Manually download the `yamnet.pth` file.
        self.yamnet_model.load_state_dict(torch.load('./yamnet.pth'))

    def forward(self, x):
        return self.yamnet_model(x)


if __name__ == '__main__':
    # one wav file as argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('wav')
    args = parser.parse_args()

    wav_file = args.wav
    waveform_for_torch, sr = torchaudio.load(wav_file, normalize=True)

    patches, spectrogram = TorchTransform().wavform_to_log_mel(waveform_for_torch, 16000)

    # tt_model = TorchTransform()
    
    # patches, spectrogram = tt_model.forward(waveform_for_torch, 16000)

    pt_model = torch_yamnet(pretrained=False)
    # Manually download the `yamnet.pth` file.
    pt_model.load_state_dict(torch.load('./yamnet.pth'))

    with torch.no_grad():
    #     tt_model.eval()
    #     torch.onnx.export(tt_model,               # model being run
    #             (waveform_for_torch, 16000),                         # model input (or a tuple for multiple inputs)
    #             'specgram.onnx',            # where to save the model (can be a file or file-like object)
    #             verbose=True)
    #     # args = (waveform_for_torch,)
    #     # kwargs = {"sample_rate": 16000}
    #     # torch.onnx.dynamo_export(tt_model, *args, **kwargs).save('specgram.onnx')
        
        pt_model.eval()
    #     # x = patches
        pt_pred = pt_model(patches)
    #     # pt_pred = pt_pred.numpy()

        torch.onnx.export(pt_model,               # model being run
                patches,                         # model input (or a tuple for multiple inputs)
                'yamnet.onnx',            # where to save the model (can be a file or file-like object)
                verbose=False,
                input_names = ['input'],
                dynamic_axes={
                    'input' : {0 : 'chunk_length'},
                    }
            )

        ov_model = ov.convert_model('yamnet.onnx',
                                    verbose=True,
                                    share_weights=False)
                    
        ov.save_model(ov_model, 'yamnet.xml', compress_to_fp16=True)