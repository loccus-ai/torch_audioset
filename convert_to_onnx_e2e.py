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
        self.epsilon=1e-10
        # Getting Mel Spectrogram on the fly
        self.spec_layer = features.STFT(n_fft=512, win_length=400, freq_bins=None,
                                           hop_length=160, window='hann',
                                           freq_scale='no', center=True,
                                           pad_mode='reflect', fmin=125,
                                           fmax=7500, sr=16000, trainable=False,
                                           output_format='Magnitude')
        # self.mel_spec_layer = features.mel.MelSpectrogram(sr=16000, n_fft=512, win_length=400, n_mels=64,
        #                                                   hop_length=160, window='hann',
        #                                                   center=True, pad_mode='reflect',
        #                                                   power = 2.0, htk = True,
        #                                                   fmin = 125, fmax = 7500, norm = None)
        
        
        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels = 64, sample_rate = 16000, f_min = 125, f_max = 7500, n_stft=512 // 2 + 1, norm = None, mel_scale = 'htk'
        )
        self.yamnet_model = torch_yamnet(pretrained=False)
        # Manually download the `yamnet.pth` file.
        self.yamnet_model.load_state_dict(torch.load('./yamnet.pth'))
        
    def _get_mel_specgram(self,x):
        specgram = self.spec_layer(x)
        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + 0.001)
        x = mel_specgram.squeeze(dim=0).T 
        spectrogram = x

        window_size_in_frames = int(round(
            0.96 / 0.010
        ))
        # num_chunks = x.shape[0] // window_size_in_frames

        # reshape into chunks of non-overlapping sliding window
        # num_frames_to_use = num_chunks * window_size_in_frames
        # x = x[:num_frames_to_use]
        # # [num_chunks, 1, window_size, num_freq]
        # x = x.reshape(num_chunks, 1, window_size_in_frames, x.shape[-1])

        patch_hop_in_frames = int(round(
            1.0 / 0.010
        ))
        # TODO performance optimization with zero copy
        patch_hop_num_chunks = (x.shape[0] - window_size_in_frames) // patch_hop_in_frames + 1
        num_frames_to_use = window_size_in_frames + (patch_hop_num_chunks - 1) * patch_hop_in_frames
        x = x[:num_frames_to_use]
        x_in_frames = x.reshape(-1, x.shape[-1])
        x_output = torch.empty((patch_hop_num_chunks, window_size_in_frames, x.shape[-1]))
        for i in range(patch_hop_num_chunks):
            start_frame = i * patch_hop_in_frames
            x_output[i] = x_in_frames[start_frame: start_frame + window_size_in_frames]
        x = x_output.reshape(patch_hop_num_chunks, 1, window_size_in_frames, x.shape[-1])
        # x = torch.tensor(x, dtype=torch.float32)

        # z = self.mel_spec_layer(x)
        return x, specgram

    def forward(self, x):
        mel_specgram, spectrogram = self._get_mel_specgram(x)
        return self.yamnet_model(mel_specgram), spectrogram


if __name__ == '__main__':
    # one wav file as argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('wav')
    args = parser.parse_args()

    wav_file = args.wav
    waveform_for_torch, sr = torchaudio.load(wav_file, normalize=True)

    # tt_model = TorchTransform()
    
    # patches, spectrogram = tt_model.forward(waveform_for_torch, 16000)

    # pt_model = torch_yamnet(pretrained=False)
    # # Manually download the `yamnet.pth` file.
    # pt_model.load_state_dict(torch.load('./yamnet.pth'))

    pt_model = Model()

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
        pt_pred, _ = pt_model(waveform_for_torch)
    #     # pt_pred = pt_pred.numpy()

        torch.onnx.export(pt_model,               # model being run
                waveform_for_torch,                         # model input (or a tuple for multiple inputs)
                'yamnet_e2e.onnx',            # where to save the model (can be a file or file-like object)
                verbose=False)

        ov_model = ov.convert_model('yamnet_e2e.onnx',
                                    verbose=True,
                                    share_weights=False,
                                    input=[1, -1])
                    
        ov.save_model(ov_model, 'yamnet_e2e.xml', compress_to_fp16=True)