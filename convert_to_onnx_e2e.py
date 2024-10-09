# -*- coding: utf-8 -*-
import argparse

import torch
import torchaudio

import openvino as ov

import custom_yamnet_model

if __name__ == '__main__':
    # one wav file as argument.
    parser = argparse.ArgumentParser()
    parser.add_argument('wav')
    args = parser.parse_args()

    waveform_for_torch, sr = torchaudio.load(args.wav, normalize=True)

    pt_model = custom_yamnet_model.ModelE2E()

    with torch.no_grad():
        pt_model.eval()

        torch.onnx.export(pt_model,               # model being run
                waveform_for_torch,                         # model input (or a tuple for multiple inputs)
                'yamnet_e2e.onnx',            # where to save the model (can be a file or file-like object)
                verbose=False,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input' : {1 : 'input_length'},
                              'output' : {0: 'output_length'}})

        ov_model = ov.convert_model('yamnet_e2e.onnx',
                                    verbose=True,
                                    share_weights=False,
                                    input=[1, -1])
                    
        ov.save_model(ov_model, 'yamnet_e2e.xml', compress_to_fp16=True)