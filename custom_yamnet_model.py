import torch
import torchaudio
from nnAudio import features

from torch_audioset.yamnet.model import yamnet as torch_yamnet

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
        return x, spectrogram

    def forward(self, x):
        mel_specgram, spectrogram = self._get_mel_specgram(x)
        return self.yamnet_model(mel_specgram), spectrogram