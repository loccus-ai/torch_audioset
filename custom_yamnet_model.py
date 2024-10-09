import torch
import torchaudio
from nnAudio import features

from torch_audioset.yamnet.model import yamnet as torch_yamnet

    
@torch.jit.script
def reshape_mel_specgram(x: torch.Tensor):

    window_size_in_frames = int(round(0.96 / 0.010))

    patch_hop_in_frames = int(round(1.0 / 0.010))
    
    # TODO performance optimization with zero copy
    # patch_hop_num_chunks = (x.shape[0] - window_size_in_frames) // patch_hop_in_frames + 1
    patch_hop_num_chunks = (x.size(0) - window_size_in_frames) // patch_hop_in_frames + 1
    num_frames_to_use = window_size_in_frames + (patch_hop_num_chunks - 1) * patch_hop_in_frames
    x = x[:num_frames_to_use]
    x_in_frames = x.reshape(-1, x.size(-1))
            
    # # Modified: avoid for loop        
    # y = x_in_frames.unfold(0, window_size_in_frames, patch_hop_in_frames)
    # yt = torch.transpose(y, 1, 2)
    # x = yt.reshape(patch_hop_num_chunks, 1, window_size_in_frames, x.shape[-1])

    # Original
    x_output = torch.empty(patch_hop_num_chunks, window_size_in_frames, x.size(-1))
    for i in range(patch_hop_num_chunks):
        start_frame = patch_hop_in_frames * i
        x_output[i] = x_in_frames[start_frame: start_frame + window_size_in_frames]
        
    x = x_output.reshape(patch_hop_num_chunks, 1, window_size_in_frames, x.size(-1))

    return x

class ModelFrontend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Getting Mel Spectrogram on the fly
        self.spec_layer = features.STFT(n_fft=512, win_length=400, freq_bins=None,
                                           hop_length=160, window='hann',
                                           freq_scale='no', center=True,
                                           pad_mode='reflect', fmin=125,
                                           fmax=7500, sr=16000, trainable=False,
                                           output_format='Magnitude')

        self.mel_scale = torchaudio.transforms.MelScale(
            n_mels = 64, sample_rate = 16000, f_min = 125, f_max = 7500, n_stft=512 // 2 + 1, norm = None, 
            mel_scale = 'htk'
        )       
        
    def _get_mel_specgram(self,x):
        specgram = self.spec_layer(x)
        mel_specgram = self.mel_scale(specgram)
        mel_specgram = torch.log(mel_specgram + 0.001)
        x = mel_specgram.squeeze(dim=0).T 
        return reshape_mel_specgram(x=x)
    
 

    def forward(self, x):
        mel_specgram = self._get_mel_specgram(x)
        return mel_specgram
    
class ModelBackend(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.yamnet_model = torch_yamnet(pretrained=False)
        # Manually download the `yamnet.pth` file.
        self.yamnet_model.load_state_dict(torch.load('./yamnet.pth'))
            
    def forward(self, x):
        return self.yamnet_model(x, to_prob=False) #, spectrogram
    
    
class ModelE2E(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backend = ModelBackend()
        
        self.frontend = ModelFrontend()
       
    def forward(self, x):
        mel_specgram = self.frontend(x)
        pred = self.backend(mel_specgram)
        return pred
    
            