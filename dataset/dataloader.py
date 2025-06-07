from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import h5py
import librosa
#from utils.audio_utils import preprocess_audio


class MusicDataset(Dataset):
    def __init__(self, audio_path, sr=16000, n_mels=128, transform=None):
        self.data = audio_path
        self.sr = sr
        self.n_mels = n_mels
        self.transform = transform #증강

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # 오디오 데이터 전처리
        #audio = preprocess_audio(audio)
        audio, _ = librosa.load(self.data, sr=self.sr)
        # 로그멜스펙트로그램 변환
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=128
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # (128, time) -> (128, 128)로 맞추기 (center crop or pad)
        if log_mel_spec.shape[1] < 128:
            # pad
            pad_width = 128 - log_mel_spec.shape[1]
            log_mel_spec = np.pad(log_mel_spec, ((0,0),(0,pad_width)), mode='constant')
        elif log_mel_spec.shape[1] > 128:
            # center crop
            start = (log_mel_spec.shape[1] - 128) // 2
            log_mel_spec = log_mel_spec[:, start:start+128]

        if self.transform:
            log_mel_spec = self.transform(log_mel_spec)

        log_mel_spec = torch.tensor(log_mel_spec, dtype=torch.float32)
        log_mel_spec = log_mel_spec.unsqueeze(0)  # (1, 128, 128)

        label = torch.tensor(torch.randint(0, 10, (1,)))

        return log_mel_spec, label
    
    
        