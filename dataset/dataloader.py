from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import torch
import h5py
import librosa
from utils.audio_utils import preprocess_audio


class MusicDataset(Dataset):
    def __init__(self, csv_path, h5_path, sr=22050, n_mels=128, transform=None):
        self.data = pd.read_csv(csv_path)
        self.h5_path = h5_path
        self.sr = sr
        self.n_mels = n_mels
        self.transform = transform

    def __len__(self):
        return len(self.data)
    

    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        class_name = row['class_name']
        dataset_name = row['dataset_name']
        index = int(row['index'])  # ensure int

        with h5py.File(self.h5_path, 'r') as f:
            audio = f['audio'][index]
            label = f['label'][index]

        # 오디오 데이터 전처리
        audio = preprocess_audio(audio)
        
        # 로그멜스펙트로그램 변환
        # audio가 이미 float32이고, 1D numpy array라고 가정
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        if self.transform:
            log_mel_spec = self.transform(log_mel_spec)

        # (채널, 시간, 주파수) 순서로 맞추기 (PyTorch는 channel-first)
        log_mel_spec = torch.tensor(log_mel_spec, dtype=torch.float32)
        log_mel_spec = log_mel_spec.unsqueeze(0)  # (1, n_mels, time)

        label = torch.tensor(label, dtype=torch.long)

        return log_mel_spec, label
    
    
        