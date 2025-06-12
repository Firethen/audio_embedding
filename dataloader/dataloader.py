from torch.utils.data import Dataset
import torch
import torchaudio

from utils.audio_utils import preprocess_audio,log_mel_spec_preprocess
from arguments import DataArgs

'''
1. mp3파일 그대로 입력
    1.1 I/O 소요가 크다면, h5와 같은 포맷으로 변환해서 만들어 놓고, 이를 입력으로 받아, 데이터로더 안에서 unpack후 사용
2. mp3 -> wav로 변환(decoding)
3. wav 전처리(pad or crop)
4. 로그멜스펙트로그램 변환
5. CNN 입력 형식으로 맞추기 (1, n_mels, time_frame)
6. label과 함께 입력으로 넣어주기

- 데이터 증강
'''


class MusicDataset(Dataset):
    def __init__(self, args: DataArgs):
        self.data_path = args.audio_path
        self.sr = args.sr
        self.n_mels = args.n_mels
        self.time_frame = args.time_frame
        self.n_fft = args.n_fft
        self.hop_length = args.hop_length
        self.window_length = args.window_length
        self.transform = args.transform
        self.audio_length = args.audio_length

        #로그 멜스펙트로그램 변환 용도 init
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window_length=self.window_length
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80)

    def __len__(self):
        return len(self.data_path)
    
    def __getitem__(self, idx):

        #1,2. mp3 입력, wav 변환
        wav, sr = torchaudio.load(self.data_path[idx]) # audio: tensor[channel, n_samples], sr: int
        if wav.shape[0] == 2: # if stereo: -> mono
            wav = torch.mean(wav, dim=0)
        
        #3. wav 전처리
        wav = preprocess_audio(wav, self.sr, self.audio_length)
        # wav: tensor[1, target_length]

        #4. 로그멜스펙트로그램 변환
        mel_spec = self.mel_transform(wav)  # (1, n_mels, time(n_fft와 hop_length에 따라 결정됨))
        log_mel_spec = self.db_transform(mel_spec)  # (1, n_mels, time)

        #5. CNN 입력 형식으로 맞추기 -> (1, n_mels, time_frame)
        log_mel_spec = log_mel_spec_preprocess(log_mel_spec, self.time_frame)

        #6. label과 함께 입력으로 넣어주기
        label = torch.tensor(torch.randint(0, 10, (1,))) # 임시 label

        return log_mel_spec, label
    
    
        