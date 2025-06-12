from dataclasses import dataclass
from typing import List, Optional, Callable

'''
- datapath를 리스트로 만들어서 넣어주기(각 wav파일마다)
- 해당 라벨도 정해놔야 함
'''


@dataclass
class DataArgs:
    audio_path: List[str] = None
    sr: int = 16000 #sampling rate
    
    n_mels: int = 128 #number of mel bins
    time_frame: int = 128 #time frame
    n_fft: int = 1024 
    hop_length: int = 512 
    window_length: int = 1024 

    transform: Optional[Callable] = None
    audio_length: int = 5 #audio length: sec


@dataclass
class ModelArgs:
    model_name: str
    hidden_dim: int
    output_dim: int