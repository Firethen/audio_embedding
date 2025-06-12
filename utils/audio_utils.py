import random
import torch

def preprocess_audio(wav, sr, audio_length):
    target_length = audio_length * sr
    wav_len = wav.shape[1]

    if wav_len < target_length:
        # ramdomly 시간격 앞뒤로 분배해서 pad
        pad_total = target_length - wav_len
        pad_left = random.randint(0, pad_total)
        pad_right = pad_total - pad_left
        wav = torch.nn.functional.pad(wav, (pad_left, pad_right))
    elif wav_len > target_length:
        # ramdomly 시간격 앞뒤로 분배해서 crop
        max_start = wav_len - target_length
        start = random.randint(0, max_start)
        wav = wav[:, start:start + target_length]
    
    return wav
    # wav: tensor[1, target_length]


def log_mel_spec_preprocess(log_mel_spec, time_frame):
    if log_mel_spec.shape[2] < time_frame:
        pad_total = time_frame - log_mel_spec.shape[2]
        pad_left = random.randint(0, pad_total)
        pad_right = pad_total - pad_left
        log_mel_spec = torch.nn.functional.pad(log_mel_spec, (pad_left, pad_right))
    elif log_mel_spec.shape[2] > time_frame:
        max_start = log_mel_spec.shape[2] - time_frame
        start = random.randint(0, max_start)
        log_mel_spec = log_mel_spec[:, :, start:start+time_frame]
    
    return log_mel_spec
    # log_mel_spec: tensor[1, n_mels, time_frame]