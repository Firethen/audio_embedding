from train import run_model
from torch.utils.data import DataLoader
import dataloader

def main():

    audio_path = '/home/namo/downloads/no_pain.wav'
    d_loader = DataLoader(dataloader.MusicDataset(audio_path), batch_size=16, shuffle=True)
    run_model(
        train_loader=d_loader,
        valid_loader=d_loader,
        device='cuda',
        num_epochs=100
    )

if __name__ == "__main__":
    main()

'''
데이터를 직접 받아서 dataloader로 wrap
(이 과정은 dataloader.py에 구현, 여기서 raw audio를 전처리까지 다함 (모델에 넣기 직전인 log mel spec을 만들어서 넣어줌))
(전처리, 데이터 증강등의 기법은 utils폴더에 구현하자)
이후 run_model
'''