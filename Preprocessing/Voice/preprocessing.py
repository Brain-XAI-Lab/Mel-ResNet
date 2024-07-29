import librosa
import numpy as np
import torchaudio
import os
import noisereduce as nr
import torch
from PIL import Image
import matplotlib.pyplot as plt


def load_data(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def cut_audio_to_2_seconds(y, sr, duration=2):
    num_samples = int(duration * sr)
    if len(y) < num_samples:
        # 2초보다 짧으면 데이터 끝에 무음 패딩 추가
        padding = np.zeros(num_samples - len(y))
        y_cut = np.concatenate((y, padding))
    else: 
        y_cut = y[:num_samples]
    return y_cut


def convert_mel_spect(data, sampling_rate):
    mel_spect = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect


def mel_spect_to_image(mel_spect, cmap='viridis'):
    # 멜 스펙트로그램을 0에서 1 사이로 정규화
    mel_spect_norm = (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min())
    # 색상 맵을 적용하여 RGB 데이터 생성
    color_mapped = plt.get_cmap(cmap)(mel_spect_norm)[:, :, :3]  # 알파 채널 제거
    color_mapped = np.uint8(255 * color_mapped)  # 0-255 범위로 스케일링
    image = Image.fromarray(color_mapped)  # PIL 이미지로 변환
    image = image.convert("RGB")  # ResNet은 3채널 이미지를 기대하므로 RGB로 변환
    return image


def preprocess_audio(file_path):
    y, sr = load_data(file_path)
    y_cut = cut_audio_to_2_seconds(y, sr)
    mel_spect = convert_mel_spect(y_cut, 44100)
    mel_img = mel_spect_to_image(mel_spect)

    mel_tensor = torch.tensor(mel_img, dtype=torch.float32)
    mel_tensor = mel_tensor.unsqueeze(0)  # 배치 차원 추가
    return mel_tensor
