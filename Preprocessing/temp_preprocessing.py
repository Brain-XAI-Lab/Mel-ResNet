import librosa
import numpy as np
import torchaudio
import os
import noisereduce as nr
import torch
from PIL import Image


def cut_audio_to_2_seconds(y, sr, duration=2):
    num_samples = int(duration * sr)
    if len(y) < num_samples:
        # 2초보다 짧으면 데이터 끝에 무음 패딩 추가
        padding = np.zeros(num_samples - len(y))
        y_cut = np.concatenate((y, padding))
    else: 
        y_cut = y[:num_samples]
    return y_cut


def resample_audio(y, orig_sr, target_sr=22050):
    y_resampled = torchaudio.functional.resample(torch.tensor(y), orig_sr, target_sr)
    return y_resampled.numpy()


def mel_spect_to_image(mel_spect):
    mel_spect = np.uint8(255 * (mel_spect - mel_spect.min()) / (mel_spect.max() - mel_spect.min()))
    image = Image.fromarray(mel_spect)
    image = image.convert("RGB")  # ResNet은 3채널 이미지를 기대하므로 RGB로 변환
    return image


def convert_mel_spect_librosa_only(data, sampling_rate):
    mel_spect = librosa.feature.melspectrogram(y=data, sr=sampling_rate)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    image = mel_spect_to_image(mel_spect)
    return image


# temp file (delete before commit)
def preprocess_audio_temp(file_path, n_fft, win_length, hop_length, n_mel_channels, mel_fmin, mel_fmax):
    y, sr = librosa.load(file_path, sr=None)
    y_cut = cut_audio_to_2_seconds(y, sr)
    y_resampled = resample_audio(y_cut, sr)
    y_nr = nr.reduce_noise(y=y, sr=sr)(y_resampled, 22050)  # noise reduction
    mel_spect = convert_mel_spect_librosa_only(y_nr, 22050)
    return mel_spect


"""
# 파라미터 설정
n_fft = 1024
win_length = 1024
hop_length = win_length // 4
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0


def convert_mel_spect(data, sampling_rate):
    mel_basis = librosa.filters.mel(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax).astype(np.float32)
    hann_window = torch.hann_window(win_length, dtype=torch.float32)

    p = (n_fft - hop_length) // 2
    data = torch.from_numpy(data).float()
    data = torch.nn.functional.pad(data, (p, p))
    spec = torch.stft(data, n_fft, hop_length=hop_length, win_length=win_length, window=hann_window, center=False, return_complex=True)

    magnitude = torch.abs(spec)
    mel_basis = torch.from_numpy(mel_basis).float()
    mel = torch.matmul(mel_basis, magnitude)
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel.numpy()
"""