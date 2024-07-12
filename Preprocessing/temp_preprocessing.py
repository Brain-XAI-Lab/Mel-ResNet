import librosa
import numpy as np
import torchaudio
import os
import noisereduce as nr
import torch

# 파라미터 설정
n_fft = 1024
win_length = 1024
hop_length = win_length // 4
n_mel_channels = 80
mel_fmin = 0.0
mel_fmax = 8000.0

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

def resample_audio(y, orig_sr, target_sr=22050):
    y_resampled = torchaudio.functional.resample(torch.tensor(y), orig_sr, target_sr)
    return y_resampled.numpy()

def reduce_noise(y, sr):
    y_nr = nr.reduce_noise(y=y, sr=sr)
    return y_nr

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

def save_mel_spect(mel_spect, save_path):
    np.save(save_path, mel_spect)

def preprocess_audio(file_path, save_path):
    y, sr = load_data(file_path)
    y_cut = cut_audio_to_2_seconds(y, sr)
    y_resampled = resample_audio(y_cut, sr)
    y_nr = reduce_noise(y_resampled, 22050)
    mel_spect = convert_mel_spect(y_nr, 22050)
    save_mel_spect(mel_spect, save_path)

def preprocess_and_save(base_path, save_path, sr=22050):
    classes = os.listdir(base_path)
    for class_dir in classes:
        class_path = os.path.join(base_path, class_dir)
        save_class_path = os.path.join(save_path, class_dir)
        os.makedirs(save_class_path, exist_ok=True)
        
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                if filename.endswith('.wav'):
                    file_path = os.path.join(class_path, filename)
                    try:
                        mel_save_path = os.path.join(save_class_path, filename.replace('.wav', '.npy'))
                        preprocess_audio(file_path, mel_save_path)
                    except ValueError as e:
                        print(f"Error processing {file_path}: {e}")

augment_base_path = 'G:/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Augmented'
preprocessed_save_path = "C:/Users/user/Desktop/0708mel_aug"
preprocess_and_save(augment_base_path, preprocessed_save_path)
