import librosa
import numpy as np


def load_data(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


def cut_audio_to_2_seconds(audio_path, output_path, duration=2):
    # 오디오 파일 로드
    y, sr = librosa.load(audio_path, sr=None)

    # 필요한 샘플 수 계산 (duration 초 동안의 샘플 수)
    num_samples = int(duration * sr)

    # 오디오 길이가 자르려는 길이보다 짧으면 예외 처리
    if len(y) < num_samples:
        raise ValueError("The audio file is shorter than the desired cut length.")

    # 오디오 파일 자르기
    y_cut = y[:num_samples]

    return y_cut


def convert_mel_spect(data, sampling_rate):
    mel_spect = librosa.feature.melspectrogram(data, sr=sampling_rate)
    mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
    return mel_spect


def preprocess_audio(file_path):
    y, sr = load_data(file_path)
    y_cut = cut_audio_to_2_seconds(y, sr)
    mel_spect = convert_mel_spect(y_cut, sr)
    return mel_spect

# test