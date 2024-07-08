"""
This file aimed to run the code in off-line environment. (on-line learning will be implemented in the future)

The main function is as follows:
  - Generate a window to traverse the EEG data sequence three times in total.
  - Upon reaching each window, pass the data from that section to the preprocessing code
    (the preprocessing code is yet to be developed, so denote it as Preprocessing(window_data) with appropriate comments),
    and return the Mel-spectrogram (EEG is saved as CSV and loaded with the mne library).
  - Use a trained ResNet model to classify the returned Mel-spectrogram and store the value returned from the model in a buffer.
  - After performing this process three times, print the words stored in the buffer at once.
"""


import numpy as np
import mne
import torch
import torch.nn as nn
import os
from utils import *
from Preprocessing.temp_preprocessing import preprocess_audio  # 전처리 함수를 정의한 파일에서 임포트


def main():
    # eeg_file = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/EEG'  # not completed yet
    # 테스트를 위해 일시적으로 EEG 데이터가 아닌 음성 데이터 사용
    eeg_file = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Raw'
    model_path = '/Users/imdohyeon/Documents/PythonWorkspace/Mel-ResNet/Model/'  # not completed yet

    # EEG 데이터 로드
    eeg_data = load_eeg_data(eeg_file)

    # 윈도우 크기 및 이동 간격 설정
    window_size = 1000  # 예시로 윈도우 크기를 설정
    step_size = 500  # 예시로 윈도우 이동 간격을 설정

    # 학습된 모델 로드
    model = load_model(model_path)

    # 버퍼 초기화
    buffer = []

    # EEG 데이터 시퀀스를 윈도우로 순회
    num_windows = (len(eeg_data) - window_size) // step_size + 1
    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        window_data = eeg_data[:, start:end]

        # 데이터 전처리 (구체적인 전처리 내용은 preprocessing.py에서 수행)
        mel_spectrogram = preprocess_window(window_data)

        # ResNet 모델을 사용하여 Mel-spectrogram 분류
        predicted_class = classify_spectrogram(model, mel_spectrogram)

        # 버퍼에 결과 저장
        buffer.append(predicted_class)

    # 최종 결과 출력
    print("Predicted classes for each window:", buffer)


if __name__ == '__main__':
    main()