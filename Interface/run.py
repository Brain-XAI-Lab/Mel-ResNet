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

from utils import *
import librosa


def main():
    eeg_file = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Raw/Call/call1.wav'
    model_path = '/Users/imdohyeon/Documents/PythonWorkspace/Mel-ResNet/Model/model_sample.pth'  # not completed yet

    # EEG 데이터 및 모델 로드 (EEG preprocessing 완성 전까지는 wav 파일로 대체)
    sampling_rate = 44100
    data = librosa.load(eeg_file, sr=sampling_rate)
    model = load_model(model_path)

    # 버퍼 초기화
    output_buffer = []
    data_length = len(data)

    # EEG 데이터 시퀀스를 윈도우로 순회하며 분류
    window_size = int(data_length * 0.4)
    step_size = int((data_length - window_size) / 2)
    start = 0

    # 윈도우 순회 시작
    for _ in range(3):
        end = start + window_size
        sliced_data = np.array(data[start:end])

        # 전처리 및 예측
        output = predict(model, sliced_data, sampling_rate, device='cpu')

        # 결과값 버퍼에 저장 후 다음 윈도우로
        output_buffer.append(output)
        start += step_size

    # 최종 결과 출력
    print("Predicted classes for each window:", output_buffer)


if __name__ == '__main__':
    main()