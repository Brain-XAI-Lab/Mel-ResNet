import torch
from torchvision import models
from Preprocessing.temp_preprocessing import convert_mel_spect_librosa_only  # conv_mel_spect 함수는 임시
import numpy as np


def load_eeg_csv_data(file_path):
    """
    mne 라이브러리를 사용하여 CSV 파일에서 EEG 데이터를 로드합니다.
    :param file_path: EEG 데이터 파일 경로
    """
    raw_data = mne.io.read_raw_csv(file_path, preload=True)
    return raw_data.get_data()

def load_eeg_mff_data(file_path):
    # mne 라이브러리를 사용하여 .mff 파일 로드
    raw = mne.io.read_raw_egi(file_path, preload=True)
    data = raw.get_data()
    return data, raw.info


def load_model(model_path):
    """
    지정된 경로에서 커스텀 ResNet50 모델을 로드합니다.

    :param model_path: 모델 파일 경로
    :return: 모델 객체
    """
    # Load the ResNet50 model pre-trained on ImageNet
    model = models.resnet50(weights=False)

    # Modify the final fully connected layer to have 13 output classes
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 13)

    # Load the model state dictionary from the file
    # model.load_state_dict(torch.load(model_path))  # 모델 파일이 없어서 주석 처리

    return model


def predict(model, raw_data, sampling_rate, device):
    """
    새로운 데이터를 분류하는 함수

    :param model: 학습된 모델 객체
    :param raw_data: 분류할 데이터
    :param sampling_rate: 데이터의 샘플링 레이트(전처리)
    :param device: 사용할 디바이스 (CPU 또는 GPU)

    :return: 분류 결과
    """
    # 데이터 전처리
    # data = preprocess_audio(file_path)  # 전처리 함수 사용
    data = convert_mel_spect_librosa_only(raw_data, sampling_rate)

    # Mel-spectrogram 데이터 차원 맞추기 (채널 차원 추가)
    data = np.expand_dims(data, axis=0)
    # ResNet은 3채널 입력을 기대하므로, 데이터를 3채널로 복제
    data = np.repeat(data, 3, axis=0)

    # Tensor로 변환
    data = torch.tensor(data, dtype=torch.float32)
    data = data.unsqueeze(0)  # 배치 차원 추가

    # 모델을 평가 모드로 설정
    model.eval()
    model.to(device)

    with torch.no_grad():
        data = data.to(device)
        outputs = model(data)
        _, preds = torch.max(outputs, 1)

    # 라벨 맵핑 사전 정의
    label_map = {
        0: 'Call', 1: 'Camera', 2: 'Down', 3: 'Left', 4: 'Message', 5: 'Music', 6: 'Off',
        7: 'On', 8: 'Receive', 9: 'Right', 10: 'Turn', 11: 'Up', 12: 'Volume'
    }

    return label_map[preds.item()]
