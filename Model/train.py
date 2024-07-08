import os
import time
import logging
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score
from Preprocessing.temp_preprocessing import preprocess_audio
from torchvision import models


# 데이터셋 클래스 정의
class SpeechDataset(Dataset):
    """
    음성 데이터셋 클래스. 디렉토리에서 음성 데이터를 로드하여 데이터셋을 생성합니다.

    Args:
        data_dir (str): 음성 데이터가 있는 디렉토리 경로
        transform (callable, optional): 데이터 변환 함수

    Attributes:
        data_files (list): 음성 데이터 파일 경로 리스트

    Methods:
        __len__(): 데이터셋 크기 반환
        __getitem__(idx): idx에 해당하는 데이터 반환
    """
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data_files = [os.path.join(root, file)
                           for root, _, files in os.walk(data_dir)
                           for file in files if file.endswith('.wav')]

        # 라벨 맵핑 사전 정의
        self.label_map = {
            'Call': 0, 'Camera': 1, 'Down': 2, 'Left': 3, 'Message': 4, 'Music': 5, 'Off': 6,
            'On': 7, 'Receive': 8, 'Right': 9, 'Turn': 10, 'Up': 11, 'Volume': 12
        }

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        file_path = self.data_files[idx]
        data = preprocess_audio(file_path)  # 전처리 함수 사용

        # Assuming label is the parent folder name
        label_name = os.path.basename(os.path.dirname(file_path))
        label = self.label_map[label_name]  # 라벨을 정수형으로 변환

        # Mel-spectrogram 데이터 차원 맞추기 (채널 차원 추가)
        data = np.expand_dims(data, axis=0)
        # ResNet은 3채널 입력을 기대하므로, 데이터를 3채널로 복제
        data = np.repeat(data, 3, axis=0)

        if self.transform:
            data = self.transform(data)

        return data, label


# ResNet 클래스 정의
class ResNet(nn.Module):
    """
    ResNet 모델 클래스. ResNet50을 사용합니다.
    """
    def __init__(self, num_classes=13):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)


# 데이터 로드 함수
def load_data(train_dir, test_dir, batch_size):
    """
    데이터를 로드하는 함수. 데이터셋을 생성하고 DataLoader로 반환합니다.
    :param train_dir: 학습 데이터 디렉토리 경로
    :param test_dir: 테스트 데이터 디렉토리 경로
    :param batch_size: 배치 크기
    :return: train_loader, test_loader
    """
    train_dataset = SpeechDataset(train_dir, transform=transforms.ToTensor())
    test_dataset = SpeechDataset(test_dir, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 성능 평가 지표 계산 함수
def calculate_metrics(labels, preds):
    """
    정확도와 F1 점수를 계산하는 함수
    :param labels: 실제 클래스(라벨)
    :param preds: 모델이 예측한 클래스
    :return: accuracy, f1
    """
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted')
    return accuracy, f1


# 모델 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs, device, early_stopping_patience):
    """
    모델을 학습하는 함수
    :param model: 학습할 모델
    :param train_loader: 학습 데이터 로더
    :param criterion: 손실 함수
    :param optimizer: 옵티마이저
    :param num_epochs: 에포크 수
    :param device: 학습 디바이스(cuda or cpu)
    :param early_stopping_patience: 조기 종료 조건
    """

    model.to(device)
    best_loss = float('inf')
    patience_counter = 0

    # 로깅 설정
    logging.basicConfig(filename='training.log', level=logging.INFO, force=True)
    logging.info('Training started at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}')

        # 조기 종료 검사
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            logging.info(f'Early stopping at epoch {epoch + 1}')
            break

    logging.info('Training ended at ' + time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


# 성능 평가 함수
def evaluate_model(model, test_loader, criterion, device):
    """
    모델을 평가하는 함수
    :param model: 평가할 모델
    :param test_loader: 테스트 데이터 로더
    :param criterion: 손실 함수
    :param device: 평가 디바이스(cuda or cpu)
    :return: 평가 결과
    """
    model.to(device)
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy, f1 = calculate_metrics(all_labels, all_preds)

    print(f'Test Loss: {avg_loss}')
    print(f'Test Accuracy: {accuracy}')
    print(f'Test F1 Score: {f1}')

    return avg_loss, accuracy, f1


# 모델 저장 함수
def save_model(model, path):
    """
    모델을 저장하는 함수
    :param model: 저장할 모델
    :param path: 저장 경로
    """
    torch.save(model.state_dict(), path)


# main 함수
def main():
    train_dir = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Train'
    test_dir = '/Users/imdohyeon/Library/CloudStorage/GoogleDrive-dhlim1598@gmail.com/공유 드라이브/4N_PKNU/BXAI/BMI/Mel-ResNet/Voice/Test'
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    model_path = '/Users/imdohyeon/Documents/PythonWorkspace/Mel-ResNet/Model/mel_resnet_001.pth'
    early_stopping_patience = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, test_loader = load_data(train_dir, test_dir, batch_size)
    model = ResNet(num_classes=13)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs, device, early_stopping_patience)
    evaluate_model(model, test_loader, criterion, device)
    save_model(model, model_path)


if __name__ == '__main__':
    main()
