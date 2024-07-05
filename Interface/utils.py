

def load_eeg_data(file_path):
    """
    mne 라이브러리를 사용하여 CSV 파일에서 EEG 데이터를 로드합니다.
    :param file_path: EEG 데이터 파일 경로
    """
    raw_data = mne.io.read_raw_csv(file_path, preload=True)
    return raw_data.get_data()

def load_model(model_path):
    """
    ResNet 모델을 로드합니다.
    :param model_path:
    :return:
    """
    model = ResNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model