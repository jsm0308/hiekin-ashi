import pandas as pd
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from src.model import BiGRUAttentionNet

# (이 스크립트는 Colab에서 실행하는 것을 권장합니다)

def run_training():
    # 1. 설정 및 데이터 로드
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    df = pd.read_csv(os.path.join(config['data']['processed_path'], config['data']['master_file']), parse_dates=['Date'])
    
    # 2. 데이터 시계열 분할 (Train / Validation / Test)
    train_df = df[df['Date'] <= config['training']['train_end_date']]
    val_df = df[(df['Date'] > config['training']['train_end_date']) & (df['Date'] <= config['training']['validation_end_date'])]
    test_df = df[df['Date'] > config['training']['validation_end_date']]
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")

    # 3. 피처/라벨 분리 및 정규화
    feature_cols = [col for col in df.columns if col not in ['Date', 'Label']]
    
    scaler = StandardScaler()
    train_df.loc[:, feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df.loc[:, feature_cols] = scaler.transform(val_df[feature_cols])
    test_df.loc[:, feature_cols] = scaler.transform(test_df[feature_cols])

    # 4. 시퀀스 데이터 생성 함수
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[feature_cols].iloc[i:i + window_size].values)
            y.append(data['Label'].iloc[i + window_size])
        return np.array(X), np.array(y)

    WINDOW_SIZE = config['data']['window_size']
    X_train, y_train = create_sequences(train_df, WINDOW_SIZE)
    X_val, y_val = create_sequences(val_df, WINDOW_SIZE)
    X_test, y_test = create_sequences(test_df, WINDOW_SIZE)

    # 5. PyTorch Tensor 및 DataLoader 생성
    BATCH_SIZE = config['training']['batch_size']
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=BATCH_SIZE)
    test_loader = DataLoader(TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=BATCH_SIZE)

    print("데이터 준비 완료. 이제 이 DataLoader들을 사용해 모델 학습을 진행할 수 있습니다.")
    # (실제 학습 로직은 이 아래에 추가됩니다. Early Stopping, loss 계산 등)

if __name__ == '__main__':
    run_training()