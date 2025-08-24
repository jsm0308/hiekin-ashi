import optuna
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import yaml
import os
import sys

# 프로젝트 루트 경로를 시스템 경로에 추가
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model import BiGRUAttentionNet
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

# --- 전역 변수 및 헬퍼 함수 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_sequences(features, labels, window_size):
    X, y = [], []
    for i in range(len(features) - window_size):
        X.append(features[i:i + window_size])
        y.append(labels[i + window_size])
    return np.array(X), np.array(y)

# --- Optuna Objective 함수 ---
def objective(trial):
    # 1. 설정 로드
    with open('config.yaml', 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    target_ticker = config['data']['optimization_target']
    filename = f"master_features_{target_ticker}.csv"
    filepath = os.path.join(config['data']['processed_path'], filename)
    df = pd.read_csv(filepath, parse_dates=['Date'])
    # 2. 데이터 분할 및 전처리
    train_df = df[df['Date'] <= config['training']['train_end_date']]
    val_df = df[(df['Date'] > config['training']['train_end_date']) & (df['Date'] <= config['training']['validation_end_date'])]
    
    feature_cols = [col for col in df.columns if col not in ['Date', 'Label']]
    scaler = StandardScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])

    # 3. 하이퍼파라미터 탐색 공간 정의
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        'hidden_size': trial.suggest_categorical('hidden_size', [32, 64, 128, 256]),
        'num_layers': trial.suggest_int('num_layers', 1, 3),
        'dropout': trial.suggest_float('dropout', 0.1, 0.7),
    }

    # 4. 모델 및 데이터 로더 생성
    WINDOW_SIZE = config['data']['window_size']
    X_train, y_train = create_sequences(train_df[feature_cols].values, train_df['Label'].values, WINDOW_SIZE)
    X_val, y_val = create_sequences(val_df[feature_cols].values, val_df['Label'].values, WINDOW_SIZE)

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)), batch_size=config['training']['batch_size'])

    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = BiGRUAttentionNet(
        input_size=config['model']['input_size'],
        hidden_size=params['hidden_size'],
        num_layers=params['num_layers'],
        num_classes=config['model']['num_classes'],
        dropout=params['dropout']
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

    # 5. 모델 학습 (간소화된 루프)
    for epoch in range(20): # 최적화 시에는 에포크를 줄여서 빠르게 탐색
        model.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # 6. 검증 및 F1 스코어 반환
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    
    f1 = f1_score(val_targets, val_preds, average='weighted')
    return f1

# --- Optuna Study 실행 ---
if __name__ == "__main__":
    study = optuna.create_study(direction='maximize') # F1 Score를 최대화하는 방향으로 탐색
    study.optimize(objective, n_trials=50) # 50번의 다른 조합으로 실험

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print("  Value (F1 Score): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))