import pandas as pd
import numpy as np
import os
import yaml
from ta import add_all_ta_features # 기술적 지표 라이브러리 (pip install ta)
from src.utils import convert_to_ha

# 설정 파일 로드
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 설정값
RAW_DATA_PATH = config['data']['raw_path']
PROCESSED_DATA_PATH = config['data']['processed_path']
MASTER_FILE = config['data']['master_file']

def create_features():
    print("피처 엔지니어링 시작...")
    # 1. SPY 데이터 로드 및 헤이킨아시 변환
    spy_df = pd.read_csv(os.path.join(RAW_DATA_PATH, 'SPY.csv'), parse_dates=['Date'])
    spy_ha_df = convert_to_ha(spy_df)
    spy_ha_df.rename(columns={
        'HA_Open': 'SPY_HA_Open', 'HA_High': 'SPY_HA_High', 
        'HA_Low': 'SPY_HA_Low', 'HA_Close': 'SPY_HA_Close'
    }, inplace=True)
    
    # 2. SPY 헤이킨아시 데이터 기반 기술적 지표 생성
    # ta 라이브러리는 open, high, low, close 컬럼명이 필요하므로 잠시 이름 변경
    temp_df = spy_ha_df.rename(columns={
        'SPY_HA_Open': 'Open', 'SPY_HA_High': 'High',
        'SPY_HA_Low': 'Low', 'SPY_HA_Close': 'Close'
    })
    # 'Volume' 추가 (원본 데이터에서 가져옴)
    temp_df['Volume'] = spy_df['Volume']
    
    ta_df = add_all_ta_features(temp_df, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)
    
    # 사용할 기술적 지표만 선택
    selected_ta_features = [
        'momentum_rsi',       # RSI
        'trend_macd_diff',    # MACD diff
        'volatility_bbh',     # 볼린저밴드 상단
        'volatility_bbl',     # 볼린저밴드 하단
    ]
    features_df = ta_df[['Date'] + selected_ta_features]
    features_df = pd.merge(features_df, spy_ha_df, on='Date', how='left')

    # 3. 외부 지표(VIXY, TLT, GLD) 로드 및 피처 생성
    for ticker in ['VIXY', 'TLT', 'GLD']:
        df = pd.read_csv(os.path.join(RAW_DATA_PATH, f'{ticker}.csv'), parse_dates=['Date'])
        features_df[f'{ticker}_change'] = df['Close'].pct_change().fillna(0)
        features_df[f'{ticker}_ma5_change'] = df['Close'].rolling(window=5).mean().pct_change().fillna(0)

    # 4. 조합 피처 생성
    features_df['SPY_TLT_ratio'] = spy_df['Close'] / pd.read_csv(os.path.join(RAW_DATA_PATH, 'TLT.csv'))['Close']
    
    # 5. Label(정답) 생성: 다음날 헤이킨아시 종가가 시가보다 높으면 1 (상승), 아니면 0 (하락)
    features_df['Label'] = np.where(features_df['SPY_HA_Close'].shift(-1) > features_df['SPY_HA_Open'].shift(-1), 1, 0)

    # 6. 불필요한 데이터 정리 및 저장
    # 마지막 행은 다음날 정보가 없으므로 Label이 NaN이 됨 -> 제거
    final_df = features_df.dropna()
    final_df['Label'] = final_df['Label'].astype(int)

    # Date와 Label 제외한 모든 컬럼을 피처로 사용
    feature_columns = [col for col in final_df.columns if col not in ['Date', 'Label']]
    print(f"총 {len(feature_columns)}개의 피처 생성 완료: {feature_columns}")
    
    os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)
    output_path = os.path.join(PROCESSED_DATA_PATH, MASTER_FILE)
    final_df.to_csv(output_path, index=False)
    print(f"최종 데이터 저장 완료: {output_path}")

if __name__ == "__main__":
    create_features()