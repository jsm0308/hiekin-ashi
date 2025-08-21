import pandas as pd

def convert_to_ha(df):
    """
    일반 OHLC 데이터를 헤이킨아시 OHLC 데이터로 변환합니다.
    입력: Date, Open, High, Low, Close 컬럼을 포함한 DataFrame
    출력: Date, HA_Open, HA_High, HA_Low, HA_Close 컬럼을 포함한 DataFrame
    """
    ha_df = df[['Date', 'Open', 'High', 'Low', 'Close']].copy()

    ha_df['HA_Close'] = (ha_df['Open'] + ha_df['High'] + ha_df['Low'] + ha_df['Close']) / 4

    # HA_Open 초기값 계산
    ha_df.loc[0, 'HA_Open'] = (ha_df.loc[0, 'Open'] + ha_df.loc[0, 'Close']) / 2

    # 나머지 HA_Open 계산
    for i in range(1, len(ha_df)):
        ha_df.loc[i, 'HA_Open'] = (ha_df.loc[i-1, 'HA_Open'] + ha_df.loc[i-1, 'HA_Close']) / 2

    ha_df['HA_High'] = ha_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)

    return ha_df[['Date', 'HA_Open', 'HA_High', 'HA_Low', 'HA_Close']]