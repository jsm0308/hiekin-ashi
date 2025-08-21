import yfinance as yf
import pandas as pd
import os
import yaml

# 설정 파일 로드
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 설정값 가져오기
TICKERS = config['data']['tickers']
START_DATE = config['data']['start_date']
END_DATE = config['data']['end_date']
RAW_DATA_PATH = config['data']['raw_path']

# 데이터 다운로드 함수
def download_data():
    os.makedirs(RAW_DATA_PATH, exist_ok=True)
    for ticker in TICKERS:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE)
            if df.empty:
                print(f"[실패] {ticker}: 데이터를 다운로드할 수 없습니다.")
                continue
            
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index()
            out_fp = os.path.join(RAW_DATA_PATH, f"{ticker}.csv")
            df.to_csv(out_fp, index=False)
            print(f"[성공] {ticker}: {out_fp} 에 저장 완료 (shape: {df.shape})")
        except Exception as e:
            print(f"[에러] {ticker}: {e}")

if __name__ == "__main__":
    download_data()