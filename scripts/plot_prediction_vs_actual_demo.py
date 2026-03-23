"""
데모용: SPY 일봉을 내려받아 프로젝트와 동일한 Heikin-Ashi 기반 라벨을 만든 뒤,
간단 분류기(RandomForest)로 학습·추론한 결과로 누적 수익 곡선을 그려 PNG로 저장한다.
- Actual  : 실제 라벨(상승=1)로 매수/관망한 경우의 누적 수익
- Predicted: 모델 예측으로 매수/관망한 경우의 누적 수익
(BiGRU 학습 가중치가 없을 때도 README용 그래프를 만들기 위한 용도)
"""
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

for _name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(_name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False

import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import convert_to_ha


def build_labels_and_features(df: pd.DataFrame) -> pd.DataFrame:
    ha = convert_to_ha(df)
    merged = df.merge(ha, on="Date", how="inner")
    merged["Label"] = (
        merged["HA_Close"].shift(-1) > merged["HA_Open"].shift(-1)
    ).astype(float)
    merged = merged.iloc[:-1].copy()
    merged["Label"] = merged["Label"].fillna(0).astype(int)
    merged = merged.dropna(subset=["Label"])

    merged["feat_ha_body"] = (merged["HA_Close"] - merged["HA_Open"]).abs()
    merged["feat_ha_range"] = merged["HA_High"] - merged["HA_Low"]
    merged["feat_close_to_ha"] = merged["Close"] - merged["HA_Close"]
    for w in (5, 10, 20):
        merged[f"feat_ret_{w}"] = merged["Close"].pct_change(w)
    merged = merged.dropna()
    return merged


def cumulative_return(signal: np.ndarray, daily_ret: np.ndarray) -> np.ndarray:
    """signal=1이면 해당 일 수익률 반영, 0이면 0 (관망)으로 누적 수익 계산."""
    strategy_ret = np.where(signal == 1, daily_ret, 0.0)
    return (1 + strategy_ret).cumprod()


def drawdown(cum: np.ndarray) -> np.ndarray:
    """누적 수익률 시계열의 낙폭(peak 대비 하락률) 계산."""
    peak = np.maximum.accumulate(cum)
    return (cum / peak) - 1.0


def main():
    out_dir = os.path.join(ROOT, "docs", "figures")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "prediction_vs_actual_demo.png")

    raw = yf.download("SPY", start="2018-01-01", auto_adjust=True, progress=False)
    if raw.empty:
        raise SystemExit("SPY 데이터를 받지 못했습니다. 네트워크를 확인하세요.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.rename_axis("Date").reset_index()
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Open", "High", "Low", "Close"])

    feat_df = build_labels_and_features(raw)
    feature_cols = [c for c in feat_df.columns if c.startswith("feat_")]

    n = len(feat_df)
    split = int(n * 0.85)
    train_df = feat_df.iloc[:split]
    test_df = feat_df.iloc[split:].copy()

    clf = RandomForestClassifier(
        n_estimators=120, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(train_df[feature_cols], train_df["Label"])
    pred = clf.predict(test_df[feature_cols])

    # 테스트 구간 일별 수익률
    test_df["daily_ret"] = test_df["Close"].pct_change().fillna(0)
    actual = test_df["Label"].values
    daily_ret = test_df["daily_ret"].values
    dates = test_df["Date"].values

    cum_actual = cumulative_return(actual, daily_ret)
    cum_pred = cumulative_return(pred, daily_ret)
    # Buy & Hold 기준선
    cum_bh = (1 + daily_ret).cumprod()

    # 성능을 과장하지 않도록 전체 범위를 유지하면서, 로그 스케일로
    # 상대적 성장률 차이를 읽기 쉽게 표현한다.
    dd_bh = drawdown(cum_bh)
    dd_actual = drawdown(cum_actual)
    dd_pred = drawdown(cum_pred)

    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )

    ax1.plot(dates, cum_bh, color="#9e9e9e", lw=1, linestyle="--", label="Buy & Hold")
    ax1.plot(dates, cum_actual, color="#1565c0", lw=1.5, label="Actual label strategy")
    ax1.plot(dates, cum_pred, color="#e65100", lw=1.5, label="Predicted label strategy")
    ax1.set_yscale("log")
    ax1.set_title(
        "Cumulative Return (log scale) — Actual vs Predicted (test split, demo: RandomForest)",
        fontsize=11,
    )
    ax1.set_ylabel("Cumulative Return (start = 1.0, log)")
    ax1.legend(fontsize=9, loc="upper left")
    ax1.grid(True, alpha=0.3, which="both")

    ax2.plot(dates, dd_bh, color="#9e9e9e", lw=1, linestyle="--", label="Buy & Hold drawdown")
    ax2.plot(dates, dd_actual, color="#1565c0", lw=1.5, label="Actual drawdown")
    ax2.plot(dates, dd_pred, color="#e65100", lw=1.5, label="Predicted drawdown")
    ax2.axhline(0.0, color="#666666", lw=0.8, alpha=0.7)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
