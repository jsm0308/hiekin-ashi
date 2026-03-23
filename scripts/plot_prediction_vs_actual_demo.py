"""
데모용: SPY 일봉을 내려받아 프로젝트와 동일한 Heikin-Ashi 기반 라벨을 만든 뒤,
간단 분류기(RandomForest)로 학습·추론한 결과를 실제 라벨과 비교해 PNG로 저장한다.
(BiGRU 학습 가중치가 없을 때도 README용 그래프를 만들기 위한 용도)
"""
import os
import sys

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Windows에서 한글 레이블 렌더링
for _name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(_name in f.name for f in fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _name
        break
plt.rcParams["axes.unicode_minus"] = False
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# 프로젝트 루트
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.utils import convert_to_ha


def build_labels_and_features(df: pd.DataFrame) -> pd.DataFrame:
    ha = convert_to_ha(df)
    merged = df.merge(ha, on="Date", how="inner")
    # 다음 봉 HA 종가 > 다음 봉 HA 시가 이면 1 (프로젝트 라벨 정의와 동일)
    merged["Label"] = (
        merged["HA_Close"].shift(-1) > merged["HA_Open"].shift(-1)
    ).astype(float)
    merged = merged.iloc[:-1].copy()
    merged["Label"] = merged["Label"].fillna(0).astype(int)
    merged = merged.dropna(subset=["Label"])

    # 간단 피처: 당일 HA OHLC 비율·변화
    merged["feat_ha_body"] = (merged["HA_Close"] - merged["HA_Open"]).abs()
    merged["feat_ha_range"] = merged["HA_High"] - merged["HA_Low"]
    merged["feat_close_to_ha"] = merged["Close"] - merged["HA_Close"]
    for w in (5, 10, 20):
        merged[f"feat_ret_{w}"] = merged["Close"].pct_change(w)
    merged = merged.dropna()
    return merged


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

    # 시계열 분할: 마지막 15%를 테스트 (시각화 구간)
    n = len(feat_df)
    split = int(n * 0.85)
    train_df = feat_df.iloc[:split]
    test_df = feat_df.iloc[split:]

    X_train, y_train = train_df[feature_cols], train_df["Label"]
    X_test, y_test = test_df[feature_cols], test_df["Label"]

    clf = RandomForestClassifier(
        n_estimators=120, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    dates = test_df["Date"].values
    actual = y_test.values
    close = test_df["Close"].values

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 4), sharey=False)
    fig.suptitle(
        "Actual vs Predicted — SPY next-bar direction (Up=1 / Down=0)",
        fontsize=12,
    )

    # 왼쪽: 실제 라벨 (종가 컬러 구분)
    colors_actual = ["#1565c0" if v == 1 else "#c62828" for v in actual]
    ax0.bar(dates, close, color=colors_actual, width=1.5, alpha=0.7)
    ax0.set_title("Actual Label")
    ax0.set_ylabel("SPY Close Price")
    ax0.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax0.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax0.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax0.grid(True, alpha=0.25, axis="y")

    # 오른쪽: 예측 라벨 (종가 컬러 구분)
    colors_pred = ["#1565c0" if v == 1 else "#c62828" for v in pred]
    ax1.bar(dates, close, color=colors_pred, width=1.5, alpha=0.7)
    ax1.set_title("Predicted Label")
    ax1.set_ylabel("SPY Close Price")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax1.grid(True, alpha=0.25, axis="y")

    # 범례 (fig 공유)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1565c0", alpha=0.7, label="Up (1)"),
        Patch(facecolor="#c62828", alpha=0.7, label="Down (0)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"저장: {out_path}")


if __name__ == "__main__":
    main()
