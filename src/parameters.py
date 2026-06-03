# ============================================================
# 閾値・パラメータ設定
# ============================================================
import pandas as pd
from typing import List, Optional

# Pandas表示設定
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)

# === 閾値設定 ===
SC_MIN = 0.7           # 類似度閾値（RAG検索結果フィルタリング用）
TH_HIGH = 0.99         # THスコア高閾値
TH_LOW = 0.005         # THスコア低閾値（回答不可判定用）
TH_MID_LOW = 0.90      # THスコア中間下限（情報補足更問用）
MAX_LOOP = 1           # 最大ループ回数

# === 評価指標の重み ===
F_WEIGHT = 0.1  # Faithfulness
G_WEIGHT = 0.1  # Groundedness
R_WEIGHT = 0.8  # Relevance

# === 検索パラメータ ===
TOP_K = 10
SEARCH_MODE = "hybrid"
SOURCE_TYPE_FILTER: Optional[List[str]] = None


def reset_conversation_history() -> dict:
    """会話履歴を初期化して返す"""
    return {
        "original_query": "",
        "additional_info": [],
        "iteration": 0,
        "last_candidates": [],
        "last_action": ""
    }


# グローバル会話履歴
conversation_history = reset_conversation_history()
