"""照会クエリ正解例作成ワークショップ用モジュール。

`notebooks/04_query_exemplar_workshop.py` から呼ばれ、以下を担う:

1. 1 候補クエリの per-query 評価（取得 chunk_id / hit rank / Recall@k / RR）
2. 複数候補の横並び比較
3. 採用された正解例の永続化（`data/eval/query_exemplars.json`）

`src.evaluator` の純関数（`recall_at_k` / `reciprocal_rank`）を再利用し、
指標計算ロジックの重複実装は避ける。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from src.config import EXEMPLAR_DATASET_PATH
from src.evaluator import recall_at_k, reciprocal_rank
from src.retriever import RetrievalResult

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass
class QueryCandidate:
    """評価対象の候補クエリ。

    Args:
        query: ユーザーが入力する想定の照会文字列。
        target_chunk_ids: ヒットさせたい正解 chunk_id のリスト（OR 判定）。
        notes: 表現意図のメモ（任意）。
    """

    query: str
    target_chunk_ids: list[str]
    notes: str | None = None


@dataclass
class QueryScore:
    """1 クエリ × retriever の per-query 評価結果。"""

    query: str
    retrieved_ids: list[str]
    target_chunk_ids: list[str]
    hit_rank: int | None
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float


@dataclass
class QueryExemplar:
    """採用された照会クエリ正解例。JSON 永続化用。"""

    query_id: str
    query: str
    target_chunk_ids: list[str]
    target_document: str
    category: str
    recall_at_5: float
    mrr: float
    variations_tried: list[str] = field(default_factory=list)
    notes: str = ""
    created_at: str = ""


# ---------------------------------------------------------------------------
# スコアリング
# ---------------------------------------------------------------------------


def _first_hit_rank(
    retrieved_ids: list[str], target_chunk_ids: list[str]
) -> int | None:
    """target が最初に現れる 1-based rank。なければ None。"""
    target_set = set(target_chunk_ids)
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in target_set:
            return i
    return None


def score_query(
    candidate: QueryCandidate,
    retriever_fn: Callable[[str], list[RetrievalResult]],
) -> QueryScore:
    """1 候補クエリを retriever に渡し、per-query メトリクスを返す。

    Args:
        candidate: 評価対象の `QueryCandidate`。
        retriever_fn: `query -> list[RetrievalResult]` を返す callable。
            通常は `functools.partial(retriever.search, top_k=..., mode=...)` を渡す。

    Returns:
        `QueryScore` インスタンス。

    Raises:
        ValueError: `target_chunk_ids` が空の場合。
    """
    if not candidate.target_chunk_ids:
        raise ValueError(
            "target_chunk_ids が空です。評価対象の正解 chunk_id を指定してください。"
        )

    results = retriever_fn(candidate.query)
    retrieved_ids = [r.chunk_id for r in results]
    targets = list(candidate.target_chunk_ids)

    return QueryScore(
        query=candidate.query,
        retrieved_ids=retrieved_ids,
        target_chunk_ids=targets,
        hit_rank=_first_hit_rank(retrieved_ids, targets),
        recall_at_1=recall_at_k(retrieved_ids, targets, 1),
        recall_at_3=recall_at_k(retrieved_ids, targets, 3),
        recall_at_5=recall_at_k(retrieved_ids, targets, 5),
        recall_at_10=recall_at_k(retrieved_ids, targets, 10),
        reciprocal_rank=reciprocal_rank(retrieved_ids, targets),
    )


def compare_query_variations(
    candidates: list[QueryCandidate],
    retriever_fn: Callable[[str], list[RetrievalResult]],
) -> list[QueryScore]:
    """複数の候補クエリを順にスコアリングし結果リストを返す。

    1 候補が例外を投げてもループは継続し、当該候補だけ警告ログを出して
    結果リストには含めない。
    """
    scores: list[QueryScore] = []
    for cand in candidates:
        try:
            scores.append(score_query(cand, retriever_fn))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "compare_query_variations: %r をスキップしました: %s",
                cand.query,
                exc,
            )
    return scores


def to_dataframe(scores: list[QueryScore]) -> pd.DataFrame:
    """`QueryScore` のリストを 1 行 1 query の `pandas.DataFrame` に整形する。

    `pandas` は notebook 環境で確実に入っているため遅延 import で扱う。
    """
    import pandas as pd

    rows = [
        {
            "query": s.query,
            "hit_rank": s.hit_rank,
            "Recall@1": s.recall_at_1,
            "Recall@3": s.recall_at_3,
            "Recall@5": s.recall_at_5,
            "Recall@10": s.recall_at_10,
            "RR": s.reciprocal_rank,
            "retrieved_top5": s.retrieved_ids[:5],
        }
        for s in scores
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 永続化（query_exemplars.json）
# ---------------------------------------------------------------------------


def load_exemplars(
    path: str | Path = EXEMPLAR_DATASET_PATH,
) -> list[QueryExemplar]:
    """`query_exemplars.json` から正解例を読み込む。

    ファイル不在時は空リストを返す（初回実行の利便性のため）。
    """
    file_path = Path(path)
    if not file_path.exists():
        return []
    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"query_exemplars.json は JSON 配列である必要があります: {path}"
        )
    exemplars: list[QueryExemplar] = []
    for entry in raw:
        exemplars.append(
            QueryExemplar(
                query_id=entry["query_id"],
                query=entry["query"],
                target_chunk_ids=list(entry.get("target_chunk_ids", [])),
                target_document=entry.get("target_document", ""),
                category=entry.get("category", ""),
                recall_at_5=float(entry.get("recall_at_5", 0.0)),
                mrr=float(entry.get("mrr", 0.0)),
                variations_tried=list(entry.get("variations_tried", [])),
                notes=entry.get("notes", ""),
                created_at=entry.get("created_at", ""),
            )
        )
    return exemplars


def save_exemplar(
    exemplar: QueryExemplar,
    path: str | Path = EXEMPLAR_DATASET_PATH,
) -> None:
    """正解例 1 件を `query_exemplars.json` に追記する。

    既に同じ `query_id` のエントリがある場合は上書きする（冪等）。
    `created_at` が空なら現在時刻 (UTC, ISO8601) を自動付与する。
    """
    if not exemplar.created_at:
        exemplar.created_at = datetime.now(timezone.utc).isoformat()

    file_path = Path(path)
    existing = load_exemplars(file_path)
    updated = [e for e in existing if e.query_id != exemplar.query_id]
    updated.append(exemplar)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        json.dumps(
            [asdict(e) for e in updated],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(
        "save_exemplar: %s を %s に保存しました（累計 %d 件）",
        exemplar.query_id,
        path,
        len(updated),
    )
