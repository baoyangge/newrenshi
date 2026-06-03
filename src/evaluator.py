"""評価指標の計算と MLflow への記録。

純関数（`recall_at_k` / `reciprocal_rank` / `dcg_at_k` / `idcg_at_k` / `ndcg_at_k`）は
外部依存なしでユニットテスト可能。`evaluate` は集約 + 内訳算出を担う。
"""

from __future__ import annotations

import json
import logging
import math
import tempfile
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from src.config import IndexMode
from src.data_loader import EvaluationQuestion
from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 純関数: 指標計算
# ---------------------------------------------------------------------------


def recall_at_k(retrieved_ids: list[str], gt_ids: list[str], k: int) -> float:
    """正解が上位 k 件に 1 件以上含まれれば 1.0、含まれなければ 0.0。"""
    top_k = retrieved_ids[:k]
    return 1.0 if any(gid in top_k for gid in gt_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], gt_ids: list[str]) -> float:
    """正解が最初に現れる順位の逆数（無ければ 0.0）。"""
    for i, cid in enumerate(retrieved_ids, start=1):
        if cid in gt_ids:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved_ids: list[str], gt_ids: list[str], k: int) -> float:
    """DCG@k（バイナリ判定）。"""
    score = 0.0
    for i, cid in enumerate(retrieved_ids[:k], start=1):
        if cid in gt_ids:
            score += 1.0 / math.log2(i + 1)
    return score


def idcg_at_k(gt_count: int, k: int) -> float:
    """IDCG@k（バイナリ判定）。"""
    ideal = min(gt_count, k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal + 1))


def ndcg_at_k(retrieved_ids: list[str], gt_ids: list[str], k: int) -> float:
    """nDCG@k = DCG@k / IDCG@k。IDCG=0 のときは 0.0。"""
    idcg = idcg_at_k(len(gt_ids), k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(retrieved_ids, gt_ids, k) / idcg


# ---------------------------------------------------------------------------
# 評価結果データクラス
# ---------------------------------------------------------------------------


@dataclass
class EvaluationMetrics:
    """評価結果サマリ（カテゴリ別・難易度別の内訳含む）。"""

    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    ndcg_at_5: float = 0.0
    by_category: dict[str, EvaluationMetrics] = field(default_factory=dict)
    by_difficulty: dict[str, EvaluationMetrics] = field(default_factory=dict)
    failed_cases: list[str] = field(default_factory=list)


@dataclass
class EvaluationRun:
    """MLflow Run 1 件分のパラメータ。

    本改修で `mode: IndexMode` フィールドを追加し、faq / doc / both の
    どのインデックスモードで評価したかを MLflow params に記録できるようにした。
    既存フィールド（run_id / distance_metric / source_type_filter 等）は維持する。
    """

    run_id: str
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 10
    distance_metric: str = "cosine"
    search_mode: Literal["vector", "hybrid"] = "hybrid"
    embedding_model: str = "text-embedding-3-small"
    source_type_filter: list[str] | None = None
    # 本改修で追加: 検索モード切替（FAQ Index / DOC Index / 両方）
    mode: IndexMode = "both"


# ---------------------------------------------------------------------------
# 評価本体
# ---------------------------------------------------------------------------


def _aggregate(
    pairs: list[tuple[list[str], list[str]]],
) -> tuple[float, float, float, float, float, float]:
    """`(retrieved_ids, gt_ids)` のペアから 6 指標の平均を計算する。"""
    if not pairs:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    n = len(pairs)
    r1 = sum(recall_at_k(r, g, 1) for r, g in pairs) / n
    r3 = sum(recall_at_k(r, g, 3) for r, g in pairs) / n
    r5 = sum(recall_at_k(r, g, 5) for r, g in pairs) / n
    r10 = sum(recall_at_k(r, g, 10) for r, g in pairs) / n
    mrr_val = sum(reciprocal_rank(r, g) for r, g in pairs) / n
    ndcg5 = sum(ndcg_at_k(r, g, 5) for r, g in pairs) / n
    return r1, r3, r5, r10, mrr_val, ndcg5


def _make_metrics(pairs: list[tuple[list[str], list[str]]]) -> EvaluationMetrics:
    r1, r3, r5, r10, mrr_val, ndcg5 = _aggregate(pairs)
    return EvaluationMetrics(
        recall_at_1=r1,
        recall_at_3=r3,
        recall_at_5=r5,
        recall_at_10=r10,
        mrr=mrr_val,
        ndcg_at_5=ndcg5,
    )


def evaluate(
    dataset: list[EvaluationQuestion],
    retriever_fn: Callable[[str], list[RetrievalResult]],
    k_list: list[int] | None = None,
    source_type_filter: list[str] | None = None,
) -> EvaluationMetrics:
    """評価データセット全問に対して検索→指標計算を行う。

    Args:
        dataset: 評価データセット。
        retriever_fn: `query -> list[RetrievalResult]` の callable。
        k_list: Recall@k を算出する k のリスト。デフォルト `[1,3,5,10]`。
        source_type_filter: 評価対象を `source_type` で絞る場合に指定（メタ情報、
            MLflow 記録に流す）。

    Returns:
        `EvaluationMetrics` インスタンス。`failed_cases` には「retriever は正常に応答したが
        正解チャンクが上位 max(k_list) 件に含まれなかった質問」の question_id のみが入る。
        retriever が例外を投げた質問は `WARNING` ログのうえ集計から除外され、`failed_cases`
        にも含めない。
    """
    if k_list is None:
        k_list = [1, 3, 5, 10]

    pairs: list[tuple[list[str], list[str]]] = []
    failed_cases: list[str] = []
    by_category_pairs: dict[str, list[tuple[list[str], list[str]]]] = defaultdict(list)
    by_difficulty_pairs: dict[str, list[tuple[list[str], list[str]]]] = defaultdict(
        list
    )

    for idx, question in enumerate(dataset, start=1):
        try:
            results = retriever_fn(question.question)
        except Exception as exc:  # noqa: BLE001 — 1 問の失敗で全体を止めない
            logger.warning(
                "Skipped question %s due to retriever error: %s",
                question.question_id,
                exc,
            )
            continue

        retrieved_ids = [r.chunk_id for r in results]
        gt_ids = list(question.ground_truth_chunk_ids)
        pairs.append((retrieved_ids, gt_ids))
        by_category_pairs[question.category].append((retrieved_ids, gt_ids))
        by_difficulty_pairs[question.difficulty].append((retrieved_ids, gt_ids))

        max_k = max(k_list) if k_list else 10
        if not any(gid in retrieved_ids[:max_k] for gid in gt_ids):
            failed_cases.append(question.question_id)

        if idx % 10 == 0 or idx == len(dataset):
            logger.info("Evaluated %d/%d questions", idx, len(dataset))

    metrics = _make_metrics(pairs)
    metrics.by_category = {
        cat: _make_metrics(p) for cat, p in by_category_pairs.items()
    }
    metrics.by_difficulty = {
        diff: _make_metrics(p) for diff, p in by_difficulty_pairs.items()
    }
    metrics.failed_cases = failed_cases
    logger.info(
        "Evaluation done: Recall@5=%.3f / MRR=%.3f / nDCG@5=%.3f (filter=%s)",
        metrics.recall_at_5,
        metrics.mrr,
        metrics.ndcg_at_5,
        source_type_filter,
    )
    return metrics


# ---------------------------------------------------------------------------
# MLflow 記録
# ---------------------------------------------------------------------------


def _metrics_to_dict(metrics: EvaluationMetrics) -> dict:
    """`EvaluationMetrics` をネスト辞書化（JSON 出力用）。"""
    payload = asdict(metrics)
    payload["by_category"] = {k: asdict(v) for k, v in metrics.by_category.items()}
    payload["by_difficulty"] = {k: asdict(v) for k, v in metrics.by_difficulty.items()}
    return payload


def log_to_mlflow(metrics: EvaluationMetrics, run: EvaluationRun) -> None:
    """MLflow Tracking に params / metrics / artifacts を記録する。

    呼び出し側で `mlflow.start_run()` 済みであることを前提とする。
    artifact として `evaluation_result.json` を一時ファイル経由でアップロードする。
    """
    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("mlflow is required for log_to_mlflow") from exc

    params = {
        "chunk_size": run.chunk_size,
        "chunk_overlap": run.chunk_overlap,
        "top_k": run.top_k,
        "distance_metric": run.distance_metric,
        "search_mode": run.search_mode,
        "embedding_model": run.embedding_model,
        "source_type_filter": (
            ",".join(run.source_type_filter) if run.source_type_filter else "all"
        ),
        # 本改修で追加: インデックス対象切替モード
        "mode": run.mode,
    }
    mlflow.log_params(params)

    metric_values = {
        "recall_at_1": metrics.recall_at_1,
        "recall_at_3": metrics.recall_at_3,
        "recall_at_5": metrics.recall_at_5,
        "recall_at_10": metrics.recall_at_10,
        "mrr": metrics.mrr,
        "ndcg_at_5": metrics.ndcg_at_5,
    }
    mlflow.log_metrics(metric_values)

    with tempfile.TemporaryDirectory() as tmpdir:
        artifact_path = Path(tmpdir) / "evaluation_result.json"
        artifact_path.write_text(
            json.dumps(_metrics_to_dict(metrics), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        mlflow.log_artifact(str(artifact_path))

        if metrics.failed_cases:
            failed_path = Path(tmpdir) / "failed_cases.json"
            failed_path.write_text(
                json.dumps(metrics.failed_cases, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            mlflow.log_artifact(str(failed_path))

    logger.info("Logged evaluation to MLflow run %s", run.run_id)
