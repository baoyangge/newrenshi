"""Vector Search で類似チャンクを検索するアプリケーション層モジュール。

本改修で FAQ Vector Index / DOC Vector Index の 2 系統に対応し、
`mode="faq" / "doc" / "both"` の 3 モードを `search()` 引数で切り替えられるようにした。
"`both`" モード時は両インデックスを順次（同期）呼び出し、RRF (Reciprocal Rank Fusion,
k=60) で統合する。クエリ embedding は `both` モードでも 1 回のみ計算される。
"""

from __future__ import annotations

import dataclasses
import logging
import os
from dataclasses import dataclass

from src.config import (
    DEFAULT_SEARCH_MODE,
    DEFAULT_TOP_K,
    DOC_HEADING_DEPTH_NULL_SENTINEL,
    DOC_INDEX_NAME,
    FAQ_INDEX_NAME,
    RRF_CANDIDATE_MULTIPLIER,
    RRF_K,
    VECTOR_SEARCH_ENDPOINT_NAME,
    IndexKind,
    IndexMode,
    SearchMode,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """1 件の検索結果。

    FAQ Vector Index / DOC Vector Index の両方に対応するため、共通フィールドに
    加えてインデックスごとの Optional フィールドを保持する。
    """

    chunk_id: str
    score: float
    content: str
    source_document: str
    index_kind: IndexKind = "faq"

    # FAQ ヒット時のみ値あり
    case_no: str | None = None
    question: str | None = None
    answer: str | None = None
    category_l1: str | None = None
    category_l2: str | None = None
    category_l3: str | None = None

    # DOC ヒット時のみ値あり
    heading_path: str | None = None
    heading_depth: int | None = None
    parent_heading_path: str | None = None
    section: str | None = None

    # DEPRECATED 互換維持
    source_type: str | None = None


_client = None  # 型: VectorSearchClient | None


def _get_client():  # type: ignore[no-untyped-def]
    """VectorSearchClient を遅延初期化する。

    DATABRICKS_HOST と DATABRICKS_TOKEN の環境変数（または Databricks Notebook 上の
    Workspace 認証）が利用できる前提。未設定時は明示的な RuntimeError を送出する。
    """
    global _client
    if _client is not None:
        return _client

    databricks_host = os.environ.get("DATABRICKS_HOST")
    databricks_token = os.environ.get("DATABRICKS_TOKEN")

    if not databricks_host or not databricks_token:
        raise RuntimeError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN must be set before using retriever."
            " Set them via .env file or dbutils.secrets in Databricks."
        )

    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("databricks-vectorsearch is required") from exc

    _client = VectorSearchClient(
        workspace_url=databricks_host,
        personal_access_token=databricks_token,
        disable_notice=True,
    )
    return _client


def set_client(client: object) -> None:
    """テスト・Notebook から外部初期化済みクライアントを差し込むためのフック。"""
    global _client
    _client = client


# FAQ Vector Index から取得するカラム
_FAQ_RESULT_COLUMNS = (
    "chunk_index",
    "source_id",
    "case_no",
    "category_l1",
    "category_l2",
    "category_l3",
    "question",
    "answer",
    "content",
)

# DOC Vector Index から取得するカラム
_DOC_RESULT_COLUMNS = (
    "chunk_index",
    "source_id",
    "heading_path",
    "heading_depth",
    "parent_heading_path",
    "section",
    "content",
)


def _extract_score(record: dict) -> float:
    raw_score = record.get("score") or record.get("_score") or 0.0
    return float(raw_score) if isinstance(raw_score, (int, float, str)) else 0.0


def _parse_faq_hit(row: list[object], columns: list[str]) -> RetrievalResult:
    record: dict[str, object] = dict(zip(columns, row, strict=False))
    return RetrievalResult(
        chunk_id=str(record.get("chunk_index", "")),
        score=_extract_score(record),
        content=str(record.get("content", "")),
        source_document=str(record.get("source_id", "")),
        index_kind="faq",
        case_no=_as_optional_str(record.get("case_no")),
        question=_as_optional_str(record.get("question")),
        answer=_as_optional_str(record.get("answer")),
        category_l1=_as_optional_str(record.get("category_l1")),
        category_l2=_as_optional_str(record.get("category_l2")),
        category_l3=_as_optional_str(record.get("category_l3")),
        source_type="faq",
    )


def _parse_doc_hit(row: list[object], columns: list[str]) -> RetrievalResult:
    record: dict[str, object] = dict(zip(columns, row, strict=False))
    return RetrievalResult(
        chunk_id=str(record.get("chunk_index", "")),
        score=_extract_score(record),
        content=str(record.get("content", "")),
        source_document=str(record.get("source_id", "")),
        index_kind="doc",
        heading_path=_as_optional_str(record.get("heading_path")),
        heading_depth=_parse_heading_depth(record.get("heading_depth")),
        parent_heading_path=_as_optional_str(record.get("parent_heading_path")),
        section=_as_optional_str(record.get("section")),
        source_type="doc",
    )


def _parse_heading_depth(value: object) -> int | None:
    """Vector Search の戻り値から heading_depth を int | None で取り出す。

    Direct Vector Access Index は NULL を扱えないため、書き込み時に
    `DOC_HEADING_DEPTH_NULL_SENTINEL`（-1）を「未設定」として保存している。
    読み出し時はセンチネル値を None に戻す。

    int / float / 数値文字列に対応。変換不能な場合は None を返す。
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # bool は int のサブクラスなので明示的に弾く
        return None
    if isinstance(value, (int, float)):
        depth = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            depth = int(stripped)
        except ValueError:
            try:
                depth = int(float(stripped))
            except ValueError:
                return None
    else:
        return None
    if depth == DOC_HEADING_DEPTH_NULL_SENTINEL:
        return None
    return depth


def _as_optional_str(value: object) -> str | None:
    """空文字を None に正規化する。"""
    if value is None:
        return None
    s = str(value)
    return s if s else None


def _build_search_kwargs(
    query_vector: list[float],
    query_text: str,
    top_k: int,
    columns: list[str],
    hybrid: bool,
    filters: dict | None,
) -> dict:
    kwargs: dict[str, object] = {
        "query_vector": query_vector,
        "columns": columns,
        "num_results": top_k,
    }
    if hybrid:
        kwargs["query_type"] = "HYBRID"
        kwargs["query_text"] = query_text
    if filters:
        kwargs["filters"] = filters
    return kwargs


def _search_single(
    query_vector: list[float],
    query_text: str,
    top_k: int,
    index_name: str,
    index_kind: IndexKind,
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
    hybrid: bool = True,
    filters: dict | None = None,
) -> list[RetrievalResult]:
    """単一のインデックスに対して類似検索を実行する。"""
    client = _get_client()
    index = client.get_index(  # type: ignore[attr-defined]
        endpoint_name=endpoint_name, index_name=index_name
    )

    if index_kind == "faq":
        columns_list = list(_FAQ_RESULT_COLUMNS)
        parse_hit = _parse_faq_hit
    else:
        columns_list = list(_DOC_RESULT_COLUMNS)
        parse_hit = _parse_doc_hit

    kwargs = _build_search_kwargs(
        query_vector=query_vector,
        query_text=query_text,
        top_k=top_k,
        columns=columns_list,
        hybrid=hybrid,
        filters=filters,
    )
    raw = index.similarity_search(**kwargs)
    result_payload = raw.get("result") if isinstance(raw, dict) else None
    if not isinstance(result_payload, dict):
        logger.warning("Unexpected search response shape: %s", raw)
        return []

    data_rows = result_payload.get("data_array", [])
    manifest = result_payload.get("manifest", {})
    column_specs = manifest.get("columns", [])
    columns = [col.get("name", "") for col in column_specs if isinstance(col, dict)]
    if not columns:
        columns = columns_list + ["score"]

    return [parse_hit(list(row), columns) for row in data_rows]


def _rrf_merge(
    faq_results: list[RetrievalResult],
    doc_results: list[RetrievalResult],
    top_k: int,
    k: int = RRF_K,
) -> list[RetrievalResult]:
    """Reciprocal Rank Fusion で 2 つの検索結果を統合する。

    RRF スコア（Σ 1 / (k + rank_i)）で順位を決定するが、
    最終的に返す score は元の類似度スコア（cosine similarity 等）を保持する。
    これにより類似度閾値によるフィルタリングが正しく機能する。

    同一 chunk_id が両インデックスに存在する場合：
    - RRF スコアは両方の項を加算
    - 原始スコアは高い方を採用
    - RetrievalResult は FAQ 側を優先
    """
    rrf_scores: dict[str, float] = {}
    original_scores: dict[str, float] = {}
    sources: dict[str, RetrievalResult] = {}

    for rank, r in enumerate(faq_results, start=1):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        original_scores[r.chunk_id] = max(
            original_scores.get(r.chunk_id, 0.0), r.score
        )
        sources[r.chunk_id] = r

    for rank, r in enumerate(doc_results, start=1):
        rrf_scores[r.chunk_id] = rrf_scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
        original_scores[r.chunk_id] = max(
            original_scores.get(r.chunk_id, 0.0), r.score
        )
        sources.setdefault(r.chunk_id, r)

    sorted_ids = sorted(sources.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    merged = [
        dataclasses.replace(sources[cid], score=original_scores[cid])
        for cid in sorted_ids
    ]
    return merged[:top_k]



def search(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    mode: IndexMode = DEFAULT_SEARCH_MODE,
    search_mode: SearchMode = "hybrid",
    source_type_filter: list[str] | None = None,
    faq_index_name: str = FAQ_INDEX_NAME,
    doc_index_name: str = DOC_INDEX_NAME,
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
) -> list[RetrievalResult]:
    """質問文から類似チャンクを検索する。

    本改修で 2 インデックス対応と検索モード切替を導入した。

    Args:
        query: 質問テキスト。空はエラー。
        top_k: 取得件数。
        mode: 検索対象（`"faq"` / `"doc"` / `"both"`）。
            `"both"` の場合は FAQ Index と DOC Index を順次（同期）呼び出し、
            RRF (k=60) で統合する。デフォルトは config.py の DEFAULT_SEARCH_MODE。
        search_mode: 検索アルゴリズム（`"vector"` ANN / `"hybrid"` BM25併用）。既存。
        source_type_filter: 後方互換のため引数は残置するが **無視される**。
            手順書 / 規程 の区別を廃止したため、フィルタ対象が存在しない。
            指定された場合は warning ログを出力する。インデックス切替は `mode` で行う。

    Returns:
        スコア降順の RetrievalResult リスト。最大 top_k 件。

    Raises:
        ValueError: query が空 / 未知の mode が指定された場合。
        RuntimeError: Databricks 認証情報が未設定の場合。
    """
    if not query:
        raise ValueError("query must be non-empty")
    if mode not in ("faq", "doc", "both"):
        raise ValueError(f"unknown mode: {mode!r}")

    # クエリ embedding は 1 回のみ計算（both モードでも）
    from src import embedder

    query_vector = embedder.embed_text(query)

    hybrid = search_mode == "hybrid"
    if source_type_filter:
        # 本改修で 手順書/規程 の doc_type 区別を廃止したため、
        # source_type_filter は実質的に無効化されている（引数は後方互換のため残置）。
        logger.warning(
            "source_type_filter=%s is ignored: procedure/regulation distinction "
            'has been removed. Use `mode="faq" / "doc" / "both"` to select indexes.',
            source_type_filter,
        )

    if mode == "faq":
        results = _search_single(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            index_name=faq_index_name,
            index_kind="faq",
            endpoint_name=endpoint_name,
            hybrid=hybrid,
        )
        logger.info(
            "search mode=faq returned %d hits (top_k=%d, hybrid=%s)",
            len(results),
            top_k,
            hybrid,
        )
        return results

    if mode == "doc":
        results = _search_single(
            query_vector=query_vector,
            query_text=query,
            top_k=top_k,
            index_name=doc_index_name,
            index_kind="doc",
            endpoint_name=endpoint_name,
            hybrid=hybrid,
        )
        logger.info(
            "search mode=doc returned %d hits (top_k=%d, hybrid=%s)",
            len(results),
            top_k,
            hybrid,
        )
        return results

    # mode == "both"
    # RRF 統合時、片方で低ランクだが両方に出現する候補を取り逃さないよう
    # 各インデックスからは top_k * RRF_CANDIDATE_MULTIPLIER 件取得してから統合する。
    candidate_k = max(top_k, top_k * RRF_CANDIDATE_MULTIPLIER)
    faq_results = _search_single(
        query_vector=query_vector,
        query_text=query,
        top_k=candidate_k,
        index_name=faq_index_name,
        index_kind="faq",
        endpoint_name=endpoint_name,
        hybrid=hybrid,
    )
    doc_results = _search_single(
        query_vector=query_vector,
        query_text=query,
        top_k=candidate_k,
        index_name=doc_index_name,
        index_kind="doc",
        endpoint_name=endpoint_name,
        hybrid=hybrid,
    )
    merged = _rrf_merge(faq_results, doc_results, top_k=top_k)
    logger.info(
        "search mode=both returned %d hits (faq=%d, doc=%d, candidate_k=%d, top_k=%d)",
        len(merged),
        len(faq_results),
        len(doc_results),
        candidate_k,
        top_k,
    )
    return merged
