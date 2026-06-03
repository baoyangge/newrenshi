"""Databricks Vector Search のインデックス操作。

本改修で FAQ Vector Index と DOC Vector Index の 2 系統に対応する。
Delta Table への書き込みと Direct Vector Access Index の管理を担う。

冪等性確保のため、`create_*_index` は存在確認後に作成スキップする。
再作成は呼び出し側で `delete_index` を明示する。
"""

from __future__ import annotations

import logging
import time
from typing import Literal

from src.chunker import Chunk
from src.config import (
    DELTA_TABLE_NAME,
    DOC_DELTA_TABLE_NAME,
    DOC_HEADING_DEPTH_NULL_SENTINEL,
    DOC_INDEX_NAME,
    EMBEDDING_DIM,
    FAQ_DELTA_TABLE_NAME,
    FAQ_INDEX_NAME,
    VECTOR_SEARCH_ENDPOINT_NAME,
    VECTOR_SEARCH_INDEX_NAME,
    VECTOR_SEARCH_UPSERT_BATCH_SIZE,
    VECTOR_SEARCH_UPSERT_MAX_RETRIES,
    VECTOR_SEARCH_UPSERT_RETRY_WAIT_SEC,
    IndexKind,
)

logger = logging.getLogger(__name__)


_client = None  # 型: VectorSearchClient | None


# FAQ Vector Index のスキーマ定義（本改修で導入）
FAQ_SCHEMA: dict[str, str] = {
    "chunk_index": "string",
    "source_id": "string",
    "case_no": "string",
    "category_l1": "string",
    "category_l2": "string",
    "category_l3": "string",
    "question": "string",
    "answer": "string",
    "content": "string",
    "embedding": "array<float>",
}

# DOC Vector Index のスキーマ定義（本改修で導入、後の改修で doc_type を廃止）
DOC_SCHEMA: dict[str, str] = {
    "chunk_index": "string",
    "source_id": "string",
    "heading_path": "string",
    "heading_depth": "int",
    "parent_heading_path": "string",
    "section": "string",
    "content": "string",
    "embedding": "array<float>",
}


def _get_client():  # type: ignore[no-untyped-def]
    """VectorSearchClient を遅延初期化する。"""
    global _client
    if _client is not None:
        return _client
    try:
        from databricks.vector_search.client import VectorSearchClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("databricks-vectorsearch is required") from exc
    _client = VectorSearchClient()
    return _client


def set_client(client: object) -> None:
    """テストや外部初期化済みクライアントを差し込むためのフック。"""
    global _client
    _client = client


def _quote_table_name(table_name: str) -> str:
    """テーブル名をバックティックでクォートする。

    catalog-name.schema.table のようなハイフンを含む識別子に対応する。
    """
    parts = table_name.split(".")
    quoted_parts = [f"`{part}`" for part in parts]
    return ".".join(quoted_parts)


def _index_exists(client: object, endpoint_name: str, index_name: str) -> bool:
    """list_indexes で対象インデックスの存在確認を行う。"""
    try:
        result = client.list_indexes(name=endpoint_name)  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_indexes failed: %s", exc)
        return False
    indexes = result.get("vector_indexes", []) if isinstance(result, dict) else []
    return any(idx.get("name") == index_name for idx in indexes)


def _chunk_to_row(chunk: Chunk, index_kind: IndexKind) -> dict:
    """Chunk dataclass を index_kind に応じた dict 形式へ変換する。

    FAQ Vector Index には FAQ_SCHEMA のカラムのみを、
    DOC Vector Index には DOC_SCHEMA のカラムのみを抽出する。
    文字列カラムの None は空文字に正規化する（Vector Search は NULL を扱えないため）。
    `heading_depth` は int カラムのため空文字を入れられず、未設定時は
    `DOC_HEADING_DEPTH_NULL_SENTINEL`（-1）で「未設定」を表現する。
    読み出し側 `retriever._parse_heading_depth` で None に復元する。
    """
    if index_kind == "faq":
        return {
            "chunk_index": str(chunk.chunk_id),
            "source_id": chunk.source_document or "",
            "case_no": chunk.case_no or "",
            "category_l1": chunk.category_l1 or "",
            "category_l2": chunk.category_l2 or "",
            "category_l3": chunk.category_l3 or "",
            "question": chunk.question or "",
            "answer": chunk.answer or "",
            "content": chunk.content or "",
            "embedding": list(chunk.embedding),
        }
    return {
        "chunk_index": str(chunk.chunk_id),
        "source_id": chunk.source_document or "",
        "heading_path": chunk.heading_path or "",
        "heading_depth": (
            int(chunk.heading_depth)
            if chunk.heading_depth is not None
            else DOC_HEADING_DEPTH_NULL_SENTINEL
        ),
        "parent_heading_path": chunk.parent_heading_path or "",
        "section": chunk.section or "",
        "content": chunk.content or "",
        "embedding": list(chunk.embedding),
    }


def write_chunks_to_delta(
    spark: object,
    chunks: list[Chunk],
    table_name: str,
    index_kind: IndexKind,
    mode: Literal["overwrite", "append"] = "overwrite",
) -> None:
    """Chunk リストを Delta Table に書き込む。

    Args:
        spark: SparkSession インスタンス。
        chunks: 書き込む Chunk のリスト。各 Chunk は embedding を含むこと。
        table_name: 書き込み先 Delta Table 名（例: `FAQ_DELTA_TABLE_NAME`）。
        index_kind: "faq" / "doc"。FAQ_SCHEMA または DOC_SCHEMA でカラムを抽出する。
        mode: 書き込みモード。`"overwrite"` は schema 更新付きで全件置換、
            `"append"` は既存行を残して追記。incremental 処理で使う際は
            事前に `delete_chunks_by_source` で対象 `source_id` を消してから
            `"append"` する想定。
    """
    rows = [_chunk_to_row(c, index_kind) for c in chunks]

    # Delta Table 側の embedding は ARRAY<FLOAT>、heading_depth は INT で定義されているが、
    # Spark の型推論だと Python list は ARRAY<DOUBLE>、Python int は BIGINT になり、
    # append 時に DELTA_FAILED_TO_MERGE_FIELDS で失敗する。FAQ_SCHEMA / DOC_SCHEMA から
    # 明示 DDL を組み立てて createDataFrame に渡し、Delta 側と完全一致させる。
    schema_dict = FAQ_SCHEMA if index_kind == "faq" else DOC_SCHEMA
    schema_ddl = ", ".join(f"`{col}` {dtype}" for col, dtype in schema_dict.items())
    df = spark.createDataFrame(rows, schema=schema_ddl)  # type: ignore[attr-defined]

    quoted_table_name = _quote_table_name(table_name)
    writer = df.write.format("delta").mode(mode)
    if mode == "overwrite":
        writer = writer.option("overwriteSchema", "true")
    writer.saveAsTable(quoted_table_name)
    logger.info(
        "Wrote %d rows to Delta Table: %s (index_kind=%s, mode=%s)",
        len(rows),
        table_name,
        index_kind,
        mode,
    )


def enable_change_data_feed(
    spark: object,
    table_name: str = DELTA_TABLE_NAME,
) -> None:
    """Delta Table で Change Data Feed を有効化する。

    注意: Direct Vector Access Index では不要ですが、
    Delta Sync Index に切り替える場合に備えて残しています。
    """
    quoted_table_name = _quote_table_name(table_name)
    spark.sql(  # type: ignore[attr-defined]
        f"ALTER TABLE {quoted_table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
    )
    logger.info("Enabled Change Data Feed on: %s", table_name)


def create_direct_access_index(
    index_name: str = VECTOR_SEARCH_INDEX_NAME,
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
    primary_key: str = "chunk_index",
    embedding_dimension: int = EMBEDDING_DIM,
    embedding_vector_column: str = "embedding",
    schema: dict | None = None,
) -> None:
    """Direct Vector Access Index を冪等に作成する。

    既に同名インデックスが存在する場合は何もしない。

    Args:
        index_name: 作成するインデックス名
        endpoint_name: Vector Search エンドポイント名
        primary_key: プライマリキーカラム名
        embedding_dimension: Embedding ベクトルの次元数
        embedding_vector_column: Embedding ベクトルが格納されているカラム名
        schema: インデックスのスキーマ定義（省略時はデフォルトスキーマを使用）
    """
    client = _get_client()
    if _index_exists(client, endpoint_name, index_name):
        logger.info("Vector Search index already exists, skip: %s", index_name)
        return

    # デフォルトスキーマ（省略時）
    if schema is None:
        schema = {
            primary_key: "string",
            "source_id": "string",
            "source_type": "string",
            "content": "string",
            embedding_vector_column: "array<float>",
        }

    client.create_direct_access_index(  # type: ignore[attr-defined]
        endpoint_name=endpoint_name,
        index_name=index_name,
        primary_key=primary_key,
        embedding_dimension=embedding_dimension,
        embedding_vector_column=embedding_vector_column,
        schema=schema,
    )
    logger.info("Created Direct Vector Access Index: %s", index_name)


def create_faq_index(
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
    embedding_dimension: int = EMBEDDING_DIM,
) -> None:
    """FAQ Vector Index を `FAQ_SCHEMA` で作成する（冪等）。"""
    create_direct_access_index(
        index_name=FAQ_INDEX_NAME,
        endpoint_name=endpoint_name,
        primary_key="chunk_index",
        embedding_dimension=embedding_dimension,
        embedding_vector_column="embedding",
        schema=FAQ_SCHEMA,
    )


def create_doc_index(
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
    embedding_dimension: int = EMBEDDING_DIM,
) -> None:
    """DOC Vector Index を `DOC_SCHEMA` で作成する（冪等）。"""
    create_direct_access_index(
        index_name=DOC_INDEX_NAME,
        endpoint_name=endpoint_name,
        primary_key="chunk_index",
        embedding_dimension=embedding_dimension,
        embedding_vector_column="embedding",
        schema=DOC_SCHEMA,
    )


def upsert_chunks(
    chunks: list[Chunk],
    index_name: str,
    index_kind: IndexKind,
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
    batch_size: int = VECTOR_SEARCH_UPSERT_BATCH_SIZE,
    max_retries: int = VECTOR_SEARCH_UPSERT_MAX_RETRIES,
    retry_wait_sec: int = VECTOR_SEARCH_UPSERT_RETRY_WAIT_SEC,
) -> None:
    """Chunk リストを Vector Search Index に upsert する。

    index_kind に応じて FAQ_SCHEMA / DOC_SCHEMA のカラムだけを抽出してから
    `index.upsert()` に渡す。バッチ化・リトライを含む（SSL 接続安定性対策）。

    Args:
        chunks: 投入する Chunk のリスト。embedding を含むこと。
        index_name: 投入先 Vector Index 名。
        index_kind: "faq" / "doc"。カラム抽出の切替に使用。
        endpoint_name: Vector Search エンドポイント名。
        batch_size: 1 回の upsert で送るチャンク数。
        max_retries: ネットワークエラー時のバッチごとリトライ回数。
        retry_wait_sec: リトライ時の待機秒数（attempt * この値）。
    """
    if not chunks:
        logger.info("upsert_chunks: no chunks to upsert (index=%s)", index_name)
        return

    client = _get_client()
    index = client.get_index(  # type: ignore[attr-defined]
        endpoint_name=endpoint_name, index_name=index_name
    )

    payload = [_chunk_to_row(c, index_kind) for c in chunks]
    total = len(payload)
    total_batches = (total + batch_size - 1) // batch_size
    logger.info(
        "upsert_chunks: starting %d chunks in %d batches (index=%s, kind=%s)",
        total,
        total_batches,
        index_name,
        index_kind,
    )

    for i in range(0, total, batch_size):
        batch = payload[i : i + batch_size]
        batch_num = i // batch_size + 1
        last_exc: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                index.upsert(batch)
                last_exc = None
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = attempt * retry_wait_sec
                logger.warning(
                    "upsert batch %d/%d failed (attempt %d/%d): %s. Retrying in %ds",
                    batch_num,
                    total_batches,
                    attempt,
                    max_retries,
                    exc,
                    wait,
                )
                time.sleep(wait)
        if last_exc is not None:
            raise RuntimeError(
                f"upsert_chunks: batch {batch_num}/{total_batches} failed after {max_retries} attempts"
            ) from last_exc

    logger.info(
        "upsert_chunks: done %d chunks (index=%s, kind=%s)",
        total,
        index_name,
        index_kind,
    )


def delete_chunks_by_source(
    source_ids: list[str],
    index_name: str,
    delta_table_name: str,
    spark: object,
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
    batch_size: int = VECTOR_SEARCH_UPSERT_BATCH_SIZE,
) -> None:
    """指定 `source_id` 群に紐づく chunk を Vector Index と Delta Table から削除する。

    incremental indexing で Volume から削除された孤児ソースをクリーンアップする際、
    あるいは差分追加前に対象ソースを "クリーンスレート" する際に呼ぶ。

    手順:
        1. Delta Table から ``SELECT chunk_index ... WHERE source_id IN (...)`` で
           削除対象の chunk_id を引く。
        2. ``index.delete(primary_keys=...)`` を `batch_size` 単位でバッチ実行。
        3. ``DELETE FROM <delta> WHERE source_id IN (...)`` で Delta も同期削除。

    Args:
        source_ids: 削除対象 `source_id` の一覧。空なら no-op。
        index_name: Vector Search Index 名（`FAQ_INDEX_NAME` / `DOC_INDEX_NAME`）。
        delta_table_name: 削除対象 chunk_id を引く Delta Table 名。
        spark: SparkSession。
        endpoint_name: Vector Search エンドポイント名。
        batch_size: ``index.delete()`` 1 回あたりの最大 primary_key 数。
    """
    if not source_ids:
        logger.info(
            "delete_chunks_by_source: source_ids が空のためスキップ (index=%s)",
            index_name,
        )
        return

    quoted_table = _quote_table_name(delta_table_name)
    # SQL の IN リストに渡すため、source_id 内のシングルクォートをエスケープ
    sanitized = [s.replace("'", "''") for s in source_ids]
    in_list = ", ".join(f"'{s}'" for s in sanitized)

    # 1. Delta から削除対象の chunk_index を取得
    try:
        rows = spark.sql(  # type: ignore[attr-defined]
            f"SELECT chunk_index FROM {quoted_table} " f"WHERE source_id IN ({in_list})"
        ).collect()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "delete_chunks_by_source: Delta から chunk_index を取得できません"
            "（%s）: %s",
            delta_table_name,
            exc,
        )
        return

    chunk_ids = [row["chunk_index"] for row in rows if row["chunk_index"]]
    if not chunk_ids:
        logger.info(
            "delete_chunks_by_source: 削除対象 chunk が Delta に存在しません "
            "(sources=%s)",
            source_ids,
        )
        return

    # 2. Vector Index から削除（バッチ実行）
    client = _get_client()
    index = client.get_index(  # type: ignore[attr-defined]
        endpoint_name=endpoint_name, index_name=index_name
    )
    total = len(chunk_ids)
    for i in range(0, total, batch_size):
        batch = chunk_ids[i : i + batch_size]
        index.delete(primary_keys=batch)
    logger.info(
        "delete_chunks_by_source: Vector Index から %d 件削除 (index=%s)",
        total,
        index_name,
    )

    # 3. Delta Table からも削除
    spark.sql(  # type: ignore[attr-defined]
        f"DELETE FROM {quoted_table} WHERE source_id IN ({in_list})"
    )
    logger.info(
        "delete_chunks_by_source: Delta から DELETE 完了 " "(table=%s, sources=%d)",
        delta_table_name,
        len(source_ids),
    )


def sync_index(
    index_name: str = VECTOR_SEARCH_INDEX_NAME,
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
) -> None:
    """TRIGGERED パイプラインの同期を手動で開始する。

    注意: Direct Vector Access Index では sync() は使用できません。
    Delta Sync Index にのみ適用されます。
    """
    client = _get_client()
    index = client.get_index(endpoint_name=endpoint_name, index_name=index_name)  # type: ignore[attr-defined]
    index.sync()
    logger.info("Triggered sync for index: %s", index_name)


def delete_index(
    index_name: str = VECTOR_SEARCH_INDEX_NAME,
    endpoint_name: str = VECTOR_SEARCH_ENDPOINT_NAME,
) -> None:
    """インデックスを明示的に削除する（呼び出し側の意図確認を前提）。"""
    client = _get_client()
    client.delete_index(endpoint_name=endpoint_name, index_name=index_name)  # type: ignore[attr-defined]
    logger.info("Deleted Vector Search index: %s", index_name)


# DELTA_TABLE_NAME / DOC_DELTA_TABLE_NAME / FAQ_DELTA_TABLE_NAME を
# 利用箇所で参照するため import を維持
__all__ = [
    "FAQ_SCHEMA",
    "DOC_SCHEMA",
    "create_direct_access_index",
    "create_faq_index",
    "create_doc_index",
    "write_chunks_to_delta",
    "upsert_chunks",
    "delete_chunks_by_source",
    "delete_index",
    "sync_index",
    "enable_change_data_feed",
    "set_client",
    "FAQ_DELTA_TABLE_NAME",
    "DOC_DELTA_TABLE_NAME",
]
