"""Volume と Delta Table の `source_id` を突き合わせて差分を検出する。

ファイル名（`Path.stem`）ベースで「新規追加されたソース」と
「Volume から削除された孤児ソース」を抽出する。`notebooks/01_build_index`
の incremental モードのエントリポイントとなる。
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _quote_table_name(table_name: str) -> str:
    """テーブル名をバックティックでクォートする（ハイフン入り識別子対策）。"""
    parts = table_name.split(".")
    return ".".join(f"`{p}`" for p in parts)


def detect_source_diff(
    volume_paths: list[str],
    delta_table_name: str,
    spark: object,
) -> tuple[list[str], list[str]]:
    """Volume 上のファイルと Delta Table の登録済 `source_id` を突き合わせる。

    ファイル名（拡張子を除いた `Path.stem`）を `source_id` と見なし、
    集合演算で「新規」「削除」を抽出する。差分処理 (incremental indexing) で
    `target_paths` を `chunk` 化対象に、`removed_source_ids` を
    `indexer.delete_chunks_by_source` の入力に渡す想定。

    Args:
        volume_paths: Volume 上に現在存在するファイルパスの一覧
            （例: ``glob.glob(f"{VOLUME_CSV_PATH}/*.csv")`` の結果）。
        delta_table_name: 登録済 `source_id` を取得する Delta Table 名
            （例: ``FAQ_DELTA_TABLE_NAME``）。
        spark: SparkSession。`sql()` のみ使用する。

    Returns:
        ``(new_source_paths, removed_source_ids)``

        - ``new_source_paths``: Volume にあるが Delta に無いファイルパス。
        - ``removed_source_ids``: Delta にあるが Volume に無い `source_id`。

        Delta Table 自体が存在しない初回実行時は ``(volume_paths, [])`` を返す。
    """
    path_to_source: dict[str, str] = {p: Path(p).stem for p in volume_paths}
    volume_sources: set[str] = set(path_to_source.values())

    quoted = _quote_table_name(delta_table_name)
    try:
        rows = spark.sql(  # type: ignore[attr-defined]
            f"SELECT DISTINCT source_id FROM {quoted}"
        ).collect()
        indexed_sources: set[str] = {
            row["source_id"] for row in rows if row["source_id"]
        }
    except Exception as exc:  # noqa: BLE001
        # Delta Table が未作成（初回実行）など。全件新規扱いとする。
        logger.info(
            "detect_source_diff: Delta Table %s が読めません"
            "（初回実行とみなします）: %s",
            delta_table_name,
            exc,
        )
        return list(volume_paths), []

    new_sources = volume_sources - indexed_sources
    removed_sources = indexed_sources - volume_sources

    new_source_paths = [p for p, s in path_to_source.items() if s in new_sources]
    removed_source_ids = sorted(removed_sources)

    logger.info(
        "detect_source_diff: volume=%d, indexed=%d, "
        "new=%d, removed=%d, unchanged=%d",
        len(volume_sources),
        len(indexed_sources),
        len(new_sources),
        len(removed_sources),
        len(volume_sources & indexed_sources),
    )
    return new_source_paths, removed_source_ids
