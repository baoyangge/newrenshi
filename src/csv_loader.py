"""FAQをまとめたCSVファイルから Chunk リストを生成する。

UTF-8 BOM付き CSV（案件番号/照会/回答/区分Ⅰ・Ⅱ・Ⅲ を含む）を読み込み、
`embedder.embed_batch` + `indexer.upsert_chunks` へ直接渡せる `list[Chunk]` を返す。
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

from src.chunker import Chunk

logger = logging.getLogger(__name__)


def load_chunks_from_faq_csv(csv_path: str | Path) -> list[Chunk]:
    """FAQ CSV から Chunk リストを読み込む。

    本改修で Chunk に case_no / category_l1/l2/l3 / question / answer を
    フィールド分離して保持するように変更した。section への連結文字列出力は廃止。

    Args:
        csv_path: FAQ CSVファイルへのパス（UTF-8 BOM付き、RFC 4180 準拠）。
            必須列: ``照会``, ``回答``, ``案件番号``, ``区分Ⅰ``, ``区分Ⅱ``, ``区分Ⅲ``

    Returns:
        FAQ の 1 レコード 1 チャンクの Chunk リスト。
        照会・回答のどちらかが空のレコードはスキップする。
        embedding は空リストのまま返す（呼び出し側で embed_batch を実行すること）。

    Raises:
        FileNotFoundError: csv_path が存在しない場合。
    """
    path = Path(csv_path)
    source_document = path.stem
    chunks: list[Chunk] = []
    skipped = 0
    seen: dict[str, int] = {}  # 案件番号の出現回数を追跡し chunk_id 重複を防ぐ

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = (row.get("照会") or "").strip()
            answer = (row.get("回答") or "").strip()
            if not question or not answer:
                skipped += 1
                continue

            case_no = (row.get("案件番号") or "").strip()
            if case_no:
                count = seen.get(case_no, 0)
                seen[case_no] = count + 1
                suffix = f"_{count}" if count > 0 else ""
                chunk_id = f"faq_csv_{case_no}{suffix}"
            else:
                chunk_id = f"faq_csv_{len(chunks)}"

            category_l1 = (row.get("区分Ⅰ") or "").strip() or None
            category_l2 = (row.get("区分Ⅱ") or "").strip() or None
            category_l3 = (row.get("区分Ⅲ") or "").strip() or None

            content = f"## Q: {question}\n\n{answer}"

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    content=content,
                    source_document=source_document,
                    index_kind="faq",
                    case_no=case_no or None,
                    category_l1=category_l1,
                    category_l2=category_l2,
                    category_l3=category_l3,
                    question=question,
                    answer=answer,
                    # 互換維持
                    source_type="faq",
                )
            )

    logger.info(
        "Loaded %d FAQ chunks from %s (skipped %d rows with empty 照会/回答)",
        len(chunks),
        csv_path,
        skipped,
    )
    return chunks
