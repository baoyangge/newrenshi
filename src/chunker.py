"""種別別のチャンク分割。

FAQ は Q&A 単位、DOC（手順書・規程混在）は見出し / 条文単位で 1 チャンクとする。
本改修で 手順書 と 規程 の区別を廃止し、`chunk_doc` 1 関数に統合した。
PDF 入力に対しては `chunk_pdf` がエントリポイントとなり、`pdf_parser` の
両構造抽出（規程 / 手順書）を試して **チャンク数が多い方を採用** する。
両方とも 0 件の場合は `chunk_fallback` で `RecursiveCharacterTextSplitter` に
フォールバックする。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    IndexKind,
    SourceType,
)

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """ベクトルストア登録単位の構造化データ。

    FAQ Vector Index と DOC Vector Index の両方を表す共通 dataclass。
    インデックスごとに使用するフィールドのみ値を持ち、他は None とする。
    サブクラス化はせず、`index_kind` で区別する設計。

    必須フィールド: chunk_id, content, source_document, index_kind, embedding
    FAQ 固有: case_no, category_l1, category_l2, category_l3, question, answer
    DOC 固有: heading_path, heading_depth, parent_heading_path, section

    Deprecated: source_type（互換維持。新コードは index_kind を参照すること）
    """

    chunk_id: str
    content: str
    source_document: str
    index_kind: IndexKind = "faq"
    embedding: list[float] = field(default_factory=list)

    # FAQ 固有フィールド（DOC では None）
    case_no: str | None = None
    category_l1: str | None = None
    category_l2: str | None = None
    category_l3: str | None = None
    question: str | None = None
    answer: str | None = None

    # DOC 固有フィールド（FAQ では None）
    heading_path: str | None = None
    heading_depth: int | None = None
    parent_heading_path: str | None = None
    section: str | None = None

    # DEPRECATED: 互換維持。index_kind への完全移行が終わったら削除する。
    source_type: SourceType | None = None


def _slugify(name: str) -> str:
    """ファイル名やタイトルからスラッグを生成する。"""
    slug = re.sub(r"\s+", "_", name.strip()).lower()
    slug = re.sub(r"[^a-z0-9_一-龯ぁ-んァ-ヶー]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "doc"


def chunk_faq(faq: dict) -> list[Chunk]:
    """FAQ dict から Q&A 単位の Chunk を返す。

    主に Markdown ベースの FAQ モックデータ用。CSV からの FAQ 読み込みは
    `src/csv_loader.py::load_chunks_from_faq_csv()` を使う。

    Args:
        faq: `{source_type, source_document, section, content}` を持つ辞書。
            `content` は 1 件の Q&A の Markdown ブロックを想定する。

    Returns:
        FAQ 1 件あたり 1 チャンクのリスト。
    """
    if faq.get("source_type") != "faq":
        raise ValueError(
            f"chunk_faq expects source_type=='faq', got {faq.get('source_type')!r}"
        )
    doc = faq["source_document"]
    content = faq["content"].strip()
    if not content:
        return []
    slug = _slugify(doc)
    chunk_id = f"faq_{slug}_chunk_0"
    return [
        Chunk(
            chunk_id=chunk_id,
            content=content,
            source_document=doc,
            index_kind="faq",
            source_type="faq",
        )
    ]


def chunk_doc(doc_input: dict) -> list[Chunk]:
    """DOC dict から見出し / 条文単位の Chunk を返す。

    手順書 / 規程の区別なく統一的に扱う。入力 dict は以下のいずれかから生成される:
    - `pdf_parser.iter_procedure_chunks_input()`
    - `pdf_parser.iter_regulation_chunks_input()`
    - `data_loader.load_procedure()` / `load_regulation()`（モック Markdown 読み込み）

    Args:
        doc_input: 以下のキーを持つ dict
            - source_type: "doc"
            - source_document: str
            - section: str | None
            - content: str
            - heading_path: str | None（PDF 由来時のみ）
            - heading_depth: int | None（同上）
            - parent_heading_path: str | None（同上）
    """
    if doc_input.get("source_type") != "doc":
        raise ValueError(
            f"chunk_doc expects source_type=='doc', got {doc_input.get('source_type')!r}"
        )
    doc = doc_input["source_document"]
    section = doc_input.get("section")
    heading_path = doc_input.get("heading_path") or section or "section"
    content = doc_input["content"].strip()
    if not content:
        return []
    slug = _slugify(doc)
    heading_slug = _slugify(heading_path)
    chunk_id = f"doc_{slug}_{heading_slug}"
    return [
        Chunk(
            chunk_id=chunk_id,
            content=content,
            source_document=doc,
            index_kind="doc",
            heading_path=heading_path,
            heading_depth=doc_input.get("heading_depth"),
            parent_heading_path=doc_input.get("parent_heading_path"),
            section=section,
            source_type="doc",
        )
    ]


def chunk_pdf(pdf_path: str, source_document: str) -> list[Chunk]:
    """PDF から DOC チャンクを生成する統合エントリポイント。

    1. `pdf_parser.parse_regulation()` と `parse_procedure()` を両方走らせる
    2. 得られたチャンク数が多い方の構造を採用（圧勝した方）
    3. 両方とも 0 件の場合は `chunk_fallback` でサイズ分割

    Args:
        pdf_path: 解析対象 PDF のフルパス。
        source_document: Chunk の `source_document` に格納する論理名（通常はファイル名）。

    Returns:
        生成された Chunk のリスト。embedding は空のまま返す
        （呼び出し側で `embedder.embed_batch` を実行すること）。
    """
    from src import pdf_parser

    articles = pdf_parser.parse_regulation(pdf_path)
    sections = pdf_parser.parse_procedure(pdf_path)

    reg_inputs = list(
        pdf_parser.iter_regulation_chunks_input(articles, source_document)
    )
    proc_inputs = list(
        pdf_parser.iter_procedure_chunks_input(sections, source_document)
    )

    if len(reg_inputs) >= len(proc_inputs) and len(reg_inputs) > 0:
        winner_inputs = reg_inputs
        logger.info(
            "chunk_pdf: %s -> regulation structure (%d chunks)",
            source_document,
            len(reg_inputs),
        )
    elif len(proc_inputs) > 0:
        winner_inputs = proc_inputs
        logger.info(
            "chunk_pdf: %s -> procedure structure (%d chunks)",
            source_document,
            len(proc_inputs),
        )
    else:
        # 両構造で 0 件 → 全文を取得してサイズ分割
        logger.warning(
            "chunk_pdf: %s -> no structure detected, falling back to size-based chunking",
            source_document,
        )
        raw_text = _read_pdf_full_text(pdf_path)
        return chunk_fallback(raw_text, "doc", source_document)

    chunks: list[Chunk] = []
    for input_dict in winner_inputs:
        chunks.extend(chunk_doc(input_dict))
    return chunks


def _read_pdf_full_text(pdf_path: str) -> str:
    """PDF から全文をプレーンテキストで取得する（フォールバック用）。"""
    from src import pdf_parser

    lines, _ = pdf_parser._read_pdf_lines(pdf_path)
    return "\n".join(lines)


def chunk_fallback(
    text: str,
    source_type: SourceType,
    source_document: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """`RecursiveCharacterTextSplitter` で素朴に分割する保険ルート。"""
    if not text.strip():
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "langchain-text-splitters is required for chunk_fallback"
        ) from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    pieces = splitter.split_text(text)
    slug = _slugify(source_document)
    kind: IndexKind = "faq" if source_type == "faq" else "doc"
    return [
        Chunk(
            chunk_id=f"{source_type}_{slug}_fallback_{idx}",
            content=piece,
            source_document=source_document,
            index_kind=kind,
            source_type=source_type,
        )
        for idx, piece in enumerate(pieces)
    ]


_DISPATCHERS = {
    "faq": chunk_faq,
    "doc": chunk_doc,
}


def chunk_all(sources: list[dict]) -> list[Chunk]:
    """`data_loader.load_all_sources` の戻り値を一括チャンク化する。"""
    chunks: list[Chunk] = []
    for src in sources:
        source_type = src.get("source_type")
        if not isinstance(source_type, str):
            logger.warning("Missing source_type, skipping: %s", src)
            continue
        dispatcher = _DISPATCHERS.get(source_type)
        if dispatcher is None:
            logger.warning("Unknown source_type %s, skipping", source_type)
            continue
        chunks.extend(dispatcher(src))
    logger.info("Produced %d chunks from %d sources", len(chunks), len(sources))
    return chunks
