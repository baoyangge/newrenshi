"""サンプル PDF（規程・手順書）の構造化抽出。

`pdfplumber` を使用。Databricks 上での実行を前提とし、ローカルでも `pdfplumber`
を導入していれば動作する。import は関数内に閉じ込め、未インストール環境で
モジュール import 時に失敗しないようにする。
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ParsedArticle:
    """規程 PDF から抽出した 1 条分の情報。"""

    chapter: str | None
    article_number: str
    article_title: str | None
    body: str


@dataclass
class ParsedSection:
    """手順書 PDF から抽出した 1 セクション分の情報。"""

    level: int
    heading: str
    body: str
    tables: list[list[list[str]]] = field(default_factory=list)


_CHAPTER_RE = re.compile(r"^第(\d+)章\s*(.*)$")
_ARTICLE_RE = re.compile(r"^第(\d+)条(?:[（(]([^）)]+)[）)])?\s*(.*)$")

_LEVEL1_RE = re.compile(r"^([ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])[\.．]?\s*(.+)$")
_LEVEL2_RE = re.compile(r"^[（(](\d+)[）)]\s*(.+)$")
_LEVEL3_RE = re.compile(r"^([アイウエオカキクケコ])[\.．]?\s*(.+)$")

_FOOTER_PATTERNS = (
    re.compile(r"^[BＢ]?[\(（]一般[）\)]"),
    re.compile(r"^頁\s*\d+/\d+"),
    re.compile(r"^\d+/\d+$"),
)


def _read_pdf_lines(pdf_path: str) -> tuple[list[str], list[list[list[list[str]]]]]:
    """PDF からページ単位のテキスト行と表データを抽出する。"""
    try:
        import pdfplumber
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber is required for pdf parsing") from exc

    all_lines: list[str] = []
    all_tables: list[list[list[list[str]]]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            page_lines = []
            for line in text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                if any(p.match(stripped) for p in _FOOTER_PATTERNS):
                    continue
                page_lines.append(stripped)
            all_lines.extend(page_lines)
            try:
                page_tables = page.extract_tables() or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Table extraction failed on %s: %s", pdf_path, exc)
                page_tables = []
            all_tables.append(
                [
                    [
                        [str(cell) if cell is not None else "" for cell in row]
                        for row in tbl
                    ]
                    for tbl in page_tables
                ]
            )
    return all_lines, all_tables


def parse_regulation(pdf_path: str) -> list[ParsedArticle]:
    """規程 PDF を「第N条」単位で構造化する。"""
    lines, _ = _read_pdf_lines(pdf_path)
    articles: list[ParsedArticle] = []
    current_chapter: str | None = None
    current_article_number: str | None = None
    current_article_title: str | None = None
    body_buffer: list[str] = []

    def _flush() -> None:
        if current_article_number is None:
            return
        articles.append(
            ParsedArticle(
                chapter=current_chapter,
                article_number=current_article_number,
                article_title=current_article_title,
                body="\n".join(body_buffer).strip(),
            )
        )

    for line in lines:
        chapter_match = _CHAPTER_RE.match(line)
        if chapter_match:
            current_chapter = (
                f"第{chapter_match.group(1)}章 {chapter_match.group(2)}".strip()
            )
            continue
        article_match = _ARTICLE_RE.match(line)
        if article_match:
            _flush()
            current_article_number = f"第{article_match.group(1)}条"
            current_article_title = article_match.group(2)
            trailing = article_match.group(3).strip()
            body_buffer = [trailing] if trailing else []
            continue
        if current_article_number is None:
            continue
        body_buffer.append(line)
    _flush()
    logger.info("Parsed %d articles from %s", len(articles), pdf_path)
    return articles


def parse_procedure(pdf_path: str) -> list[ParsedSection]:
    """手順書 PDF を「Ⅰ/Ⅱ → (1)(2) → ア/イ」の階層で構造化する。"""
    lines, page_tables = _read_pdf_lines(pdf_path)
    flat_tables = [tbl for page_tbls in page_tables for tbl in page_tbls]

    sections: list[ParsedSection] = []
    current: ParsedSection | None = None

    def _flush() -> None:
        nonlocal current
        if current is not None:
            current.body = current.body.strip()
            sections.append(current)
            current = None

    for line in lines:
        for level, regex in ((1, _LEVEL1_RE), (2, _LEVEL2_RE), (3, _LEVEL3_RE)):
            m = regex.match(line)
            if m:
                _flush()
                heading = line
                current = ParsedSection(level=level, heading=heading, body="")
                break
        else:
            if current is not None:
                current.body = (current.body + "\n" + line) if current.body else line
    _flush()

    if flat_tables and sections:
        sections[-1].tables = flat_tables

    logger.info("Parsed %d sections from %s", len(sections), pdf_path)
    return sections


def iter_procedure_chunks_input(
    sections: list[ParsedSection],
    source_document: str,
) -> Iterator[dict]:
    """ParsedSection リストから chunker.chunk_doc が受け取る dict を生成する。

    階層（Ⅰ → (1) → ア）を追跡し、heading_path / heading_depth / parent_heading_path
    を組み立てる。各セクションを 1 件の dict として yield する。
    本改修で 手順書 / 規程 の区別を廃止したため、`source_type` は常に `"doc"` を返す。

    Yields:
        {
            "source_type": "doc",
            "source_document": str,
            "heading_path": str,        # 例: "Ⅰ. 申込受付 / (1) 必要書類 / ア. 本人確認"
            "heading_depth": int,       # 1, 2, 3
            "parent_heading_path": str | None,
            "section": str,             # heading そのまま
            "content": str,             # body
        }
    """
    # レベルごとの現在見出しスタック（index 0..2 が level 1..3 に対応）
    stack: list[str | None] = [None, None, None]
    for sec in sections:
        # 自分以下の階層をクリアし、自階層を更新
        stack[sec.level - 1] = sec.heading
        for i in range(sec.level, 3):
            stack[i] = None
        path_parts = [s for s in stack[: sec.level] if s]
        heading_path = " / ".join(path_parts)
        parent_parts = [s for s in stack[: sec.level - 1] if s]
        parent_heading_path = " / ".join(parent_parts) if parent_parts else None

        body = sec.body.strip()
        if not body:
            continue

        yield {
            "source_type": "doc",
            "source_document": source_document,
            "heading_path": heading_path,
            "heading_depth": sec.level,
            "parent_heading_path": parent_heading_path,
            "section": sec.heading,
            "content": body,
        }


def iter_regulation_chunks_input(
    articles: list[ParsedArticle],
    source_document: str,
) -> Iterator[dict]:
    """ParsedArticle リストから chunker.chunk_doc が受け取る dict を生成する。

    本改修で 手順書 / 規程 の区別を廃止したため、`source_type` は常に `"doc"` を返す。

    Yields:
        {
            "source_type": "doc",
            "source_document": str,
            "heading_path": str,        # 例: "第1章 解約 / 第3条 解約手続"
            "heading_depth": int,       # 章=1 / 条=2 を固定で扱う
            "parent_heading_path": str | None,
            "section": str,             # 例: "第3条"
            "content": str,             # body
        }
    """
    for art in articles:
        body = art.body.strip()
        if not body:
            continue

        article_label = art.article_number
        if art.article_title:
            article_label = f"{art.article_number} {art.article_title}"

        if art.chapter:
            heading_path = f"{art.chapter} / {article_label}"
            parent_heading_path: str | None = art.chapter
        else:
            heading_path = article_label
            parent_heading_path = None

        yield {
            "source_type": "doc",
            "source_document": source_document,
            "heading_path": heading_path,
            "heading_depth": 2,
            "parent_heading_path": parent_heading_path,
            "section": art.article_number,
            "content": body,
        }
