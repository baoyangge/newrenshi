"""`pdf_parser` の出力をモック Markdown に変換して書き出す。

`chunker` が解釈できる Markdown フォーマットに整形する。
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.pdf_parser import (
    ParsedArticle,
    ParsedSection,
    parse_procedure,
    parse_regulation,
)

logger = logging.getLogger(__name__)


def regulation_to_markdown(articles: list[ParsedArticle], doc_title: str) -> str:
    """`ParsedArticle` リストを規程 Markdown へ整形する。"""
    lines: list[str] = [f"# {doc_title}", ""]
    last_chapter: str | None = None
    for art in articles:
        if art.chapter and art.chapter != last_chapter:
            lines.extend(["", f"## {art.chapter}", ""])
            last_chapter = art.chapter
        heading = art.article_number
        if art.article_title:
            heading = f"{art.article_number}（{art.article_title}）"
        lines.extend([f"### {heading}", "", art.body.strip(), ""])
    return "\n".join(lines).strip() + "\n"


def procedure_to_markdown(sections: list[ParsedSection], doc_title: str) -> str:
    """`ParsedSection` リストを手順書 Markdown へ整形する。"""
    lines: list[str] = [f"# {doc_title}", ""]
    for sec in sections:
        prefix = "#" * (sec.level + 1)
        lines.extend([f"{prefix} {sec.heading}", "", sec.body.strip(), ""])
        for table in sec.tables:
            lines.extend(_table_to_markdown(table))
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def _table_to_markdown(table: list[list[str]]) -> list[str]:
    """二次元配列を GitHub Flavored Markdown のテーブル行に変換する。"""
    if not table:
        return []
    header = table[0]
    body = table[1:] if len(table) > 1 else []
    lines = [
        "| " + " | ".join(cell.replace("\n", " ") for cell in header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for row in body:
        cells = row + [""] * (len(header) - len(row))
        lines.append(
            "| " + " | ".join(cell.replace("\n", " ") for cell in cells) + " |"
        )
    return lines


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def write_regulation_mock(
    pdf_path: str, output_path: str, doc_title: str | None = None
) -> str:
    """規程 PDF を解析し、モック Markdown を書き出す。"""
    out = Path(output_path)
    articles = parse_regulation(pdf_path)
    title = doc_title or Path(pdf_path).stem
    md = regulation_to_markdown(articles, title)
    _write(out, md)
    logger.info("Wrote regulation mock: %s", out)
    return str(out)


def write_procedure_mock(
    pdf_path: str, output_path: str, doc_title: str | None = None
) -> str:
    """手順書 PDF を解析し、モック Markdown を書き出す。"""
    out = Path(output_path)
    sections = parse_procedure(pdf_path)
    title = doc_title or Path(pdf_path).stem
    md = procedure_to_markdown(sections, title)
    _write(out, md)
    logger.info("Wrote procedure mock: %s", out)
    return str(out)
