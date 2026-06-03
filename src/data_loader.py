"""モックデータ・評価データセットの読み込み。

`data/mock/` 配下の Markdown と `data/eval/questions.json` を構造化辞書に変換する。
ファイルシステム以外への依存はなく、Databricks／ローカル双方で動作する。
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.config import (
    EVAL_DATASET_PATH,
    MOCK_DATA_BASE_PATH,
    QuestionSourceType,
    SourceType,
)

logger = logging.getLogger(__name__)


@dataclass
class EvaluationQuestion:
    """評価データセット1問分の構造化データ。"""

    question_id: str
    question: str
    category: str
    source_type: QuestionSourceType
    ground_truth_chunk_ids: list[str]
    ground_truth_document: str
    difficulty: str
    question_type: str
    expected_answer: str | None = None


# ---------------------------------------------------------------------------
# Markdown パース内部関数
# ---------------------------------------------------------------------------

_FAQ_BOUNDARY_RE = re.compile(r"^##\s+Q:\s*(.+?)\s*$")
_ARTICLE_BOUNDARY_RE = re.compile(r"^###\s+(第\d+条)(?:[（(]([^）)]+)[）)])?\s*(.*)$")
_PROCEDURE_HEADING_RE = re.compile(r"^(#{2,4})\s+(.+?)\s*$")


def _read_markdown(path: Path) -> str:
    """Markdown ファイルを UTF-8 で読み込む。"""
    return path.read_text(encoding="utf-8")


def _strip_doc_title(text: str) -> tuple[str, str]:
    """冒頭の `# <title>` 行を切り出し、残りの本文とともに返す。"""
    lines = text.splitlines()
    title = ""
    body_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            title = stripped[2:].strip()
            body_start = i + 1
            break
        if stripped.startswith("##"):
            body_start = i
            break
    body = "\n".join(lines[body_start:])
    return title, body


# ---------------------------------------------------------------------------
# 公開関数
# ---------------------------------------------------------------------------


def load_faq(path: str) -> list[dict]:
    """単一の FAQ Markdown を Q&A 単位の辞書リストとして返す。

    Args:
        path: FAQ Markdown ファイルへのパス。

    Returns:
        各要素が `{source_type, source_document, section, content}` を持つ dict のリスト。
        FAQ は section を持たないため `None` を設定する。
    """
    file_path = Path(path)
    text = _read_markdown(file_path)
    doc_title, body = _strip_doc_title(text)
    if not doc_title:
        doc_title = file_path.stem

    results: list[dict] = []
    current_question: str | None = None
    current_buffer: list[str] = []

    def _flush() -> None:
        if current_question is None:
            return
        content = "\n".join([f"## Q: {current_question}", "", *current_buffer]).strip()
        results.append(
            {
                "source_type": "faq",
                "source_document": doc_title,
                "section": None,
                "content": content,
            }
        )

    for line in body.splitlines():
        boundary = _FAQ_BOUNDARY_RE.match(line)
        if boundary:
            _flush()
            current_question = boundary.group(1)
            current_buffer = []
            continue
        if current_question is None:
            continue
        if line.strip() == "---":
            continue
        current_buffer.append(line)
    _flush()
    return results


def load_procedure(path: str) -> list[dict]:
    """手順書 Markdown を見出し単位の辞書リストとして返す。

    `##` / `###` / `####` の見出しごとに 1 チャンクの候補を切り出す。
    """
    file_path = Path(path)
    text = _read_markdown(file_path)
    doc_title, body = _strip_doc_title(text)
    if not doc_title:
        doc_title = file_path.stem

    sections: list[dict] = []
    current_heading: str | None = None
    current_buffer: list[str] = []

    def _flush() -> None:
        if current_heading is None:
            return
        content_lines = [current_heading, ""] + current_buffer
        content = "\n".join(content_lines).strip()
        if not content:
            return
        sections.append(
            {
                "source_type": "doc",
                "source_document": doc_title,
                "section": current_heading.lstrip("#").strip(),
                "content": content,
            }
        )

    for line in body.splitlines():
        heading_match = _PROCEDURE_HEADING_RE.match(line)
        if heading_match:
            _flush()
            current_heading = line.rstrip()
            current_buffer = []
            continue
        if current_heading is None:
            continue
        current_buffer.append(line)
    _flush()
    return sections


def load_regulation(path: str) -> list[dict]:
    """規程 Markdown を条文（第N条）単位の辞書リストとして返す。"""
    file_path = Path(path)
    text = _read_markdown(file_path)
    doc_title, body = _strip_doc_title(text)
    if not doc_title:
        doc_title = file_path.stem

    articles: list[dict] = []
    current_article: str | None = None
    current_buffer: list[str] = []

    def _flush() -> None:
        if current_article is None:
            return
        content = "\n".join([current_article, "", *current_buffer]).strip()
        articles.append(
            {
                "source_type": "doc",
                "source_document": doc_title,
                "section": current_article.lstrip("#").strip().split("（")[0].strip(),
                "content": content,
            }
        )

    for line in body.splitlines():
        article_match = _ARTICLE_BOUNDARY_RE.match(line)
        if article_match:
            _flush()
            current_article = line.rstrip()
            current_buffer = []
            continue
        if current_article is None:
            continue
        current_buffer.append(line)
    _flush()
    return articles


def _load_directory(
    base: Path,
    sub: str,
    loader: Callable[[str], list[dict]],
    expected: SourceType,
) -> list[dict]:
    """`base/sub/*.md` を `loader` で読み込み、`_draft` 配下はスキップする。"""
    target_dir = base / sub
    if not target_dir.exists():
        return []
    items: list[dict] = []
    for md_path in sorted(target_dir.glob("*.md")):
        try:
            items.extend(loader(str(md_path)))
        except Exception as exc:
            logger.warning("Failed to load %s: %s", md_path, exc)
    # 読み込み済みの dict に source_type が無いケースを保険で補う
    for item in items:
        item.setdefault("source_type", expected)
    return items


def load_all_sources(base_path: str = MOCK_DATA_BASE_PATH) -> list[dict]:
    """`base_path` 配下の FAQ/手順書/規程を一括ロードする。

    Returns:
        `source_type` 付きの dict のリスト。`_draft/` 配下はスキップ。
    """
    base = Path(base_path)
    results: list[dict] = []
    results.extend(_load_directory(base, "faq", load_faq, "faq"))
    results.extend(_load_directory(base, "procedure", load_procedure, "doc"))
    results.extend(_load_directory(base, "regulation", load_regulation, "doc"))
    logger.info("Loaded %d source items from %s", len(results), base_path)
    return results


def load_evaluation_dataset(path: str = EVAL_DATASET_PATH) -> list[EvaluationQuestion]:
    """評価データセット JSON を読み込む。"""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"Evaluation dataset must be a JSON array: {path}")
    dataset: list[EvaluationQuestion] = []
    for entry in raw:
        dataset.append(
            EvaluationQuestion(
                question_id=entry["question_id"],
                question=entry["question"],
                category=entry["category"],
                source_type=entry["source_type"],
                ground_truth_chunk_ids=list(entry.get("ground_truth_chunk_ids", [])),
                ground_truth_document=entry.get("ground_truth_document", ""),
                difficulty=entry.get("difficulty", "medium"),
                question_type=entry.get("question_type", "single"),
                expected_answer=entry.get("expected_answer"),
            )
        )
    logger.info("Loaded %d evaluation questions from %s", len(dataset), path)
    return dataset
