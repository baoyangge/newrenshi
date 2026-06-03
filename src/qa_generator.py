"""LLM による FAQ 逆生成（PDF 系統モック生成の補助）。"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from src.chunker import Chunk
from src.config import (
    AZURE_OPENAI_API_VERSION,
    FAQ_PER_CHUNK,
    LLM_MODEL,
    MOCK_GEN_LLM_TEMPERATURE,
    MOCK_GEN_MAX_TOKENS,  # LLM_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class QaGenerationError(Exception):
    """FAQ 逆生成のエラー。"""


_FAQ_PROMPT_TEMPLATE = """\
あなたは生命保険会社のコールセンター FAQ 作成担当者です。
以下の手順書/規程の一節をもとに、お客さま/社員が実際に問い合わせそうな質問を {num} 件作成し、
同じ情報源から引ける回答を併記してください。

【制約】
- 質問はキーワード直マッチ/言い換え/文脈理解の3タイプを均等に含める
- 個人情報・実名は含めない
- 回答は情報源の内容のみから構成し、推測を含めない

【情報源】
{content}

【出力形式】
JSON 配列: [{{"question": "...", "answer": "...", "q_type": "direct|paraphrase|context"}}, ...]
"""


_client = None  # 型: openai.AzureOpenAI | None
_deployment_name: str | None = None


def _get_client():  # type: ignore[no-untyped-def]
    global _client
    if _client is not None:
        return _client
    try:
        from openai import AzureOpenAI
    except ImportError as exc:  # pragma: no cover
        raise QaGenerationError("openai package is required") from exc
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION)
    if not endpoint or not api_key:
        raise QaGenerationError("Azure OpenAI credentials missing for qa_generator.")
    _client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return _client


def set_client(client: object, deployment_name: str | None = None) -> None:
    """テスト・Notebook から外部初期化済みクライアントを差し込む。"""
    global _client, _deployment_name
    _client = client
    if deployment_name is not None:
        _deployment_name = deployment_name


def _deployment() -> str:
    if _deployment_name:
        return _deployment_name
    return os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT", LLM_MODEL)


def _extract_json_array(text: str) -> list[dict]:
    """LLM 出力から JSON 配列を抽出する。"""
    text = text.strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON array found in LLM output: {text[:200]}")
    return json.loads(match.group(0))


def _call_llm(prompt: str) -> str:
    """Chat Completions を 1 回呼び出して文字列を返す（リトライ 1 回）。"""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            client = _get_client()
            response = client.chat.completions.create(
                model=_deployment(),
                messages=[{"role": "user", "content": prompt}],
                temperature=MOCK_GEN_LLM_TEMPERATURE,
                max_completion_tokens=MOCK_GEN_MAX_TOKENS,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.warning("FAQ generation attempt %d failed: %s", attempt + 1, exc)
    raise QaGenerationError(f"LLM call failed: {last_error}") from last_error


def generate_faq_from_chunks(
    chunks: list[Chunk],
    num_qa_per_chunk: int = FAQ_PER_CHUNK,
    topic_name: str = "general",
) -> list[dict]:
    """チャンクから FAQ を逆生成する。

    生成失敗したチャンクはログ警告のうえスキップし、残りは継続する。
    """
    qa_list: list[dict] = []
    for chunk in chunks:
        prompt = _FAQ_PROMPT_TEMPLATE.format(
            num=num_qa_per_chunk, content=chunk.content
        )
        try:
            raw = _call_llm(prompt)
            parsed = _extract_json_array(raw)
        except (QaGenerationError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Skipped chunk %s: %s", chunk.chunk_id, exc)
            continue
        for item in parsed:
            qa_list.append(
                {
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "q_type": item.get("q_type", "direct"),
                    "source_chunk_id": chunk.chunk_id,
                    "source_document": chunk.source_document,
                    "topic_name": topic_name,
                }
            )
    logger.info(
        "Generated %d Q&A pairs from %d chunks (topic=%s)",
        len(qa_list),
        len(chunks),
        topic_name,
    )
    return qa_list


def write_faq_mock(qa_list: list[dict], output_path: str, doc_title: str) -> str:
    """Q&A 辞書リストを FAQ Markdown 形式で書き出す。"""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {doc_title}", ""]
    for qa in qa_list:
        question = qa.get("question", "").strip()
        answer = qa.get("answer", "").strip()
        if not question or not answer:
            continue
        lines.extend(
            [
                "---",
                "",
                f"## Q: {question}",
                "",
                f"A: {answer}",
                "",
            ]
        )
    out.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    logger.info("Wrote FAQ mock: %s", out)
    return str(out)
