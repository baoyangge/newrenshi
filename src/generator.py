"""Azure OpenAI Chat Completions による回答生成。

`generate_response` は1回のLLM呼び出しで回答生成と即答/更問の判断を同時に行う。
LLM は JSON 形式 {"type": "answer"|"clarification", "content": "..."} を返す。
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import (
    AZURE_OPENAI_API_VERSION,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
)
from src.retriever import RetrievalResult

logger = logging.getLogger(__name__)


class GenerationError(Exception):
    """LLM 回答生成のエラー。"""


@dataclass
class GenerationResult:
    """LLM の応答種別と本文を保持するデータクラス。"""

    type: Literal["answer", "clarification"]
    content: str


_SYSTEM_PROMPT = (
    Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
).read_text(encoding="utf-8")

_FORCE_ANSWER_SUFFIX = "\n（これ以上の確認は行わず、現時点で得られている情報をもとに最善の回答をしてください。）"

_NO_CONTEXT_MESSAGE = "関連するドキュメントが見つかりませんでした。"


_client = None  # 型: openai.AzureOpenAI | None
_deployment_name: str | None = None


def _get_client():  # type: ignore[no-untyped-def]
    """`AzureOpenAI` クライアントを遅延初期化する。"""
    global _client
    if _client is not None:
        return _client
    try:
        from openai import AzureOpenAI
    except ImportError as exc:  # pragma: no cover
        raise GenerationError("openai package is required") from exc

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION)
    if not endpoint or not api_key:
        raise GenerationError("Azure OpenAI credentials missing for chat completion.")
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


def _build_context_block(retrieved: list[RetrievalResult]) -> str:
    """検索結果を LLM プロンプトの Context 部分に整形する。"""
    lines: list[str] = []
    for i, hit in enumerate(retrieved, start=1):
        source_label = hit.source_document
        if hit.section:
            source_label = f"{hit.source_document} / {hit.section}"
        lines.append(f"[{i}] {hit.content}（出典: {source_label}）")
    return "\n".join(lines)


def _build_messages(
    query: str,
    retrieved: list[RetrievalResult],
    conversation_history: list[dict] | None,
    force_answer: bool = False,
) -> list[dict]:
    """messages リストを構築する。

    初回（conversation_history=None）:
        [system, user(検索結果+質問)]

    2回目（会話履歴あり、history=[user:original, assistant:clarification]):
        [system, user(検索結果+original), assistant(clarification), user(query=補足)]
    """
    system_content = _SYSTEM_PROMPT + (_FORCE_ANSWER_SUFFIX if force_answer else "")
    system_msg: dict = {"role": "system", "content": system_content}
    context = _build_context_block(retrieved)

    if not conversation_history:
        context_msg: dict = {
            "role": "user",
            "content": f"[検索結果]\n{context}\n\n[質問]\n{query}",
        }
        return [system_msg, context_msg]

    # 会話履歴がある場合: 最初のユーザーメッセージを検索結果と合わせて再構築
    original_query = conversation_history[0].get("content", query)
    context_msg = {
        "role": "user",
        "content": f"[検索結果]\n{context}\n\n[質問]\n{original_query}",
    }
    prior: list[dict] = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in conversation_history[1:]
    ]
    supplement_msg: dict = {"role": "user", "content": query}
    return [system_msg, context_msg] + prior + [supplement_msg]


def generate_response(
    query: str,
    retrieved: list[RetrievalResult],
    conversation_history: list[dict] | None = None,
    force_answer: bool = False,
) -> GenerationResult:
    """LLM が判断・生成を一体で行い `GenerationResult` を返す（マルチターン対応）。

    LLM は JSON {"type": "answer"|"clarification", "content": "..."} を返す。
    JSON のパースに失敗した場合は type="answer" としてフォールバックする。

    Args:
        query: ユーザーの入力テキスト（初回は質問、2回目は補足情報）。
        retrieved: `retriever.search` の戻り値。
        conversation_history: 直前までの会話メッセージリスト（role/content）。
            初回は None、2回目以降は [user:original, assistant:clarification, ...] を渡す。
        force_answer: True のとき、LLM に更問せず回答するよう強制する。
            更問上限到達時に app.py から渡される。

    Returns:
        `GenerationResult`。type が "answer" なら回答テキスト、"clarification" なら更問テキスト。
    """
    if not query:
        raise ValueError("query must be non-empty")
    if not retrieved:
        return GenerationResult(type="answer", content=_NO_CONTEXT_MESSAGE)

    messages = _build_messages(
        query, retrieved, conversation_history, force_answer=force_answer
    )
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=_deployment(),
            messages=messages,  # type: ignore[arg-type]
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Chat completion failed: %s", exc)
        raise GenerationError(f"Chat completion failed: {exc}") from exc

    try:
        raw_content: str = response.choices[0].message.content or ""
    except (AttributeError, IndexError) as exc:
        logger.error("Unexpected chat response shape: %s", response)
        raise GenerationError("Unexpected chat response shape") from exc

    try:
        parsed = json.loads(raw_content)
        result_type = parsed.get("type", "answer")
        content = parsed.get("content", raw_content)
        if result_type not in ("answer", "clarification"):
            logger.warning("Unexpected type '%s', defaulting to 'answer'", result_type)
            result_type = "answer"
    except json.JSONDecodeError:
        logger.warning(
            "LLM returned non-JSON; treating as plain answer: %.100s", raw_content
        )
        result_type = "answer"
        content = raw_content

    return GenerationResult(type=result_type, content=content)


def generate_answer(query: str, retrieved: list[RetrievalResult]) -> str:
    """シングルターン互換ラッパー。`generate_response` の content を返す。

    Args:
        query: 質問テキスト。
        retrieved: `retriever.search` の戻り値。

    Returns:
        回答テキスト。
    """
    return generate_response(query, retrieved).content
