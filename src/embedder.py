"""Azure OpenAI Service の Embedding API ラッパー。

`AzureOpenAI` クライアントは遅延初期化し、モジュール import 時には外部接続を行わない。
バッチ呼び出し・リトライ・スリープでレート制限対策を行う。
"""

from __future__ import annotations

import logging
import os
import time

from src.config import (
    AZURE_OPENAI_API_VERSION,
    EMBEDDING_BATCH_SIZE,
    EMBEDDING_BATCH_SLEEP_SEC,
    EMBEDDING_MAX_RETRIES,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


class AzureOpenAIError(Exception):
    """Azure OpenAI API 呼び出しのエラー。"""


_client = None  # 型: openai.AzureOpenAI | None
_deployment_name: str | None = None


def _get_client():  # type: ignore[no-untyped-def]
    """`AzureOpenAI` クライアントを遅延初期化する。

    環境変数（ローカル開発）または事前にセットされた `set_client()` を優先する。
    どちらも無ければ実行時に `AzureOpenAIError` を送出する。
    """
    global _client
    if _client is not None:
        return _client
    try:
        from openai import AzureOpenAI
    except ImportError as exc:  # pragma: no cover
        raise AzureOpenAIError("openai package is required") from exc

    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION)
    if not endpoint or not api_key:
        raise AzureOpenAIError(
            "Azure OpenAI credentials missing. "
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY, or call set_client()."
        )
    _client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )
    return _client


def set_client(client: object, deployment_name: str | None = None) -> None:
    """テストや Databricks Notebook から外部初期化したクライアントを差し込む。"""
    global _client, _deployment_name
    _client = client
    if deployment_name is not None:
        _deployment_name = deployment_name


def _deployment() -> str:
    """Embedding デプロイ名（Azure OpenAI のモデル指定）を返す。"""
    if _deployment_name:
        return _deployment_name
    return os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", EMBEDDING_MODEL)


def _call_with_retry(texts: list[str]) -> list[list[float]]:
    """embeddings.create をリトライ付きで呼び出す。

    8192トークンを超えるテキストは自動的に切り詰める。
    """
    # トークン数上限を超える可能性のあるテキストを切り詰める
    # 簡易的に文字数で制限（1トークン ≈ 0.75文字として、8192トークン ≈ 6000文字）
    MAX_CHARS = 6000
    truncated_texts = []
    for i, text in enumerate(texts):
        if len(text) > MAX_CHARS:
            logger.warning(
                "Text[%d] exceeds %d chars (%d chars), truncating",
                i,
                MAX_CHARS,
                len(text),
            )
            truncated_texts.append(text[:MAX_CHARS])
        else:
            truncated_texts.append(text)

    last_error: Exception | None = None
    for attempt in range(1, EMBEDDING_MAX_RETRIES + 1):
        try:
            client = _get_client()
            response = client.embeddings.create(
                model=_deployment(), input=truncated_texts
            )
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001 — API 系の広い例外を捕捉
            last_error = exc
            wait = 2 ** (attempt - 1)
            logger.warning(
                "Azure OpenAI embedding attempt %d/%d failed: %s",
                attempt,
                EMBEDDING_MAX_RETRIES,
                exc,
            )
            if attempt < EMBEDDING_MAX_RETRIES:
                time.sleep(wait)
    raise AzureOpenAIError(
        f"Embedding failed after retries: {last_error}"
    ) from last_error


def embed_text(text: str) -> list[float]:
    """単一テキストのベクトルを返す。

    Args:
        text: 空文字でないテキスト。

    Raises:
        ValueError: 入力が空。
        AzureOpenAIError: リトライ後も失敗した場合。
    """
    if not text:
        raise ValueError("text must be non-empty")
    return _call_with_retry([text])[0]


def embed_batch(texts: list[str]) -> list[list[float]]:
    """複数テキストをバッチでベクトル化する。

    バッチサイズ・スリープは `config.EMBEDDING_BATCH_SIZE` と
    `EMBEDDING_BATCH_SLEEP_SEC` に従う。
    """
    if not texts:
        return []
    vectors: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, EMBEDDING_BATCH_SIZE):
        batch = texts[start : start + EMBEDDING_BATCH_SIZE]
        vectors.extend(_call_with_retry(batch))
        logger.info(
            "Embedded %d/%d texts", min(start + EMBEDDING_BATCH_SIZE, total), total
        )
        if start + EMBEDDING_BATCH_SIZE < total:
            time.sleep(EMBEDDING_BATCH_SLEEP_SEC)
    return vectors
