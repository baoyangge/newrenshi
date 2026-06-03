"""カテゴリ駆動モックデータ自動生成（Feature #6）。

YAML スキーマから「規程 → 手順書 → FAQ」を相互参照可能な形で連鎖生成する。
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from src.config import (
    AZURE_OPENAI_API_VERSION,
    CATEGORY_FAQ_MAX,
    CATEGORY_FAQ_MIN,
    CATEGORY_SCHEMA_PATH,
    LLM_MODEL,
    MOCK_DATA_BASE_PATH,
    MOCK_GEN_LLM_TEMPERATURE,
    MOCK_GEN_MAX_TOKENS,
)

logger = logging.getLogger(__name__)


class CategoryGenerationError(Exception):
    """カテゴリ駆動生成のエラー。"""


# ---------------------------------------------------------------------------
# Pydantic スキーマ
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
_RESERVED_SLUGS = {"draft", "rejected", "_draft"}


def _validate_slug(value: str) -> str:
    """slug 命名規則を検証する。"""
    if not _SLUG_RE.match(value):
        raise ValueError(f"slug must match {_SLUG_RE.pattern}, got {value!r}")
    if value in _RESERVED_SLUGS:
        raise ValueError(f"slug {value!r} is reserved")
    if value.startswith("_") or value.endswith("_"):
        raise ValueError(f"slug {value!r} must not start or end with underscore")
    return value


class TopicSchema(BaseModel):
    """トピック = FAQ 1 ファイルの単位。"""

    slug: str
    name: str
    description: str
    keywords: list[str] = Field(default_factory=list)

    @field_validator("slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        return _validate_slug(v)


class SubcategorySchema(BaseModel):
    """サブカテゴリ = 手順書 1 ファイルの単位。"""

    slug: str
    name: str
    description: str
    topics: list[TopicSchema]

    @field_validator("slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        return _validate_slug(v)

    @field_validator("topics")
    @classmethod
    def _topics_non_empty(cls, v: list[TopicSchema]) -> list[TopicSchema]:
        if not v:
            raise ValueError("subcategory must have at least one topic")
        slugs = [t.slug for t in v]
        if len(slugs) != len(set(slugs)):
            raise ValueError("topic slugs must be unique within a subcategory")
        return v


class CategorySchema(BaseModel):
    """カテゴリ = 規程 1 ファイルの単位。"""

    slug: str
    name: str
    description: str
    regulation_title: str
    subcategories: list[SubcategorySchema]

    @field_validator("slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        return _validate_slug(v)

    @field_validator("subcategories")
    @classmethod
    def _subs_non_empty(cls, v: list[SubcategorySchema]) -> list[SubcategorySchema]:
        if not v:
            raise ValueError("category must have at least one subcategory")
        slugs = [s.slug for s in v]
        if len(slugs) != len(set(slugs)):
            raise ValueError("subcategory slugs must be unique within a category")
        return v


class CategorySchemaRoot(BaseModel):
    """YAML ファイル全体のルート。"""

    categories: list[CategorySchema]

    @field_validator("categories")
    @classmethod
    def _categories_non_empty(cls, v: list[CategorySchema]) -> list[CategorySchema]:
        if not v:
            raise ValueError("at least one category is required")
        slugs = [c.slug for c in v]
        if len(slugs) != len(set(slugs)):
            raise ValueError("category slugs must be unique")
        return v


# ---------------------------------------------------------------------------
# YAML ローダ
# ---------------------------------------------------------------------------


def load_category_schema(yaml_path: str = CATEGORY_SCHEMA_PATH) -> CategorySchemaRoot:
    """YAML を読み込んで Pydantic で検証する。"""
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise CategoryGenerationError("pyyaml is required") from exc

    raw = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8")) or {}
    schema = CategorySchemaRoot.model_validate(raw)
    logger.info("Loaded category schema with %d categories", len(schema.categories))
    return schema


# ---------------------------------------------------------------------------
# Azure OpenAI クライアント
# ---------------------------------------------------------------------------


_client = None  # 型: openai.AzureOpenAI | None
_deployment_name: str | None = None


def _get_client():  # type: ignore[no-untyped-def]
    global _client
    if _client is not None:
        return _client
    try:
        from openai import AzureOpenAI
    except ImportError as exc:  # pragma: no cover
        raise CategoryGenerationError("openai package is required") from exc
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", AZURE_OPENAI_API_VERSION)
    if not endpoint or not api_key:
        raise CategoryGenerationError(
            "Azure OpenAI credentials missing for category_generator."
        )
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


# ---------------------------------------------------------------------------
# プロンプト構築
# ---------------------------------------------------------------------------


def _format_subcategory_block(cat: CategorySchema) -> str:
    lines: list[str] = []
    for sub in cat.subcategories:
        lines.append(f"- サブカテゴリ: {sub.name} — {sub.description}")
        for topic in sub.topics:
            keywords = "、".join(topic.keywords) if topic.keywords else "（指定なし）"
            lines.append(
                f"  - トピック: {topic.name} — {topic.description}（キーワード: {keywords}）"
            )
    return "\n".join(lines)


def _regulation_prompt(category: CategorySchema) -> str:
    block = _format_subcategory_block(category)
    return (
        f"あなたは生命保険会社の規程策定担当です。\n"
        f"以下のカテゴリ構造に基づき、「{category.regulation_title}」の条文を Markdown で出力してください。\n\n"
        "【出力フォーマット】\n"
        f"# {category.regulation_title}\n\n"
        "## 第1章 総則\n"
        "### 第1条（目的）\n"
        "本規程は...\n\n"
        f"## 第2章 <サブカテゴリ名>\n"
        "### 第N条（<トピック名>に関する規程）\n"
        "...\n\n"
        "【カテゴリ構造】\n"
        f"{category.description}\n"
        f"{block}\n\n"
        "【制約】\n"
        "- 各トピックに 1 条以上の条文を割り当てる\n"
        "- 条番号は通し番号（章をまたいで連続）\n"
        "- 個人情報・実名は含めない\n"
    )


def _procedure_prompt(
    category: CategorySchema,
    subcategory: SubcategorySchema,
    regulation_context: str,
) -> str:
    topic_lines = "\n".join(
        f"- トピック: {t.name} — {t.description}" for t in subcategory.topics
    )
    return (
        f"あなたは生命保険会社の業務手順書担当です。\n"
        f"以下の「{subcategory.name}」について、手順書 Markdown を作成してください。\n"
        "条文との整合を保つため、下記の規程抜粋を根拠として引用してください。\n\n"
        "【規程抜粋】\n"
        f"{regulation_context}\n\n"
        "【サブカテゴリ】\n"
        f"{subcategory.description}\n"
        f"{topic_lines}\n\n"
        "【出力フォーマット】\n"
        f"# {subcategory.name} 手順書\n"
        "## Ⅰ 基本的考え方\n"
        "## Ⅱ 手続きの流れ\n"
        "### (1) 受付\n"
        "### (2) 確認\n\n"
        "【制約】\n"
        f'- 各 "## Ⅰ" 直下の説明文で、根拠規程として「{category.regulation_title} 第N条」を 1 箇所以上参照\n'
        "- 表形式の手続きフローは GitHub Flavored Markdown の table を使用\n"
    )


def _faq_prompt(
    category: CategorySchema,
    subcategory: SubcategorySchema,
    topic: TopicSchema,
    regulation_context: str,
    procedure_context: str,
) -> str:
    keywords = "、".join(topic.keywords) if topic.keywords else "（指定なし）"
    return (
        f"あなたは生命保険会社のコールセンター FAQ 作成担当です。\n"
        f"以下のトピックについて、実際に問い合わせが想定される Q&A を {CATEGORY_FAQ_MIN}〜{CATEGORY_FAQ_MAX} 件作成してください。\n"
        "規程・手順書との用語一致を担保するため、下記抜粋を参考にしてください。\n\n"
        "【規程抜粋】\n"
        f"{regulation_context}\n\n"
        "【手順書抜粋】\n"
        f"{procedure_context}\n\n"
        "【トピック】\n"
        f"{topic.description}\n"
        f"キーワード: {keywords}\n\n"
        "【出力フォーマット】\n"
        f"# {topic.name} FAQ\n\n"
        "## Q: <質問文>\n\n"
        "A: <回答本文>\n\n"
        "---\n\n"
        "## Q: ...\n\n"
        "A: ...\n\n"
        "【制約】\n"
        "- 出力は必ず上記フォーマットに従う（`## Q:` 質問見出し + 空行 + `A: ` 回答）\n"
        "- 質問タイプは「直接検索」「言い換え」「文脈理解」を混在させる\n"
        f"- 手順書の手続きステップ名、{category.regulation_title} の条番号を回答に自然に含める\n"
        "- 個人情報・実名は含めない\n"
    )


# ---------------------------------------------------------------------------
# 生成 API
# ---------------------------------------------------------------------------


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
            logger.warning(
                "category_generator LLM attempt %d failed: %s", attempt + 1, exc
            )
    raise CategoryGenerationError(f"LLM call failed: {last_error}") from last_error


def generate_regulation_markdown(
    category: CategorySchema, dry_run: bool = False
) -> str:
    """カテゴリ 1 件から規程 Markdown を生成する。"""
    prompt = _regulation_prompt(category)
    if dry_run:
        print("===== [dry_run] regulation prompt =====")
        print(prompt)
        return ""
    return _call_llm(prompt)


def generate_procedure_markdown(
    category: CategorySchema,
    subcategory: SubcategorySchema,
    regulation_context: str,
    dry_run: bool = False,
) -> str:
    """サブカテゴリ 1 件から手順書 Markdown を生成する。"""
    prompt = _procedure_prompt(category, subcategory, regulation_context)
    if dry_run:
        print("===== [dry_run] procedure prompt =====")
        print(prompt)
        return ""
    return _call_llm(prompt)


def generate_faq_markdown(
    category: CategorySchema,
    subcategory: SubcategorySchema,
    topic: TopicSchema,
    regulation_context: str,
    procedure_context: str,
    dry_run: bool = False,
) -> str:
    """トピック 1 件から FAQ Markdown を生成する。"""
    prompt = _faq_prompt(
        category, subcategory, topic, regulation_context, procedure_context
    )
    if dry_run:
        print("===== [dry_run] faq prompt =====")
        print(prompt)
        return ""
    return _call_llm(prompt)


def _write_markdown(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.strip() + "\n", encoding="utf-8")


def generate_mock_set(
    schema: CategorySchemaRoot,
    output_base: str = MOCK_DATA_BASE_PATH,
    category_filter: list[str] | None = None,
    dry_run: bool = False,
) -> list[Path]:
    """スキーマ全体を走査して規程 → 手順書 → FAQ をセット生成する。

    生成物はすべて `output_base/{regulation,procedure,faq}/_draft/` 配下に
    書き出される。`dry_run=True` のとき書き出しと LLM 呼び出しを行わない。
    """
    base = Path(output_base)
    written: list[Path] = []

    for category in schema.categories:
        if category_filter is not None and category.slug not in category_filter:
            continue

        try:
            reg_md = generate_regulation_markdown(category, dry_run=dry_run)
        except CategoryGenerationError as exc:
            logger.error("Regulation generation failed for %s: %s", category.slug, exc)
            continue
        reg_path = base / "regulation" / "_draft" / f"regulation_{category.slug}.md"
        if not dry_run:
            _write_markdown(reg_path, reg_md)
            written.append(reg_path)

        for sub in category.subcategories:
            try:
                proc_md = generate_procedure_markdown(
                    category, sub, regulation_context=reg_md, dry_run=dry_run
                )
            except CategoryGenerationError as exc:
                logger.error(
                    "Procedure generation failed for %s/%s: %s",
                    category.slug,
                    sub.slug,
                    exc,
                )
                continue
            proc_path = (
                base
                / "procedure"
                / "_draft"
                / f"procedure_{category.slug}_{sub.slug}.md"
            )
            if not dry_run:
                _write_markdown(proc_path, proc_md)
                written.append(proc_path)

            for topic in sub.topics:
                try:
                    faq_md = generate_faq_markdown(
                        category,
                        sub,
                        topic,
                        regulation_context=reg_md,
                        procedure_context=proc_md,
                        dry_run=dry_run,
                    )
                except CategoryGenerationError as exc:
                    logger.error(
                        "FAQ generation failed for %s/%s/%s: %s",
                        category.slug,
                        sub.slug,
                        topic.slug,
                        exc,
                    )
                    continue
                faq_path = (
                    base
                    / "faq"
                    / "_draft"
                    / f"faq_{category.slug}_{sub.slug}_{topic.slug}.md"
                )
                if not dry_run:
                    _write_markdown(faq_path, faq_md)
                    written.append(faq_path)

    logger.info(
        "Mock set generation done. Wrote %d files (dry_run=%s)", len(written), dry_run
    )
    return written
