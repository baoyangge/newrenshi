""" パラメータ・定数の一元管理 """

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

config_dir = Path(__file__).parent
project_root = config_dir.parent
env_file = project_root / ".env"

if env_file.exists():
    load_dotenv(dotenv_path=env_file)
elif Path("/Workspace").exists():
    # Databricks 環境のフォールバック。/Workspace が存在する Linux ランタイムのみ。
    # （Windows ローカルでは relative_to("/") が失敗するため `/Workspace` の存在で守る）
    try:
        workspace_env_file = Path("/Workspace") / project_root.relative_to("/") / ".env"
        if workspace_env_file.exists():
            load_dotenv(dotenv_path=workspace_env_file)
    except ValueError:
        # POSIX 絶対パスでない場合は黙ってスキップ（環境変数が未ロードでも以降は default 値で動く）
        pass

# 型エイリアス（全モジュールで共通利用）
# PDF は手順書 / 規程の区別をしないため、source_type は "faq" / "doc" の 2 値のみ。
SourceType = Literal["faq", "doc"]
QuestionSourceType = Literal["faq", "doc", "cross_source"]

# インデックス対象の切替（FAQ Vector Index / DOC Vector Index / 両方）
IndexMode = Literal["faq", "doc", "both"]
# Chunk が属する Vector Index 種別
IndexKind = Literal["faq", "doc"]
# 既存。検索アルゴリズム切替（vector ANN / BM25 ハイブリッド）
SearchMode = Literal["vector", "hybrid"]

# Databricks 接続情報
# SECRETS_SCOPE_NAME のデフォルト値は -2 サフィックス付き（旧版 RAG 検索システムとの衝突回避のため）
SECRETS_SCOPE_NAME: str = os.getenv("SECRETS_SCOPE_NAME", "rag-prototype-2")
DATABRICKS_HOST: str = os.getenv("DATABRICKS_HOST", "")
DATABRICKS_TOKEN: str = os.getenv("DATABRICKS_TOKEN", "")

# Azure 接続情報
AZURE_OPENAI_ENDPOINT: str = os.getenv(
    "AZURE_OPENAI_ENDPOINT", "https://oai-hb-aidatabricks.openai.azure.com"
)
AZURE_OPENAI_API_KEY_SECRET_KEY: str = os.getenv(
    "AZURE_OPENAI_API_KEY_SECRET_KEY", "azure-openai-api-key"
)
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "")

# モデル
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "")
EMBEDDING_DIM: int = 1536
LLM_MODEL: str = os.getenv("LLM_MODEL", "")

# Unity Catalog / Volumes / Delta Table
# SCHEMA_NAME は -2 サフィックス付き（旧版 RAG 検索システムとの衝突回避のため）
CATALOG_NAME: str = "catalog-19300"
SCHEMA_NAME: str = "rag_inquiry_2"
VOLUME_NAME: str = "source_documents"
VOLUME_BASE_PATH: str = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"
VOLUME_PDF_PATH: str = f"{VOLUME_BASE_PATH}/pdfs"
VOLUME_CSV_PATH: str = f"{VOLUME_BASE_PATH}/csvs"

# FAQ / DOC で Delta Table を 2 本に分割
FAQ_DELTA_TABLE_NAME: str = f"{CATALOG_NAME}.{SCHEMA_NAME}.source_chunks_faq"
DOC_DELTA_TABLE_NAME: str = f"{CATALOG_NAME}.{SCHEMA_NAME}.source_chunks_doc"

# DEPRECATED: 単一インデックス時代の Delta Table。
# 旧版 RAG 検索システム（rag_inquiry スキーマ）との共存方針により、
# 本定数は実行時には参照されない。00_environment_setup.py の旧定義クリーンアップ用に残置。
DELTA_TABLE_NAME: str = f"{CATALOG_NAME}.rag_inquiry.source_chunks"

# Vector Search
# エンドポイント名は -2 サフィックス付き（旧版 RAG 検索システムとの衝突回避のため）
VECTOR_SEARCH_ENDPOINT_NAME: str = "rag-inquiry-dev-lab-2"

# FAQ / DOC で Vector Index を 2 本に分割
FAQ_INDEX_NAME: str = f"{CATALOG_NAME}.{SCHEMA_NAME}.faq_chunks"
DOC_INDEX_NAME: str = f"{CATALOG_NAME}.{SCHEMA_NAME}.doc_chunks"

# DEPRECATED: 単一インデックス時代の Vector Index。
# 旧版 RAG 検索システム（rag_inquiry スキーマ）との共存方針により、
# 本定数は実行時には参照されない。残置するのみ。
VECTOR_SEARCH_INDEX_NAME: str = "catalog-19300.rag_inquiry.chunks"

# チャンク・検索パラメータ
DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 50
DEFAULT_TOP_K: int = 10
DEFAULT_DISTANCE_METRIC: str = "cosine"

# 検索モードのデフォルト（app.py 後方互換のため "both"）
DEFAULT_SEARCH_MODE: IndexMode = "both"

# RRF (Reciprocal Rank Fusion) の定数 k
# both モードで FAQ Index と DOC Index の結果を統合する際に使用
RRF_K: int = 60

# both モードで各インデックスから取得する候補件数の倍率。
# top_k=5 / multiplier=2 なら各インデックスから 10 件取得し RRF 統合後に上位 5 件へ絞る。
# 「片方で低ランクだがもう片方で高ランク」な統合スコア最強候補を取り逃さないためのバッファ。
RRF_CANDIDATE_MULTIPLIER: int = 2

# DOC Vector Index で heading_depth が未設定の場合に書き込むセンチネル値。
# Direct Vector Access Index は NULL を扱えないため、明示的に -1 で表現する。
# 読み出し側 (retriever._parse_doc_hit) で None に正規化する。
DOC_HEADING_DEPTH_NULL_SENTINEL: int = -1

# 照会画面固有（応答速度優先でコンテキスト量を抑える）
QUERY_TOP_K: int = 5
LLM_TEMPERATURE: float = 0.2
LLM_MAX_TOKENS: int = 1000

# Embedding レート制限対策（Azure OpenAI API呼び出し時）
EMBEDDING_BATCH_SIZE: int = 100  # embedder.embed_batch()で一度に処理するテキスト数
EMBEDDING_BATCH_SLEEP_SEC: float = 0.1  # バッチ間の待機時間（レート制限回避）
EMBEDDING_MAX_RETRIES: int = 3  # Embedding API呼び出しのリトライ回数

# Vector Search upsert パラメータ（indexer.upsert_chunks()実行時）
VECTOR_SEARCH_UPSERT_BATCH_SIZE: int = (
    100  # 1回のAPI呼び出しで送信するchunk数（SSL接続安定性のため）
)
VECTOR_SEARCH_UPSERT_MAX_RETRIES: int = (
    3  # バッチごとのリトライ回数（ネットワークエラー対策）
)
VECTOR_SEARCH_UPSERT_RETRY_WAIT_SEC: int = 2  # リトライ時の待機秒数（attempt * この値）

# Vector Search クライアント初期化パラメータ（VectorSearchClient()実行時）
VECTOR_SEARCH_CLIENT_INIT_RETRIES: int = (
    3  # クライアント初期化のリトライ回数（接続エラー対策）
)
VECTOR_SEARCH_CLIENT_INIT_WAIT_SEC: int = (
    2  # 初回接続前の待機秒数（ネットワーク安定化）
)
VECTOR_SEARCH_CLIENT_RETRY_WAIT_SEC: int = 3  # リトライ時の待機秒数（attempt * この値）

# データパス
MOCK_DATA_BASE_PATH: str = "data/mock"
SAMPLE_SOURCES_PATH: str = "data/sample_sources"
CATEGORY_SCHEMA_PATH: str = "data/category_schema.yaml"
EVAL_DATASET_PATH: str = "data/eval/questions.json"
EXEMPLAR_DATASET_PATH: str = "data/eval/query_exemplars.json"

# モックデータ自動生成パラメータ
FAQ_PER_CHUNK: int = 2
MOCK_GEN_LLM_TEMPERATURE: float = 0.4
CATEGORY_FAQ_MIN: int = 3
CATEGORY_FAQ_MAX: int = 5
MOCK_GEN_MAX_TOKENS: int = (
    4000  # 規程・手順書は複数条文・章を一度に生成するため照会画面より大きく取る
)

# MLFlow実験パス
# -2 サフィックス付き（旧版 RAG 検索システムとの実験ラン混在を防ぐため）
MLFLOW_EXPERIMENT_NAME: str = (
    "/Workspace/Users/sh1-kawai@meijiyasuda.co.jp/20260520_動作確認テスト/rag_prototype_eval_2"
)


# 照会画面の入力制約
MAX_QUERY_LENGTH: int = 1000

# 更問の最大ターン数（無限ループ防止）
MAX_CLARIFICATION_TURNS: int = 1
