# ============================================================
# クライアント初期化
# ============================================================
import os
from openai import AzureOpenAI

# モジュールレベル変数
azure_client: AzureOpenAI = None
CHAT_DEPLOYMENT: str = None


def init_azure_client(dbutils) -> tuple[AzureOpenAI, str]:
    """
    Azure OpenAIクライアントを初期化する
    
    Args:
        dbutils: Databricks dbutils オブジェクト
    
    Returns:
        (azure_client, CHAT_DEPLOYMENT)
    """
    global azure_client, CHAT_DEPLOYMENT
    
    # Databricks認証情報
    os.environ["DATABRICKS_HOST"] = dbutils.secrets.get(
        scope="rag-prototype", key="databricks-host"
    )
    if "DATABRICKS_TOKEN" not in os.environ:
        raise ValueError("DATABRICKS_TOKEN が .env ファイルに設定されていません")

    # Azure OpenAI認証情報
    os.environ["AZURE_OPENAI_ENDPOINT"] = dbutils.secrets.get(
        scope="rag-prototype", key="azure-openai-endpoint"
    )
    os.environ["AZURE_OPENAI_API_KEY"] = dbutils.secrets.get(
        scope="rag-prototype", key="azure-openai-api-key"
    )
    os.environ["AZURE_OPENAI_API_VERSION"] = dbutils.secrets.get(
        scope="rag-prototype", key="azure-openai-api-version"
    )
    os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = dbutils.secrets.get(
        scope="rag-prototype", key="azure-openai-embedding-deployment"
    )
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] = dbutils.secrets.get(
        scope="rag-prototype", key="azure-openai-chat-deployment"
    )

    azure_client = AzureOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    )

    CHAT_DEPLOYMENT = os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not CHAT_DEPLOYMENT:
        raise ValueError("AZURE_OPENAI_CHAT_DEPLOYMENT が取得できませんでした")
    
    print(f"✅ Chat Deployment: {CHAT_DEPLOYMENT}")
    
    return azure_client, CHAT_DEPLOYMENT


def get_client() -> AzureOpenAI:
    """クライアントを取得"""
    if azure_client is None:
        raise RuntimeError("azure_client が初期化されていません。init_azure_client() を先に呼び出してください。")
    return azure_client


def get_chat_deployment() -> str:
    """Chat Deployment名を取得"""
    if CHAT_DEPLOYMENT is None:
        raise RuntimeError("CHAT_DEPLOYMENT が初期化されていません。init_azure_client() を先に呼び出してください。")
    return CHAT_DEPLOYMENT
