# ============================================================
# ヘルパー関数定義
# ============================================================
import json
from typing import Any, List
import pandas as pd  

from src.clients import get_client, get_chat_deployment
from src.parameters import F_WEIGHT, G_WEIGHT, R_WEIGHT

# === プロンプト定義 ===
# === 質問タイプ判定用プロンプト（NEW） ===
QUESTION_TYPE_PROMPT = """
以下の質問が「手順提示型」か「標準回答型」かを判定してください。

質問:
{question}

判定基準:
- 手順提示型: 申請方法、設定手順、業務プロセス、操作案内に関する質問（「方法」「手順」「やり方」「設定」「申請」などのキーワードを含む）
- 標準回答型: 事実確認、仕様確認、定義、締切、手続きの可否など、手順提示型以外の質問

出力形式(JSON):
{{"type": "procedure" または "standard", "reason": "判定理由"}}
""".strip()

# === 手順提示型の回答プロンプト（NEW） === #20260526更新、single turn type verification
PROCEDURE_ANSWER_PROMPT = """
あなたは保険業務の照会対応AIです。
以下の資料にのみ基づいて、質問に対して**簡潔に**回答してください。

【回答形式 - 必ず以下の構成で回答】

■ 回答サマリー
（2～3文で回答の概要を簡潔に記載）

■ 手順
1. （手順1のタイトルのみ - 詳細説明は書かない）
   → 詳細は「ファイル名」chunk_index: xxx を参照

2. （手順2のタイトルのみ）
   → 詳細は「ファイル名」chunk_index: xxx を参照

（以下同様。各手順はタイトルと参照先のみ記載）

■ 補足（任意）
（重要な注意事項があれば1点のみ。なければ省略）

【注意事項】
- 各手順の「詳細説明」は書かない。タイトルと参照chunkのみ記載
- chunk_indexは資料の【】内に記載されている番号を正確に引用
- 資料にない内容は「資料からは確認できません」と明記
- 回答全体を200文字以内に収める
""".strip()


# === 標準回答型の回答プロンプト（NEW） ===
STANDARD_ANSWER_PROMPT = """
あなたは保険業務の照会対応AIです。
以下の資料にのみ基づいて、質問に対して**簡潔に**回答してください。

【回答形式 - 必ず以下の構成で回答】

■ 回答サマリー
（2～3文で結論を簡潔に記載）

■ 根拠
（「ファイル名」chunk_index: xxx を参照 - 複数ある場合は列挙）

■ 補足（任意）
（補足情報があれば1点のみ。なければ省略）

【注意事項】
- chunk_indexは資料の【】内に記載されている番号を正確に引用
- 資料にない内容は「資料からは確認できません」と明記
- 回答全体を150文字以内に収める
""".strip()





EVAL_PROMPT = """
以下の仮回答について、資料とユーザー質問との忠実性(Faithfulness)、資料に根拠があるか(Groundedness)、質問への適切さ(Relevance)を0.0〜1.0のスケールで評価してください。
出力はJSON形式で、keysは"faithfulness", "groundedness", "relevance"としてください。

仮回答:
{answer}

資料:
{evidence}

質問:
{question}
""".strip()

CONDITION_QUESTION_PROMPT = """
以下の複数候補について、どの回答が適切かを特定するための確認質問を生成してください。
ユーザーに追加情報を求める質問を1つ生成してください。

候補一覧:
{candidates}

元の質問:
{question}

出力形式: 確認質問のみを出力してください。
""".strip()

SUPPLEMENT_QUESTION_PROMPT = """
以下の仮回答と元の質問に基づいて、2つの情報を生成してください：

1. 現時点でわかっていること（仮回答から確実に言える内容を簡潔に要約）
2. より正確な回答のための確認質問（最大3問まで）

現在の仮回答:
{answer}

元の質問:
{question}

THスコア: {th_score}（1.0が最高、現在は中程度の信頼度）

出力形式(JSON):
{{
    "current_understanding": "現時点でわかっていることの要約（箇条書きまたは簡潔な文章）",
    "questions": [
        "確認質問1",
        "確認質問2",
        "確認質問3"
    ]
}}

注意：
- current_understanding には、仮回答から確実に言える内容を簡潔にまとめてください。不確実な部分は含めないでください。
- questions は最大3問とし、回答の精度を上げるために本当に必要な情報だけを求めてください。
- 質問は具体的で答えやすいものにしてください。
- 不要な質問は省略し、必要最小限の質問数にしてください。
""".strip()

def is_question_content(content: str, max_length: int = 500) -> dict:
    """コンテンツが質問形式かどうかをLLMで判定する"""
    azure_client = get_client()
    CHAT_DEPLOYMENT = get_chat_deployment()
    
    truncated_content = content[:max_length] if len(content) > max_length else content
    
    try:
        response = azure_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "あなたはテキスト分類AIです。JSON形式で回答してください。"},
                {"role": "user", "content": IS_QUESTION_PROMPT.format(content=truncated_content)}
            ],
            temperature=0.0,
        )
        
        result_json = response.choices[0].message.content.strip()
        if result_json.startswith("```"):
            result_json = result_json.split("```")[1]
            if result_json.startswith("json"):
                result_json = result_json[4:]
        
        result = json.loads(result_json)
        return {
            "is_question": result.get("is_question", False),
            "reason": result.get("reason", "")
        }
    except Exception as e:
        print(f"   ⚠️ 質問判定エラー: {e}")
        return {"is_question": False, "reason": f"判定エラー: {e}"}


def filter_question_candidates(results: list) -> tuple[list, list]:
    """質問形式の候補をフィルタリングして除外する"""
    filtered = []
    removed = []
    
    for r in results:
        judgment = is_question_content(r.content)
        if judgment["is_question"]:
            removed.append({
                "chunk_id": r.chunk_id,
                "section": r.section,
                "source_document": r.source_document,
                "content": r.content,
                "content_excerpt": r.content[:80],
                "reason": judgment["reason"]
            })
        else:
            filtered.append(r)
    
    return filtered, removed


def get_following_chunks_from_delta(chunk_id: str, file_name: str, num_chunks: int = 2) -> List[str]:
    """Delta Tableから、同一ファイル内の後続chunkを取得する"""
    try:
        from src.config import DOC_DELTA_TABLE_NAME
        
        quoted_table_name = ".".join([f"`{part}`" for part in DOC_DELTA_TABLE_NAME.split(".")])
        
        query = f"""
        SELECT content, chunk_index
        FROM {quoted_table_name}
        WHERE source_id = '{file_name}'
        AND CAST(chunk_index AS INT) > {int(chunk_id)}
        ORDER BY CAST(chunk_index AS INT)
        LIMIT {num_chunks}
        """
        # Note: spark は Databricks 環境で自動的に利用可能
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        result = spark.sql(query).collect()
        return [row.content for row in result]
    except Exception as e:
        print(f"   ⚠️ 後続チャンク取得エラー: {e}")
        return []


def build_evidence_text(items: List[Any]) -> str:
    """証拠テキストを構築する（chunk_id付き）"""
    blocks = []
    for r in items:
        content = r.content[:1200]
        # chunk_id を取得（retrieverではchunk_indexをchunk_idに格納）
        chunk_idx = getattr(r, 'chunk_id', 'N/A')
        # sectionがない場合（FAQの場合）はcategory_l1を使用
        section_info = getattr(r, 'section', None) or getattr(r, 'category_l1', None) or 'N/A'
        blocks.append(f"【{r.source_document} (chunk_index: {chunk_idx}) / {section_info}】\n{content}")
    return "\n\n".join(blocks)

#20260526
#20260526
def determine_question_type(question: str) -> dict:
    """
    質問タイプを判定する（手順提示型 vs 標準回答型）
    
    Returns:
        {"type": "procedure" または "standard", "reason": "判定理由"}
    """
    azure_client = get_client()           # ← 追加
    CHAT_DEPLOYMENT = get_chat_deployment()  # ← 追加
    
    try:
        response = azure_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "あなたは質問分類AIです。JSON形式で回答してください。"},
                {"role": "user", "content": QUESTION_TYPE_PROMPT.format(question=question)}
            ],
            temperature=0.0,
        )
        
        result_json = response.choices[0].message.content.strip()
        if result_json.startswith("```"):
            result_json = result_json.split("```")[1]
            if result_json.startswith("json"):
                result_json = result_json[4:]
        
        result = json.loads(result_json)
        return {
            "type": result.get("type", "standard"),
            "reason": result.get("reason", "")
        }
    except Exception as e:
        print(f"   ⚠️ 質問タイプ判定エラー: {e}")
        return {"type": "standard", "reason": f"判定エラー: {e}"}


def generate_provisional_answer(question: str, evidence: str) -> tuple[str, str]:
    """
    仮回答を生成する（回答タイプに応じたフォーマット）
    
    Returns:
        tuple: (回答テキスト, 質問タイプ)
    """
    azure_client = get_client()           # ← 追加
    CHAT_DEPLOYMENT = get_chat_deployment()  # ← 追加
    
    # 質問タイプを判定
    question_type_result = determine_question_type(question)
    question_type = question_type_result["type"]
    print(f"   📋 質問タイプ: {'手順提示型' if question_type == 'procedure' else '標準回答型'}")
    
    # タイプに応じたプロンプトを選択
    if question_type == "procedure":
        system_prompt = PROCEDURE_ANSWER_PROMPT
    else:
        system_prompt = STANDARD_ANSWER_PROMPT
    
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"質問:\n{question}\n\n資料:\n{evidence}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip(), question_type


def generate_provisional_answer(question: str, evidence: str) -> tuple[str, str]:
    """
    仮回答を生成する（回答タイプに応じたフォーマット）
    
    Returns:
        tuple: (回答テキスト, 質問タイプ)
    """
    azure_client = get_client()           # ← 追加
    CHAT_DEPLOYMENT = get_chat_deployment()  # ← 追加
    
    # 質問タイプを判定
    question_type_result = determine_question_type(question)
    question_type = question_type_result["type"]
    print(f"   📋 質問タイプ: {'手順提示型' if question_type == 'procedure' else '標準回答型'}")
    
    # タイプに応じたプロンプトを選択
    if question_type == "procedure":
        system_prompt = PROCEDURE_ANSWER_PROMPT
    else:
        system_prompt = STANDARD_ANSWER_PROMPT
    
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"質問:\n{question}\n\n資料:\n{evidence}"},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip(), question_type


def evaluate_answer(question: str, evidence: str, answer: str) -> dict:
    """回答を評価してF/G/Rスコアを返す"""
    azure_client = get_client()
    CHAT_DEPLOYMENT = get_chat_deployment()
    
    prompt = EVAL_PROMPT.format(question=question, evidence=evidence, answer=answer)
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "あなたは評価者です。JSON形式で回答してください。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
    )
    try:
        eval_json = response.choices[0].message.content.strip()
        if eval_json.startswith("```"):
            eval_json = eval_json.split("```")[1]
            if eval_json.startswith("json"):
                eval_json = eval_json[4:]
        scores = json.loads(eval_json)
        
        result = {}
        for k in ["faithfulness", "groundedness", "relevance"]:
            val = scores.get(k, 0.0)
            if isinstance(val, dict):
                val = val.get("score", val.get("value", 0.0))
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    val = 0.0
            if not isinstance(val, (int, float)):
                val = 0.0
            result[k] = max(0.0, min(1.0, float(val)))
        return result
    except Exception as e:
        print(f"評価解析失敗: {e}")
        return {"faithfulness": 0.0, "groundedness": 0.0, "relevance": 0.0}


def calc_th_score(f: float, g: float, r: float) -> float:
    """THスコアを計算する"""
    return f * F_WEIGHT + g * G_WEIGHT + r * R_WEIGHT


def generate_condition_question(candidates: list, question: str) -> str:
    """条件特定のための更問を生成する"""
    azure_client = get_client()
    CHAT_DEPLOYMENT = get_chat_deployment()
    
    candidates_text = "\n".join([f"- {c['answer'][:100]}..." for c in candidates])
    prompt = CONDITION_QUESTION_PROMPT.format(candidates=candidates_text, question=question)
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def generate_supplement_question(answer: str, question: str, th_score: float) -> str:
    """情報補足のための更問を生成する"""
    azure_client = get_client()
    CHAT_DEPLOYMENT = get_chat_deployment()
    
    prompt = SUPPLEMENT_QUESTION_PROMPT.format(answer=answer, question=question, th_score=th_score)
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def filter_by_similarity(results: list, threshold: float) -> list:
    """類似度閾値でフィルタリングする"""
    return [r for r in results if r.score >= threshold]


def display_df(df: pd.DataFrame, title: str = ""):
    """DataFrameを美しく表示する"""
    try:
        from IPython.display import display, HTML
        if title:
            display(HTML(f"<b>{title}</b>"))
        display(df)
    except ImportError:
        if title:
            print(title)
        print(df.to_string())


def display_message(message: str, style: str = "info"):
    """メッセージを美しく表示する"""
    styles = {
        "info": "background-color: #e7f3fe; padding: 10px; border-left: 4px solid #2196F3;",
        "success": "background-color: #ddffdd; padding: 10px; border-left: 4px solid #4CAF50;",
        "warning": "background-color: #ffffcc; padding: 10px; border-left: 4px solid #ffeb3b;",
        "error": "background-color: #ffdddd; padding: 10px; border-left: 4px solid #f44336;",
    }
    try:
        from IPython.display import display, HTML
        css = styles.get(style, styles["info"])
        display(HTML(f'<div style="{css}">{message}</div>'))
    except ImportError:
        print(message)



# ============================================================
# 回答タイプ判定（回答内容ベース）- NEW
# ============================================================

ANSWER_TYPE_PROMPT = """
以下の回答内容が「手順提示型」か「標準回答型」かを判定してください。

回答内容:
{answer}

判定基準:
- 手順提示型：申請方法、設定手順、業務プロセス、操作案内を説明している回答（番号付き手順、ステップ形式などを含む）
- 標準回答型：事実確認、仕様確認、定義、締切、手続きの可否などを回答している、手順提示型以外の回答

出力形式(JSON):
{{"type": "procedure" または "standard", "reason": "判定理由"}}
""".strip()


def determine_answer_type(answer: str) -> dict:
    """
    回答内容に基づいてタイプを判定する（手順提示型 vs 標準回答型）
    
    Returns:
        {"type": "procedure" または "standard", "reason": "判定理由"}
    """
    azure_client = get_client()
    CHAT_DEPLOYMENT = get_chat_deployment()
    
    try:
        response = azure_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "あなたは回答分類AIです。JSON形式で回答してください。"},
                {"role": "user", "content": ANSWER_TYPE_PROMPT.format(answer=answer[:1500])}
            ],
            temperature=0.0,
        )
        
        result_json = response.choices[0].message.content.strip()
        if result_json.startswith("```"):
            result_json = result_json.split("```")[1]
            if result_json.startswith("json"):
                result_json = result_json[4:]
        
        result = json.loads(result_json)
        return {
            "type": result.get("type", "standard"),
            "reason": result.get("reason", "")
        }
    except Exception as e:
        print(f"   ⚠️ 回答タイプ判定エラー: {e}")
        return {"type": "standard", "reason": f"判定エラー: {e}"}





