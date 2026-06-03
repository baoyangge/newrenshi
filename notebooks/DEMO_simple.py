# Databricks notebook source
# DBTITLE 1,環境変数設定
# ============================================================
# Cell 1: 環境変数
# ============================================================
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# 1. まずクラスター環境変数から REPO_ROOT を取得
REPO_ROOT = os.getenv("REPO_ROOT")

# 2. 環境変数に無い場合は、ノートブックパスから .env の場所を推定
if not REPO_ROOT:
    
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    notebook_path_obj = Path(notebook_path)
    
    # notebooks ディレクトリの親がプロジェクトルート
    repo_root_without_workspace = notebook_path_obj.parent.parent
    
    # /Workspace プレフィックスを追加してファイルシステムアクセス可能にする
    inferred_root = Path("/Workspace") / Path(*repo_root_without_workspace.parts[1:])
    env_file_path = inferred_root / ".env"
    
    print(f"📍 ノートブックパス: {notebook_path}")
    print(f"📂 推定された .env パス: {env_file_path}")
    
    if env_file_path.exists():
        load_dotenv(dotenv_path=env_file_path)
        REPO_ROOT = os.getenv("REPO_ROOT")
        if REPO_ROOT:
            print(f"✅ REPO_ROOT を .env ファイルから読み込みました: {REPO_ROOT}")
        else:
            raise EnvironmentError(
                f".env ファイル ({env_file_path}) には REPO_ROOT が定義されていません。"
            )
    else:
        raise FileNotFoundError(
            f".env ファイルが見つかりません: {env_file_path}\n"
            "クラスター環境変数に REPO_ROOT を設定するか、.env ファイルを作成してください。"
        )
else:
    print(f"✅ REPO_ROOT をクラスター環境変数から取得: {REPO_ROOT}")

# 3. sys.path に追加
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)


# COMMAND ----------

# DBTITLE 1,Secrets 取得・クライアント初期化
import importlib
from src import clients, embedder, retriever, parameters

importlib.reload(clients)
importlib.reload(parameters)

# 初始化Azure客户端
azure_client, CHAT_DEPLOYMENT = clients.init_azure_client(dbutils)

# 设置Embedding
embedder.set_client(
    azure_client,
    deployment_name=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
)

# COMMAND ----------

# DBTITLE 1,import
import importlib
from src import helpers, pipeline, matching

importlib.reload(helpers)
importlib.reload(pipeline)
importlib.reload(matching)

from src.pipeline import run_rag_pipeline
from src.matching import process_condition_matching
from src.parameters import conversation_history

print("✅ 全てのモジュールをロードしました")

def print_sources(sources: list, title: str = "📚 根拠ファイル:"):
    print(title)
    for src in sources:
        if isinstance(src, dict):
            doc = src.get("source_document", "N/A")
            chunk = src.get("chunk_index", "N/A")
            print(f"   • {doc} (chunk_index: {chunk})")
        else:
            print(f"   • {src}")


# COMMAND ----------

# DBTITLE 1,実行例 - 初回質問
# ============================================================
# Cell 8: 実行例 - 初回質問
# ============================================================

# === ユーザー質問 ===
#OLD
#QUERY = "Web-WiDEの一括帳票初期画面から収納ホーム画面に戻る方法を教えてください。"    #完全FAQ一致の質問  VVVVVVVVVVVVVVV　#唯一回答パターン1
#QUERY = " みんなのMYポータルでロック解除した場合、登録メールアドレスにロック解除完了の通知メールは届きますか"    #完全FAQ一致の質問  VVVVVVVVV #唯一回答パターン2
#QUERY = " MYポータルでロック解除後、登録メールアドレスに解除完了のメールは届きますか"    #完全FAQ一致の質問  VVVVVVVVV #唯一回答パターン2
#QUERY = " みんなのＭＹポータル方法について教えてください。"    #1372601052　追加情報:積立金残高はどこで確認できますか     #追加情報パターン
#QUERY = "テスラ今のCEOはだれですか？"    #回答できない質問XXXXXXXXXX  追加情報：テスラはNASDAQのTSLAです   #回答できないパターン
#QUERY = "  退職後商品のについて教えてください"    #複数候補パターン　　　追加情報：契約の更新条件（更新可能年齢や保険料の再計算など）
#OLD


#New
QUERY = "団体がん保障保険の試算の際、漢字氏名・カナ氏名は必要か"  #唯一回答  標準回答型
#QUERY = "契約申込にあたっての重要事項の説明、および定款、約款、ご契約のしおりの事前配布の手順は具体的にどのようになっていますか？" #手順提示型　追加：前に出力して配布する必要がある
#QUERY = "団体がん保障保険「試算ソフト」に表示される「算定基準」とは何ですか"  #条件特定 :保保険金額を指します。一律50万円・一律100万円などの具体的な保険金額の設定区分について
#QUERY = "団体がん保障保険"  #条件補足  :団体がん保障保険の新設申請（事前申請）の方法を教えてください。
#QUERY = "テスラ今のCEOはだれですか?"  #回答不能
#QUERY = "車両保険を申請する時に、フルカバー型の場合は車両種類により申請可能判断を行いますか"
result = run_rag_pipeline(QUERY)

print("="*60)
print("🚀 RAGパイプライン実行開始")
print("="*60)
if result["action"] == "output_answer":
    print(f"✅ 回答出力")
    q_type = result.get('question_type', 'N/A')
    q_type_display = '手順提示型' if q_type == 'procedure' else '標準回答型'
    print(f"🏷️ 回答タイプ: {q_type_display}")
    print(f"📝 回答: {result['answer']}")
    print(f"📊 THスコア: {result.get('th_score', 'N/A')}")
    print(f"📄 ソース: {result.get('source', 'N/A')}")
    # ↓↓↓ 追加：根拠ファイル表示 ↓↓↓
    print("-" * 60)
    print_sources(result.get('sources', []))




elif result["action"] == "request_condition":
    print(f"❓ 条件特定更問")
    print(f"📝 質問: {result['message']}")
    print(f"   候補数: {len(result.get('candidates', []))}")
    # ↓↓↓ 追加 ↓↓↓
    print("-" * 60)
    print_sources(result.get('sources', []))




elif result["action"] == "request_supplement":
    print(f"❓ 情報補足更問")
    print(f"📝 質問: {result['message']}")
    print(f"📊 現在のTHスコア: {result.get('th_score', 'N/A')}")
    # ↓↓↓ 追加 ↓↓↓
    print("-" * 60)
    print_sources(result.get('sources', []))


elif result["action"] == "cannot_answer":
    print(f"❌ 回答不可")
    print(f"📝 メッセージ: {result.get('message', 'N/A')}")

elif result["action"] == "low_confidence_answer":
    print(f"⚠️ 低信頼度回答")
    print(f"📝 回答: {result['answer']}")
    print(f"📊 THスコア: {result.get('th_score', 'N/A')}")
    print(f"⚠️ 警告: {result.get('warning', '')}")

    print("-" * 60)
    print_sources(result.get('sources', []))



# COMMAND ----------

# DBTITLE 1,追加情報入力後の再処理
# ============================================================
# Cell 9: 追加情報入力後の再処理（ループ対応版）
# ============================================================

# === 追加情報がある場合の処理例 ===

def process_retry(result: dict, user_additional_info: str):
    """
    追加情報を受けて再処理を行う
    """
    global conversation_history
    
    print("\n" + "="*60)
    print("🔄 追加情報による再処理開始")
    print("="*60)
    print(f"追加情報: {user_additional_info}")
    print(f"現在のイテレーション: {conversation_history['iteration']}")
    
    if result["action"] == "request_condition":
        # Case A: 条件特定更問への回答の場合 → マッチング処理してTHスコア再評価
        print("\n📌 条件特定更問への回答を処理...")
        
        matching_result = process_condition_matching(
            candidates=result["candidates"],
            user_response=user_additional_info,
            original_query=conversation_history["original_query"]
        )

        matching_result['sources'] = result.get('sources', [])
        
        if matching_result["action"] == "output_answer":
            return matching_result
        elif matching_result["action"] == "need_reevaluation":
            # 再評価が必要な場合、Step 3に戻る（実質的に再検索）
            print("\n🔄 再評価のため再検索を実行...")
            return run_rag_pipeline(
                query=conversation_history["original_query"],
                additional_info=user_additional_info,
                is_retry=True
            )
        else:
            return matching_result
    
    elif result["action"] == "request_supplement":
        # Case B: 情報補足更問への回答の場合 → Step 2に戻り再検索
        print("\n📌 情報補足更問への回答を処理...")
        print("   追加情報と最初の質問を組み合わせて再検索")
        
        return run_rag_pipeline(
            query=conversation_history["original_query"],
            additional_info=user_additional_info,
            is_retry=True
        )
    
    else:
        print(f"⚠️ 予期しないaction: {result['action']}")
        return result


# === 実行例 ===
if result["action"] in ["request_condition", "request_supplement"]:
    # ユーザーからの追加情報（実際の運用ではユーザー入力を受け取る）
    #USER_ADDITIONAL_INFO = "受給権者本人の住所変更です"
    #USER_ADDITIONAL_INFO = "積立金残高はどこで確認できますか"
    #USER_ADDITIONAL_INFO = " 団体の住所変更（会社移転）"
    #USER_ADDITIONAL_INFO = " 積立金残高はどこで確認できますか"
    USER_ADDITIONAL_INFO = "団体がん保障保険の新設申請（事前申請）の方法を教えてください。"
    
    # 再処理実行
    retry_result = process_retry(result, USER_ADDITIONAL_INFO)
    
    print("\n" + "="*60)
    print("📋 再処理後の結果:")
    print("="*60)
    print(f"Action: {retry_result['action']}")
    print(f"Iteration: {conversation_history['iteration']}")
    
    if retry_result["action"] == "output_answer":
        print(f"\n✅ 回答:\n{retry_result['answer']}")
        print(f"\nTHスコア: {retry_result['th_score']:.3f}")
        if retry_result.get('original_th'):
            print(f"元THスコア: {retry_result['original_th']:.3f}")
        if retry_result.get('note'):
            print(f"備考: {retry_result['note']}")
        print("-" * 60)
        print_sources(result.get('sources', []))


    
    # さらに追加情報が必要な場合、ループを続ける
    loop_count = 0
    while retry_result["action"] in ["request_condition", "request_supplement"] and loop_count < 2:
        loop_count += 1
        print(f"\n{'='*60}")
        print(f"🔄 追加ループ {loop_count}")
        print(f"{'='*60}")
        print(f"質問: {retry_result['message']}")
        
        # 追加のユーザー入力（実際の運用では動的に取得）
        additional_info_loop = f"追加情報{loop_count}: より具体的な情報"
        retry_result = process_retry(retry_result, additional_info_loop)
        
        print(f"\nAction: {retry_result['action']}")
        if retry_result["action"] == "output_answer":
            print(f"✅ 回答:\n{retry_result['answer']}")
            print(f"THスコア: {retry_result['th_score']:.3f}")
            print("-" * 60)
            print_sources(result.get('sources', []))



else:
    print("\n✅ 追加処理は不要です")


# COMMAND ----------

# ============================================================
# Cell 9: 追加情報入力後の再処理
# ============================================================

# ╔════════════════════════════════════════════════════════════╗
# ║  👇 追加情報をここに入力してください                           ║
# ╚════════════════════════════════════════════════════════════╝

def process_retry(result: dict, user_additional_info: str):
    """
    追加情報を受けて再処理を行う
    """
    global conversation_history
    
    print("\n" + "="*60)
    print("🔄 追加情報による再処理開始")
    print("="*60)
    print(f"追加情報: {user_additional_info}")
    print(f"現在のイテレーション: {conversation_history['iteration']}")
    
    if result["action"] == "request_condition":
        # Case A: 条件特定更問への回答の場合 → マッチング処理してTHスコア再評価
        print("\n📌 条件特定更問への回答を処理...")
        
        matching_result = process_condition_matching(
            candidates=result["candidates"],
            user_response=user_additional_info,
            original_query=conversation_history["original_query"]
        )
        
        matching_result['sources'] = result.get('sources', [])
        if matching_result["action"] == "output_answer":
            return matching_result
        elif matching_result["action"] == "need_reevaluation":
            # 再評価が必要な場合、Step 3に戻る（実質的に再検索）
            print("\n🔄 再評価のため再検索を実行...")
            return run_rag_pipeline(
                query=conversation_history["original_query"],
                additional_info=user_additional_info,
                is_retry=True
            )
        else:
            return matching_result
    
    elif result["action"] == "request_supplement":
        # Case B: 情報補足更問への回答の場合 → Step 2に戻り再検索
        print("\n📌 情報補足更問への回答を処理...")
        print("   追加情報と最初の質問を組み合わせて再検索")
        
        return run_rag_pipeline(
            query=conversation_history["original_query"],
            additional_info=user_additional_info,
            is_retry=True
        )
    
    else:
        print(f"⚠️ 予期しないaction: {result['action']}")
        return result



USER_ADDITIONAL_INFO = "保保険金額を指します。一律50万円・一律100万円などの具体的な保険金額の設定区分について"  # ← ここに追加情報を入力


# 例:
# USER_ADDITIONAL_INFO = "契約移管はどのように反映されますか？"
# USER_ADDITIONAL_INFO = "対象期間は2024年4月です"

# ╔════════════════════════════════════════════════════════════╗
# ║  以下は自動実行（編集不要）                                   ║
# ╚════════════════════════════════════════════════════════════╝

def execute_additional_processing():
    """追加情報処理を実行"""
    global result, conversation_history
    
    # 入力チェック
    if not USER_ADDITIONAL_INFO.strip():
        print("=" * 60)
        print("⚠️  USER_ADDITIONAL_INFO に追加情報を入力してから再実行してください")
        print("=" * 60)
        print("\n現在の状態:")
        print(f"  - Action: {result.get('action', 'N/A')}")
        if result.get('message'):
            print(f"  - LLMからの質問: {result['message']}")
        return
    
    # アクション確認
    current_action = result.get('action', '')
    if current_action not in ["request_condition", "request_supplement"]:
        print("=" * 60)
        print("✅ 追加処理は不要です")
        print("=" * 60)
        print(f"\n前回のアクション: {current_action}")
        if current_action == "output_answer":
            print(f"回答: {result.get('answer', 'N/A')}")
        elif current_action == "cannot_answer":
            print(f"メッセージ: {result.get('message', 'N/A')}")
        return
    
    # 処理開始
    print("=" * 60)
    print("🔄 追加情報処理を開始します")
    print("=" * 60)
    print(f"\n📝 追加情報: {USER_ADDITIONAL_INFO}")
    print(f"📊 現在のイテレーション: {conversation_history['iteration']}")
    print(f"🔖 前回のアクション: {current_action}")
    print("-" * 60)
    
    # 再処理実行
    result = process_retry(result, USER_ADDITIONAL_INFO)
    
    # 結果表示
    print("\n" + "=" * 60)
    print("📋 処理結果")
    print("=" * 60)
    print(f"Action: {result['action']}")
    print(f"Iteration: {conversation_history['iteration']}")
    
    if result["action"] == "output_answer":
        print("\n" + "-" * 60)
        print("✅ 回答:")
        print("-" * 60)
        print(result['answer'])
        print("-" * 60)
        print(f"THスコア: {result['th_score']:.3f}")
        if result.get('note'):
            print(f"備考: {result['note']}")
        print("-" * 60)
        print_sources(result.get('sources', []))


        print("\n🎉 処理完了しました")
        print("\n🎉 処理完了しました")
    
    elif result["action"] == "cannot_answer":
        print("\n" + "-" * 60)
        print("❌ 回答不可:")
        print("-" * 60)
        print(result['message'])
        print("\n処理を終了します")
    
    elif result["action"] in ["request_condition", "request_supplement"]:
        print("\n" + "-" * 60)
        print("🔄 さらに追加情報が必要です:")
        print("-" * 60)
        print(f"質問: {result['message']}")
        print("-" * 60)
        print("\n💡 次のステップ:")
        print("   1. 上部の USER_ADDITIONAL_INFO を新しい情報に更新")
        print("   2. このセル(Cell 9)を再実行")
    
    elif result["action"] == "low_confidence_answer":
        print("\n" + "-" * 60)
        print("⚠️ 低信頼度回答:")
        print("-" * 60)
        print(result['answer'])
        print("-" * 60)
        print(f"THスコア: {result['th_score']:.3f}")
        if result.get('note'):
            print(f"備考: {result['note']}")
        print("-" * 60)
        print_sources(result.get('sources', []))



# 実行
execute_additional_processing()
