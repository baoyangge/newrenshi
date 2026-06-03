# ============================================================
# 条件特定マッチング処理
# ============================================================
import json

from src.clients import get_client, get_chat_deployment
from src.parameters import MAX_LOOP, TH_HIGH, conversation_history


def process_condition_matching(candidates: list, user_response: str, original_query: str) -> dict:
    """
    条件特定更問後のマッチング処理
    """
    azure_client = get_client()
    CHAT_DEPLOYMENT = get_chat_deployment()
    
    print(f"\n{'='*60}")
    print(f"🔄 条件特定マッチング処理開始...")
    print(f"   イテレーション: {conversation_history['iteration']} / {MAX_LOOP}")
    print(f"   ユーザー回答: {user_response}")
    print(f"{'='*60}")
    
    conversation_history["iteration"] += 1
    conversation_history["additional_info"].append(user_response)
    
    # ループ上限チェック
    if conversation_history["iteration"] >= MAX_LOOP:
        print(f"\n⚠️ ループ回数が上限({MAX_LOOP})に達しました。")
        best_candidate = max(candidates, key=lambda x: x["th_score"])
        return {
            "action": "output_answer",
            "answer": best_candidate["answer"],
            "th_score": best_candidate["th_score"],
            "source": best_candidate.get("section", "N/A"),
            "note": f"ループ上限到達により最高TH候補を出力"
        }
    
    MATCHING_PROMPT = """
以下の候補の中から、ユーザーの追加情報に最も適合するものを選択してください。
選択した候補のインデックス(0始まり)とその理由を返してください。

ユーザーの元の質問: {original_query}
ユーザーの追加情報: {user_response}

候補一覧:
{candidates_text}

出力形式(JSON):
{{"selected_index": 0, "reason": "選択理由", "confidence": 0.95}}
""".strip()
    
    candidates_text = "\n".join([
        f"[{i}] {c['answer'][:200]}... (TH: {c['th_score']:.3f})"
        for i, c in enumerate(candidates)
    ])
    
    response = azure_client.chat.completions.create(
        model=CHAT_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "あなたは分析AIです。JSON形式で回答してください。"},
            {"role": "user", "content": MATCHING_PROMPT.format(
                original_query=original_query,
                user_response=user_response,
                candidates_text=candidates_text
            )}
        ],
        temperature=0.1,
    )
    
    try:
        result_json = response.choices[0].message.content.strip()
        if result_json.startswith("```"):
            result_json = result_json.split("```")[1]
            if result_json.startswith("json"):
                result_json = result_json[4:]
        result = json.loads(result_json)
        
        selected_idx = result.get("selected_index", 0)
        selected_candidate = candidates[selected_idx]
        matching_confidence = result.get("confidence", 0.9)
        
        adjusted_th = selected_candidate["th_score"] * matching_confidence
        
        if adjusted_th >= TH_HIGH:
            return {
                "action": "output_answer",
                "answer": selected_candidate["answer"],
                "th_score": adjusted_th,
                "original_th": selected_candidate["th_score"],
                "matching_reason": result.get("reason", ""),
                "source": selected_candidate.get("section", "N/A")
            }
        else:
            return {
                "action": "need_reevaluation",
                "selected_candidate": selected_candidate,
                "adjusted_th": adjusted_th,
                "iteration": conversation_history["iteration"]
            }
            
    except Exception as e:
        print(f"❌ マッチング処理エラー: {e}")
        best_candidate = max(candidates, key=lambda x: x["th_score"])
        return {
            "action": "output_answer",
            "answer": best_candidate["answer"],
            "th_score": best_candidate["th_score"],
            "source": best_candidate.get("section", "N/A"),
            "warning": "マッチング処理に問題が発生しました"
        }
