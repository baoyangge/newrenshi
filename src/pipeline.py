# ============================================================
# メイン処理 - RAGパイプライン実行
# ============================================================
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.clients import get_client, get_chat_deployment
from src.parameters import (
    SC_MIN, TH_HIGH, TH_LOW, TH_MID_LOW, MAX_LOOP,
    TOP_K, SEARCH_MODE, SOURCE_TYPE_FILTER,
    conversation_history
)
from src.helpers import (
    # is_question_content,  # ← 削除：不要になったためコメントアウト
    build_evidence_text,
    generate_provisional_answer,
    evaluate_answer,
    calc_th_score,
    generate_condition_question,
    generate_supplement_question,
    filter_by_similarity,
    display_df,
    determine_answer_type,
)
from src.retriever import search


# ============================================================
# ソース情報収集
# ============================================================

def _collect_sources(candidates: list) -> list[dict]:
    """候補リストからソース情報（source_document + chunk_id）を収集する"""
    sources = []
    seen = set()
    for c in candidates:
        result_obj = c.get("result_obj")
        if result_obj:
            src_doc = getattr(result_obj, 'source_document', None) or 'N/A'
            chunk_id = getattr(result_obj, 'chunk_id', None) or 'N/A'
            section = getattr(result_obj, 'section', None) or getattr(result_obj, 'category_l1', None) or ''
            
            key = f"{src_doc}_{chunk_id}"
            if key not in seen:
                sources.append({
                    "source_document": src_doc,
                    "chunk_index": chunk_id,
                    "section": section
                })
                seen.add(key)
    return sources


def run_rag_pipeline(query: str, additional_info: str = None, is_retry: bool = False):
    """
    RAGパイプラインを実行する
    """
    azure_client = get_client()
    CHAT_DEPLOYMENT = get_chat_deployment()
    
    # === Step 1: クエリの準備 ===
    if not is_retry:
        conversation_history["original_query"] = query
        conversation_history["additional_info"] = []
        conversation_history["iteration"] = 0
        conversation_history["last_candidates"] = []
        conversation_history["last_action"] = ""
        full_query = query
    else:
        if additional_info:
            conversation_history["additional_info"].append(additional_info)
        conversation_history["iteration"] += 1
        full_query = conversation_history["original_query"]
        if conversation_history["additional_info"]:
            full_query += "\n\n追加情報: " + " ".join(conversation_history["additional_info"])
    
    print(f"{'='*60}")
    print(f"📝 Step 1: ユーザー質問開始")
    print(f"   イテレーション: {conversation_history['iteration']} / {MAX_LOOP}")
    print(f"   元のクエリ: {conversation_history['original_query']}")
    print(f"   完全なクエリ: {full_query}")
    print(f"{'='*60}")
    
    # === ループ回数チェック ===
    if conversation_history["iteration"] >= MAX_LOOP:
        print(f"\n⚠️ ループ回数が上限({MAX_LOOP})に達しました。")
        if conversation_history["last_candidates"]:
            best_candidate = max(conversation_history["last_candidates"], key=lambda x: x["th_score"])
            
            answer_text = best_candidate["answer"]
            answer_type_result = determine_answer_type(answer_text)
            final_answer_type = answer_type_result["type"]
            print(f"   📋 回答タイプ（内容ベース）: {'手順提示型' if final_answer_type == 'procedure' else '標準回答型'}")
            
            return {
                "action": "output_answer",
                "answer": answer_text,
                "th_score": best_candidate["th_score"],
                "source": best_candidate.get("section", "N/A"),
                "note": f"ループ上限到達により最高TH候補を出力",
                "question_type": final_answer_type,
                "sources": _collect_sources([best_candidate])
            }

    
    # === Step 2: RAG検索 ===
    print(f"\n🔍 Step 2: RAG検索実行中...")
    results = search(
        full_query,
        top_k=TOP_K,
        search_mode=SEARCH_MODE,
        source_type_filter=SOURCE_TYPE_FILTER,
    )
    print(f"   検索結果: {len(results)} 件取得")
    
    df_search = pd.DataFrame([
        {
            "rank": i + 1,
            "score": round(r.score, 4),
            "source_document": r.source_document,
            "section": r.section,
            "content_excerpt": r.content[:80],
        }
        for i, r in enumerate(results)
    ])
    display_df(df_search, "📊 検索結果一覧:")
    
    # === Step 3: 類似度判定 ===
    print(f"\n📊 Step 3: 類似度判定 (閾値: {SC_MIN})...")
    max_similarity = max([r.score for r in results]) if results else 0
    print(f"   最大類似度: {max_similarity:.4f}")
    
    if max_similarity < SC_MIN:
        print(f"   ❌ 全ての候補の類似度が閾値 {SC_MIN} 以下です")
        return {
            "action": "cannot_answer",
            "message": "申し訳ございませんが、ご質問に適した情報が見つかりませんでした。",
            "max_similarity": max_similarity,
            "results": results
        }
    
    # === Step 4: SC_MIN未満を除外 ===
    print(f"\n🗑️ Step 4: SC_MIN未満の候補を除外...")
    filtered_results = filter_by_similarity(results, SC_MIN)
    removed_by_similarity = len(results) - len(filtered_results)
    print(f"   フィルタ前: {len(results)} 件 → フィルタ後: {len(filtered_results)} 件")
    
    # === Step 4.5: 削除（質問形式チェックは不要）===
    # 注意：質問判定ロジックを削除しました
    print(f"\n📋 Step 4.5: スキップ（質問形式チェック不要）")
    print(f"   最終候補数: {len(filtered_results)} 件")
    
    if len(filtered_results) == 0:
        print(f"\n   ❌ フィルタリング後、有効な候補がありません")
        return {
            "action": "cannot_answer",
            "message": "申し訳ございませんが、ご質問に適した回答情報が見つかりませんでした。",
            "removed_by_similarity": removed_by_similarity,
        }
    
    # === Step 5: 仮回答生成・評価 ===
    print(f"\n🤖 Step 5: 仮回答生成・F/G/R評価・THスコア算出...")
    provisional_answers = []
    
    # 回答不可キーワード定義
    CANNOT_ANSWER_KEYWORDS = ["回答できない", "回答できません", "判断できない", "判断できません", "情報がありません", "見つかりません","記載はありません", 
        "確認できません", "確認できませんでした",
        "資料からは確認できません",
        "該当する情報がありません","記載がありません","含まれていません",]

    def _process_one_candidate(r):
        """1つの候補を処理し、回答不可の場合はNoneを返す"""
        evidence = build_evidence_text([r])
        answer, q_type = generate_provisional_answer(full_query, evidence)
        
        # 回答不可キーワードチェック → 含まれていれば除外
        if any(kw in answer for kw in CANNOT_ANSWER_KEYWORDS):
            return None

        scores = evaluate_answer(full_query, evidence, answer)
        th_score = calc_th_score(scores["faithfulness"], scores["groundedness"], scores["relevance"])

        return {
            "chunk_id": r.chunk_id,
            "section": r.section,
            "similarity_score": r.score,
            "answer": answer,
            "question_type": q_type,
            "faithfulness": scores["faithfulness"],
            "groundedness": scores["groundedness"],
            "relevance": scores["relevance"],
            "th_score": th_score,
            "result_obj": r
        }

    # 並列処理
    max_workers = min(10, len(filtered_results))
    excluded_count = 0  # 除外された候補数をカウント
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_process_one_candidate, r) for r in filtered_results]
        for idx, fut in enumerate(as_completed(futures), 1):
            try:
                cand = fut.result()
                if cand is None:
                    excluded_count += 1
                    print(f"   除外 {idx}/{len(filtered_results)} → 回答不可キーワードを含む")
                    continue
                provisional_answers.append(cand)
                print(f"   完了 {idx}/{len(filtered_results)} → TH:{cand['th_score']:.3f}")
            except Exception as e:
                print(f"   ⚠️ 候補処理エラー: {e}")

    # 除外結果を表示
    if excluded_count > 0:
        print(f"\n   📊 回答不可キーワードにより除外: {excluded_count} 件")

    provisional_answers.sort(key=lambda x: x["th_score"], reverse=True)
    
    if len(provisional_answers) == 0:
        print(f"\n   ❌ 有効な仮回答候補がありません → 回答不可")
        conversation_history["last_action"] = "cannot_answer"
        return {
            "action": "cannot_answer",
            "message": "申し訳ございませんが、ご質問に対する適切な回答を生成できませんでした。",
            "reason": "全ての候補が回答不可キーワードを含んでいたため除外されました"
        }
    
    conversation_history["last_candidates"] = provisional_answers

    df_eval = pd.DataFrame([
        {
            "section": ((pa.get("section") or "")[:30] + "...") if len(pa.get("section") or "") > 30 else (pa.get("section") or ""),
            "similarity": round(pa.get("similarity_score", 0), 4),
            "F": round(pa.get("faithfulness", 0), 3),
            "G": round(pa.get("groundedness", 0), 3),
            "R": round(pa.get("relevance", 0), 3),
            "TH": round(pa.get("th_score", 0), 3),
            "answer_excerpt": ((pa.get("answer") or "")[:50] + "...") if len(pa.get("answer") or "") > 50 else (pa.get("answer") or ""),
        }
        for pa in provisional_answers
    ])

    display_df(df_eval, "📊 評価結果一覧:")
    
    # === Step 6: THスコアによる分岐 === 
    print(f"\n🔀 Step 6: THスコアによる条件分岐...")
    
    top1 = provisional_answers[0]
    high_th_candidates = [pa for pa in provisional_answers if pa["th_score"] >= TH_HIGH]
    all_below_low = all(pa["th_score"] < TH_LOW for pa in provisional_answers)
    
    print(f"   TOP1 THスコア: {top1['th_score']:.3f}")
    print(f"   TH≥{TH_HIGH}の候補数: {len(high_th_candidates)}")
    
    # Case 1: 全候補TH < TH_LOW
    if all_below_low:
        print(f"\n   ❌ Case 1: 全候補のTHスコアが {TH_LOW} 未満 → 回答不可")
        no_answer_response = azure_client.chat.completions.create(
            model=CHAT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": "あなたは丁寧なカスタマーサポートAIです。"},
                {"role": "user", "content": f"以下の質問に対して、適切な回答が見つからなかったことを丁寧に伝えてください。\n\n質問: {full_query}"}
            ],
            temperature=0.3,
        )
        conversation_history["last_action"] = "cannot_answer"
        return {
            "action": "cannot_answer",
            "message": no_answer_response.choices[0].message.content.strip(),
            "th_scores": [pa["th_score"] for pa in provisional_answers]
        }
    
    # Case 2: 唯一高スコア
    if len(high_th_candidates) == 1:
        print(f"\n   ✅ Case 2: 唯一の高スコア候補 → 回答出力")
        conversation_history["last_action"] = "output_answer"
        
        answer_text = high_th_candidates[0]["answer"]
        answer_type_result = determine_answer_type(answer_text)
        final_answer_type = answer_type_result["type"]
        print(f"   📋 回答タイプ（内容ベース）: {'手順提示型' if final_answer_type == 'procedure' else '標準回答型'}")
        
        return {
            "action": "output_answer",
            "answer": answer_text,
            "th_score": high_th_candidates[0]["th_score"],
            "source": high_th_candidates[0]["section"],
            "question_type": final_answer_type,
            "sources": _collect_sources(high_th_candidates)
        }

    
    # Case 3: 複数高スコア
    if len(high_th_candidates) > 1:
        print(f"\n   🔄 Case 3: 複数の高スコア候補 → 条件特定更問")
        condition_question = generate_condition_question(high_th_candidates, full_query)
        conversation_history["last_action"] = "request_condition"
        return {
            "action": "request_condition",
            "message": condition_question,
            "candidates": high_th_candidates,
            "iteration": conversation_history["iteration"],
            "instruction": "ユーザーからの追加情報を元にマッチング処理を行う",
            "sources": _collect_sources(high_th_candidates)
        }
    
    # Case 4: 中間スコア
    if TH_MID_LOW <= top1["th_score"] < TH_HIGH:
        print(f"\n   🔄 Case 4: TOP1のTHスコアが中間範囲 → 情報補足更問")
        supplement_question = generate_supplement_question(top1["answer"], full_query, top1["th_score"])
        conversation_history["last_action"] = "request_supplement"
        return {
            "action": "request_supplement",
            "message": supplement_question,
            "current_answer": top1["answer"],
            "th_score": top1["th_score"],
            "iteration": conversation_history["iteration"],
            "instruction": "追加情報と最初の質問を組み合わせて再検索",
            "sources": _collect_sources([top1])
        }
    
    # Case 5: 低スコア回答
    print(f"\n   ⚠️ Case 5: TOP1のTHスコアが低い → 低信頼度回答")
    conversation_history["last_action"] = "low_confidence_answer"

    answer_text = top1["answer"]
    answer_type_result = determine_answer_type(answer_text)
    final_answer_type = answer_type_result["type"]
    print(f"   📋 回答タイプ（内容ベース）: {'手順提示型' if final_answer_type == 'procedure' else '標準回答型'}")

    return {
        "action": "low_confidence_answer",
        "answer": answer_text,
        "th_score": top1["th_score"],
        "warning": "この回答は信頼度が低い可能性があります",
        "question_type": final_answer_type
    }
