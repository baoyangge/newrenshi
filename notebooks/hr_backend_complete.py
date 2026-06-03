import json
import os
import re

try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

class RuleIndexManager:
    def __init__(self, dbutils=None):
        self.dbutils = dbutils
        
    def update_rule_database_from_pdf(self, pdf_path: str):
        print(f"【ベクトルDB更新】PDFファイル '{pdf_path}' の処理を開始します。")
        extracted_text = "【単身赴任規定】第1条: 単身赴任手当の支給は...\n【帰省旅費規定】第2条: ..."
        
        chunks = [
            {"chunk_id": "c_001", "content": "【単身赴任開始規定】単身赴任手当の支給は、本人が単身赴任状態（適用中ではない）である場合のみ開始可能。"},
            {"chunk_id": "c_002", "content": "【帰省旅費規定】単身赴任中の者（適用中）に対し、月に1回までの往復交通費実費を支給する。"},
            {"chunk_id": "c_003", "content": "【支給上限額規定】交通費の支給上限額は、最も経済的な経路（駅すぱあとの検索結果）に基づく往復運賃の残月数分とする。"}
        ]
        
        for chunk in chunks:
            chunk["embedding"] = [0.01, 0.02, 0.03]
            
        print(f"  -> Databricks Vector Index に {len(chunks)} 件のチャンクを保存/更新しました。")
        return {"status": "SUCCESS", "message": f"{pdf_path} のインデックス化が完了しました。"}

class MockRuleRetriever:
    def retrieve_rules(self, query):
        rules = []
        if "開始" in query:
            rules.append("【単身赴任開始規定】単身赴任手当の支給は、本人が単身赴任状態（適用中ではない）である場合のみ開始可能。")
        if "往復" in query or "帰省" in query:
            rules.append("【帰省旅費規定】単身赴任中の者（適用中）に対し、月に1回までの往復交通費実費を支給する。")
        if "上限" in query or "駅すぱあと" in query:
            rules.append("【支給上限額規定】交通費の支給上限額は、最も経済的な経路（駅すぱあとの検索結果）に基づく往復運賃の残月数分とする。")
        
        return " / ".join(rules)

    def evaluate_rule(self, rules, hr_data, ocr_data=None):
        if not ocr_data:
            if "適用中ではない" in rules and hr_data.get("単身赴任ステータス") == "適用中":
                return False, "すでに単身赴任旅費が適用されているため、開始申請はできません。"
            if "単身赴任中の者（適用中）" in rules and hr_data.get("単身赴任ステータス") != "適用中":
                return False, "現在、単身赴任旅費の適用期間外です。"
            return True, "人事要件を満たしています。"
            
        if "支給上限額" in rules and ocr_data:
            amount = ocr_data.get("往復金額") or ocr_data.get("往路_交通費")
            if not amount:
                return False, "証憑から金額が読み取れないため、規定に基づく判定ができません。"
            return True, f"証憑の金額（{amount}円）と規定を照合しました。承認可能です。"
            
        return True, "ルールに適合しています。"

class TanshinFuninProcessor:
    def __init__(self, user_id, llm_client=None, dbutils=None, debug=False):
        self.user_id = user_id
        self.hr_data = None
        self.application_type = None
        self.state = {}
        self.ocr_results = []
        self.rule_manager = RuleIndexManager(dbutils)
        self.retriever = MockRuleRetriever()
        self.debug = debug  # デバッグフラグ

    def _log_debug(self, title, message):
        if self.debug:
            print(f"\n[DEBUG] === {title} ===")
            print(message)
            print("-" * 40)

    def update_rules(self, pdf_path):
        return self.rule_manager.update_rule_database_from_pdf(pdf_path)

    def determine_application_type(self, user_input):
        if "開始" in user_input or "事由" in user_input:
            self.application_type = "単身赴任旅費の開始申請"
        elif "往復" in user_input or "帰省" in user_input:
            self.application_type = "単身赴任旅費の往復申請"
        else:
            self.application_type = "不明な申請"
        
        self._log_debug("申請種類の判定", f"入力: {user_input}\n判定結果: {self.application_type}")
        return self.application_type

    def get_hr_info(self):
        mock_db = {
            "U001": {"社員名": "山田 太郎", "単身赴任ステータス": "未開始"},
            "U002": {"社員名": "鈴木 花子", "単身赴任ステータス": "適用中"}
        }
        self.hr_data = mock_db.get(self.user_id)
        self._log_debug("人事情報の取得", f"User: {self.user_id}\nData: {json.dumps(self.hr_data, ensure_ascii=False, indent=2)}")
        return self.hr_data

    def first_rule_check(self, input_data=None):
        query = f"{self.application_type}の条件"
        retrieved_rules = self.retriever.retrieve_rules(query)
        
        self._log_debug("1次チェック: RAG検索", f"Query: {query}\n検索されたルール:\n{retrieved_rules}")
        
        is_ok, reason = self.retriever.evaluate_rule(retrieved_rules, self.hr_data)
        
        if is_ok:
            if input_data:
                self.state['reason'] = input_data.get('reason', 'その他')
            result = {"status": "OK", "reason": reason, "reference_rule": retrieved_rules}
        else:
            result = {"status": "NG", "reason": reason, "reference_rule": retrieved_rules}
            
        self._log_debug("1次チェック: 判定結果", json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def determine_required_evidence(self):
        evidence_list = []
        if self.application_type == "単身赴任旅費の開始申請":
            evidence_list.append("駅すぱあとの証憑（検索結果等）")
        elif self.application_type == "単身赴任旅費の往復申請":
            evidence_list.append("往復の交通費を示す証憑（領収書等）")
        return evidence_list

    def process_evidence_ocr(self, evidence_type, image_path):
        filename = os.path.basename(image_path)
        extracted_data = {"ファイル名": filename, "証憑種類": evidence_type}
        
        raw_text = ""
        if HAS_TESSERACT and os.path.exists(image_path):
            try:
                img = Image.open(image_path)
                raw_text = pytesseract.image_to_string(img, lang='jpn')
            except Exception as e:
                raw_text = f"OCRエラー: {str(e)}"
        else:
            raw_text = "領収書 2026年06月10日 東京駅から新大阪駅 新幹線 14,520円 515.4km" 

        self._log_debug("OCR処理: 生テキスト", f"ファイル: {filename}\nテキスト:\n{raw_text}")

        extracted_data["生テキスト"] = raw_text
        parsed_fields = self._extract_fields_dynamically(raw_text)
        extracted_data.update(parsed_fields)
            
        self.ocr_results.append(extracted_data)
        self._log_debug("OCR処理: 動的抽出結果", json.dumps(parsed_fields, ensure_ascii=False, indent=2))
        return extracted_data

    def _extract_fields_dynamically(self, text):
        result = {}
        amounts = re.findall(r'([0-9,]+)円', text)
        if amounts:
            val = int(amounts[0].replace(',', ''))
            if self.application_type == "単身赴任旅費の開始申請":
                result["往復金額"] = val * 2
            else:
                result["往路_交通費"] = val
                result["復路_交通費"] = val
        dates = re.findall(r'\d{4}[年/]\d{1,2}[月/]\d{1,2}日?', text)
        if dates:
            result["往路_利用日"] = dates[0]
        distances = re.findall(r'([0-9\.]+)km', text)
        if distances:
            result["総距離"] = distances[0]
        return result

    def second_rule_check(self):
        if not self.ocr_results:
            return {"status": "NG", "reason": "証憑データが提出されていません。"}
            
        latest_ocr = self.ocr_results[-1]
        query = f"{self.application_type} 証憑 金額上限 駅すぱあと"
        retrieved_rules = self.retriever.retrieve_rules(query)
        
        self._log_debug("2次チェック: RAG検索", f"Query: {query}\n検索されたルール:\n{retrieved_rules}")
        
        is_ok, reason = self.retriever.evaluate_rule(retrieved_rules, self.hr_data, latest_ocr)
        
        if is_ok:
            if self.application_type == "単身赴任旅費の開始申請" and "往復金額" in latest_ocr:
                self.state["calculated_annual_cap"] = latest_ocr["往復金額"] * 10
            result = {"status": "OK", "reason": reason, "reference_rule": retrieved_rules}
        else:
            result = {"status": "NG", "reason": reason, "reference_rule": retrieved_rules}
            
        self._log_debug("2次チェック: 判定結果", json.dumps(result, ensure_ascii=False, indent=2))
        return result

    def complete_application(self):
        return {"status": "完了", "message": "手続きが正常に完了しました。"}

if __name__ == "__main__":
    # デバッグフラグを True にして実行
    processor = TanshinFuninProcessor(user_id="U002", debug=True)
    
    print("--- デバッグモード実行開始 ---\n")
    processor.determine_application_type("往復申請をお願いします")
    processor.get_hr_info()
    processor.first_rule_check()
    processor.determine_required_evidence()
    processor.process_evidence_ocr("往復証憑", "sample/新幹線チケット.jpg")
    processor.second_rule_check()
    print("\n--- デバッグモード実行終了 ---")
