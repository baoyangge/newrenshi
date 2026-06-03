import json
import os
import re
from datetime import datetime
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

class TanshinFuninProcessor:
    def __init__(self, user_id, llm_client=None):
        self.user_id = user_id
        self.llm_client = llm_client
        self.hr_data = None
        self.application_type = None
        self.state = {}
        self.ocr_results = []

    def determine_application_type(self, user_input):
        if "開始" in user_input or "事由" in user_input:
            self.application_type = "単身赴任旅費の開始申請"
        elif "往復" in user_input or "帰省" in user_input:
            self.application_type = "単身赴任旅費の往復申請"
        else:
            self.application_type = "不明な申請"
        return self.application_type

    def get_hr_info(self):
        mock_db = {
            "U001": {"社員名": "山田 太郎", "単身赴任ステータス": "未開始"},
            "U002": {"社員名": "鈴木 花子", "単身赴任ステータス": "適用中"}
        }
        self.hr_data = mock_db.get(self.user_id)
        if not self.hr_data:
            raise ValueError(f"ユーザーID {self.user_id} の人事情報が見つかりません。")
        return self.hr_data

    def first_rule_check(self, input_data=None):
        if self.application_type == "単身赴任旅費の開始申請":
            if self.hr_data.get("単身赴任ステータス") == "適用中":
                return {"status": "NG", "reason": "すでに単身赴任旅費が適用されています。"}
            if input_data:
                self.state['reason'] = input_data.get('reason', 'その他')
            return {"status": "OK", "reason": "単身赴任開始の要件を満たしています。"}
        elif self.application_type == "単身赴任旅費の往復申請":
            if self.hr_data.get("単身赴任ステータス") != "適用中":
                return {"status": "NG", "reason": "単身赴任旅費の適用期間外です。"}
            return {"status": "OK", "reason": "単身赴任旅費の往復申請が可能です。"}
        return {"status": "NG"}

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
            # Tesseractがない環境用のテストテキスト
            raw_text = "領収書 2026年06月10日 東京駅から新大阪駅 新幹線 14,520円 515.4km" 

        extracted_data["生テキスト"] = raw_text

        # ---------------------------------------------------------
        # 【重要】動的抽出ロジック（固定値の排除）
        # 本番(Databricks)では、ここのロジックをLLMプロンプトに置き換えます。
        # 例: LLMに対して「以下の生テキストから金額、日付、距離をJSONで抽出して」と指示する
        # ここではLLMの代わりに正規表現を用いてOCRテキストから「動的」に値を拾います。
        # ---------------------------------------------------------
        parsed_fields = self._extract_fields_dynamically(raw_text)
        extracted_data.update(parsed_fields)
            
        self.ocr_results.append(extracted_data)
        return extracted_data

    def _extract_fields_dynamically(self, text):
        """
        OCRで読み取った生テキストから正規表現等を用いて動的に値を抽出する
        """
        result = {}
        
        # 金額の抽出（例: "14,520円" -> 14520）
        amounts = re.findall(r'([0-9,]+)円', text)
        if amounts:
            # 見つかった金額を数値化
            val = int(amounts[0].replace(',', ''))
            if self.application_type == "単身赴任旅費の開始申請":
                result["往復金額"] = val * 2  # 片道分と仮定して往復計算するなど
            else:
                result["往路_交通費"] = val
                result["復路_交通費"] = val
                
        # 日付の抽出（例: "2026年06月10日"）
        dates = re.findall(r'\d{4}[年/]\d{1,2}[月/]\d{1,2}日?', text)
        if dates:
            result["往路_利用日"] = dates[0]
            
        # 距離の抽出（例: "515.4km"）
        distances = re.findall(r'([0-9\.]+)km', text)
        if distances:
            result["総距離"] = distances[0]
            
        return result

    def second_rule_check(self):
        if not self.ocr_results:
            return {"status": "NG", "reason": "証憑データが提出されていません。"}
            
        latest = self.ocr_results[-1]
        
        if self.application_type == "単身赴任旅費の開始申請":
            if "往復金額" not in latest:
                return {"status": "NG", "reason": "OCRから金額を読み取れませんでした。"}
            annual_cap = latest["往復金額"] * 10
            return {"status": "OK", "reason": f"今年度支給上限額は {annual_cap} 円で設定されます。"}

        elif self.application_type == "単身赴任旅費の往復申請":
            if "往路_交通費" not in latest:
                return {"status": "NG", "reason": "OCRから交通費を読み取れませんでした。"}
            return {"status": "OK", "reason": f"交通費 {latest['往路_交通費']}円 の申請を受理しました。"}

    def complete_application(self):
        return {"status": "完了", "message": "手続きが正常に完了しました。"}

if __name__ == "__main__":
    processor = TanshinFuninProcessor(user_id="U002")
    processor.determine_application_type("往復申請をお願いします")
    processor.get_hr_info()
    processor.first_rule_check()
    processor.determine_required_evidence()
    
    # 実際の画像パスを渡してOCR実行（Tesseractがあれば実際の画像を読み取る）
    result = processor.process_evidence_ocr("往復証憑", "sample/新幹線チケット.jpg")
    print("【動的抽出されたデータ】", result)
    
    print(processor.second_rule_check())
