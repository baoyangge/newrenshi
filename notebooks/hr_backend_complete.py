import json
import os

class HRApplicationProcessor:
    def __init__(self, user_id, llm_client=None):
        """
        初期化メソッド
        :param user_id: 申請を行うユーザーのID
        :param llm_client: RAG/LLM処理用のクライアント
        """
        self.user_id = user_id
        self.llm_client = llm_client
        self.hr_data = None
        self.application_type = None
        self.ocr_results = []

    # ① ユーザからの入力によって、どの申請を進めるかを判断する
    def determine_application_type(self, user_input):
        if "新幹線" in user_input or "出張" in user_input or "精算" in user_input:
            self.application_type = "交通費精算"
        elif "休" in user_input or "風邪" in user_input or "病院" in user_input:
            self.application_type = "休暇申請"
        else:
            self.application_type = "その他申請"
        return self.application_type

    # ② 人事情報を取得してくる
    def get_hr_info(self):
        mock_db = {
            "U001": {"社員名": "山田 太郎", "部署": "営業部", "役職": "マネージャー", "有給残日数": 12},
            "U002": {"社員名": "鈴木 花子", "部署": "開発部", "役職": "一般社員", "有給残日数": 0}
        }
        self.hr_data = mock_db.get(self.user_id)
        if not self.hr_data:
            raise ValueError(f"ユーザーID {self.user_id} の人事情報が見つかりません。")
        return self.hr_data

    # ③ 申請を進めてよいかどうかのルールチェックを行なう（人事情報をもとにした１次チェック）
    def first_rule_check(self):
        if self.application_type == "休暇申請" and self.hr_data["有給残日数"] <= 0:
            return {"status": "NG", "reason": f"有給残日数が不足しています（残: {self.hr_data['有給残日数']}日）。"}
        return {"status": "OK", "reason": "1次チェック通過。申請処理を進めます。"}

    # ④ 何の証憑データを提出してもらうかを判断し、ユーザに提出を求める
    def determine_required_evidence(self):
        if self.application_type == "交通費精算":
            return ["交通機関の領収書（新幹線のチケット等）"]
        elif self.application_type == "休暇申請":
            return ["医師の診断書"]
        return []

    # ⑤ 提出された証憑データが何の種類かを識別し、必要な項目をOCRで抽出する
    def process_evidence_ocr(self, image_path):
        """
        ※本来は pytesseract や Azure Document Intelligence 等を呼び出す箇所
        今回はモックとしてファイル名から抽出結果をシミュレート
        """
        filename = os.path.basename(image_path)
        extracted_data = {}
        
        if "新幹線" in filename:
            extracted_data = {"種類": "新幹線チケット", "テキスト": "東京発 大阪着 乗車券・特急券 14,520円"}
        elif "診断書" in filename:
            extracted_data = {"種類": "医療診断書", "テキスト": "病名：インフルエンザ。3日間の休養を要する。"}
        else:
            extracted_data = {"種類": "不明", "テキスト": "読み取りエラー"}
            
        self.ocr_results.append(extracted_data)
        return extracted_data

    # ⑥ 申請を進めてよいかのルールチェックを行なう（2次チェック）
    def second_rule_check(self):
        if not self.ocr_results:
            return {"status": "NG", "reason": "証憑データがアップロードされていません。"}
            
        latest_ocr = self.ocr_results[-1]
        text = latest_ocr.get("テキスト", "")
        
        if self.application_type == "交通費精算":
            if "新幹線" in latest_ocr["種類"] and "円" in text:
                return {"status": "OK", "reason": "領収書の金額と区間が確認できました。2次チェック通過。"}
            return {"status": "NG", "reason": "必要な情報（金額・区間）が読み取れませんでした。"}
            
        elif self.application_type == "休暇申請":
            if "診断書" in latest_ocr["種類"]:
                return {"status": "OK", "reason": "診断書の内容が規定を満たしています。2次チェック通過。"}
            return {"status": "NG", "reason": "有効な診断書が確認できませんでした。"}

    # ⑦ ユーザに確認をしてもらい、申請を完了する
    def complete_application(self):
        return {
            "status": "完了",
            "message": f"お疲れ様です。{self.hr_data['社員名']}さんの{self.application_type}の手続きが正常に完了しました。"
        }

# --- テスト実行用 ---
if __name__ == "__main__":
    processor = HRApplicationProcessor(user_id="U001")
    processor.determine_application_type("新幹線で出張に行きました")
    processor.get_hr_info()
    processor.first_rule_check()
    processor.determine_required_evidence()
    processor.process_evidence_ocr("sample/新幹線チケット.jpg")
    processor.second_rule_check()
    result = processor.complete_application()
    print(result)
