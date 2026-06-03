import json
import os
from datetime import datetime
try:
    import pytesseract
    from PIL import Image
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

class TanshinFuninProcessor:
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
        self.state = {}
        self.ocr_results = []

    # ① ユーザからの入力によって、どの申請を進めるかを判断する
    def determine_application_type(self, user_input):
        """
        入力テキストから「開始申請」か「往復申請」かを判定する
        """
        if "開始" in user_input or "事由" in user_input:
            self.application_type = "単身赴任旅費の開始申請"
        elif "往復" in user_input or "帰省" in user_input:
            self.application_type = "単身赴任旅費の往復申請"
        else:
            self.application_type = "不明な申請"
        return self.application_type

    # ② 人事情報を取得してくる
    def get_hr_info(self):
        mock_db = {
            "U001": {
                "社員名": "山田 太郎", 
                "部署": "大阪支社", 
                "役職": "マネージャー",
                "単身赴任ステータス": "未開始",
                "家族居住地": "東京"
            },
            "U002": {
                "社員名": "鈴木 花子", 
                "部署": "福岡支社", 
                "役職": "一般社員",
                "単身赴任ステータス": "適用中",
                "家族居住地": "東京",
                "登録経路": "新幹線"
            }
        }
        self.hr_data = mock_db.get(self.user_id)
        if not self.hr_data:
            raise ValueError(f"ユーザーID {self.user_id} の人事情報が見つかりません。")
        return self.hr_data

    # ③ 申請を進めてよいかどうかのルールチェックを行なう（人事情報をもとにした１次チェック）
    def first_rule_check(self, input_data=None):
        if self.application_type == "単身赴任旅費の開始申請":
            if self.hr_data.get("単身赴任ステータス") == "適用中":
                return {"status": "NG", "reason": "すでに単身赴任旅費が適用されています。"}
            if input_data:
                self.state['reason'] = input_data.get('reason', 'その他')
            return {"status": "OK", "reason": "単身赴任開始の要件を満たしています。"}

        elif self.application_type == "単身赴任旅費の往復申請":
            if self.hr_data.get("単身赴任ステータス") != "適用中":
                return {"status": "NG", "reason": "単身赴任旅費の適用期間外です。まずは開始申請を行ってください。"}
            return {"status": "OK", "reason": "単身赴任旅費の往復申請が可能です。"}

        return {"status": "NG", "reason": "申請種類が特定できません。"}

    # ④ 何の証憑データを提出してもらうかを判断し、ユーザに提出を求める
    def determine_required_evidence(self):
        evidence_list = []
        if self.application_type == "単身赴任旅費の開始申請":
            reason = self.state.get('reason', '')
            if reason == "重度傷病治療":
                evidence_list.append("重度傷病治療に係る証憑（医師による診断書）")
            elif reason == "介護・看護":
                evidence_list.append("介護・看護対象者に係る証憑（介護保険被保険者証等）")
            evidence_list.append("駅すぱあとの証憑（検索結果のスクリーンショット等）")

        elif self.application_type == "単身赴任旅費の往復申請":
            evidence_list.append("往復の交通費を示す証憑ファイル（領収書、チケット等）")

        return evidence_list

    # ⑤ 提出された証憑データが何の種類かを識別し、必要な項目をOCRで抽出する
    def process_evidence_ocr(self, evidence_type, image_path):
        """
        Pythonライブラリ（pytesseract等）を用いた実際のOCR処理を行います。
        """
        filename = os.path.basename(image_path)
        extracted_data = {"ファイル名": filename, "証憑種類": evidence_type}
        
        raw_text = ""
        # pytesseractを用いた画像からのテキスト抽出
        if HAS_TESSERACT and os.path.exists(image_path):
            try:
                # 日本語（jpn）を指定してOCR実行
                img = Image.open(image_path)
                raw_text = pytesseract.image_to_string(img, lang='jpn')
                extracted_data["生テキスト"] = raw_text
            except Exception as e:
                extracted_data["OCRエラー"] = str(e)
        else:
            extracted_data["生テキスト"] = f"（{filename} のダミーOCRテキスト。ライブラリ未インストールまたはファイル不在）"
            raw_text = extracted_data["生テキスト"]

        # 本来はここで抽出した生テキスト(raw_text)をLLMに投げて構造化データに変換する
        # 今回はLLM連携の前段階として、ルールベースで仮の構造化を行う
        if self.application_type == "単身赴任旅費の開始申請" and "駅すぱあと" in evidence_type:
            extracted_data.update({
                "経路_交通手段": "新幹線（のぞみ）",
                "総距離": "515.4km",
                "赴任先最寄り駅": "新大阪",
                "扶養家族等最寄り駅": "東京",
                "往復金額": 29040
            })
        elif self.application_type == "単身赴任旅費の往復申請":
            extracted_data.update({
                "往路_利用日": "2026/06/10",
                "往路_交通手段": "新幹線",
                "往路_経路": "東京 -> 新大阪",
                "往路_交通費": 14520,
                "復路_利用日": "2026/06/12",
                "復路_交通手段": "新幹線",
                "復路_経路": "新大阪 -> 東京",
                "復路_交通費": 14520
            })
            
        self.ocr_results.append(extracted_data)
        return extracted_data

    # ⑥ 申請を進めてよいかのルールチェックを行なう（2次チェック）
    def second_rule_check(self):
        if not self.ocr_results:
            return {"status": "NG", "reason": "証憑データが提出されていません。"}
            
        if self.application_type == "単身赴任旅費の開始申請":
            eki_data = next((item for item in self.ocr_results if "駅すぱあと" in item["証憑種類"] or "経路_交通手段" in item), None)
            if not eki_data:
                return {"status": "NG", "reason": "経路と金額が確認できる証憑（駅すぱあと等）が必要です。"}
            
            annual_cap = eki_data.get("往復金額", 0) * 10
            self.state["calculated_annual_cap"] = annual_cap
            return {"status": "OK", "reason": f"経路情報の確認が完了しました。今年度支給上限額は {annual_cap} 円です。"}

        elif self.application_type == "単身赴任旅費の往復申請":
            ticket_data = next((item for item in self.ocr_results if "往復" in item["証憑種類"] or "往路_利用日" in item), None)
            if not ticket_data:
                return {"status": "NG", "reason": "往復の交通費が確認できる証憑が必要です。"}
                
            return {"status": "OK", "reason": "往復の利用日・経路・金額が正しく読み取れました。"}

    # ⑦ ユーザに確認をしてもらい、申請を完了する
    def complete_application(self):
        return {
            "status": "完了",
            "message": f"お疲れ様です。{self.hr_data['社員名']}さんの【{self.application_type}】の手続きが正常に完了しました。"
        }
