import json
import os
from datetime import datetime

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
        """
        人事情報をモックデータベースから取得する
        単身赴任の資格や現在の赴任状況などを確認するため
        """
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
        """
        人事情報に基づく1次ルールチェック
        input_dataには開始申請の場合「事由発生年月日」や「申請事由」が含まれる想定
        """
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
        """
        現在の申請内容と事由に基づいて、必要な証憑データを判定する
        """
        evidence_list = []
        if self.application_type == "単身赴任旅費の開始申請":
            reason = self.state.get('reason', '')
            if reason == "重度傷病治療":
                evidence_list.append("重度傷病治療に係る証憑（医師による診断書）")
            elif reason == "介護・看護":
                evidence_list.append("介護・看護対象者に係る証憑（介護保険被保険者証等）")
            # 全ての事由で最終的に必要なもの
            evidence_list.append("駅すぱあとの証憑（検索結果のスクリーンショット等）")

        elif self.application_type == "単身赴任旅費の往復申請":
            evidence_list.append("往復の交通費を示す証憑ファイル（領収書、チケット等）")
            # 宿泊が伴う場合は別途「宿泊証憑」などが必要だが今回は基本の移動とする

        return evidence_list

    # ⑤ 提出された証憑データが何の種類かを識別し、必要な項目をOCRで抽出する
    def process_evidence_ocr(self, evidence_type, image_path):
        """
        ファイル名と要求された証憑種類に基づくOCR抽出のモック
        本来はLLMやOCR APIを用いて画像から情報を読み取る
        """
        filename = os.path.basename(image_path)
        extracted_data = {"ファイル名": filename, "証憑種類": evidence_type}
        
        if "駅すぱあと" in evidence_type or "新幹線" in filename:
            # 開始申請用の駅すぱあと抽出モック
            extracted_data.update({
                "経路_交通手段": "新幹線（のぞみ）",
                "総距離": "515.4km",
                "赴任先最寄り駅": "新大阪",
                "扶養家族等最寄り駅": "東京",
                "往復金額": 29040
            })
        elif "往復" in evidence_type or "領収書" in filename:
            # 往復申請用のチケット抽出モック
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
        elif "診断書" in filename:
            extracted_data.update({"記載内容": "重度傷病による治療を要する", "判定": "有効"})
        elif "保険" in filename:
            extracted_data.update({"記載内容": "要介護認定済み", "判定": "有効"})
            
        self.ocr_results.append(extracted_data)
        return extracted_data

    # ⑥ 申請を進めてよいかのルールチェックを行なう（2次チェック）
    def second_rule_check(self):
        """
        OCR抽出結果と人事データを突き合わせて、最終的な承認可否を判定
        """
        if not self.ocr_results:
            return {"status": "NG", "reason": "証憑データが提出されていません。"}
            
        if self.application_type == "単身赴任旅費の開始申請":
            # 駅すぱあとのデータがあるかチェック
            eki_data = next((item for item in self.ocr_results if "駅すぱあと" in item["証憑種類"] or "経路_交通手段" in item), None)
            if not eki_data:
                return {"status": "NG", "reason": "経路と金額が確認できる証憑（駅すぱあと等）が必要です。"}
            
            # 年間支給上限額などの計算（モック）
            annual_cap = eki_data["往復金額"] * 10  # 仮に残り10ヶ月とする
            self.state["calculated_annual_cap"] = annual_cap
            return {"status": "OK", "reason": f"経路情報の確認が完了しました。今年度支給上限額は {annual_cap} 円です。"}

        elif self.application_type == "単身赴任旅費の往復申請":
            ticket_data = next((item for item in self.ocr_results if "往復" in item["証憑種類"] or "往路_利用日" in item), None)
            if not ticket_data:
                return {"status": "NG", "reason": "往復の交通費が確認できる証憑が必要です。"}
                
            return {"status": "OK", "reason": "往復の利用日・経路・金額が正しく読み取れました。規定内の経路です。"}

    # ⑦ ユーザに確認をしてもらい、申請を完了する
    def complete_application(self):
        return {
            "status": "完了",
            "message": f"お疲れ様です。{self.hr_data['社員名']}さんの【{self.application_type}】の手続きが正常に完了しました。"
        }

# --- テスト実行用 ---
if __name__ == "__main__":
    print("=== テスト1: 単身赴任旅費の開始申請 (重度傷病治療) ===")
    processor1 = TanshinFuninProcessor(user_id="U001")
    app_type1 = processor1.determine_application_type("単身赴任の開始申請をしたいです")
    processor1.get_hr_info()
    rule1 = processor1.first_rule_check({"reason": "重度傷病治療"})
    req_evidences = processor1.determine_required_evidence()
    print(f"要求される証憑: {req_evidences}")
    
    # 診断書のアップロード
    processor1.process_evidence_ocr(req_evidences[0], "sample/診断書.jpg")
    # 駅すぱあとのアップロード
    processor1.process_evidence_ocr(req_evidences[1], "sample/駅すぱあと.png")
    
    rule2_1 = processor1.second_rule_check()
    print(f"2次チェック結果: {rule2_1}")
    print(processor1.complete_application())

    print("\n=== テスト2: 単身赴任旅費の往復申請 ===")
    processor2 = TanshinFuninProcessor(user_id="U002")
    app_type2 = processor2.determine_application_type("先週末に帰省したので往復申請をお願いします")
    processor2.get_hr_info()
    rule1_2 = processor2.first_rule_check()
    req_evidences2 = processor2.determine_required_evidence()
    print(f"要求される証憑: {req_evidences2}")
    
    # 新幹線チケット領収書のアップロード
    processor2.process_evidence_ocr(req_evidences2[0], "sample/新幹線チケット.jpg")
    rule2_2 = processor2.second_rule_check()
    print(f"2次チェック結果: {rule2_2}")
    print(processor2.complete_application())
