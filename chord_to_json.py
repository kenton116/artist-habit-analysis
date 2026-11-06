
!pip install -q google-genai python-dotenv

import os
import sys
import json
import textwrap
from google.colab import userdata
# from google.colab import drive
# drive.mount('/content/drive/MyDrive/ssh/input')

try:
    from google import genai
except ImportError:
    sys.exit("[ERROR] google-genai is not installed.")
try:
    os.environ["GEMINI_API_KEY"] = userdata.get('GEMINI_API_KEY')
except Exception:
    print("[WARNING] Colabのシークレットに 'GEMINI_API_KEY' が見つかりません。")
SYSTEM_PROMPT = textwrap.dedent(
    """\
    あなたは音楽理論とデータ分析に詳しいアシスタントです。
    ユーザーが送る「歌詞+コード混在テキスト」を解析し、
    コード進行の統計分析（遷移確率や出現頻度の集計）が最も行いやすいよう、
    小節や拍の概念をすべて排除して、以下のシンプルなJSONフォーマットで出力してください。

    ## 抽出ルール
    - `chord_sequence` 配列には、そのセクション内で**コードが変化した順番**のまま、コード記号の文字列をフラットに並べてください（小節ごとに配列を分けないでください）。
    - コード記号は入力テキストの表記をそのまま使用してください（例: "G/B", "Cadd9", "Am7(11)"）。
    - 連続して同じコードが続く場合は、1つにまとめてください（例: Gが2小節続いても ["G"] と表記）。ただし、別のコードを挟んで戻る場合は当然記録します。
    - キメやブレイク等でコードが鳴っていない、または N.C. と書かれている箇所は "N.C." として記録してください。
    - セクション名（section_type）は構造から判断して英語（intro, verse, pre-chorus, chorus, bridge, solo, outro, interlude）で記入してください。

    ## 出力JSONスキーマ
    {
      "song_meta": {
        "title": str,
        "artist": str,
        "bpm": int | null,
        "time_signature": str | null,
        "key": str | null
      },
      "sections": [
        {
          "section_type": "intro" | "verse" | "pre-chorus" | "chorus" | "bridge" | "solo" | "outro" | "interlude",
          "chord_sequence": list[str]  // 例: ["G/B", "Cadd9", "Dsus4", "Em7", "G/B", "Cadd9", "Dsus4", "G"]
        }
      ]
    }
""")

def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("[ERROR] GEMINI_API_KEY が設定されていません。")
    return key

def call_gemini(chord_text: str, model_name: str = "gemini-2.5-flash"):
    client = genai.Client(api_key=load_api_key())
    config = {
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "max_output_tokens": 65536
    }
    response = client.models.generate_content(
        model=model_name,
        contents=chord_text,
        config=config
    )
    return response.text

input_chord_text = """

"""

def main():
    raw_response = call_gemini(input_chord_text)

    try:
        result = json.loads(raw_response)
        json_output = json.dumps(result, ensure_ascii=False, indent=2)
        print(json_output)

        with open("flat_chords.json", "w", encoding="utf-8") as f:
            f.write(json_output)
    except Exception as e:
        print(f"[ERROR] JSONのパースに失敗しました: {e}")

if __name__ == "__main__":
    main()