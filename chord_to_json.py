"""
Usage:
    python chord_to_json.py <input.txt>
Setup:
    pip install google-genai python-dotenv
"""

import os
import sys
import json
import argparse
import textwrap
import time
from pathlib import Path

try:
    from google import genai
except ImportError:
    sys.exit("[ERROR] google-genai is not installed. Run: pip install google-genai")

try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    print("[INFO] python-dotenv not installed. Using system environment variables.")

SYSTEM_PROMPT = textwrap.dedent(
    """\
    あなたは音楽理論に詳しいアシスタントです。
    ユーザーが送る「歌詞+コード混在テキスト」を解析し、
    以下のJSONフォーマット仕様に厳密に従ってJSONを出力してください。

    ## 基本方針
    - 判断できない項目は null を入れる。推測で埋めない
    - コード記号は入力テキストの表記をそのまま使用する（正規化しない）
    - セクション名が明記されていない場合は構造から推測して英語で記入する
      （例: intro / verse / pre-chorus / chorus / interlude / bridge / solo / outro）

    ## 楽器レイヤー
    - 入力に楽器の指示がある場合のみ記録する（表記例: E.Gt / A.Gt / E.Ba / Dr / Vo）
    - "All" と書かれている場合はそれまでに登場した主要楽器をすべて列挙する
    - 指示がない箇所は null

    ## コード長さ
    - "----" のようなリズム表記がある場合: "-" 1つ = 8分音符 = 0.5拍
    - リズム表記がない場合: 直前の同じコード進行を持つセクションのリズムパターンをそのまま継承する
    （例: イントロが G/B=2.0 / Cadd9=1.5 / Dsus4=0.5シンコペ なら、
            同じコード進行のVerseも同じ拍割りで記録する）
    - N.C. は chord_symbol = "N.C." として記録
    - N.C. の duration_beats も前後の "-" 表記から同様に計算すること
    （例: "--N.C.--" なら N.C. 自体の長さ + 前後の "-" 分も N.C. に含める）
    （例: "B7>= --N.C.--" → B7=0.5拍、N.C.=1.0拍）

    ## シンコペーション（小節跨ぎ）
    - 前の小節末尾（例: beat_position=4.5）に 0.5拍分のエントリを置く
    - 次の小節頭に同じコードを再エントリして残り時間を duration_beats に記録
    - シンコペ継続エントリの chord_symbol は必ず前小節と同じコード名を入れること。null にしてはいけない

    ## section_label
    - 元テキストに「Aメロ」「サビ」「ギターソロ」等のラベルが明記されている場合は必ず記入する
    - 明記されていない場合のみ null

    ## 転調
    - 曲全体のデフォルトは song_meta の key_root / key_mode に記録
    - 転調するセクションのみ section_key_root / section_key_mode に上書き値を記録
    - 転調なしのセクションは null

    ## 繰り返し
    - "×2" "D.C." などがある場合は repeat: true にしてコードを展開して記録

    ## statistics
    - 全フィールドを空値（null または {}）のまま出力すること。計算しない

    ## 出力JSONスキーマ
    {
      "song_meta": {
        "title": str,
        "artist": str,
        "composer": str | null,
        "lyricist": str | null,
        "album": str | null,
        "year": int | null,
        "bpm": int | null,
        "time_signature": str,   // 例: "4/4"
        "key_root": str,         // 例: "G", "Bb", "F#"
        "key_mode": "major" | "minor"
      },
      "sections": [
        {
          "section_id": str,          // 例: "S01"
          "section_type": str,        // intro|verse|pre-chorus|chorus|bridge|solo|outro|interlude
          "section_label": str | null,
          "repeat": bool,
          "instrument_layer": list[str] | null,
          "measure_start": int,
          "measure_end": int,
          "section_key_root": str | null,
          "section_key_mode": str | null,
          "chords": [
            {
              "measure_number": int,
              "beat_position": float,   // 1.0始まり、0.5刻み
              "chord_symbol": str,
              "duration_beats": float,
              "lyric_syllable": str | null
            }
          ]
        }
      ],
      "statistics": {
        "total_measures": null,
        "total_chord_events": null,
        "chord_frequency": {},
        "degree_frequency": {},
        "transition_matrix": {},
        "most_common_progression": [],
        "unique_chords": [],
        "unique_degrees": [],
        "secondary_dominants": [],
        "borrowed_chords": [],
        "cadence_types": {"authentic": null, "half": null, "plagal": null, "deceptive": null}
      }
    }
""")

def load_api_key():
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        sys.exit("[ERROR] GEMINI_API_KEY is not set in .env or environment variables.")
    return key

def call_gemini(chord_text: str, model_name: str = "gemini-2.5-flash", retries: int = 3):
    client = genai.Client(api_key=load_api_key())
    
    config = {
        "system_instruction": SYSTEM_PROMPT,
        "response_mime_type": "application/json",
        "max_output_tokens": 65536
    }

    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Calling Gemini API ({attempt}/{retries})")
            response = client.models.generate_content(
                model=model_name,
                contents=chord_text,
                config=config
            )
            return response.text
        except Exception as e:
            print(f"[ERROR] {e}")
            if attempt < retries:
                wait = 5 * attempt
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)
            else:
                sys.exit(f"[ERROR] API request failed.")

def parse_json_response(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        debug_path = Path("debug_raw_response.txt")
        debug_path.write_text(raw, encoding="utf-8")
        sys.exit(f"[ERROR] {e}\n Saved to {debug_path}")

def strip_lyrics(data: dict):
    for section in data.get('sections', []):
        for chord in section.get('chords', []):
            chord.pop('lyric_syllable', None)
    return data

def fix_syncopation(data: dict):
    for section in data.get('sections', []):
        chords = section.get('chords', [])
        for i, c in enumerate(chords):
            if c['chord_symbol'] is None and i > 0:
                c['chord_symbol'] = chords[i - 1]['chord_symbol']
    return data

def resolve_output_path(input_path: Path | None):
    if input_path:
        return input_path.with_suffix(".json")
    return Path("output.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            sys.exit(f"[ERROR] Not found: {input_path}")
        chord_text = input_path.read_text(encoding="utf-8")
        print(f"[INFO] Success: {input_path}")
    else:
        parser.print_help()
        sys.exit(1)

    raw_response = call_gemini(chord_text)
    result = parse_json_response(raw_response)
    result = strip_lyrics(result)
    result = fix_syncopation(result)

    output_path = resolve_output_path(input_path)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"[INFO] Successfully saved to {output_path}")

if __name__ == "__main__":
    main()