"""
generate_data.py : 「思考の系譜」評価用の架空メモデータ生成

出力 (--out ディレクトリ)
  memos.json    : メモ本体(正解ラベル付き。ベクトルは含まない)
  queries.json  : 評価クエリ(いま書いている下書き)と正解メモ
  stats.json    : 件数などの集計

v3 での変更（外部レビューの指摘への対応）
- 下書きは、系列のメモを作った「後」に、別の呼び出しで作る。k番目の時点の下書きには、
  k番目までのメモしか渡さない（後続メモが LLM に見えない = 未来情報が混入しない）
- ペルソナを 10 人に増やし（職業を分散）、dev / test の割り当てをシード付きでランダムにする
  （既定: 5人 dev / 5人 test = 8系列 x 5人 = 40系列がテスト）
- 下書きの種類（続きの考え / 新しい疑問 / 次にやること）を変えて、パターンを偏らせない
- 人手確認用のシートの書き出し・取り込みは import_generated_dataset.py が行う

v2 での変更（評価データの人工性を減らす）
- 各メモのタイトルは「そのメモ固有」にする（系列で共通のテーマ名を付けない）
- 下書き（クエリ）にテーマ名そのものを書かせない
- 下書きは「各メモの直後」に1つずつ作り、過去時点クエリ(thought_before)にも使う
- ペルソナ4人 x 各10系列（40系列）

設計
- 関係の型は 4 つ: root(最初のメモ) / extends(発展) / revises(修正・撤回) / revisits(再燃)
  + distractor(語彙は似ているが別件) / filler(無関係なメモ)
- 時刻は LLM に任せず、プログラムで割り当てる(関係ごとに間隔を変える)
- 分割は「ペルソナ単位」で dev / test に分ける(重みの調整が test に漏れないようにする)
- LLM の出力は cache/ に保存し、同じプロンプトは再呼び出ししない(再現性・無料枠対策)
- --mock で API を使わず、パイプラインだけ確認できる

使い方
  # Gemini で生成
  python generate_data.py --out eval_data
  # ローカル LLM(Ollama)で生成。API キー・利用制限・外部送信なし
  ollama pull <モデル名>
  python generate_data.py --backend ollama --model <モデル名> --out eval_data_local --legacy-models ""
  # API なしの動作確認
  python generate_data.py --out eval_data_mock --mock

生成した LLM と、後で評価に使う埋め込み・判定のモデルが同じ系統だと、結果が実際より良く
見える（自己贔屓）。別系統で作る（例: データはローカルLLM、埋め込みは Gemini）と、その偏りを減らせる。
注意: 生成と評価の審判には、別のモデルを使うのが望ましい(自己贔屓を避けるため)。
"""
import argparse
import datetime
import hashlib
import json
import os
import random
import sys
import time
import uuid

GEN_MODEL = "gemini-3.8-flash"   # あなたの環境で応答が得られたモデル名。公式の最新名称は要確認
# 旧キャッシュ(モデル名をキーに含めていた版)を引き継ぐ対象。これまでに使ったモデル名を並べる。
LEGACY_MODELS = ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-2.5-flash")

# ---------- ペルソナ(固定。メモの種類を混ぜる) ----------
PERSONAS = [
    {"name": "情報系の大学生",
     "themes": ["卒業研究のテーマ", "ゼミ合宿の準備", "アルバイトのシフト", "就職活動の方針",
                "研究室の発表資料", "引っ越しの検討", "英語の勉強計画", "サークルの新歓イベント",
                "卒業旅行の計画", "ノートPCの買い替え"]},
    {"name": "Webエンジニア",
     "themes": ["社内ツールの設計方針", "障害対応の手順", "新規プロジェクトの技術選定", "勉強会の登壇準備",
                "転職の検討", "チームの開発ルール", "副業アプリのアイデア", "資格試験の学習",
                "在宅勤務の環境づくり", "ブログ記事の執筆習慣"]},
    {"name": "子育て中の会社員",
     "themes": ["家族旅行の計画", "保育園の行事", "家の模様替え", "週末の買い物リスト",
                "習い事の選定", "資産運用の方針", "実家への帰省", "職場の異動の相談",
                "子どもの進学の検討", "家庭菜園の始め方"]},
    {"name": "フリーランスのデザイナー",
     "themes": ["クライアント案件の進め方", "ポートフォリオの更新", "新しい機材の購入", "個展の企画",
                "料金表の見直し", "コワーキングスペース選び", "ブログの連載計画", "確定申告の準備",
                "海外クライアントとの契約", "作業用デスクの改善"]},
]
PERSONAS += [
    {"name": "総合病院の看護師",
     "themes": ["夜勤シフトの調整", "認定資格の勉強", "職場の人間関係の相談", "引っ越しの検討",
                "親の介護の準備", "休日の過ごし方", "転職の検討", "体調管理の習慣"]},
    {"name": "高校の数学教師",
     "themes": ["授業の進め方の見直し", "文化祭の出し物", "部活動の指導方針", "定期試験の問題作り",
                "保護者面談の準備", "研究授業の準備", "教材のデジタル化", "異動の希望"]},
    {"name": "飲食店のオーナー",
     "themes": ["新メニューの開発", "仕入れ先の見直し", "アルバイトの採用", "値上げの検討",
                "SNS集客の方針", "店舗の改装", "2号店の検討", "年末の営業計画"]},
    {"name": "生命科学系の大学院生",
     "themes": ["実験計画の立て直し", "学会発表の準備", "論文の投稿先", "共同研究者との連絡",
                "試薬の発注管理", "博士課程への進学", "留学の検討", "研究費の申請"]},
    {"name": "定年後の趣味人",
     "themes": ["畑の作付け計画", "写真教室の選択", "夫婦旅行の計画", "孫の誕生日の準備",
                "健康診断の結果への対応", "地域ボランティアへの参加", "蔵書の整理", "自転車の購入"]},
    {"name": "メーカーの営業職",
     "themes": ["大口顧客への提案", "四半期目標の見直し", "新人の指導", "出張の手配",
                "社内資格の勉強", "競合製品の調査", "展示会の準備", "営業資料の改善"]},
]
MAX_THREADS = 8   # 全ペルソナで共通に使える系列数
DRAFT_MODES = ["続きの考え(前のメモを踏まえて考えを進める)",
               "新しい疑問(ふと浮かんだ疑問や迷い)",
               "次にやること(これから取りかかる作業や確認事項)"]

OFF_DOMAIN = ["火星探査の最新ニュース", "古代ローマの建築技術", "深海魚の生態", "クラシック音楽の作曲法",
              "宇宙エレベーターの構想", "江戸時代の食文化", "チェスの終盤理論", "火山の噴火メカニズム"]

REL_DEF = {
    "extends": "発展: 前のメモの考えを深めた・具体化した・広げた(前の内容は否定しない)",
    "revises": "修正: 前のメモの内容を変更・撤回・やめた(前の内容と食い違う)",
    "revisits": "再燃: 長い空白のあと、同じテーマに戻って新しい視点で書き足した",
}
GAP_DAYS = {"extends": (1, 14), "revises": (3, 60), "revisits": (90, 240)}


# ---------- APIキー ----------
def get_api_key():
    """環境変数 GEMINI_API_KEY → .env(プロジェクトのルート/カレント)の順に探す。"""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        from dotenv import load_dotenv
        here = os.path.dirname(os.path.abspath(__file__))
        for path in (os.path.join(here, ".env"), os.path.join(os.getcwd(), ".env")):
            if os.path.exists(path):
                load_dotenv(path)
        key = os.environ.get("GEMINI_API_KEY")
    except ImportError:
        pass
    if key:
        return key
    raise SystemExit(
        "GEMINI_API_KEY が見つかりません。次のどちらかを行ってください。\n"
        "  (1) PowerShell:  $env:GEMINI_API_KEY=\"あなたのキー\"   (このウィンドウでのみ有効)\n"
        "  (2) プロジェクトのルートの .env に  GEMINI_API_KEY=あなたのキー  と書く\n"
        "      (python-dotenv が必要: pip install python-dotenv)\n"
        "API を使わず動作だけ確認するなら:  python generate_data.py --out eval_data_mock --mock"
    )


def parse_json(text):
    """モデルの出力から JSON を取り出す。```json ... ``` や前置きが付いていても読む（小型モデル対策）。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
    i, j = t.find("{"), t.rfind("}")
    if i == -1 or j == -1 or j < i:
        raise ValueError("JSON が見つかりません")
    return json.loads(t[i:j + 1])


# ---------- LLM 呼び出し(キャッシュ・リトライ付き) ----------
class LLM:
    """
    Gemini 呼び出し。結果は cache/ に保存し、同じプロンプトは再呼び出ししない。

    キャッシュのキーは「プロンプトだけ」から作る（モデル名を含めない）。
    こうすると、レート制限などでモデルを切り替えても、完了済みの結果を捨てずに済む。
    どのモデルで作ったかは、各キャッシュ内の _meta.model に記録する。
    以前の版（キーにモデル名を含めていた）のキャッシュは、legacy_models を試して引き継ぐ。
    """

    def __init__(self, cache_dir, mock=False, model=GEN_MODEL, legacy_models=LEGACY_MODELS,
                 sleep=2.0, max_retries=10, rate_wait=30, total=None,
                 backend="gemini", ollama_host="http://127.0.0.1:11434", timeout=600):
        self.cache_dir, self.mock, self.model = cache_dir, mock, model
        self.backend = backend
        self.legacy_models = tuple(legacy_models)
        self.sleep, self.max_retries, self.rate_wait = sleep, max_retries, rate_wait
        self.total, self.calls = total, 0
        self.stats = {"cache": 0, "migrated": 0, "api": 0, "retries": 0}
        os.makedirs(cache_dir, exist_ok=True)
        self.client = None
        if not mock and backend == "ollama":
            try:
                import ollama
            except ImportError:
                raise SystemExit("ollama パッケージがありません。  pip install ollama  を実行してください。")
            self.client = ollama.Client(host=ollama_host, timeout=timeout)
        elif not mock:
            api_key = get_api_key()          # キーが無ければ、先に分かりやすいエラーで止める
            from google import genai
            self.client = genai.Client(api_key=api_key)

    @staticmethod
    def _key(prompt, model=""):
        return hashlib.sha1((model + prompt).encode("utf-8")).hexdigest()

    def _path(self, key):
        return os.path.join(self.cache_dir, key + ".json")

    def _progress(self, tag, how):
        self.calls += 1
        total = f"/{self.total}" if self.total else ""
        print(f"[{self.calls}{total}] {tag:<10} {how}", flush=True)

    def _call(self, prompt, key):
        """バックエンドに1回問い合わせて、出力テキストを返す。"""
        if self.backend == "ollama":
            r = self.client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}], format="json",
                # プロンプトごとに固定のシード（再現性）。温度は多様性のため高め。
                options={"temperature": 0.9, "seed": int(key[:8], 16)})
            return r["message"]["content"]
        r = self.client.models.generate_content(
            model=self.model, contents=prompt,
            config={"temperature": 0.9, "response_mime_type": "application/json"})
        return r.text

    def json(self, prompt, validate, tag):
        key = self._key(prompt)
        path = self._path(key)
        if os.path.exists(path):
            self.stats["cache"] += 1
            self._progress(tag, "キャッシュ")
            return json.load(open(path, encoding="utf-8"))
        # 旧形式のキャッシュ（モデル名入りキー）を探して引き継ぐ
        for m in self.legacy_models:
            old = self._path(self._key(prompt, m))
            if os.path.exists(old):
                d = json.load(open(old, encoding="utf-8"))
                d.setdefault("_meta", {"model": m, "migrated": True})
                json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
                self.stats["migrated"] += 1
                self._progress(tag, f"キャッシュ(旧形式 {m} から引き継ぎ)")
                return d
        if self.mock:
            self._progress(tag, "mock")
            return self._mock(tag, prompt)

        last = None
        for i in range(self.max_retries):
            try:
                d = parse_json(self._call(prompt, key))
                if validate(d):
                    d["_meta"] = {"model": self.model, "backend": self.backend,
                                  "generated_at": datetime.datetime.now().isoformat(timespec="seconds")}
                    json.dump(d, open(path, "w", encoding="utf-8"), ensure_ascii=False)
                    self.stats["api"] += 1
                    self._progress(tag, f"{self.backend} ({self.model})")
                    time.sleep(self.sleep)        # 利用制限に当たりにくくする
                    return d
                last = "出力の形式が想定と違います（検証に失敗）"
            except Exception as e:
                last = str(e)
            low = (last or "").lower()
            if self.backend == "ollama" and ("connect" in low or "refused" in low):
                raise RuntimeError("Ollama に接続できません。Ollama を起動してから(アプリを開く、または  ollama serve)、"
                                   "同じコマンドを再実行してください。完了分は保存済みです。")
            if self.backend == "ollama" and "not found" in low:
                raise RuntimeError(f"モデル '{self.model}' が見つかりません。  ollama pull {self.model}  を実行してください。")
            self.stats["retries"] += 1
            if "429" in low or "resource_exhausted" in low:
                if "perday" in low or "per day" in low or "daily" in low:
                    raise RuntimeError(
                        "1日の利用上限に達した可能性があります。時間をおいて(上限のリセット後に)"
                        f"同じコマンドを再実行してください。完了分は保存済みです。\n詳細: {last[:200]}")
                print(f"  [{tag}] 利用制限(429)。{self.rate_wait}秒待って再試行します ({i + 1}/{self.max_retries})", flush=True)
                time.sleep(self.rate_wait)
            else:
                print(f"  [{tag}] 失敗: {last[:120]} 。再試行します ({i + 1}/{self.max_retries})", flush=True)
                time.sleep(min(2 ** i, 30))
        raise RuntimeError(f"Gemini の呼び出しに失敗しました ({tag}): {last}\n"
                           "完了分は保存済みです。同じコマンドで続きから再開できます。")

    def _mock(self, tag, prompt):
        n = int(prompt.split("<<N=")[1].split(">>")[0]) if "<<N=" in prompt else 0
        h = hashlib.sha1(prompt.encode()).hexdigest()[:6]
        if tag == "thread":
            return {"memos": [{"title": f"[mock {h}] 思考{i}", "content": f"mock内容 {h}-{i}"} for i in range(n)]}
        if tag == "draft":
            return {"draft": f"mock下書き {h}"}
        if tag in ("distractor", "filler"):
            return {"memos": [{"title": f"[mock {tag} {h}] {i}", "content": f"mock {tag} {h}-{i}"} for i in range(n)]}
        return {"drafts": [f"mock無関係下書き {h}-{i}" for i in range(n)]}


def ok_draft(d):
    return isinstance(d.get("draft"), str) and bool(d["draft"].strip())


def ok_memos(n):
    def v(d):
        m = d.get("memos")
        return isinstance(m, list) and len(m) == n and all(
            isinstance(x, dict) and x.get("content", "").strip() for x in m)
    return v


# ---------- プロンプト ----------
STYLE = ("メモは1〜4行の短い断片で、日記ではなくメモらしい書き方にする(箇条書き・体言止め可)。"
         "日付や『前に書いた』という表現は入れない。")


def thread_prompt(persona, theme, rels):
    """系列のメモだけを作る（下書きは作らない。下書きは draft_prompt で、その時点までのメモだけを見て作る）。"""
    n = len(rels) + 1
    lines = ["1. (最初のメモ) テーマについて最初に書いた考え・予定・決定・アイデア"]
    for i, r in enumerate(rels, 2):
        lines.append(f"{i}. [{r}] {REL_DEF[r]}")
    return f"""あなたは「{persona}」の個人メモを作る担当です。
テーマ「{theme}」について、時間をおいて書かれた{n}件のメモを、下の関係に従って順に作ってください。
{chr(10).join(lines)}

条件:
- {STYLE}
- 各メモは前のメモの話題と同じ対象を指すこと。ただし言い回しは変える。
- 各メモの title は、そのメモの内容に固有の短い題にする。メモ同士で同じ題・共通のテーマ名を使い回さない。テーマ名「{theme}」そのものを title や本文にそのまま書かない。
出力: JSONのみ。{{"memos":[{{"title":"...","content":"..."}} ...(<<N={n}>>件)]}}"""


def draft_prompt(persona, contents, mode):
    """contents は『その時点までに書かれたメモ』（古い順）だけ。後続メモは渡さない。"""
    listed = "\n".join(f"({i}) {c}" for i, c in enumerate(contents, 1))
    return f"""あなたは「{persona}」本人です。次は、あなたが時間をおいて書いた個人メモです(古い順)。
{listed}

最後のメモを書いた『直後』に、あなたが新しく書き始めた下書き(1〜2文)を作ってください。
種類: {mode}
条件:
- {STYLE}
- 上のメモに書かれている内容だけを知っている前提で書く。上に書かれていない出来事や決定を、事実として書かない。
- 過去のメモと同じ文言をそのまま使わない。
出力: JSONのみ。{{"draft":"..."}}"""


def distractor_prompt(persona, root_text):
    return f"""あなたは「{persona}」の個人メモを作る担当です。
次のメモと『同じような単語・話題領域』を使うが、『全く別の件(別の対象・別の予定)』を扱うメモを2件作ってください。
元のメモ:
{root_text}

条件: {STYLE} 元のメモの続きや修正にならないこと。
出力: JSONのみ。{{"memos":[{{"title":"...","content":"..."}}, ...(<<N=2>>件)]}}"""


def filler_prompt(persona, n):
    return f"""あなたは「{persona}」の個人メモを作る担当です。
日常の短いメモ(買い物リスト、ふと思いついたこと、やることリスト、調べたことの要点など)を{n}件作ってください。
互いに話題がばらけるようにし、同じ話題を繰り返さないこと。{STYLE}
出力: JSONのみ。{{"memos":[{{"title":"...","content":"..."}}, ...(<<N={n}>>件)]}}"""


def nomatch_prompt(topics):
    return f"""次の各トピックについて、ユーザーがメモを書き始める時の下書きを1〜2文ずつ作ってください。
トピック: {json.dumps(topics, ensure_ascii=False)}
出力: JSONのみ。{{"drafts":["...", ...(<<N={len(topics)}>>件、トピックと同じ順)]}}"""


# ---------- 時刻 ----------
START, END = datetime.datetime(2026, 1, 5), datetime.datetime(2026, 9, 30)


def assign_times(rng, rels, dormant):
    """root からの累積間隔で時刻を割り当てる。-> (memo時刻リスト, クエリ時刻)"""
    gaps = [rng.randint(*GAP_DAYS[r]) for r in rels]
    total = sum(gaps)
    span = max(1, (END - START).days - total)
    t = START + datetime.timedelta(days=rng.randint(0, span), hours=rng.randint(7, 22), minutes=rng.randint(0, 59))
    times = [t]
    for g in gaps:
        t = t + datetime.timedelta(days=g, hours=rng.randint(-3, 3))
        # 系列のメモは、必ず「別の日」に書く（日付単位の as_of で、系列の途中を切れるようにする）
        if t.date() <= times[-1].date():
            t = datetime.datetime.combine(times[-1].date() + datetime.timedelta(days=1), t.time())
        times.append(t)
    q_gap = rng.randint(100, 200) if dormant else rng.randint(1, 30)
    return times, times[-1] + datetime.timedelta(days=q_gap)


def sample_rels(rng):
    steps = [rng.choice(["extends", "revises"]) for _ in range(rng.randint(2, 3))]
    if "revises" not in steps and rng.random() < 0.6:
        steps[rng.randrange(len(steps))] = "revises"
    if rng.random() < 0.4:
        steps.append("revisits")       # 再燃は最後に1回だけ
    return steps


# ---------- 本体 ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval_data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=MAX_THREADS, help=f"ペルソナごとの系譜数(最大{MAX_THREADS})")
    ap.add_argument("--personas", type=int, default=len(PERSONAS), help="使うペルソナの数(最大%d)" % len(PERSONAS))
    ap.add_argument("--test-personas", type=int, default=None, help="test に回すペルソナの数(既定: 半数)")
    ap.add_argument("--split-seed", type=int, default=None, help="dev/test の割り当てシード(既定: --seed と同じ)")
    ap.add_argument("--fillers", type=int, default=15, help="ペルソナごとの無関係メモ数")
    ap.add_argument("--dormant-ratio", type=float, default=0.3)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--backend", choices=["gemini", "ollama"], default="gemini")
    ap.add_argument("--ollama-host", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=float, default=600, help="Ollama 1回あたりの待ち時間(秒)。CPUだと長めに")
    ap.add_argument("--sleep", type=float, default=None, help="成功後の待ち時間(秒)。既定: gemini 2秒 / ollama 0秒")
    ap.add_argument("--max-retries", type=int, default=10)
    ap.add_argument("--rate-wait", type=float, default=30, help="429(利用制限)のときの待ち時間(秒)")
    ap.add_argument("--legacy-models", default=",".join(LEGACY_MODELS),
                    help="旧形式キャッシュを引き継ぐモデル名(カンマ区切り)。不要なら空文字")
    ap.add_argument("--model", default=GEN_MODEL)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    os.makedirs(a.out, exist_ok=True)
    n_threads = min(a.threads, MAX_THREADS)
    personas = PERSONAS[:max(2, min(a.personas, len(PERSONAS)))]
    # 1系列あたり: メモ生成1 + 下書き2(現在/過去時点) + 紛らわしいメモ1。ペルソナごとに 無関係メモ1 + 該当なし下書き1
    total_calls = len(personas) * (4 * n_threads + 2)
    legacy = tuple(x for x in a.legacy_models.split(",") if x)
    if a.backend == "ollama" and a.model == GEN_MODEL and not a.mock:
        raise SystemExit("--backend ollama では --model でローカルのモデル名を指定してください（例: --model <モデル名>）。\n"
                         "  先に  ollama pull <モデル名>  でモデルを取得します。")
    sleep = a.sleep if a.sleep is not None else (0.0 if a.backend == "ollama" else 2.0)
    llm = LLM(os.path.join(a.out, "cache"), mock=a.mock, model=a.model, legacy_models=legacy,
              sleep=sleep, max_retries=a.max_retries, rate_wait=a.rate_wait, total=total_calls,
              backend=a.backend, ollama_host=a.ollama_host, timeout=a.timeout)

    memos, queries = [], []
    # dev / test をペルソナ単位で、シード付きランダムに割り当てる（職業と split を結びつけない）
    n_test = a.test_personas if a.test_personas is not None else len(personas) // 2
    n_test = max(1, min(n_test, len(personas) - 1))
    order = list(range(len(personas)))
    random.Random(a.split_seed if a.split_seed is not None else a.seed).shuffle(order)
    test_set = set(order[:n_test])
    split_of = {pi: ("test" if pi in test_set else "dev") for pi in range(len(personas))}

    for pi, persona in enumerate(personas):
        split = split_of[pi]
        pname = persona["name"]
        themes = persona["themes"][:n_threads]

        for ti, theme in enumerate(themes):
            rels = sample_rels(rng)
            dormant = rng.random() < a.dormant_ratio
            times, q_time = assign_times(rng, rels, dormant)
            n = len(rels) + 1
            d = llm.json(thread_prompt(pname, theme, rels), ok_memos(n), "thread")
            thread_id = f"p{pi}_t{ti}"
            ids, rel_labels = [], ["root"] + rels
            for k in range(n):
                mid = str(uuid.UUID(int=rng.getrandbits(128)))
                ids.append(mid)
                memos.append({
                    "id": mid, "persona": pname, "split": split, "thread_id": thread_id, "theme": theme,
                    "relation": rel_labels[k], "parent_id": ids[k - 1] if k > 0 else None,
                    "created_at": times[k].isoformat(timespec="seconds"),
                    "title": d["memos"][k].get("title", "").strip() or "無題",
                    "content": d["memos"][k]["content"].strip(),
                })
            turning_all = [ids[k] for k in range(n) if rel_labels[k] == "revises"]
            contents = [x["content"].strip() for x in d["memos"]]
            # (1) 現在の下書き: 最後のメモの直後。過去のメモ全部が正解。全メモを見て書いた下書き。
            dp = llm.json(draft_prompt(pname, contents, rng.choice(DRAFT_MODES)), ok_draft, "draft")
            queries.append({
                "qid": f"q_{thread_id}_past", "persona": pname, "split": split, "type": "thought_past",
                "theme": theme, "as_of": q_time.isoformat(timespec="seconds"),
                "text": dp["draft"].strip(),
                "gold_ids": ids, "turning_ids": turning_all, "root_id": ids[0],
                "dormant": dormant, "gap_days": (q_time - times[-1]).days, "thread_id": thread_id,
            })
            # (2) 過去時点の下書き: k番目のメモの直後(1 <= k <= n-2)。その時点までのメモが正解。
            #     下書きの生成には contents[:k+1] だけを渡す（後続メモは LLM に見せない）。
            k = rng.randint(1, n - 2)
            db = llm.json(draft_prompt(pname, contents[:k + 1], rng.choice(DRAFT_MODES)), ok_draft, "draft")
            # k番目のメモと同じ日の終わり。次のメモは必ず別の日なので、日付単位でも k までが「存在する」状態になる
            as_of_k = datetime.datetime.combine(times[k].date(), datetime.time(23, 59))
            queries.append({
                "qid": f"q_{thread_id}_before", "persona": pname, "split": split, "type": "thought_before",
                "theme": theme, "as_of": as_of_k.isoformat(timespec="seconds"),
                "text": db["draft"].strip(),
                "gold_ids": ids[:k + 1], "turning_ids": [i for i in turning_all if i in ids[:k + 1]],
                "root_id": ids[0], "dormant": False, "k": k,
                "gap_days": (as_of_k - times[k]).days, "thread_id": thread_id,
            })

            # 語彙は似ているが別件(紛らわしいメモ)
            dd = llm.json(distractor_prompt(pname, d["memos"][0]["content"]), ok_memos(2), "distractor")
            for x in dd["memos"]:
                t = START + datetime.timedelta(days=rng.randint(0, (END - START).days), hours=rng.randint(7, 22))
                memos.append({
                    "id": str(uuid.UUID(int=rng.getrandbits(128))), "persona": pname, "split": split,
                    "thread_id": None, "relation": "distractor", "parent_id": None,
                    "created_at": t.isoformat(timespec="seconds"),
                    "title": x.get("title", "").strip() or "無題", "content": x["content"].strip(),
                    "distractor_of": thread_id,
                })

        # 無関係なメモ
        ff = llm.json(filler_prompt(pname, a.fillers), ok_memos(a.fillers), "filler")
        for x in ff["memos"]:
            t = START + datetime.timedelta(days=rng.randint(0, (END - START).days), hours=rng.randint(7, 22))
            memos.append({
                "id": str(uuid.UUID(int=rng.getrandbits(128))), "persona": pname, "split": split,
                "thread_id": None, "relation": "filler", "parent_id": None,
                "created_at": t.isoformat(timespec="seconds"),
                "title": x.get("title", "").strip() or "無題", "content": x["content"].strip(),
            })

        # 該当なしクエリ(ペルソナのメモに存在しない話題)
        topics = rng.sample(OFF_DOMAIN, 3)
        nd = llm.json(nomatch_prompt(topics), lambda d: isinstance(d.get("drafts"), list) and len(d["drafts"]) == 3, "nomatch")
        for k, txt in enumerate(nd["drafts"]):
            queries.append({
                "qid": f"q_p{pi}_nm{k}", "persona": pname, "split": split, "type": "no_match",
                "as_of": END.isoformat(timespec="seconds"), "text": str(txt).strip(),
                "gold_ids": [], "turning_ids": [], "root_id": None, "dormant": False, "gap_days": None,
                "thread_id": None, "topic": topics[k],
            })

    memos.sort(key=lambda m: m["created_at"])
    json.dump(memos, open(os.path.join(a.out, "memos.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    json.dump(queries, open(os.path.join(a.out, "queries.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    from collections import Counter
    stats = {
        "memos": len(memos), "queries": len(queries),
        "relations": dict(Counter(m["relation"] for m in memos)),
        "memos_by_split": dict(Counter(m["split"] for m in memos)),
        "queries_by_type": dict(Counter(q["type"] for q in queries)),
        "personas": [{"name": pe["name"], "split": split_of[i]} for i, pe in enumerate(personas)],
        "threads_by_split": dict(Counter(q["split"] for q in queries if q["type"] == "thought_past")),
        "draft_generation": "別呼び出し(その時点までのメモのみ入力)",
        "distinct_titles_in_threads": len({m["title"] for m in memos if m["thread_id"]}),
        "thread_memos": sum(1 for m in memos if m["thread_id"]),
        "queries_by_split": dict(Counter(q["split"] for q in queries)),
        "dormant_queries": sum(1 for q in queries if q["dormant"]),
        "model": a.model if not a.mock else "MOCK", "backend": a.backend if not a.mock else "mock", "seed": a.seed,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    stats["cache_usage"] = llm.stats
    json.dump(stats, open(os.path.join(a.out, "stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n中断しました。完了分は cache/ に保存済みです。同じコマンドで続きから再開できます。")
        sys.exit(130)
    except RuntimeError as e:
        print(f"\n停止しました: {e}")
        sys.exit(1)
