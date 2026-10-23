"""
評価用データセットの生成（API不要・シード固定で再現可能・堅牢性強化版）

「更新・訂正・取り消しを含む個人メモ」を人工的に作り、正解ラベルを付ける。

  data/eval/memos_eval.json     メモ（ベクトルなし。supersedes は正解ラベル）
  data/eval/queries_eval.json   クエリ（ベクトルなし）
                                種類: current / as_of / windowed（期間指定つき）
"""

import argparse
import datetime
import json
import random

from src.config import EVAL_DIR

BASE_DATE = datetime.date(2026, 4, 1)

# ---------------------------------------------------------------- 語彙

RESCHEDULE_EVENTS = [
    "歯医者の予約", "ゼミの中間発表", "研究室の飲み会", "卒論の中間報告会",
    "就職面接", "美容室の予約", "引っ越しの立ち会い", "運転免許の更新",
    "健康診断", "結婚式の二次会", "サークルの新歓", "英会話のレッスン",
    "家族の食事会", "ボランティア説明会", "自動車の点検", "旅行の集合",
    "インターンの説明会", "就活セミナー", "同窓会", "アルバイトの面談",
    "研究室のミーティング", "指導教員との面談", "映画の約束", "親戚の法事",
    "ゴルフコンペ", "OB会", "読書会", "勉強会", "内定者懇親会", "留学説明会",
]

PLACES = [
    "駅前のカフェ", "大学の第二講義棟", "渋谷の居酒屋", "研究棟305室",
    "市立図書館", "新宿のオフィス", "自宅近くの病院", "千葉駅の改札前",
    "オンライン(Zoom)", "学生会館", "幕張のホール", "本館2階の会議室",
    "東京駅の八重洲口", "実家", "近所の公民館", "横浜の貸し会議室",
]

TIMES = ["9:00", "10:30", "13:00", "14:30", "16:00", "18:30", "19:00"]

CORRECTION_ITEMS = [
    "コーヒー豆", "参考書", "定期券", "学会の参加費", "ノートPC",
    "ゼミ合宿の費用", "研究室の会費", "引っ越しの見積もり", "英会話の月謝",
    "卒論の印刷代", "ワイヤレスイヤホン", "図書カード", "電気代", "家賃",
    "ジムの月会費", "スマホの修理代", "新幹線の切符", "資格試験の受験料",
    "運動靴", "プリンターのインク",
]

CANCEL_EVENTS = [
    "英会話の体験会", "学内の就活ガイダンス", "友人とのキャンプ", "野球観戦",
    "美術館の見学", "研究室の花見", "日帰りのバス旅行", "ハッカソン",
    "地域の清掃活動", "映画の試写会", "サイクリング", "料理教室",
]

TASKS = [
    "卒論の第2章", "英語のレポート", "確定申告", "奨学金の書類", "研究計画書",
    "実験データの整理", "履歴書", "エントリーシート", "引っ越しの手続き",
    "住民票の取得", "部屋の掃除", "実験ノートの提出", "論文の要約",
    "学会の発表スライド", "参考文献リスト", "保険の手続き", "免許更新の予約",
    "自己分析シート", "資格の勉強", "プログラミング課題", "統計学の課題",
    "卒論の図表", "同期への連絡", "会計報告", "引き継ぎ資料",
    "奨学金の返還手続き", "図書館の予約本", "ゼミの発表資料", "旅費の精算",
    "アンケートの集計",
]

FOODS = ["カレー", "うどん", "ラーメン", "オムライス", "サラダ", "焼き魚", "パスタ"]
BOOKS = ["推理小説", "統計の入門書", "エッセイ", "歴史の本", "技術書", "漫画"]
HOBBIES = ["ランニング", "散歩", "ギター", "料理", "読書", "ジム"]

# ---------------------------------------------------------------- 文面テンプレート（表現の多様性を拡張）

RESCHEDULE_V1 = [
    "{ev}は{d}の{t}から{pl}。忘れないこと。",
    "{d} {t}に{pl}で{ev}がある。",
    "予定メモ：{ev}、{d}、{t}、場所は{pl}。",
    "そういえば{ev}、{d}に{pl}で開催されるらしい。",
]
RESCHEDULE_UPDATE = [
    "{ev}の日程が変わった。新しい日程は{d}の{t}、場所は{pl}。",
    "【変更】{ev}は{d}の{t}に{pl}で行うことになった。",
    "{ev}、{d}の{t}に変更。場所は{pl}で確定。",
    "連絡あり。{ev}のスケジュールがずれて{d}の{t}（{pl}）になった。",
]
RESCHEDULE_QUERY = [
    "{ev}はいつどこだっけ",
    "{ev}の集合時間と場所を確認したい",
    "{ev}の予定を教えて",
]

CORRECTION_V1 = [
    "{it}は{p}円。",
    "{it}の金額は{p}円だった。",
    "{it}、{p}円で払った。",
    "メモ：{it}の価格は{p}円。",
]
CORRECTION_UPDATE = [
    "訂正：{it}は{p}円だった。",
    "{it}の金額を間違えていた。正しくは{p}円。",
    "{it}は{p}円に訂正。",
    "レシートを確認したら{it}の値段は{p}円だったわ。",
]
CORRECTION_QUERY = [
    "{it}はいくらだったっけ",
    "{it}にかかった費用を知りたい",
    "{it}の金額",
]

CANCEL_V1 = [
    "{ev}に参加する。日程は{d}。",
    "{ev}に申し込んだ。{d}に行く予定。",
    "予定：{ev}（{d}）",
]
CANCEL_UPDATE = [
    "{ev}は中止になった。参加しない。",
    "{ev}が取り消しになったので、{d}の予定は空いた。",
    "取り消し：{ev}。{d}の予定は白紙に戻す。",
    "残念だけど{ev}は見送りになった。",
]
CANCEL_QUERY = [
    "{ev}に行くんだっけ",
    "{ev}の予定はどうなった",
    "{ev}って結局どうなった？",
]

TASK_STAGE = {
    0: [
        "{task}はまだ着手していない。{d}までに終わらせる。",
        "{task}の締め切りは{d}。まだ手を付けていない。",
    ],
    1: [
        "{task}に取りかかった。半分ほど進んだ。",
        "{task}は進行中。あと半分くらい。",
    ],
    2: [
        "{task}を提出した。完了。",
        "{task}は終わった。{d}に提出済み。",
    ],
}
TASK_QUERY = [
    "{task}の進み具合",
    "{task}は終わったっけ",
    "{task}ってどうなってる？",
]

# 通常の日常ノイズ
NOISE_TEMPLATES = [
    "今日は{pl}で昼食に{food}を食べた。おいしかった。",
    "{book}を読み終えた。続きが気になる。",
    "週末は{hobby}をして過ごした。{pl}まで足を延ばした。",
    "{t}に起きられなかった。明日は早く寝る。",
    "{pl}の近くに新しい店ができていた。今度{food}を食べに行く。",
    "最近{hobby}を始めた。続けられるか不安。",
    "{book}を買った。{d}までに読みたい。",
    "友達と{pl}で会った。{food}の話で盛り上がった。",
]

# 【新規追加】引っ掛けノイズ（トピックや語彙・イベント名が似ているが更新関係にはないノイズ）
HARD_NOISE_TEMPLATES = [
    "去年の{ev}は{pl}ですごく楽しかったな。",
    "そういえば以前話題になった{it}について、後輩から質問された。",
    "過去に検討した{task}の資料を整理していた。",
    "{ev}の過去の記録を見返した。",
]

# ---------------------------------------------------------------- 生成

def fmt_date(d):
    return f"{d.month}月{d.day}日"

def _fill(template, **kw):
    return template.format(**kw)

def _pick_distinct(rng, options, n):
    pool = options[:]
    rng.shuffle(pool)
    return [pool[i % len(pool)] for i in range(n)]

def _make_memo(mid, title, content, created, event_date, supersedes,
               chain_id=None, version=None, kind=None):
    return {
        "id": mid,
        "title": title,
        "content": content,
        "date": created.strftime("%Y/%m/%d"),
        "created_at": created.isoformat(timespec="seconds"),
        "event_date": event_date.isoformat() if event_date else None,
        "supersedes": supersedes,
        "status": "active",
        "chain_id": chain_id,
        "version": version,
        "kind": kind,
    }

def _random_created(rng, start_offset, span):
    day = BASE_DATE + datetime.timedelta(
        days=start_offset + rng.randint(0, span)
    )
    return datetime.datetime.combine(
        day,
        datetime.time(rng.randint(7, 22), rng.choice([0, 12, 30, 45])),
    )

def build_chains(rng, n_resched=28, n_correct=16, n_cancel=12, n_task=24):
    memos = []
    chain_meta = []
    chain_no = 0

    def n_versions():
        return 3 if rng.random() < 0.3 else 2

    # ---- 予定変更
    for ev in _pick_distinct(rng, RESCHEDULE_EVENTS, n_resched):
        chain_no += 1
        cid = f"c{chain_no:03d}"
        nv = n_versions()
        created = _random_created(rng, 0, 40)
        event_day = created.date() + datetime.timedelta(days=rng.randint(20, 45))
        ids = []

        for v in range(1, nv + 1):
            if v > 1:
                created += datetime.timedelta(days=rng.randint(3, 12))
                event_day += datetime.timedelta(days=rng.randint(3, 10))

            tmpl = rng.choice(RESCHEDULE_V1 if v == 1 else RESCHEDULE_UPDATE)
            content = _fill(
                tmpl, ev=ev, d=fmt_date(event_day),
                t=rng.choice(TIMES), pl=rng.choice(PLACES),
            )

            mid = f"{cid}-v{v}"
            memos.append(_make_memo(
                mid, ev if v == 1 else f"{ev}（変更）", content, created,
                event_day, [ids[-1]] if ids else [], cid, v, "reschedule",
            ))
            ids.append(mid)

        chain_meta.append({"chain_id": cid, "kind": "reschedule", "entity": ev, "ids": ids})

    # ---- 訂正
    for it in _pick_distinct(rng, CORRECTION_ITEMS, n_correct):
        chain_no += 1
        cid = f"c{chain_no:03d}"
        nv = n_versions()
        created = _random_created(rng, 0, 50)
        ids = []

        for v in range(1, nv + 1):
            if v > 1:
                created += datetime.timedelta(days=rng.randint(2, 10))

            tmpl = rng.choice(CORRECTION_V1 if v == 1 else CORRECTION_UPDATE)
            price = rng.randrange(500, 60000, 100)
            content = _fill(tmpl, it=it, p=f"{price:,}")

            mid = f"{cid}-v{v}"
            memos.append(_make_memo(
                mid, it if v == 1 else f"{it}（訂正）", content, created, None,
                [ids[-1]] if ids else [], cid, v, "correction",
            ))
            ids.append(mid)

        chain_meta.append({"chain_id": cid, "kind": "correction", "entity": it, "ids": ids})

    # ---- 取り消し
    for ev in _pick_distinct(rng, CANCEL_EVENTS, n_cancel):
        chain_no += 1
        cid = f"c{chain_no:03d}"
        created = _random_created(rng, 0, 40)
        event_day = created.date() + datetime.timedelta(days=rng.randint(15, 40))
        ids = []

        for v in (1, 2):
            if v == 2:
                created += datetime.timedelta(days=rng.randint(3, 12))

            tmpl = rng.choice(CANCEL_V1 if v == 1 else CANCEL_UPDATE)
            content = _fill(tmpl, ev=ev, d=fmt_date(event_day))

            mid = f"{cid}-v{v}"
            memos.append(_make_memo(
                mid, ev if v == 1 else f"{ev}（取り消し）", content, created,
                event_day if v == 1 else None,
                [ids[-1]] if ids else [], cid, v, "cancel",
            ))
            ids.append(mid)

        chain_meta.append({"chain_id": cid, "kind": "cancel", "entity": ev, "ids": ids})

    # ---- 進捗
    for task in _pick_distinct(rng, TASKS, n_task):
        chain_no += 1
        cid = f"c{chain_no:03d}"
        stages = [0, 1, 2] if rng.random() < 0.3 else [0, 2]
        created = _random_created(rng, 0, 40)
        due = created.date() + datetime.timedelta(days=rng.randint(10, 30))
        ids = []

        for v, stage in enumerate(stages, start=1):
            if v > 1:
                created += datetime.timedelta(days=rng.randint(3, 10))

            content = _fill(rng.choice(TASK_STAGE[stage]), task=task, d=fmt_date(due))

            mid = f"{cid}-v{v}"
            memos.append(_make_memo(
                mid, task, content, created, None,
                [ids[-1]] if ids else [], cid, v, "progress",
            ))
            ids.append(mid)

        chain_meta.append({"chain_id": cid, "kind": "progress", "entity": task, "ids": ids})

    return memos, chain_meta


def build_noise(rng, n, chain_meta):
    memos = []
    # 通常ノイズと引っ掛けノイズを半々に分ける
    n_hard = n // 2
    n_normal = n - n_hard

    # 通常ノイズ
    for i in range(1, n_normal + 1):
        created = _random_created(rng, 0, 90)
        content = _fill(
            rng.choice(NOISE_TEMPLATES),
            pl=rng.choice(PLACES), food=rng.choice(FOODS),
            book=rng.choice(BOOKS), hobby=rng.choice(HOBBIES),
            t=rng.choice(TIMES),
            d=fmt_date(created.date() + datetime.timedelta(days=rng.randint(3, 20))),
        )
        memos.append(_make_memo(f"n{i:04d}", "日常メモ", content, created, None, [], None, None, "noise"))

    # 引っ掛けノイズ（既存のチェーンのエンティティ名を流用して文脈違いのメモを作る）
    entities = [meta["entity"] for meta in chain_meta]
    for i in range(1, n_hard + 1):
        created = _random_created(rng, 0, 90)
        ent = rng.choice(entities)
        tmpl = rng.choice(HARD_NOISE_TEMPLATES)
        content = _fill(tmpl, ev=ent, it=ent, task=ent, pl=rng.choice(PLACES))
        memos.append(_make_memo(f"hn{i:04d}", "引っかけメモ", content, created, None, [], None, None, "hard_noise"))

    return memos


QUERY_TEMPLATES = {
    "reschedule": RESCHEDULE_QUERY,
    "correction": CORRECTION_QUERY,
    "cancel": CANCEL_QUERY,
    "progress": TASK_QUERY,
}

def _query_text(rng, kind, entity):
    tmpl = rng.choice(QUERY_TEMPLATES[kind])
    key = {"reschedule": "ev", "cancel": "ev", "correction": "it", "progress": "task"}[kind]
    return tmpl.format(**{key: entity})


def build_queries(rng, memos, chain_meta):
    by_id = {m["id"]: m for m in memos}
    queries = []

    for meta in chain_meta:
        ids = meta["ids"]

        queries.append({
            "qid": f'{meta["chain_id"]}-current',
            "type": "current",
            "text": _query_text(rng, meta["kind"], meta["entity"]),
            "as_of": None,
            "relevant_id": ids[-1],
            "stale_ids": ids[:-1],
            "chain_id": meta["chain_id"],
            "kind": meta["kind"],
        })

        if len(ids) >= 2:
            v2_date = datetime.date.fromisoformat(by_id[ids[1]]["created_at"][:10])
            v1_date = datetime.date.fromisoformat(by_id[ids[0]]["created_at"][:10])
            as_of = v2_date - datetime.timedelta(days=1)

            # 厳密な時系列順序のバリデーション（アサーション）
            assert as_of >= v1_date, f"時系列エラー: chain {meta['chain_id']} の as_of ({as_of}) が v1_date ({v1_date}) より過去です。"

            queries.append({
                "qid": f'{meta["chain_id"]}-asof',
                "type": "as_of",
                "text": _query_text(rng, meta["kind"], meta["entity"]),
                "as_of": as_of.isoformat(),
                "relevant_id": ids[0],
                "stale_ids": [],
                "chain_id": meta["chain_id"],
                "kind": meta["kind"],
            })

    return queries


def build_windowed_queries(rng, memos, chain_meta):
    """
    期間指定つきクエリ（マジックナンバーのオフセットを排除し、渡された専用rngを使う）
    """
    by_id = {m["id"]: m for m in memos}
    out = []

    for meta in chain_meta:
        if meta["kind"] != "reschedule":
            continue

        ids = meta["ids"]
        latest_event_str = by_id[ids[-1]]["event_date"]
        if not latest_event_str:
            continue

        latest = datetime.date.fromisoformat(latest_event_str)
        half = datetime.timedelta(days=2)

        out.append({
            "qid": f'{meta["chain_id"]}-win',
            "type": "windowed",
            "text": _query_text(rng, meta["kind"], meta["entity"]),
            "as_of": None,
            "time_from": (latest - half).isoformat(),
            "time_to": (latest + half).isoformat(),
            "relevant_id": ids[-1],
            "stale_ids": ids[:-1],
            "chain_id": meta["chain_id"],
            "kind": meta["kind"],
        })

    return out


def build_dataset(seed=42, n_noise=150):
    rng = random.Random(seed)
    
    # ウィンドウクエリ専用の独立した乱数ジェネレータ（マジックナンバーのオフセット廃止）
    window_rng = random.Random(seed + 999)

    chain_memos, chain_meta = build_chains(rng)
    noise = build_noise(rng, n_noise, chain_meta)

    memos = chain_memos + noise
    rng.shuffle(memos)

    queries = build_queries(rng, memos, chain_meta)
    queries += build_windowed_queries(window_rng, memos, chain_meta)

    return memos, queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=int, default=150)
    args = parser.parse_args()

    memos, queries = build_dataset(args.seed, args.noise)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    with open(EVAL_DIR / "memos_eval.json", "w", encoding="utf-8") as f:
        json.dump(memos, f, ensure_ascii=False, indent=2)

    with open(EVAL_DIR / "queries_eval.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    n_chain = len({m["chain_id"] for m in memos if m["chain_id"]})
    counts = {
        t: sum(q["type"] == t for q in queries)
        for t in ("current", "as_of", "windowed")
    }

    print(f"メモ {len(memos)}件（チェーン{n_chain}本 + ノイズ{args.noise}件[通常+ハード]）")
    print(f"クエリ {len(queries)}件（" + " / ".join(f"{t} {n}" for t, n in counts.items()) + "）")
    print(f"保存先: {EVAL_DIR}")


if __name__ == "__main__":
    main()