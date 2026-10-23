# 個人メモの検索・推薦システム（卒業研究）

意味検索（埋め込み）・類似グラフ上のPersonalized PageRank・時間整合・
**更新済みメモの減点（supersedes）** を組み合わせた、個人メモの検索システムと、その評価実験。

## セットアップ
```bash
pip install -r requirements.txt
cp .env.example .env        # GEMINI_API_KEY を記入
```

## アプリの起動
```bash
python -m uvicorn src.main:app --reload
```

## テスト（APIキー不要）
```bash
python -m unittest discover -s tests -t .
```

## 埋め込みの次元数を切り替える
`gemini-embedding-2` の出力次元は **128〜3072**（推奨: 768 / 1536 / 3072、デフォルト 3072）。
`.env` の `EMBED_DIM` で、次のどれかを指定できる（未指定は 256）。

| 指定 | 次元 | 意味 |
|---|---|---|
| `min` | 128 | APIの最小 |
| `recommended` | 768 | 推奨値（ほかの推奨値 1536, 3072 は数値で指定） |
| `max` | 3072 | APIの最大（APIのデフォルトでもある） |
| 整数 | 128〜3072 | 任意（範囲外はエラー。黙って丸めない） |

**次元を変えたら、保存済みメモを再埋め込みする**（次元が違うと内積が計算できないため）:
```bash
python -m tools.reembed_memos --dim recommended   # 元ファイルは .bak_d<元の次元> に退避される
```
次元が合っていないと、検索は原因と対処を示すエラーで止まる（誤った結果を黙って返さない）。

## 評価実験の手順（プロジェクトのルートで実行）
```bash
python -m experiments.build_eval_dataset                          # 1. 評価データ生成（API不要・seed固定）
python -m experiments.embed_dataset --dim recommended             # 2. ベクトル化（Gemini API）
python -m experiments.run_eval --dim recommended --distractors 300  # 3. 手法比較（API不要）
python -m experiments.analyze_accuracy --dim recommended          # 4. AIの更新判定の精度（Gemini API）
```
- `--dim` は `min` / `recommended` / `max` / 128〜3072の整数。データと結果は次元ごとに別々に保存される
  （`data/eval/*_d768.json`, `results/main_d768/`）。
- 手順2は途中で止まっても、再実行で続きから進む。

### 次元の比較（推奨値すべて・最小・最大）
```bash
# 公式の推奨値すべて (768, 1536, 3072) を、ベクトル化 → 評価 → 比較表まで一括実行
python -m experiments.sweep_dims --dims recommended_all

# 最小 + 推奨すべて (128, 768, 1536, 3072)
python -m experiments.sweep_dims --dims all

# 自由な組み合わせ（カンマ区切り）
python -m experiments.sweep_dims --dims min,recommended_all,256

# 先に、実行されるコマンドとAPI呼び出し回数の目安だけ確認する
python -m experiments.sweep_dims --dims recommended_all --dry-run
```
- `--dims` に指定できるのは `recommended_all` / `all` / `min` / `recommended` / `max` / 128〜3072の整数。
  1つでも不正な値があれば、何も実行せずにエラーになる。
- API呼び出しの目安は 1次元あたり約518回（メモ330 + クエリ188）。`recommended_all` で約1,554回、`all` で約2,072回。
  （`--dry-run` は、評価データの件数から自動で数える）
- 途中で失敗したら、同じコマンドを再実行する。ベクトル化は続きから進む。
  評価だけやり直すなら `--skip-embed`。
- 結果は `results/dim_comparison.csv`（表。`recommended` 列が推奨値かどうか）と
  `results/dim_comparison.png`（図。推奨値には `*`）。
- `--distractors` の背景ノイズ（Wikipediaメモ）は256次元なので、256以外の次元では自動的に使われない
  （警告が出る）。使いたい場合は、そのファイルを再埋め込みする。

## 結果ファイル（`results/main_d<次元>/`）
| ファイル | 内容 |
|---|---|
| `dim_comparison.csv` / `.png`（`results/` 直下） | 次元ごとの結果の比較表と図（`sweep_dims` / `compare_dims`） |
| `main_results.csv` | 手法×クエリ種別の指標（MRR, Hit@k, nDCG, StaleAbove と95%信頼区間） |
| `paired_diff_current.csv` | full と各手法との対応のある差（信頼区間つき） |
| `by_kind_current.csv` | 予定変更／訂正／取り消し／進捗ごとの結果 |
| `sweep_stale_penalty.csv` | 減点係数を変えたとき |
| `robustness_supersedes.csv` | 更新リンクを一部見逃した場合の性能（AI判定の誤りの影響） |
| `supersedes_accuracy.csv` | AI判定の Precision / Recall / F1（手順4） |
| `run_config.json` | 実験条件（シード・パラメータ・モデル） |

設計の詳細と限界は `docs/evaluation_design.md`、修正点は `docs/changes.md`。

## ディレクトリ
```
src/          本体（検索・推薦・グラフ・BM25・指標・FastAPI）
tools/        保守用（保存済みメモの再埋め込み）
experiments/  評価パイプライン（legacy/ は旧版・非推奨）
tests/        単体テスト
data/         raw/（元データ） processed/（アプリが使うデータ） eval/（評価データ）
results/      実験結果
docs/         設計メモ・修正履歴
```# 個人メモの検索・推薦システム（卒業研究）

意味検索（埋め込み）・類似グラフ上のPersonalized PageRank・時間整合・
**更新済みメモの減点（supersedes）** を組み合わせた、個人メモの検索システムと、その評価実験。

## セットアップ
```bash
pip install -r requirements.txt
cp .env.example .env        # GEMINI_API_KEY を記入
```

## アプリの起動
```bash
uvicorn src.main:app --reload
```

## テスト（APIキー不要）
```bash
python -m unittest discover -s tests -t .
```

## 埋め込みの次元数を切り替える
`gemini-embedding-2` の出力次元は **128〜3072**（推奨: 768 / 1536 / 3072、デフォルト 3072）。
`.env` の `EMBED_DIM` で、次のどれかを指定できる（未指定は 256）。

| 指定 | 次元 | 意味 |
|---|---|---|
| `min` | 128 | APIの最小 |
| `recommended` | 768 | 推奨値（ほかの推奨値 1536, 3072 は数値で指定） |
| `max` | 3072 | APIの最大（APIのデフォルトでもある） |
| 整数 | 128〜3072 | 任意（範囲外はエラー。黙って丸めない） |

**次元を変えたら、保存済みメモを再埋め込みする**（次元が違うと内積が計算できないため）:
```bash
python -m tools.reembed_memos --dim recommended   # 元ファイルは .bak_d<元の次元> に退避される
```
次元が合っていないと、検索は原因と対処を示すエラーで止まる（誤った結果を黙って返さない）。

## 評価実験の手順（プロジェクトのルートで実行）
```bash
python -m experiments.build_eval_dataset                          # 1. 評価データ生成（API不要・seed固定）
python -m experiments.embed_dataset --dim recommended             # 2. ベクトル化（Gemini API）
python -m experiments.run_eval --dim recommended --distractors 300  # 3. 手法比較（API不要）
python -m experiments.analyze_accuracy --dim recommended          # 4. AIの更新判定の精度（Gemini API）
```
- `--dim` は `min` / `recommended` / `max` / 128〜3072の整数。データと結果は次元ごとに別々に保存される
  （`data/eval/*_d768.json`, `results/main_d768/`）。
- 手順2は途中で止まっても、再実行で続きから進む。

### 次元の比較（推奨値すべて・最小・最大）
```bash
# 公式の推奨値すべて (768, 1536, 3072) を、ベクトル化 → 評価 → 比較表まで一括実行
python -m experiments.sweep_dims --dims recommended_all

# 最小 + 推奨すべて (128, 768, 1536, 3072)
python -m experiments.sweep_dims --dims all

# 自由な組み合わせ（カンマ区切り）
python -m experiments.sweep_dims --dims min,recommended_all,256

# 先に、実行されるコマンドとAPI呼び出し回数の目安だけ確認する
python -m experiments.sweep_dims --dims recommended_all --dry-run
```
- `--dims` に指定できるのは `recommended_all` / `all` / `min` / `recommended` / `max` / 128〜3072の整数。
  1つでも不正な値があれば、何も実行せずにエラーになる。
- API呼び出しの目安は 1次元あたり約518回（メモ330 + クエリ188）。`recommended_all` で約1,554回、`all` で約2,072回。
  （`--dry-run` は、評価データの件数から自動で数える）
- 途中で失敗したら、同じコマンドを再実行する。ベクトル化は続きから進む。
  評価だけやり直すなら `--skip-embed`。
- 結果は `results/dim_comparison.csv`（表。`recommended` 列が推奨値かどうか）と
  `results/dim_comparison.png`（図。推奨値には `*`）。
- `--distractors` の背景ノイズ（Wikipediaメモ）は256次元なので、256以外の次元では自動的に使われない
  （警告が出る）。使いたい場合は、そのファイルを再埋め込みする。

## 結果ファイル（`results/main_d<次元>/`）
| ファイル | 内容 |
|---|---|
| `dim_comparison.csv` / `.png`（`results/` 直下） | 次元ごとの結果の比較表と図（`sweep_dims` / `compare_dims`） |
| `main_results.csv` | 手法×クエリ種別の指標（MRR, Hit@k, nDCG, StaleAbove と95%信頼区間） |
| `paired_diff_current.csv` | full と各手法との対応のある差（信頼区間つき） |
| `by_kind_current.csv` | 予定変更／訂正／取り消し／進捗ごとの結果 |
| `sweep_stale_penalty.csv` | 減点係数を変えたとき |
| `robustness_supersedes.csv` | 更新リンクを一部見逃した場合の性能（AI判定の誤りの影響） |
| `supersedes_accuracy.csv` | AI判定の Precision / Recall / F1（手順4） |
| `run_config.json` | 実験条件（シード・パラメータ・モデル） |

設計の詳細と限界は `docs/evaluation_design.md`、修正点は `docs/changes.md`。

## ディレクトリ
```
src/          本体（検索・推薦・グラフ・BM25・指標・FastAPI）
tools/        保守用（保存済みメモの再埋め込み）
experiments/  評価パイプライン（legacy/ は旧版・非推奨）
tests/        単体テスト
data/         raw/（元データ） processed/（アプリが使うデータ） eval/（評価データ）
results/      実験結果
docs/         設計メモ・修正履歴
```