<p align="right"><strong>日本語</strong> | <a href="README.en.md">English</a></p>

<p align="center">
  <img src="docs/architecture.svg" alt="JCFPC alignment reranker architecture" width="100%">
</p>

# JCFPC Alignment Reranker

**日本語・中国語の文学対訳に対して、候補生成・ニューラル再ランキング・単調動的計画法を組み合わせる研究実装です。**

[![CI](https://github.com/lzq628/jcfpc-alignment-reranker/actions/workflows/ci.yml/badge.svg)](https://github.com/lzq628/jcfpc-alignment-reranker/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![Data](https://img.shields.io/badge/data-copyright--safe-167D74)

本リポジトリは、JCFPC（Japanese-Chinese Fiction Parallel Corpus）構築過程から生まれた
**NLP 手法部分のみ**を、著作権保護された小説本文を含まない形で公開するものです。
実データ、人工注釈、学習済み LoRA、候補スコアは公開していません。代わりに、動作確認用の
合成データ、DP 実装、Qwen3-Reranker-8B の LoRA 学習・推論コード、集計済み評価結果を提供します。

## 要点

- 日本語・中国語の文群を `1:8` から `8:1` まで扱い、省略も表現できる単調 DP
- BGE-M3 cosine、Qwen3 reranker、両者のスコア融合を同一 decoder で比較
- Qwen3-Reranker-8B を、固定 query と hard negatives による listwise loss で LoRA 微調整
- 学習データと保護評価文脈の sentence-ID / 正規化テキスト重複をともにゼロに固定
- 人手で全文脈を確定した評価セット上で、BGE-M3 + DP より `+6.10` point の Unit F1
- 実験コード、ハッシュ検証、再開可能な学習、bootstrap 診断を一つの流れとして実装

## 研究上の問い

埋め込みモデルは高速ですが、文学翻訳における「意味が似ている」と
「同じ内容を過不足なく覆う」は同じではありません。長い候補の冒頭だけが一致した場合や、
語彙が大きく変わる意訳では、cosine 類似度だけで境界を決めることが難しくなります。

本実験では、候補となる日本語文群と中国語文群を同時に読む reranker を微調整し、
そのスコアを順序制約付き DP に渡しました。これにより、ニューラルモデルは局所的な
翻訳対応を評価し、DP は文書全体の順序・被覆・文群境界を管理します。

## 結果

<p align="center">
  <img src="docs/results.svg" alt="Primary human-gold evaluation results" width="88%">
</p>

主要評価は、モデル予測を見せずに人手で全文脈を確定し、学習前に保護した
20 文脈・193 gold alignment units に対して行いました。

| Scorer + decoder | Unit Precision | Unit Recall | Unit F1 |
|---|---:|---:|---:|
| Length-only DP | 62.76% | 47.15% | 53.85% |
| Frozen BGE-M3 + DP | 76.69% | 64.77% | 70.22% |
| **Fine-tuned Qwen3 reranker + DP** | **82.53%** | **70.98%** | **76.32%** |
| BGE-Qwen score fusion + DP | 76.69% | 64.77% | 70.22% |

Qwen と BGE の paired case-bootstrap 差は `+1.69` から `+11.70` point、
10,000 resamples の `99.66%` がゼロを上回りました。これは不確実性の診断であり、
正式な有意差検定ではありません。既知の難例 12 文脈は challenge set として分離し、
主要結果には混合していません。集計値は [reports/aggregate_metrics.json](reports/aggregate_metrics.json) にあります。

## Baseline と hybrid の定義

`70.22%` の baseline は **frozen BGE-M3 cosine + 同一の単調 DP** です。
Length-only baseline は `53.85%` です。

Hybrid は二段階の「BGE 召回 → Qwen 再ランキング」ではなく、全合法候補に対するスコア融合です。

```text
hybrid = 0.9 * BGE cosine + 0.1 * sigmoid(Qwen logit / 2)
```

この固定 hybrid は gold 評価上で BGE と同一経路を選びました。そのため、hybrid の改善は主張せず、
有効だった結果を **direct Qwen scorer + DP** として報告します。

## データ分離

LoRA 学習には 16,921 queries / 219,704 candidates を使用しました。保護した評価全文脈との間で、
以下を独立に検証しています。

```text
gold_train_sentence_id_overlap = 0
gold_train_normalized_sentence_text_overlap = 0
```

ただし、評価文脈は同じ五作品に由来する既往の audit contexts です。
「未知作品への汎化」や「前向きに採取した完全な blind test」とは表現しません。

## クイックスタート

合成データだけで、公開 DP と exact-unit 評価を実行できます。

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[test]"
pytest -q
python -m jcfpc_alignment.demo
```

合成例は公開動作確認専用であり、JCFPC の小説本文ではありません。

## LoRA 学習

学習には BF16 対応 GPU と、利用権限を確認した非公開データが必要です。

```bash
python -m pip install -e ".[train]"
python scripts/train_qwen_reranker.py \
  --data data/private/candidates \
  --output runs/qwen8
```

途中から再開する場合:

```bash
python scripts/train_qwen_reranker.py \
  --data data/private/candidates \
  --output runs/qwen8 \
  --resume runs/qwen8/checkpoint-000100
```

入力仕様と必須 leakage checks は [docs/DATA_SCHEMA.md](docs/DATA_SCHEMA.md) を参照してください。

## 構成

```text
src/jcfpc_alignment/       公開 DP・評価・合成 demo
scripts/                   Qwen LoRA 学習と候補 scoring
examples/                  著作権安全な合成データ
reports/                   テキストを含まない集計結果
docs/                      方法図・結果図・データ仕様
tests/                     DP と評価指標の回帰テスト
```

## 公開しないもの

- 小説および刊行翻訳の本文・断片
- 実データの `train.jsonl` / `dev.jsonl` / gold workbook
- sentence-level annotation、候補スコア、SQLite corpus
- private RunPod package、checkpoint、LoRA adapter

この区別により、方法と再現可能なソフトウェア設計を公開しながら、原著者・翻訳者・出版社の
権利と研究参加データを保護します。

## 解釈上の制限

- `76.32%` は JCFPC 全体の「正解率」ではなく、保護された全文脈での exact Unit F1 です。
- gold annotation は単独 annotator によるもので、inter-annotator agreement はありません。
- challenge set は主要結果と分離し、parameter tuning には使用していません。
- 学習済み adapter の再配布可能性と memorization risk は未評価のため、weight は公開していません。

## ライセンス

コードと本リポジトリ独自の図表は MIT License です。第三者のモデル、データ、出版物には
それぞれの利用条件が適用されます。
