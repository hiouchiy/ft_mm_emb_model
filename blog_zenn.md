---
title: "Databricks AI Runtime で SigLIP2 をファインチューニングして日本語画像タグ付けパイプラインを構築する"
emoji: "🖼️"
type: "tech"
topics: ["databricks", "siglip2", "machinelearning", "computervision", "python"]
published: false
---

:::message
この記事は、Databricks AI Runtime の Public Preview を記念して執筆しました。AI Runtime により、GPU クラスタの構築・管理なしにサーバーレスでディープラーニングワークロードを実行できるようになりました。
詳細: [Introducing AI Runtime: Scalable Serverless NVIDIA GPUs on Databricks](https://www.databricks.com/jp/blog/introducing-ai-runtime-scalable-serverless-nvidia-gpus-databricks-training-and-finetuning)
:::

## はじめに

画像に対して日本語でタグ付けやエンベディング抽出を行いたい。そんなニーズに対して、Google の **SigLIP2** をファインチューニングし、Databricks 上でエンドツーエンドのパイプラインを構築する方法を紹介します。

本記事のポイントは、**データ準備 → モデルトレーニング → サービングデプロイ → バッチ推論** という一連の ML ワークフローが、すべて Databricks のワンプラットフォーム上で完結することです。特にトレーニングには、新しく Public Preview となった **AI Runtime（サーバーレス GPU）** を使用します。

コードのフルセットは以下の GitHub リポジトリに公開しています。本記事では要点のみを取り上げます。

👉 **[GitHub: ft_mm_emb_model](https://github.com/hiouchiy/ft_mm_emb_model)** *(リポジトリ URL は後日更新)*

## AI Runtime とは

AI Runtime は、Databricks が提供するサーバーレス GPU コンピュート環境です。

- **クラスタ管理不要**: GPU ドライバやオートスケーリングの設定が不要。ノートブックから「Serverless GPU」を選ぶだけ
- **GPU 選択**: A10（中規模ワークロード向け）と H100（大規模トレーニング向け）から選択可能
- **従量課金**: アイドル時は自動停止。使った分だけのコスト
- **PyTorch / Transformers がプリインストール**: `%pip install` で追加パッケージを入れるだけですぐ使える

今回は A10 GPU 1枚で SigLIP2（約 1.1B パラメータ）のファインチューニングを実行しました。**セットアップ 55 秒、トレーニング約 7 分**で完了しています。

## アーキテクチャ全体像

```
┌─────────────────────────────────────────────────────────────┐
│                    Databricks ワンプラットフォーム             │
│                                                             │
│  ① データ準備          ② トレーニング        ③ デプロイ       │
│  ┌───────────┐       ┌──────────────┐     ┌────────────┐   │
│  │ STAIR     │──→──→ │ AI Runtime   │──→──│ Model      │   │
│  │ Captions  │ Delta │ (A10 GPU)    │ UC  │ Serving    │   │
│  │ + MS-COCO │ Table │ SigLIP2 FT   │ Model│ Endpoint  │   │
│  └───────────┘       └──────────────┘     └─────┬──────┘   │
│       │                                         │          │
│       │              ④ バッチ推論                │          │
│       │             ┌──────────────┐            │          │
│       └────────────→│ ai_query()   │←───────────┘          │
│                     │ SQL関数      │                        │
│                     └──────┬───────┘                        │
│                            ↓                                │
│                     Delta Table (結果)                       │
└─────────────────────────────────────────────────────────────┘
```

## Step 1: データ準備 — STAIR Captions

トレーニングデータには **STAIR Captions**（MS-COCO 画像に対する人手作成の日本語キャプション）を使用しました。82 万枚の画像から 1,000 枚をサンプリングしています。

```python
# STAIR Captions をダウンロード＆展開
with tarfile.open(STAIR_TAR_PATH, "r:gz") as tar:
    tar.extractall(STAIR_EXTRACT_DIR)

# MS-COCO 画像を並列ダウンロードして Volume に保存
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(download_image, img_id): img_id
               for img_id in sampled_ids}
```

データは以下のスキーマで Delta テーブルに格納します。

| カラム | 内容 |
|--------|------|
| `image_path` | Volume 上の画像パス |
| `text_positive` | 日本語キャプション（正例） |
| `text_negative` | 別画像のキャプション（負例） |
| `split` | train / valid / test |

## Step 2: SigLIP2 ファインチューニング on AI Runtime

### SigLIP2 とは

SigLIP2 は Google が開発したマルチモーダルモデルで、画像とテキストの対応関係を学習します。CLIP と異なり **sigmoid** ベースの損失関数を使用し、各画像-テキストペアを独立した二値分類として扱います。

### AI Runtime でのトレーニング

AI Runtime の Jobs API で A10 GPU を指定して実行します。

```yaml
# Databricks Asset Bundles 形式
tasks:
  - task_key: train_siglip2
    notebook_task:
      notebook_path: /Workspace/.../2_train_siglip2_from_delta
    compute:
      hardware_accelerator: GPU_1xA10
    environment_key: ai_runtime_env

environments:
  - environment_key: ai_runtime_env
    spec:
      client: "4"
      dependencies:
        - transformers==4.51.3
        - pynvml
```

### トレーニングのポイント

**Projection Head のみファインチューニング**します。全パラメータの 0.02% のみを学習するため、A10 1枚で十分です。

```python
TRAINABLE_LAYERS = ["text_model.head", "vision_model.head"]

for name, param in model.named_parameters():
    param.requires_grad = False
    if any(layer in name for layer in TRAINABLE_LAYERS):
        param.requires_grad = True
# Total: 1.1B, Trainable: 約200K (0.02%)
```

損失関数は SigLIP オリジナルの sigmoid contrastive loss です。

### トレーニング結果

STAIR Captions 1,000 枚（train: 713, valid: 137, test: 150）で 5 エポックのファインチューニングを行いました。すべてのメトリクスは MLflow に自動記録されます。

**Train Loss の推移**

| Epoch | Train Loss |
|:-----:|:----------:|
| 1 | 0.6904 |
| 2 | 0.5270 |
| 3 | 0.4986 |
| 4 | 0.4425 |
| 5 | 0.4418 |

Loss は順調に減少しており、モデルの学習は進んでいます。

**ベースライン vs ファインチューニング後**

| メトリクス | | ベースライン（FT前） | FT後 | 変化 |
|-----------|:---:|:---:|:---:|:---:|
| Triplet Accuracy | Valid | **74.5%** | 67.9% | -6.6pt |
| | Test | **58.0%** | 55.3% | -2.7pt |
| Recall@1 | Valid | **6.6%** | 5.8% | -0.8pt |
| | Test | 2.0% | 2.0% | ±0 |
| Recall@1 | Train | 1.0% | **3.1%** | +2.1pt |

- **Triplet Accuracy**: 画像に対して正解テキスト（positive）が不正解テキスト（negative）よりも高い類似度スコアを得た割合（ランダムが 50%）
- **Recall@1**: 全キャプションの中から正しいキャプションを 1 位に取得できた割合

結果を見ると、Train の Recall@1 は改善している一方、Valid/Test では精度が低下しています。これは **1,000 枚という少量データでは過学習が発生**し、汎化性能が向上しないことを示しています。ベースの SigLIP2 自体が高い汎化性能を持つため、少量データでの Projection Head のみのファインチューニングでは性能を上回ることが難しいのです。

:::message alert
今回はパイプラインのデモ目的で 1,000 枚のみ使用しています。実運用でファインチューニングの効果を得るには、**数万枚以上のドメイン固有データ**が推奨されます。日本語に特化させたい場合は [WAON](https://huggingface.co/datasets/llm-jp/WAON)（1.55 億ペア）や [STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions) 全体（約 16 万枚）の活用を検討してください。
:::

**GPU 使用状況**（MLflow System Metrics より）

| 項目 | 値 |
|------|-----|
| GPU メモリ使用量 | 11.9 GB / 24 GB（49.2%） |
| GPU 電力 | 103W（34.3%） |
| システムメモリ | 15.5 GB（23.3%） |

A10 GPU（24GB VRAM）に対してメモリ使用率は約 50% で余裕があり、バッチサイズの増加やより大きなモデルの使用も可能です。

```python
def siglip_contrastive_loss(image_embs, text_embs):
    logits = (ie @ te.T) * t + b
    labels = 2 * torch.eye(n, device=logits.device) - 1
    return -torch.nn.functional.logsigmoid(labels * logits).mean()
```

### サーバーレス GPU 環境での Tips

AI Runtime（サーバーレス GPU）で HuggingFace モデルを使う場合、`AutoProcessor.from_pretrained` が失敗することがあります。以下の2点で回避できます。

```python
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download
LOCAL_MODEL_DIR = "/tmp/google_siglip2-so400m-patch14-384"
snapshot_download(repo_id="google/siglip2-so400m-patch14-384",
                  local_dir=LOCAL_MODEL_DIR)

# ローカルパスから読み込み
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModel.from_pretrained(LOCAL_MODEL_DIR)
```

## Step 3: サービングエンドポイントへのデプロイ

ファインチューニング済みモデルを MLflow PyFunc でラップし、Unity Catalog に登録、GPU サービングエンドポイントとしてデプロイします。

### 日本語推論で重要なポイント

SigLIP2 はテキスト入力に **テンプレートが必須**です。生のタグを渡すと正しいスコアが出ません。

```python
# NG: 生のタグ
texts = ["風景", "動物", "食べ物"]

# OK: テンプレート適用
texts = ["This is a photo of 風景.",
         "This is a photo of 動物.",
         "This is a photo of 食べ物."]
```

また、`processor.tokenizer()` と `processor.image_processor()` を別々に呼ぶのではなく、**統合 processor** を使います。

```python
inputs = self.processor(
    text=templates, images=image, return_tensors="pt",
    padding="max_length", max_length=64, truncation=True,
)
outputs = self.model(**inputs)
logits = outputs.logits_per_image[0]
scores = torch.sigmoid(logits).cpu().tolist()
```

## Step 4: ai_query() でバッチ推論

デプロイしたエンドポイントに対して、`ai_query()` SQL 関数で大量の画像を一括処理できます。Spark が自動的にバッチ化・並列化してくれるので、コードは非常にシンプルです。

```sql
SELECT
  filename,
  ai_query(
    'siglip2-finetuned-serving',
    named_struct(
      'image_base64', image_base64,
      'tag_candidates_json', tag_candidates_json
    )
  ) AS prediction
FROM input_images
```

結果は Delta テーブルに保存し、タグ分布の分析やエンベディングによる類似画像検索に活用できます。

## まとめ

| ステップ | 使用機能 | 所要時間 |
|----------|----------|----------|
| データ準備 | Volume + Delta Table | 約 3 分 |
| トレーニング | AI Runtime (A10 GPU) | 約 8 分 |
| デプロイ | Model Serving (GPU_SMALL) | 約 30 分 |
| バッチ推論 | ai_query() | 画像枚数に依存 |

Databricks のワンプラットフォームなら、データの準備から始まり、サーバーレス GPU でのモデルトレーニング、エンドポイントへのデプロイ、SQL 関数によるバッチ推論まで、一貫したワークフローで完結します。インフラの管理に煩わされることなく、モデルの改善やビジネス価値の創出に集中できるのが最大のメリットです。

AI Runtime の Public Preview により、GPU クラスタを自前で管理する必要がなくなりました。ぜひ試してみてください。

## 参考リンク

- [Databricks AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/)
- [AI Runtime 発表ブログ](https://www.databricks.com/jp/blog/introducing-ai-runtime-scalable-serverless-nvidia-gpus-databricks-training-and-finetuning)
- [SigLIP2 (HuggingFace)](https://huggingface.co/google/siglip2-so400m-patch14-384)
- [STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions)
- [本記事のコード (GitHub)](https://github.com/hiouchiy/ft_mm_emb_model)
