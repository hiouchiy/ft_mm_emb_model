# SigLIP2 Fine-tuning Pipeline on Databricks

[日本語](#日本語) | [English](#english)

---

## 日本語

### 概要

Google SigLIP2 をファインチューニングして、日本語での画像タグ付け・エンベディング抽出を行うエンドツーエンドパイプラインです。データ準備からバッチ推論まで、すべて Databricks 上で完結します。

### パイプライン構成

| # | ノートブック | 内容 | コンピュート |
|---|------------|------|:----------:|
| 1 | `1_prepare_training_data.py` | STAIR Captions + MS-COCO 画像のダウンロード・Delta テーブル登録 | Serverless Notebook |
| 2 | `2_train_siglip2_from_delta.py` | SigLIP2 ファインチューニング・MLflow 記録・UC モデル登録 | **AI Runtime (GPU)** |
| 3 | `3_deploy_model.py` | Model Serving エンドポイントへのデプロイ | Serverless Notebook |
| 4 | `4_batch_inference.py` | `ai_query()` によるバッチ推論・類似画像検索 | Serverless Notebook |

### 使用技術

- **モデル**: [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) (1152次元, 384px)
- **データ**: [STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions) (MS-COCO 画像 + 日本語キャプション)
- **トレーニング**: Projection Head のみファインチューニング (SigLIP sigmoid contrastive loss)
- **サービング**: Databricks Model Serving (GPU_SMALL)
- **バッチ推論**: `ai_query()` SQL 関数

### Databricks での動かし方

#### 前提条件

- Databricks ワークスペースへのアクセス
- Unity Catalog が有効
- AI Runtime（Serverless GPU）が有効

#### 手順

1. **ノートブックのインポート**

   このリポジトリの `1_*.py` 〜 `4_*.py` を Databricks ワークスペースにインポートします。

2. **ノートブック 1: データ準備**（Serverless Notebook で実行）

   ノートブックを開き、コンピュートで **Serverless** を選択して実行します。STAIR Captions のダウンロードと Delta テーブルへの登録を行います。

3. **ノートブック 2: トレーニング**（AI Runtime で実行）

   コンピュートで **Serverless GPU** を選択し、アクセラレータに **A10** を指定して実行します。または、Jobs API で以下のように実行できます。

   ```yaml
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

4. **ノートブック 3: デプロイ**（Serverless Notebook で実行）

   Serverless Notebook で実行します。GPU サービングエンドポイントが作成されます。初回起動に最大 60 分かかる場合があります。

5. **ノートブック 4: バッチ推論**（Serverless Notebook で実行）

   エンドポイントが READY になったら、Serverless Notebook で実行します。`ai_query()` で画像を一括処理し、結果を Delta テーブルに保存します。

### カスタマイズ

- `1_prepare_training_data.py` のデータ読み込み部分をお客様のデータに差し替えてください
- タグ候補は各ノートブックの `TAG_CANDIDATES` で変更できます
- カタログ名・スキーマ名は各ノートブック冒頭の `CATALOG_NAME` / `SCHEMA_NAME` で変更してください

---

## English

### Overview

An end-to-end pipeline for fine-tuning Google SigLIP2 for Japanese image tagging and embedding extraction. The entire workflow — from data preparation to batch inference — runs on Databricks.

### Pipeline

| # | Notebook | Description | Compute |
|---|----------|-------------|:-------:|
| 1 | `1_prepare_training_data.py` | Download STAIR Captions + MS-COCO images, register as Delta table | Serverless Notebook |
| 2 | `2_train_siglip2_from_delta.py` | Fine-tune SigLIP2, log to MLflow, register UC model | **AI Runtime (GPU)** |
| 3 | `3_deploy_model.py` | Deploy to Model Serving endpoint | Serverless Notebook |
| 4 | `4_batch_inference.py` | Batch inference with `ai_query()`, similarity search | Serverless Notebook |

### Tech Stack

- **Model**: [google/siglip2-so400m-patch14-384](https://huggingface.co/google/siglip2-so400m-patch14-384) (1152-dim, 384px)
- **Data**: [STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions) (MS-COCO images + Japanese captions)
- **Training**: Projection Head only fine-tuning (SigLIP sigmoid contrastive loss)
- **Serving**: Databricks Model Serving (GPU_SMALL)
- **Batch Inference**: `ai_query()` SQL function

### Running on Databricks

#### Prerequisites

- Access to a Databricks workspace
- Unity Catalog enabled
- AI Runtime (Serverless GPU) enabled

#### Steps

1. **Import notebooks** — Upload `1_*.py` through `4_*.py` to your Databricks workspace.

2. **Notebook 1: Data Preparation** — Run on **Serverless Notebook**. Downloads STAIR Captions and registers a Delta table.

3. **Notebook 2: Training** — Run on **Serverless GPU (AI Runtime)** with A10 accelerator. Can also be submitted via the Jobs API:

   ```yaml
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

4. **Notebook 3: Deploy** — Run on **Serverless Notebook**. Creates a GPU serving endpoint (may take up to 60 minutes for initial startup).

5. **Notebook 4: Batch Inference** — Run on **Serverless Notebook** after the endpoint is READY. Processes images via `ai_query()` and saves results to a Delta table.

### Customization

- Replace the data loading section in `1_prepare_training_data.py` with your own data
- Modify `TAG_CANDIDATES` in each notebook to match your use case
- Update `CATALOG_NAME` / `SCHEMA_NAME` at the top of each notebook

### License

MIT
