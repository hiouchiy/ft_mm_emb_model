# Databricks notebook source
# MAGIC %md
# MAGIC # 4. バッチ画像推論 (`ai_query`)
# MAGIC
# MAGIC ノートブック 3 でデプロイしたサービングエンドポイントを使い、
# MAGIC Volume 内の MS-COCO 画像を一括でタグ付け＆エンベディング抽出します。
# MAGIC
# MAGIC **機能**:
# MAGIC 1. Volume 内の画像を base64 エンコード → Spark DataFrame
# MAGIC 2. `ai_query()` SQL関数でバッチ推論
# MAGIC 3. 結果を Delta テーブルに保存
# MAGIC 4. タグ分布の分析
# MAGIC 5. エンベディングを使った類似画像検索

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.1 設定

# COMMAND ----------

CATALOG_NAME = "hiroshi"
SCHEMA_NAME = "auto_labeling"

# サービングエンドポイント名
ENDPOINT_NAME = "siglip2-finetuned-serving"

# 推論対象の画像 Volume (トレーニングで使用した MS-COCO 画像を流用)
IMAGE_VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/training_images"

# バッチ推論する画像枚数
N_INFERENCE = 100

# 結果保存先の Delta テーブル
RESULT_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.image_inference_results"

# タグ候補 (お客様のユースケースに合わせてカスタマイズ)
TAG_CANDIDATES = [
    "風景", "自然", "山", "海", "森", "空",
    "都市", "建物", "道路", "夜景",
    "食べ物", "料理", "フルーツ",
    "スポーツ", "サッカー", "野球",
    "動物", "犬", "猫", "鳥",
    "人物", "グループ", "ポートレート",
]

import json
tag_candidates_json = json.dumps(TAG_CANDIDATES, ensure_ascii=False)

print(f"Endpoint:     {ENDPOINT_NAME}")
print(f"Image Volume: {IMAGE_VOLUME_PATH}")
print(f"Inference:    {N_INFERENCE} images")
print(f"Result Table: {RESULT_TABLE}")
print(f"Tags:         {len(TAG_CANDIDATES)} candidates")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.2 エンドポイントの状態確認

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
ep = w.serving_endpoints.get(ENDPOINT_NAME)
print(f"Endpoint: {ENDPOINT_NAME}")
print(f"State:    {ep.state.ready}")

if str(ep.state.ready) != "EndpointStateReady.READY":
    print("\nエンドポイントが READY ではありません。")
    print("ノートブック 3 でデプロイが完了するのを待ってから再実行してください。")
    dbutils.notebook.exit("Endpoint not ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.3 画像の読み込みと base64 エンコード
# MAGIC
# MAGIC Volume 内の MS-COCO 画像から 100 枚を選択し、
# MAGIC base64 エンコードして Spark DataFrame に変換します。

# COMMAND ----------

import os
import base64
from pyspark.sql.functions import col, lit

# Volume 内の画像ファイル一覧を取得 (先頭 N_INFERENCE 枚)
image_files = sorted([
    f for f in os.listdir(IMAGE_VOLUME_PATH)
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])[:N_INFERENCE]
print(f"Selected {len(image_files)} images from {IMAGE_VOLUME_PATH}")

# base64 エンコードして DataFrame 化
rows = []
for fname in image_files:
    fpath = os.path.join(IMAGE_VOLUME_PATH, fname)
    with open(fpath, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    rows.append((fname, b64))

df_images = spark.createDataFrame(rows, ["filename", "image_base64"])
df_images = df_images.withColumn("tag_candidates_json", lit(tag_candidates_json))

print(f"DataFrame: {df_images.count()} rows")
df_images.select("filename").show(5, truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.4 `ai_query()` でバッチ推論
# MAGIC
# MAGIC `ai_query()` は Databricks SQL のビルトイン関数で、
# MAGIC サービングエンドポイントに対してバッチリクエストを送信します。
# MAGIC Spark が自動的にバッチ化・並列化を行います。

# COMMAND ----------

df_images.createOrReplaceTempView("input_images")

# COMMAND ----------

# MAGIC %sql
# MAGIC -- ai_query でバッチ推論を実行
# MAGIC CREATE OR REPLACE TEMP VIEW inference_results AS
# MAGIC SELECT
# MAGIC   filename,
# MAGIC   ai_query(
# MAGIC     'siglip2-finetuned-serving',
# MAGIC     named_struct(
# MAGIC       'image_base64', image_base64,
# MAGIC       'tag_candidates_json', tag_candidates_json
# MAGIC     )
# MAGIC   ) AS prediction
# MAGIC FROM input_images

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 推論結果のプレビュー
# MAGIC SELECT
# MAGIC   filename,
# MAGIC   prediction.tags_json,
# MAGIC   prediction.metrics_json,
# MAGIC   LEFT(prediction.embedding, 80) AS embedding_preview
# MAGIC FROM inference_results
# MAGIC LIMIT 10

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.5 結果を Delta テーブルに保存

# COMMAND ----------

# Python から結果を保存 (テーブル名を変数で指定するため)
result_df = spark.sql("""
    SELECT
        filename,
        get_json_object(prediction.tags_json, '$.tags') AS tags,
        prediction.embedding AS embedding_json,
        prediction.metrics_json AS metrics_json
    FROM inference_results
""")

result_df.write.mode("overwrite").saveAsTable(RESULT_TABLE)
print(f"Saved {spark.table(RESULT_TABLE).count()} rows to {RESULT_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.6 タグ分布の分析

# COMMAND ----------

# MAGIC %sql
# MAGIC -- タグごとの出現回数
# MAGIC SELECT tag.tag, tag.score, COUNT(*) as count
# MAGIC FROM (
# MAGIC   SELECT explode(from_json(tags, 'ARRAY<STRUCT<tag:STRING,score:DOUBLE>>')) as tag
# MAGIC   FROM hiroshi.auto_labeling.image_inference_results
# MAGIC )
# MAGIC GROUP BY tag.tag, tag.score
# MAGIC ORDER BY count DESC
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %sql
# MAGIC -- 画像ごとのタグ数の分布
# MAGIC SELECT
# MAGIC   num_tags,
# MAGIC   COUNT(*) as image_count
# MAGIC FROM (
# MAGIC   SELECT filename, size(from_json(tags, 'ARRAY<STRUCT<tag:STRING,score:DOUBLE>>')) as num_tags
# MAGIC   FROM hiroshi.auto_labeling.image_inference_results
# MAGIC )
# MAGIC GROUP BY num_tags
# MAGIC ORDER BY num_tags

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4.7 エンベディングを使った類似画像検索
# MAGIC
# MAGIC エンベディングベクトルのコサイン類似度で、類似画像を検索できます。
# MAGIC 実運用では Databricks Vector Search を使うのが推奨です。

# COMMAND ----------

import numpy as np

# Delta テーブルから結果を読み込み
results_pdf = spark.table(RESULT_TABLE).toPandas()
embeddings = np.array([json.loads(e) for e in results_pdf["embedding_json"]])
filenames = results_pdf["filename"].tolist()

print(f"Embeddings: {embeddings.shape} ({embeddings.shape[1]}次元)")

# クエリ画像を選択 (先頭の画像)
query_idx = 0
query_emb = embeddings[query_idx]

# コサイン類似度で検索
norms = np.linalg.norm(embeddings, axis=1)
similarities = embeddings @ query_emb / (norms * np.linalg.norm(query_emb))

# Top 10 類似画像
top10 = np.argsort(similarities)[::-1][1:11]  # 自分自身を除く

print(f"\nQuery: {filenames[query_idx]}")
print(f"\nTop 10 Similar Images:")
for rank, idx in enumerate(top10):
    print(f"  {rank+1:2d}. {filenames[idx]:40s} (similarity: {similarities[idx]:.4f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC ### 結果まとめ
# MAGIC - **バッチ推論**: `ai_query()` で Volume 内の MS-COCO 画像 100 枚を一括処理
# MAGIC - **結果テーブル**: `hiroshi.auto_labeling.image_inference_results`
# MAGIC   - `filename`: 画像ファイル名
# MAGIC   - `tags`: 付与されたタグ (JSON配列、スコア付き)
# MAGIC   - `embedding_json`: 画像エンベディングベクトル (1152次元)
# MAGIC   - `metrics_json`: 推論メトリクス (推論時間、GPU使用状況)
# MAGIC - **類似画像検索**: エンベディングのコサイン類似度で検索可能
# MAGIC
# MAGIC ### ノートブック一覧
# MAGIC | # | ノートブック | 内容 |
# MAGIC |---|-----------|------|
# MAGIC | 1 | `1_prepare_training_data` | STAIR Captions から 1,000 枚のデータ準備 |
# MAGIC | 2 | `2_train_siglip2_from_delta` | SigLIP2 ファインチューニング (AI Runtime) |
# MAGIC | 3 | `3_deploy_model` | サービングエンドポイント デプロイ |
# MAGIC | 4 | `4_batch_inference` | バッチ推論 (`ai_query`) + 類似検索 |
