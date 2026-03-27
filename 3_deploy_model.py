# Databricks notebook source
# MAGIC %md
# MAGIC # 3. モデルデプロイ (サービングエンドポイント)
# MAGIC
# MAGIC `2_train_siglip2_from_delta.py` で Unity Catalog に登録したファインチューニング済み
# MAGIC SigLIP2 モデルを、Model Serving エンドポイントとしてデプロイします。
# MAGIC
# MAGIC **前提**: ノートブック 2 を事前に実行し、UC にモデルが登録済みであること
# MAGIC
# MAGIC **出力**: GPU サービングエンドポイント (リアルタイム推論 + バッチ推論 `ai_query` 対応)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.1 設定

# COMMAND ----------

CATALOG_NAME = "hiroshi"
SCHEMA_NAME = "auto_labeling"
UC_MODEL_NAME = "siglip2_finetuned_embedding"
FULL_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{UC_MODEL_NAME}"

# エンドポイント名 (お客様環境に合わせて変更)
ENDPOINT_NAME = "siglip2-finetuned-serving"

# テスト画像の Volume パス
IMAGE_VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/test_images"

print(f"UC Model : {FULL_MODEL_NAME}")
print(f"Endpoint : {ENDPOINT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.2 登録済みモデルの確認

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{FULL_MODEL_NAME}'")

if not versions:
    raise RuntimeError(
        f"モデル '{FULL_MODEL_NAME}' が見つかりません。\n"
        "先にノートブック 2 (2_train_siglip2_from_delta) を実行してください。"
    )

latest_version = max(int(v.version) for v in versions)
print(f"Model: {FULL_MODEL_NAME}")
print(f"Latest Version: {latest_version}")

# バージョン一覧
for v in sorted(versions, key=lambda x: int(x.version), reverse=True)[:5]:
    print(f"  v{v.version}: status={v.status}, run_id={v.run_id[:12]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.3 サービングエンドポイントの作成 / 更新
# MAGIC
# MAGIC **ワークロードタイプ**: `GPU_SMALL` (T4 1枚相当)
# MAGIC - SigLIP2 推論に GPU が必要
# MAGIC - `scale_to_zero_enabled=True` で未使用時はコスト節約
# MAGIC - 初回起動に最大60分かかる場合あり

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

w = WorkspaceClient()

endpoint_config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name=FULL_MODEL_NAME,
            entity_version=str(latest_version),
            workload_size="Small",
            workload_type=ServingModelWorkloadType.GPU_SMALL,
            scale_to_zero_enabled=True,
        )
    ]
)

try:
    existing = w.serving_endpoints.get(ENDPOINT_NAME)
    print(f"エンドポイント '{ENDPOINT_NAME}' は既に存在します。更新します...")
    w.serving_endpoints.update_config(
        name=ENDPOINT_NAME,
        served_entities=endpoint_config.served_entities,
    )
except Exception:
    print(f"エンドポイント '{ENDPOINT_NAME}' を新規作成します...")
    w.serving_endpoints.create(
        name=ENDPOINT_NAME,
        config=endpoint_config,
    )

print(f"\nデプロイリクエストを送信しました。")
print(f"GPU_SMALL の場合、初回起動に最大60分かかることがあります。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.4 エンドポイントの状態確認
# MAGIC
# MAGIC デプロイ完了まで定期的にポーリングします。
# MAGIC 完了しない場合は Databricks UI の「Serving」ページで確認してください。

# COMMAND ----------

import time

print("エンドポイントの状態を確認中...")
for attempt in range(60):  # 最大30分待機
    time.sleep(30)
    try:
        ep = w.serving_endpoints.get(ENDPOINT_NAME)
        state = ep.state
        print(f"  [{attempt+1}] Ready: {state.ready}, Config: {state.config_update}")
        if str(state.ready) == "EndpointStateReady.READY":
            print("\nエンドポイントが READY になりました！")
            break
    except Exception as e:
        print(f"  [{attempt+1}] Error: {e}")
else:
    print("\nタイムアウト。Databricks UI でエンドポイントの状態を確認してください。")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3.5 単体テスト (エンドポイントが READY になってから実行)

# COMMAND ----------

import json
import base64
import os

# テスト画像を1枚選択
test_files = [f for f in os.listdir(IMAGE_VOLUME_PATH) if f.endswith('.jpg')]
if not test_files:
    print(f"テスト画像が {IMAGE_VOLUME_PATH} に見つかりません。")
    dbutils.notebook.exit("No test images")

test_file = test_files[0]
with open(f"{IMAGE_VOLUME_PATH}/{test_file}", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode("utf-8")

print(f"テスト画像: {test_file} ({len(image_b64)} bytes base64)")

# COMMAND ----------

# テスト1: デフォルトタグ候補で推論
print("--- テスト1: デフォルトタグ候補 ---")
response = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=[{"image_base64": image_b64}],
)

result = response.as_dict()
if "predictions" in result:
    for pred in result["predictions"]:
        tags_data = json.loads(pred["tags_json"])
        emb = json.loads(pred["embedding"])
        metrics = json.loads(pred["metrics_json"])
        print("選択されたタグ:")
        for t in tags_data["tags"]:
            print(f"  {t['tag']}: {t['score']}")
        print(f"\n全タグスコア (上位10):")
        for t in tags_data["all_scores"][:10]:
            print(f"  {t['tag']}: {t['score']}")
        print(f"\n埋め込み次元: {len(emb)}")
        print(f"メトリクス: {json.dumps(metrics, indent=2)}")
else:
    print(json.dumps(result, indent=2, ensure_ascii=False))

# COMMAND ----------

# テスト2: カスタムタグ候補で推論
print("--- テスト2: カスタムタグ候補 ---")
custom_tags = ["風景", "ポートレート", "食べ物", "動物", "建築物", "テクノロジー", "夜景", "スポーツ"]
response2 = w.serving_endpoints.query(
    name=ENDPOINT_NAME,
    dataframe_records=[{
        "image_base64": image_b64,
        "tag_candidates_json": json.dumps(custom_tags, ensure_ascii=False),
    }],
)

result2 = response2.as_dict()
if "predictions" in result2:
    for pred in result2["predictions"]:
        tags_data = json.loads(pred["tags_json"])
        metrics = json.loads(pred["metrics_json"])
        print("選択されたタグ:")
        for t in tags_data["tags"]:
            print(f"  {t['tag']}: {t['score']}")
        print("\n全タグスコア:")
        for t in tags_data["all_scores"]:
            print(f"  {t['tag']}: {t['score']}")
        print(f"\nメトリクス: {json.dumps(metrics, indent=2)}")
else:
    print(json.dumps(result2, indent=2, ensure_ascii=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC エンドポイント `siglip2-finetuned-serving` のデプロイが完了しました。
# MAGIC
# MAGIC ### エンドポイントの使い方
# MAGIC - **リアルタイム推論**: REST API / Python SDK でリクエスト
# MAGIC - **バッチ推論**: `ai_query()` SQL関数で大量画像を一括処理
# MAGIC
# MAGIC ### 次のステップ
# MAGIC → **ノートブック 4** (`4_batch_inference.py`) でバッチ推論を実行
