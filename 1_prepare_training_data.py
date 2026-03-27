# Databricks notebook source
# MAGIC %md
# MAGIC # 1. トレーニングデータの準備
# MAGIC
# MAGIC STAIR Captions（MS-COCO 画像 + 人手作成の日本語キャプション）から
# MAGIC 1,000 枚をサンプリングし、SigLIP2 ファインチューニング用の
# MAGIC Delta テーブルとして Unity Catalog に登録します。
# MAGIC
# MAGIC **データソース**:
# MAGIC - 画像: MS-COCO 2014 (train split)
# MAGIC - キャプション: [STAIR Captions](https://github.com/STAIR-Lab-CIT/STAIR-captions) — MS-COCO 画像に対する人手作成の日本語キャプション (820K キャプション)
# MAGIC
# MAGIC **お客様環境では**: セクション 1.3 を実際のデータ読み込みに置き換えてください。

# COMMAND ----------

# MAGIC %pip install datasets

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.1 設定

# COMMAND ----------

CATALOG_NAME = "hiroshi"
SCHEMA_NAME = "auto_labeling"

# Delta テーブル名
TRAINING_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.training_image_text_pairs"
# 画像を格納する Volume
IMAGE_VOLUME = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/training_images"

# サンプル数
N_SAMPLES = 1000

print(f"Training Table : {TRAINING_TABLE}")
print(f"Image Volume   : {IMAGE_VOLUME}")
print(f"Sample Size    : {N_SAMPLES}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.2 Volume の作成

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG_NAME}.{SCHEMA_NAME}.training_images")
print("Volume ready.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.3 STAIR Captions + MS-COCO 画像のダウンロード
# MAGIC
# MAGIC 1. STAIR Captions の日本語キャプション JSON をダウンロード
# MAGIC 2. MS-COCO 画像を HuggingFace datasets から取得
# MAGIC 3. 1,000 枚をサンプリングして Volume に保存

# COMMAND ----------

import json
import urllib.request
import tarfile
import os
import random

random.seed(42)

# STAIR Captions (tar.gz) をダウンロード＆展開
STAIR_TAR_URL = "https://raw.githubusercontent.com/STAIR-Lab-CIT/STAIR-captions/master/stair_captions_v1.2.tar.gz"
STAIR_TAR_PATH = "/tmp/stair_captions_v1.2.tar.gz"
STAIR_EXTRACT_DIR = "/tmp/stair_captions"
STAIR_TRAIN_JSON = f"{STAIR_EXTRACT_DIR}/stair_captions_v1.2_train.json"

if not os.path.exists(STAIR_TRAIN_JSON):
    print("Downloading STAIR Captions tar.gz...")
    req = urllib.request.Request(STAIR_TAR_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(STAIR_TAR_PATH, "wb") as f:
        f.write(resp.read())
    print(f"Downloaded: {os.path.getsize(STAIR_TAR_PATH) / 1024 / 1024:.1f} MB")

    os.makedirs(STAIR_EXTRACT_DIR, exist_ok=True)
    with tarfile.open(STAIR_TAR_PATH, "r:gz") as tar:
        tar.extractall(STAIR_EXTRACT_DIR)
    print("Extracted.")
else:
    print(f"STAIR Captions cached at {STAIR_EXTRACT_DIR}")

with open(STAIR_TRAIN_JSON) as f:
    stair_data = json.load(f)

# image_id → キャプションリストのマッピング
caption_map = {}
for ann in stair_data["annotations"]:
    img_id = ann["image_id"]
    if img_id not in caption_map:
        caption_map[img_id] = []
    caption_map[img_id].append(ann["caption"])

print(f"STAIR Captions: {len(stair_data['annotations'])} captions for {len(caption_map)} images")

# COMMAND ----------

# MS-COCO image_id → ファイル名のマッピング
image_info = {img["id"]: img for img in stair_data["images"]}

# キャプションが複数ある画像から N_SAMPLES 枚をランダムサンプリング
available_ids = [img_id for img_id, caps in caption_map.items() if len(caps) >= 2]
sampled_ids = random.sample(available_ids, min(N_SAMPLES, len(available_ids)))
print(f"Sampled {len(sampled_ids)} images (from {len(available_ids)} available)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### MS-COCO 画像のダウンロード

# COMMAND ----------

from concurrent.futures import ThreadPoolExecutor, as_completed
import io

COCO_BASE_URL = "http://images.cocodataset.org/train2014"

def download_image(img_id):
    """MS-COCO から画像をダウンロードして Volume に保存"""
    info = image_info[img_id]
    filename = info["file_name"]
    save_path = f"{IMAGE_VOLUME}/{filename}"

    if os.path.exists(save_path):
        return img_id, save_path, True

    try:
        url = f"{COCO_BASE_URL}/{filename}"
        urllib.request.urlretrieve(url, save_path)
        return img_id, save_path, True
    except Exception as e:
        return img_id, str(e), False

print(f"Downloading {len(sampled_ids)} images to {IMAGE_VOLUME}...")

results = {}
failed = 0
with ThreadPoolExecutor(max_workers=16) as executor:
    futures = {executor.submit(download_image, img_id): img_id for img_id in sampled_ids}
    for i, future in enumerate(as_completed(futures)):
        img_id, path_or_err, success = future.result()
        if success:
            results[img_id] = path_or_err
        else:
            failed += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(sampled_ids)} done ({failed} failed)")

print(f"\nDownloaded: {len(results)}, Failed: {failed}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.4 トレーニングデータの作成
# MAGIC
# MAGIC 各画像に対して:
# MAGIC - `text_positive`: STAIR Captions のキャプション（1つ目）
# MAGIC - `text_negative`: 別の画像のキャプションをランダム割り当て
# MAGIC - `category`: MS-COCO のスーパーカテゴリ（利用可能な場合）

# COMMAND ----------

# カテゴリ情報の構築 (COCO annotations が利用可能な場合)
# STAIR Captions JSON には categories がないため、シンプルにキャプション内容で分類
records = []
all_captions = []

for img_id in results:
    captions = caption_map[img_id]
    all_captions.extend(captions)

for img_id, img_path in results.items():
    captions = caption_map[img_id]

    # 正解テキスト: 1つ目のキャプション
    text_positive = captions[0]

    # 不正解テキスト: 別画像のキャプションからランダム選択
    other_captions = [c for c in all_captions if c not in captions]
    text_negative = random.choice(other_captions)

    # split: train/valid/test = 70/15/15
    r = random.random()
    split = "train" if r < 0.7 else ("valid" if r < 0.85 else "test")

    records.append({
        "image_id": str(img_id),
        "image_path": img_path,
        "text_positive": text_positive,
        "text_negative": text_negative,
        "category": "coco",
        "split": split,
    })

print(f"Created {len(records)} records")
print(f"  Train: {sum(1 for r in records if r['split']=='train')}")
print(f"  Valid: {sum(1 for r in records if r['split']=='valid')}")
print(f"  Test:  {sum(1 for r in records if r['split']=='test')}")
print(f"\nSample:")
for r in records[:3]:
    print(f"  {r['image_id']}: {r['text_positive'][:50]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.5 Delta テーブルへの書き込み

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

schema = StructType([
    StructField("image_id", StringType(), False),
    StructField("image_path", StringType(), False),
    StructField("text_positive", StringType(), False),
    StructField("text_negative", StringType(), True),
    StructField("category", StringType(), False),
    StructField("split", StringType(), False),
])

df = spark.createDataFrame(records, schema=schema)
df.write.mode("overwrite").saveAsTable(TRAINING_TABLE)

print(f"Saved to {TRAINING_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1.6 データの確認

# COMMAND ----------

print("=== Dataset Summary ===")
spark.sql(f"""
    SELECT split, COUNT(*) as total
    FROM {TRAINING_TABLE}
    GROUP BY split
    ORDER BY split
""").show()

# サンプル確認
print("=== Sample Records ===")
spark.sql(f"""
    SELECT image_id, split,
           LEFT(text_positive, 40) AS text_positive_preview,
           LEFT(text_negative, 40) AS text_negative_preview
    FROM {TRAINING_TABLE}
    LIMIT 10
""").show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC STAIR Captions から 1,000 枚の画像-テキストペアを
# MAGIC Delta テーブル `hiroshi.auto_labeling.training_image_text_pairs` に登録しました。
# MAGIC
# MAGIC 次のステップ: **`2_train_siglip2_from_delta.py`** でファインチューニングを実行
