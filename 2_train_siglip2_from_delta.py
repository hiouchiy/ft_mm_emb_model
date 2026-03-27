# Databricks notebook source
# MAGIC %md
# MAGIC # 2. SigLIP2 ファインチューニング (Delta テーブルから)
# MAGIC
# MAGIC `1_prepare_training_data.py` で作成した Delta テーブルからデータを読み込み、
# MAGIC SigLIP2 のマルチモーダルエンベディングモデルをファインチューニングします。
# MAGIC
# MAGIC **環境要件**: Databricks GPU クラスタ (A10 x1 推奨)
# MAGIC
# MAGIC **入力**: `hiroshi.auto_labeling.training_image_text_pairs` (Delta テーブル)
# MAGIC **出力**: Unity Catalog 登録済みモデル + MLflow 実験ログ

# COMMAND ----------

# MAGIC %pip install transformers==4.51.3 pynvml

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

# huggingface_hub の snapshot_download でモデルをローカルにダウンロード
# (AutoProcessor.from_pretrained の内部ダウンロードが失敗する環境への対策)
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

from huggingface_hub import snapshot_download

HF_MODEL_NAME = "google/siglip2-so400m-patch14-384"
LOCAL_MODEL_DIR = f"/tmp/{HF_MODEL_NAME.replace('/', '_')}"

if not os.path.exists(os.path.join(LOCAL_MODEL_DIR, "config.json")):
    print(f"Downloading {HF_MODEL_NAME} to {LOCAL_MODEL_DIR}...")
    snapshot_download(
        repo_id=HF_MODEL_NAME,
        local_dir=LOCAL_MODEL_DIR,
    )
    print("Done.")
else:
    print(f"Model already cached at {LOCAL_MODEL_DIR}")

print(f"Files: {os.listdir(LOCAL_MODEL_DIR)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 設定

# COMMAND ----------

# --- モデル設定 ---
# google/siglip2-base-patch16-224   (768次元, 224px, ~400M params, A10 1枚で余裕)
# google/siglip2-so400m-patch14-384 (1152次元, 384px, ~1.1B params, A10 1枚でfp32+autocast)
HF_MODEL_NAME = "google/siglip2-so400m-patch14-384"

# --- データ設定 ---
CATALOG_NAME = "hiroshi"
SCHEMA_NAME = "auto_labeling"
TRAINING_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.training_image_text_pairs"

# --- UC モデル登録先 ---
UC_MODEL_NAME = "siglip2_finetuned_embedding"
FULL_MODEL_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.{UC_MODEL_NAME}"

# --- ハイパーパラメータ ---
NUM_EPOCHS = 5
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
TRAINABLE_LAYERS = ["text_model.head", "vision_model.head"]

# --- サービング時のデフォルトタグ候補 ---
DEFAULT_TAG_CANDIDATES = [
    "風景", "自然", "山", "海", "森", "空",
    "都市", "建物", "道路", "夜景",
    "食べ物", "料理", "フルーツ",
    "スポーツ", "サッカー", "野球",
    "動物", "犬", "猫", "鳥",
    "人物", "グループ", "ポートレート",
]

print(f"Model:     {HF_MODEL_NAME}")
print(f"Data:      {TRAINING_TABLE}")
print(f"UC Model:  {FULL_MODEL_NAME}")
print(f"Epochs:    {NUM_EPOCHS}, BS: {BATCH_SIZE}, LR: {LEARNING_RATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 環境セットアップ

# COMMAND ----------

import os
for key in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
    os.environ.pop(key, None)

import torch
import mlflow
from transformers import AutoModel, AutoProcessor
from PIL import Image
from typing import List
import io

mlflow.set_experiment(f"/Users/hiroshi.ouchiyama@databricks.com/siglip2-ft-{SCHEMA_NAME}")
mlflow.autolog(disable=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch: {torch.__version__}, Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.3 Delta テーブルからデータ読み込み

# COMMAND ----------

# Delta テーブルから Pandas に変換
df_train = spark.table(TRAINING_TABLE).filter("split = 'train'").toPandas()
df_valid = spark.table(TRAINING_TABLE).filter("split = 'valid'").toPandas()
df_test  = spark.table(TRAINING_TABLE).filter("split = 'test'").toPandas()

print(f"Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}")
print(f"\nColumns: {list(df_train.columns)}")
print(f"Categories: {sorted(df_train['category'].unique())}")
df_train.head(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.4 画像の読み込み
# MAGIC
# MAGIC Delta テーブルの `image_path` (Volume パス) から画像を PIL Image として読み込みます。

# COMMAND ----------

def load_images_from_paths(image_paths: List[str]) -> List[Image.Image]:
    """Volume パスから PIL Image を読み込む"""
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            images.append(img)
        except Exception as e:
            print(f"Failed to load {path}: {e}")
            images.append(Image.new("RGB", (384, 384), (0, 0, 0)))
    return images

# 画像をプリロード (小規模データセットの場合)
print("Loading images...")
train_images = load_images_from_paths(df_train["image_path"].tolist())
valid_images = load_images_from_paths(df_valid["image_path"].tolist())
test_images  = load_images_from_paths(df_test["image_path"].tolist())

train_texts = df_train["text_positive"].tolist()
valid_texts = df_valid["text_positive"].tolist()
test_texts  = df_test["text_positive"].tolist()

# negative テキスト (評価用)
valid_neg_texts = df_valid["text_negative"].tolist()
test_neg_texts  = df_test["text_negative"].tolist()

print(f"Images loaded: train={len(train_images)}, valid={len(valid_images)}, test={len(test_images)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.5 モデルの読み込みとパラメータフリーズ

# COMMAND ----------

print(f"Loading model: {HF_MODEL_NAME}")
processor = AutoProcessor.from_pretrained(LOCAL_MODEL_DIR)
model = AutoModel.from_pretrained(LOCAL_MODEL_DIR).to(device)

# Projection / Head 層のみ学習可能にする
for name, param in model.named_parameters():
    param.requires_grad = False
    if any(layer in name for layer in TRAINABLE_LAYERS):
        param.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total: {total_params:,}, Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.6 評価関数の定義

# COMMAND ----------

@torch.no_grad()
def encode_images(images: List, batch_size: int = 4) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(images), batch_size):
        with torch.autocast("cuda", dtype=torch.float16):
            inp = processor.image_processor(images[i:i+batch_size], return_tensors="pt")
            embs = model.get_image_features(pixel_values=inp["pixel_values"].to(device))
        all_embs.append(embs / embs.norm(dim=-1, keepdim=True))
    return torch.cat(all_embs, dim=0)

@torch.no_grad()
def encode_texts(texts: List[str], batch_size: int = 32) -> torch.Tensor:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        with torch.autocast("cuda", dtype=torch.float16):
            inp = processor.tokenizer(texts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
            embs = model.get_text_features(input_ids=inp["input_ids"].to(device))
        all_embs.append(embs / embs.norm(dim=-1, keepdim=True))
    return torch.cat(all_embs, dim=0)


def evaluate(split_name, images, pos_texts, neg_texts=None):
    """Triplet Accuracy と Recall@1 を計算"""
    model.eval()
    img_embs = encode_images(images)
    pos_embs = encode_texts(pos_texts)

    # Recall@1
    cos = torch.nn.functional.cosine_similarity(img_embs.unsqueeze(1), pos_embs.unsqueeze(0), dim=2)
    _, top_idx = torch.topk(cos, k=1, dim=1)
    recall_1 = sum(i in top_idx[i].tolist() for i in range(len(images))) / len(images)

    # Triplet Accuracy (negative テキストがある場合)
    triplet_acc = None
    if neg_texts:
        neg_embs = encode_texts(neg_texts)
        pos_sim = torch.nn.functional.cosine_similarity(img_embs, pos_embs)
        neg_sim = torch.nn.functional.cosine_similarity(img_embs, neg_embs)
        triplet_acc = (pos_sim > neg_sim).float().mean().item()

    return {"recall_at_1": recall_1, "triplet_accuracy": triplet_acc}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.7 ベースラインの評価

# COMMAND ----------

print("=== Baseline Evaluation (Before Fine-tuning) ===")
baseline = {}
for name, imgs, pos, neg in [
    ("train", train_images, train_texts, None),
    ("valid", valid_images, valid_texts, valid_neg_texts),
    ("test",  test_images,  test_texts,  test_neg_texts),
]:
    metrics = evaluate(name, imgs, pos, neg)
    baseline[name] = metrics
    print(f"  {name}: Recall@1={metrics['recall_at_1']:.4f}" +
          (f", Triplet Acc={metrics['triplet_accuracy']:.4f}" if metrics['triplet_accuracy'] is not None else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.8 ファインチューニング
# MAGIC
# MAGIC **損失関数**: SigLIP sigmoid contrastive loss
# MAGIC - 各画像-テキストペアが独立した二値分類として扱われる
# MAGIC - In-batch negatives: 同じバッチ内の他のテキストが負例になる
# MAGIC
# MAGIC **Mixed Precision**: モデルは fp32 で保持、forward は autocast fp16

# COMMAND ----------

def siglip_contrastive_loss(image_embs, text_embs):
    """SigLIP sigmoid contrastive loss (fp32で計算)"""
    ie = image_embs.float()
    te = text_embs.float()
    t = model.logit_scale.float().exp()
    b = model.logit_bias.float()
    logits = (ie @ te.T) * t + b
    n = logits.size(0)
    labels = 2 * torch.eye(n, device=logits.device) - 1
    return -torch.nn.functional.logsigmoid(labels * logits).mean()


optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
)
n_train = len(train_images)

print(f"=== Training ===")
print(f"  Model: {HF_MODEL_NAME}")
print(f"  Epochs: {NUM_EPOCHS}, Batch size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
print(f"  Trainable layers: {TRAINABLE_LAYERS}")
print(f"  Train samples: {n_train}")
print(f"  Loss: SigLIP sigmoid contrastive (fp32)")

training_log = []
best_valid_recall = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches = 0
    indices = torch.randperm(n_train).tolist()

    for start in range(0, n_train, BATCH_SIZE):
        batch_idx = indices[start:start + BATCH_SIZE]
        batch_images = [train_images[i] for i in batch_idx]
        batch_texts = [train_texts[i] for i in batch_idx]

        # Forward (autocast fp16)
        img_inp = processor.image_processor(batch_images, return_tensors="pt")
        with torch.autocast("cuda", dtype=torch.float16):
            image_embs = model.get_image_features(pixel_values=img_inp["pixel_values"].to(device))
        image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

        txt_inp = processor.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        with torch.autocast("cuda", dtype=torch.float16):
            text_embs = model.get_text_features(input_ids=txt_inp["input_ids"].to(device))
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        # Loss (fp32) + Backward
        loss = siglip_contrastive_loss(image_embs, text_embs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches

    # エポック毎の評価
    valid_metrics = evaluate("valid", valid_images, valid_texts, valid_neg_texts)
    log_entry = {
        "epoch": epoch + 1,
        "loss": avg_loss,
        **{f"valid_{k}": v for k, v in valid_metrics.items() if v is not None},
    }
    training_log.append(log_entry)

    # ベストモデルの追跡
    if valid_metrics["recall_at_1"] > best_valid_recall:
        best_valid_recall = valid_metrics["recall_at_1"]
        best_epoch = epoch + 1

    ta_str = f", Triplet={valid_metrics['triplet_accuracy']:.4f}" if valid_metrics['triplet_accuracy'] else ""
    print(f"  Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f} | Valid Recall@1: {valid_metrics['recall_at_1']:.4f}{ta_str}")

print(f"\nBest valid Recall@1: {best_valid_recall:.4f} (epoch {best_epoch})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.9 最終評価

# COMMAND ----------

print("=== Final Evaluation (After Fine-tuning) ===")
final_results = {}
for name, imgs, pos, neg in [
    ("train", train_images, train_texts, None),
    ("valid", valid_images, valid_texts, valid_neg_texts),
    ("test",  test_images,  test_texts,  test_neg_texts),
]:
    metrics = evaluate(name, imgs, pos, neg)
    final_results[name] = metrics
    print(f"  {name}: Recall@1={metrics['recall_at_1']:.4f}" +
          (f", Triplet Acc={metrics['triplet_accuracy']:.4f}" if metrics['triplet_accuracy'] is not None else ""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.10 推論テスト

# COMMAND ----------

model.eval()
# テストセットの最初のサンプルで推論
test_img = test_images[0]
test_pos = test_texts[0]
test_neg = test_neg_texts[0] if test_neg_texts else "ランダムなテキスト"

img_emb = encode_images([test_img])
txt_embs = encode_texts([test_pos, test_neg])

logit_scale = model.logit_scale.float().exp()
logit_bias = model.logit_bias.float()
logits = (logit_scale * img_emb.float() @ txt_embs.float().T + logit_bias)[0]
scores = torch.sigmoid(logits).cpu().tolist()

print(f"=== Inference Test ===")
print(f"Image: {df_test.iloc[0]['image_id']} ({df_test.iloc[0]['category']})")
print(f"Positive: {test_pos} → score: {scores[0]:.4f}")
print(f"Negative: {test_neg} → score: {scores[1]:.4f}")
print(f"Correct ranking: {scores[0] > scores[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.11 MLflow 記録 & Unity Catalog 登録

# COMMAND ----------

import json
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

# モデル保存
MODEL_SAVE_DIR = f"/tmp/siglip2_finetuned_{SCHEMA_NAME}"
print(f"Saving model to {MODEL_SAVE_DIR}...")
processor.save_pretrained(MODEL_SAVE_DIR)
model.save_pretrained(MODEL_SAVE_DIR)


# PyFunc ラッパー (サービング用)
# SigLIP 2 はテキスト入力に "This is a photo of {label}." テンプレートが必須。
# テンプレートなしで生のラベルを渡すと正しいスコアが出ない。
class SigLIP2FinetunedModel(mlflow.pyfunc.PythonModel):
    def __init__(self, tag_candidates, tag_score_threshold=0.30):
        self.tag_candidates = tag_candidates
        self.tag_score_threshold = tag_score_threshold

    def __getstate__(self):
        """MLflow 3.x の pickle 対策: load_context で生成されたオブジェクトを除外"""
        state = self.__dict__.copy()
        for key in ['model', 'processor', 'device', '_nvml_handle', '_nvml_available']:
            state.pop(key, None)
        return state

    def load_context(self, context):
        import torch
        from transformers import AutoModel, AutoProcessor
        d = context.artifacts["model_weights"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(d)
        self.model = AutoModel.from_pretrained(d).to(self.device)
        self.model.eval()

        # GPU メトリクス取得の初期化
        self._nvml_available = False
        if self.device == "cuda":
            try:
                import pynvml
                pynvml.nvmlInit()
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self._nvml_available = True
            except Exception:
                pass

        print(f"SigLIP 2 loaded from artifacts on {self.device} (nvml={self._nvml_available})")

    def _get_gpu_metrics(self):
        """GPU メトリクスを取得。取得できない項目は None。"""
        import torch
        metrics = {
            "gpu_memory_allocated_mb": None,
            "gpu_memory_peak_mb": None,
            "gpu_utilization_pct": None,
        }
        if self.device != "cuda":
            return metrics
        metrics["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated() / 1024 / 1024, 1)
        metrics["gpu_memory_peak_mb"] = round(torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        if self._nvml_available:
            try:
                import pynvml
                util = pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                metrics["gpu_utilization_pct"] = util.gpu
            except Exception:
                pass
        return metrics

    def _analyze(self, image_bytes, tag_candidates, threshold):
        """全タグのスコアを計算し、閾値以上のタグをスコア付きで返す。
        SigLIP 2 は "This is a photo of {label}." テンプレートで学習されているため、
        テキスト入力時にテンプレートを適用する必要がある。
        """
        import torch
        from PIL import Image
        import io

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # SigLIP 2 はテキストテンプレートが必須（pipeline 内部と同じ処理）
        templates = [f"This is a photo of {tag}." for tag in tag_candidates]

        inputs = self.processor(
            text=templates, images=image, return_tensors="pt",
            padding="max_length", max_length=64, truncation=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # model.forward で logits を取得（logit_scale + logit_bias が自動適用される）
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0]

            # SigLIP は sigmoid を使用（各タグ独立の確率）
            scores = torch.sigmoid(logits).cpu().tolist()

        embedding = self.model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)

        # 全タグ＋スコアのペアをスコア降順でソート
        tag_scores = sorted(
            [{"tag": tag_candidates[i], "score": round(scores[i], 4)} for i in range(len(tag_candidates))],
            key=lambda x: x["score"], reverse=True,
        )

        # 閾値以上のタグを選択（なければ上位3件）
        selected = [ts for ts in tag_scores if ts["score"] >= threshold]
        if not selected:
            selected = tag_scores[:3]

        embedding = embedding[0].cpu().tolist()
        return selected, tag_scores, embedding

    def predict(self, context, model_input):
        import base64
        import time

        results = []
        for _, row in model_input.iterrows():
            image_bytes = base64.b64decode(row["image_base64"])

            # タグ候補（リクエストで上書き可能）
            if "tag_candidates_json" in row and pd.notna(row.get("tag_candidates_json")):
                tag_candidates = json.loads(row["tag_candidates_json"])
            else:
                tag_candidates = self.tag_candidates

            # 閾値（リクエストで上書き可能）
            if "tag_score_threshold" in row and pd.notna(row.get("tag_score_threshold")):
                threshold = float(row["tag_score_threshold"])
            else:
                threshold = self.tag_score_threshold

            # GPU メモリカウンタリセット
            import torch
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()

            # 推論実行 + 時間計測
            start_time = time.time()
            selected, all_scores, embedding = self._analyze(image_bytes, tag_candidates, threshold)
            inference_time_ms = round((time.time() - start_time) * 1000, 1)

            # GPU メトリクス取得
            gpu_metrics = self._get_gpu_metrics()
            metrics = {
                "inference_time_ms": inference_time_ms,
                **gpu_metrics,
            }

            results.append({
                "tags_json": json.dumps({"tags": selected, "all_scores": all_scores}, ensure_ascii=False),
                "embedding": json.dumps(embedding),
                "metrics_json": json.dumps(metrics),
            })
        return pd.DataFrame(results)


# MLflow ログ
mlflow.set_registry_uri("databricks-uc")
signature = ModelSignature(
    inputs=Schema([
        ColSpec("string", "image_base64"),
        ColSpec("string", "tag_candidates_json", required=False),
        ColSpec("string", "tag_score_threshold", required=False),
    ]),
    outputs=Schema([
        ColSpec("string", "tags_json"),
        ColSpec("string", "embedding"),
        ColSpec("string", "metrics_json"),
    ]),
)

with mlflow.start_run(run_name=f"siglip2-ft-{SCHEMA_NAME}") as run:
    # ベースラインメトリクス (ファインチューニング前)
    for split, metrics in baseline.items():
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(f"baseline_{split}_{k}", v)

    # ファインチューニング後メトリクス
    for split, metrics in final_results.items():
        for k, v in metrics.items():
            if v is not None:
                mlflow.log_metric(f"{split}_{k}", v)
    for entry in training_log:
        mlflow.log_metric("train_loss", entry["loss"], step=entry["epoch"])

    # パラメータ
    mlflow.log_params({
        "base_model": HF_MODEL_NAME,
        "training_table": TRAINING_TABLE,
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "trainable_layers": str(TRAINABLE_LAYERS),
        "loss_function": "SigLIP_sigmoid_contrastive",
        "train_samples": len(train_images),
        "valid_samples": len(valid_images),
        "test_samples": len(test_images),
    })

    # PyFunc モデル登録
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=SigLIP2FinetunedModel(
            tag_candidates=DEFAULT_TAG_CANDIDATES,
        ),
        artifacts={"model_weights": MODEL_SAVE_DIR},
        signature=signature,
        pip_requirements=[
            "transformers==4.51.3",
            "sentencepiece",
            "protobuf",
            "torch>=2.0.0",
            "torchvision",
            "Pillow",
            "numpy",
            "pandas",
            "pynvml",
        ],
        registered_model_name=FULL_MODEL_NAME,
    )

    print(f"\nMLflow Run ID: {run.info.run_id}")
    print(f"Model URI: {model_info.model_uri}")
    print(f"Registered: {FULL_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done!
# MAGIC
# MAGIC ### 結果サマリー
# MAGIC - ベースモデル: SigLIP2 (`google/siglip2-so400m-patch14-384`)
# MAGIC - トレーニングデータ: Delta テーブル `hiroshi.auto_labeling.training_image_text_pairs`
# MAGIC - 損失関数: SigLIP sigmoid contrastive loss
# MAGIC - Mixed Precision: fp32 パラメータ + autocast fp16 forward
# MAGIC - モデル登録先: `hiroshi.auto_labeling.siglip2_finetuned_embedding`
# MAGIC
# MAGIC ### 次のステップ
# MAGIC 1. `deploy_siglip2_finetuned.py` でサービングエンドポイントにデプロイ
# MAGIC 2. `batch_inference_ai_query.py` / `batch_inference_ai_query_notebook.py` でバッチ推論
