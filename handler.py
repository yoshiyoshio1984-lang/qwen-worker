# Qwen-Image-Edit-2509 画像編集ワーカー (RunPod Serverless)
#
# 使い方: 画像 + 指示文 (例: "この女性の服を脱がせて、全裸にして") を送ると、
# 元の人物の顔・構図を保ったまま指示通りに描き直した画像を返す。
# FLUX.1-Kontext と同じ指示ベースの編集モデルだが、こちらは
# Apache 2.0 ライセンスで、Hugging Face のライセンス同意も
# HF_TOKEN も不要。20B パラメータで、顔の一貫性が特に高いと
# されている (2509 はマルチ画像編集・顔維持が改善された版)。
#
# 入力 (job["input"]):
#   prompt: str (必須, 指示文。アプリ側が日本語を英語に変換して送る)
#   image_base64: str (必須, 編集元の画像。PNG/JPEG の base64)
#   negative_prompt: str (任意。Qwen では true_cfg_scale の既定が
#     4.0 で、ネガティブがあるときだけ 4.0 を渡す。無いときは 1.0)
#   num_inference_steps: int (既定 50, Qwen-Image-Edit の既定値。
#     20B モデルで 50 ステップのため、1枚あたり 1〜2 分かかる見込み)
#   guidance_scale: float (既定 1.0, Qwen の既定値。Flux と違い
#     1.0 が標準。アプリ側が裸プロンプトで 6.0 を送ることがあるが、
#     Qwen では高すぎるため 4.0 で頭打ちにする)
#   seed: int (任意。-1 / 省略ならランダム)
#   width / height: int (任意。省略なら元画像のサイズを 16 の倍数に丸める)
#
# 出力: {"image_base64", "width", "height", "seed", "model", "mode"}
#
# 注意: モデル全体のダウンロードは約 58GB (transformer 40.9GB +
# テキストエンコーダ Qwen2.5-VL-7B 約15GB + VAE)。コンテナ
# ディスクは 50GB では足りないため、エンドポイント作成時に
# 100GB にすること。GPU は 48GB 帯で、enable_model_cpu_offload
# によりテキストエンコーダ等を CPU に逃がして動かす。
# transformers は 4.49.0 が必要。4.46.3 には Qwen2.5-VL の
# 実装が無く、テキストエンコーダが読めない (wheel で確認済み)。
# qwen-vl-utils は Qwen2.5-VL の画像処理に必要。

import base64
import io
import os
import random

import runpod
import torch
from diffusers import QwenImageEditPipeline

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen-Image-Edit-2509")

print(f"Loading {MODEL_ID} ...", flush=True)
pipe = QwenImageEditPipeline.from_pretrained(
    MODEL_ID, torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()
print("Model loaded.", flush=True)


def _round16(n):
    return max(16, (int(n) + 8) // 16 * 16)


def handler(job):
    inp = job.get("input", {}) or {}
    prompt = inp.get("prompt", "")
    if not prompt:
        return {"error": "prompt is required"}
    image_b64 = inp.get("image_base64")
    if not image_b64:
        return {"error": "image_base64 is required"}

    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")

    steps = int(inp.get("num_inference_steps", 50))
    guidance = float(inp.get("guidance_scale", 1.0))
    # Qwen の guidance は 1.0 が標準。高すぎると崩れるため頭打ち。
    guidance = min(4.0, max(0.0, guidance))
    # ネガティブプロンプト (任意): アプリ側が英語化と一緒に送る。
    # 例: "clothed, wearing clothes, dressed, underwear, bikini"
    # Qwen では true_cfg_scale の既定が 4.0。ネガティブがあるとき
    # だけ 4.0 を渡し、無いときは 1.0 にして余計な計算を省く。
    negative_prompt = inp.get("negative_prompt", "") or ""
    true_cfg_scale = 4.0 if negative_prompt else 1.0
    seed = inp.get("seed", -1)
    if seed is None or int(seed) < 0:
        seed = random.randint(0, 2**32 - 1)
    seed = int(seed)

    generator = torch.Generator(device="cpu").manual_seed(seed)

    width = _round16(inp.get("width") or img.width)
    height = _round16(inp.get("height") or img.height)

    image = pipe(
        image=img,
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        negative_prompt=negative_prompt,
        true_cfg_scale=true_cfg_scale,
        generator=generator,
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return {
        "image_base64": base64.b64encode(buf.getvalue()).decode(),
        "width": image.width,
        "height": image.height,
        "seed": seed,
        "model": MODEL_ID,
        "mode": "qwen-edit",
    }


runpod.serverless.start({"handler": handler})
