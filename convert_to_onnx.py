#!/usr/bin/env python3
"""
convert_to_onnx.py
------------------
• Qwen2.5-7B (sharded safetensors) → ONNX (fp16)
• 失敗時は exit(1)
------------------
使用例:
  python convert_to_onnx.py \
    --model-id Qwen/Qwen2.5-7B \
    --output ./onnx_qwen \
    --fp16

  # FP16を使わずにエクスポートする場合
  python convert_to_onnx.py \
    --model-id distilgpt2 \
    --output ./onnx_distilgpt2

  # ローカルに保存したモデルを指定してエクスポートする場合
  python convert_to_onnx.py \
    --model-id ./local_model_path \
    --output ./onnx_local_model
"""
import argparse
from logger_config import logging
import os
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from optimum.exporters.onnx.__main__ import main_export
import torch
from transformers import AutoTokenizer

# Hugging Face Hubのシンボリックリンクを無効にする設定
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")

def main() -> None:
    # 引数パーサーの設定
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-7B",
        help="HFリポジトリIDまたはローカルパス",  # モデルのID（Hugging FaceリポジトリIDまたはローカルパス）
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="ONNXファイル出力先ディレクトリ"  # 出力先ディレクトリ
    )
    parser.add_argument(
        "--fp16", action="store_true", help="半精度(fp16)でエクスポートします"  # FP16でエクスポートするかどうか
    )
    args = parser.parse_args()

    # FP16オプションの環境チェック
    if args.fp16 and not torch.cuda.is_available():
        # GPUがない場合、FP16オプションを無効にして、FP32でエクスポートに切り替え
        logging.warning("CUDA GPU がないため FP16 を無効化します (FP32 にフォールバック)")
        args.fp16 = False

    try:
        # 出力先ディレクトリの作成（もし存在しない場合）
        args.output.mkdir(parents=True, exist_ok=True)
        logging.info("モデル %s のダウンロードを開始します …", args.model_id)

        # Hugging Face Hub からモデルをダウンロード
        model_path = snapshot_download(
            args.model_id,
            ignore_patterns=["*.md", "*.txt"],  # メタデータやテキストファイルは無視
        )
        logging.info("ダウンロード完了: %s", model_path)

        # ---- コンバーターの引数設定 ----
        main_export(
            model_name_or_path=model_path,  # ダウンロードしたモデルのパス
            output=args.output,  # 出力先ディレクトリ
            task="text-generation",  # タスクの指定（テキスト生成）
            dtype="fp16" if args.fp16 else None,  # FP16を使用する場合は指定
            trust_remote_code=True,  # リモートコードを信頼して実行
            device="cuda" if torch.cuda.is_available() else "cpu",  # GPUまたはCPUを選択
            monolith=True,  # モデルを1つのファイルとしてエクスポート
        )

        # トークナイザーを一緒に保存
        tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok.save_pretrained(args.output / "tokenizer")  # トークナイザーを指定ディレクトリに保存

        # エクスポートが完了したことをログに記録
        logging.info("✅ ONNXエクスポート完了: %s", args.output)

    except Exception as ex:  # 例外処理：エクスポート中にエラーが発生した場合
        logging.exception("❌ ONNXエクスポート中にエラーが発生しました: %s", ex)
        sys.exit(1)  # エラー発生時に終了コード1で終了

if __name__ == "__main__":
    main()  # メイン関数を実行
