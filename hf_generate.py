#!/usr/bin/env python3
"""
hf_generate.py  ― 任意の HF CausalLM を即実行
------------------------------------------------
使用例:
  python hf_generate.py                             # デフォルト (distilgpt2)
  python hf_generate.py --model-id gpt2
  python hf_generate.py --model-id EleutherAI/gpt-neo-125M
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from logger_config import logging  # 共通ロガーを使用


def main() -> None:
    parser = argparse.ArgumentParser(description="任意のHuggingFace CausalLMでテキスト生成")
    parser.add_argument(
        "--model-id",
        default="distilgpt2",
        help="HFリポジトリIDまたはローカルディレクトリ (default: distilgpt2)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="生成トークン数 (default: 512)",
    )
    args = parser.parse_args()

    logging.info(f"モデル '{args.model_id}' をロード中…")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    ).eval()
    logging.info(f"モデルロード完了: {args.model_id}")

    # サンプル入力 (必要に応じて変更)
    requests = [
        "南硫黄島原生自然環境保全地域は、自然",
        "The capybara is a giant cavy rodent",
    ]

    for req in requests:
        logging.info(f"INPUT: {req}")
        inputs = tokenizer(req, return_tensors="pt").to(model.device)
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
        logging.info(f"OUTPUT: {output}")
        logging.info("------------------------------------------------------------")


if __name__ == "__main__":
    main()
