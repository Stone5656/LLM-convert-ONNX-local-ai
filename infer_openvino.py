#!/usr/bin/env python3
"""
infer_openvino.py
-------------------
Minimal LangChainラッパーによるOpenVINOコンパイル済LLM推論スクリプト
• テスト環境: langchain==0.2.3, openvino==2025.2.0
• デバイス: 'GPU' (Intel Arc)
• 失敗時は exit(1)
-------------------
使用例:
  python infer_openvino.py --ir ./ir_int8_distilgpt2 --device CPU

  # トークナイザーを別指定
  python infer_openvino.py --ir ./onnx_distilgpt2 --tokenizer ./onnx_distilgpt2/tokenizer --device GPU
"""

import argparse
from logger_config import logging
import sys
from pathlib import Path
from typing import Any, List, Optional

from transformers import AutoTokenizer
from langchain.schema import Generation, LLMResult

import openvino as ov
import numpy as np
from langchain.llms.base import LLM
from pydantic.v1 import PrivateAttr

class OpenVINOLLM(LLM):
    """LangChain互換のOpenVINOコンパイルモデルラッパー。"""

    model_name: str = "qwen2.5-7b-int4"
    max_tokens: int = 256

    # Pydanticプライベート属性
    _model: ov.CompiledModel = PrivateAttr()
    _tok: Any = PrivateAttr()
    _request: ov.InferRequest = PrivateAttr()

    def __init__(self, compiled_model: ov.CompiledModel, tokenizer: Any, **kwargs):
        super().__init__(**kwargs)
        self._model = compiled_model
        self._tok = tokenizer
        self._request = self._model.create_infer_request()

    @property
    def _llm_type(self) -> str:
        return "openvino_int4"

    def _call(self, prompt: str, **kwargs) -> str:
        # 入力トークン化
        input_ids = self._tok(prompt, return_tensors="np").input_ids
        generated = input_ids

        ids_name  = self._model.inputs[0].get_any_name()
        mask_name = self._model.inputs[1].get_any_name()
        pos_name  = self._model.inputs[2].get_any_name()
        out_name  = self._model.outputs[0].get_any_name()

        logging.info("推論開始: トークン数初期=%d", generated.shape[1])
        for step in range(self.max_tokens):
            attn_mask    = np.ones_like(generated, dtype="int64")
            position_ids = np.arange(generated.shape[1], dtype="int64")[None, :]
    
            results = self._request.infer({
                ids_name:  generated,
                mask_name: attn_mask,
                pos_name:  position_ids,
            })
            logits = results[out_name][:, -1, :]  # (1, V)
            # 温度付きサンプリングに変更
            from scipy.special import softmax
            temperature = 1.0
            probs = softmax(logits[0] / temperature)
            chosen = np.random.choice(len(probs), p=probs)
            next_id = np.array([[chosen]], dtype="int64")
            generated = np.concatenate([generated, next_id], axis=-1)

            if next_id[0, 0] == self._tok.eos_token_id:
                logging.info("<EOS>検出、推論終了"); break

        text = self._tok.decode(generated[0], skip_special_tokens=True)
        logging.info("生成テキスト長=%d", len(text))
        return text

    def generate(self, prompts: List[str], **kwargs) -> LLMResult:
        # LangChain 0.2+ API
        gens = [Generation(text=self._call(p)) for p in prompts]
        return LLMResult(generations=[gens])


def build_llm(
    model_path: Path,
    device: str = "GPU",
    tokenizer_path: Optional[Path] = None,
) -> OpenVINOLLM:
    """
    model_path    : IRディレクトリ or .xml or .onnxファイル
    tokenizer_path: トークナイザーディレクトリを明示する場合
    """
    core = ov.Core()

    # モデルファイル取得
    if model_path.is_dir():
        xmls = list(model_path.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError("モデルXMLがディレクトリ内に見つかりません")
        model_file = xmls[0]
    else:
        model_file = model_path

    logging.info("モデル読み込み: %s", model_file)
    model = core.read_model(str(model_file))
    compiled = core.compile_model(model, device)
    logging.info("モデルコンパイル完了: デバイス=%s", device)

    # トークナイザーパス決定
    if tokenizer_path is None:
        guess = model_path / "tokenizer" if model_path.is_dir() else model_path.parent / "tokenizer"
        if guess.exists(): tokenizer_path = guess
        else:
            raise FileNotFoundError(
                "トークナイザーが見つかりません。--tokenizer オプションで指定してください"
            )
    logging.info("トークナイザー読み込み: %s", tokenizer_path)
    tok = AutoTokenizer.from_pretrained(str(tokenizer_path), trust_remote_code=True)
    return OpenVINOLLM(compiled, tok)


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenVINO LLMで推論実行")
    parser.add_argument("--ir", required=True, type=Path, help="IRファイル/ディレクトリパス")
    parser.add_argument("--device", default="GPU", help="OpenVINOデバイス名 (GPU or CPU)")
    parser.add_argument(
        "--tokenizer", type=Path, default=None,
        help="トークナイザーディレクトリ (bare .onnx時に必須)"
    )
    args = parser.parse_args()

    try:
        llm = build_llm(args.ir, args.device, tokenizer_path=args.tokenizer)
        prompt = "フランスの首都はどこ？"
        logging.info("プロンプト: %s", prompt)
        answer = llm.invoke(prompt)
        print(f"回答: {answer}")
    except Exception as e:
        logging.exception("❌ 推論中にエラーが発生しました: %s", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
