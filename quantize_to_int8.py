#!/usr/bin/env python3
"""
quantize_to_int8.py
-------------------
1. ONNX → OpenVINO IR
2. INT8量子化 (Neural Network Compression Framework via NNCF)
3. `{model}_int8.xml/.bin` を --output に保存
• 失敗時は exit(1)
-------------------
使用例:
  python quantize_to_int8.py \
    --onnx ./onnx_distilgpt2/model.onnx \
    --output ./ir_int8_distilgpt2
"""

import argparse
import sys
from pathlib import Path

import openvino as ov
from nncf import compress_weights, CompressWeightsMode  # NNCF経由で重み量子化
from logger_config import logging

core = ov.Core()
core.set_property({"LOG_LEVEL": "WARNING"})  # ログ抑制

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ONNXモデルをIR変換し、NNCFでINT8量子化"
    )
    parser.add_argument("--onnx", required=True, type=Path, help="Path to model.onnx")
    parser.add_argument("--output", required=True, type=Path, help="Dir for IR+INT8")
    args = parser.parse_args()

    try:
        args.output.mkdir(parents=True, exist_ok=True)
        logging.info("ONNX→IR変換中…")
        model_ir = ov.convert_model(str(args.onnx))

        logging.info("NNCFでINT8量子化実行…")
        model_int8 = compress_weights(
            model_ir,
            # CompressWeightsModeは['INT8_SYM', 'INT8_ASYM', 'INT4_SYM', 'INT4_ASYM', 'NF4', 'INT8', 'E2M1']のどれかにして
            mode=CompressWeightsMode.INT8_SYM
        )

        out_xml = args.output / "model_int8.xml"
        ov.save_model(model_int8, str(out_xml))
        logging.info("✅ 量子化モデル保存: %s", out_xml)
    except Exception as ex:
        logging.exception("❌ 量子化失敗: %s", ex)
        sys.exit(1)

if __name__ == "__main__":
    main()
