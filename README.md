# HuggingFace AI → ONNX → OpenVINO‑INT8 / INT4 （Windows 10 / 11）

> **このガイドはローカル GPU（Intel Arc / A770 など）で LLM を動かす最短ルート**をまとめたものです。
> 2025‑07 時点のツールチェーン（uv + Transformers + Optimum + OpenVINO 2025.x）に合わせてあります。

---

## 0. 前提 & 注意点

| 懸念点                 | 内容                                                                                                        |
| ------------------- | --------------------------------------------------------------------------------------------------------- |
| **GPU ドライバ**        | Intel Arc 用ドライバは **31.0.101.5590** 以降必須。それ以前のバージョンでは Level‑Zero がクラッシュします。                                |
| **長いパス**            | `MAX_PATH = 260` の制限が残っていると `build\onnx\tokenizer\…` が 260 文字を超えて失敗します。`LongPathsEnabled = 1` を有効化してください。 |
| **uv on Windows**   | `uv venv` で作成される仮想環境は **`\.venv\Scripts\activate.ps1`** でアクティブ化します。                                       |
| **OpenVINO 2025.x** | Model Optimizer は **廃止**。代わりに `openvino.convert_model()` API で ONNX → IR 変換＆量子化を行います。                     |

---

## 1. ツールチェインのセットアップ

```powershell
# 1‑1. uv をグローバルにインストール（初回のみ）
pip install --upgrade uv
# or: winget install AstralSoftware.uv
```

### 1‑2. プロジェクト用仮想環境

```powershell
uv venv                      # .\.venv を作成
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1  # PowerShell
# (cmd.exe の場合) .\.venv\Scripts\activate.bat
uv pip install -r requirements.txt
```

---

## 2. ONNX 変換

```powershell
python convert_to_onnx.py \
    --model-id rinna/japanese-gpt2-small \
    --output ./onnx_rinna
```

---

## 3. IR 生成 + 量子化（INT8 / INT4）

```powershell
python quantize_to_int8.py \
    --onnx   ./onnx_rinna/model.onnx \
    --output ./ir_int8_rinna
```

> **目安**
> ⏱️ 所要時間 : 3 〜 10 分
> 🖥️ メモリ    : 32 GB 以上推奨

量子化モードは `quantize_to_int8.py` の **44 行目**

```python
mode = CompressWeightsMode.INT8_SYM
```

を変更すれば INT4 等に切り替えられます。

---

## 4. LangChain で GPU 推論

```powershell
python infer_openvino.py \
    --ir        ./ir_int8_rinna \
    --tokenizer ./onnx_rinna/tokenizer \
    --device    GPU
```

期待される挙動例：

```
Prompt: 日本の首都はどこ？
Answer: 日本の首都は東京です。
```

> **注意**: `rinna/japanese-gpt2-small` は軽量ですが **精度は高くありません**。実運用ではより大きなモデルを推奨します。

---

## 5. 生成済みファイル構成

```
.
├─ .gitignore
├─ convert_to_onnx.py
├─ infer_openvino.py
├─ logger_config.py
├─ quantize_to_int8.py
├─ README.md
├─ requirements.txt
└─ tests/
```

---

## 6. 参考リンク

* [uv ドキュメント — *Using a virtual environment* (2025‑06 更新)](https://docs.uvtools.io/en/latest/guide/virtual-env.html)
* [Intel® GPU 向け OpenVINO™ Configurations (docs.openvino.ai, 2025‑06‑20)](https://docs.openvino.ai/2025/openvino_docs_GPU_DevGuide_Configurations.html)
* [Windows のパス長制限 (Microsoft Learn, 2024‑07)](https://learn.microsoft.com/windows/win32/fileio/maximum-file-path-limitation?tabs=registry)
* [OpenVINO™ 2025.2 Release Notes — Model Optimizer 廃止 など](https://docs.openvino.ai/2025/openvino_release_notes_2025_2.html)
