import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = ROOT / "models" / "ad-classifier"
MAX_SEQUENCE_LENGTH = 512
ONNX_OPSET_VERSION = 18


class OnnxClassifierWrapper(nn.Module):
    def __init__(self, model: DistilBertForSequenceClassification) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def validate_model_dir(model_dir: Path) -> None:
    if not model_dir.is_dir():
        raise SystemExit(f"Model folder not found: {model_dir}")

    required_files = ("config.json",)
    missing = [name for name in required_files if not (model_dir / name).exists()]
    if missing:
        raise SystemExit(f"Model folder is missing required files: {', '.join(missing)}")

    weight_files = (
        model_dir / "model.safetensors",
        model_dir / "pytorch_model.bin",
    )
    if not any(path.exists() for path in weight_files):
        raise SystemExit(
            f"No model weights found in {model_dir} "
            "(expected model.safetensors or pytorch_model.bin)"
        )


def export_to_onnx(model_dir: Path, onnx_path: Path) -> None:
    model = OnnxClassifierWrapper(
        DistilBertForSequenceClassification.from_pretrained(model_dir)
    )
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    model.eval()

    sample = tokenizer(
        "example transcript segment",
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQUENCE_LENGTH,
    )

    torch.onnx.export(
        model,
        (sample["input_ids"], sample["attention_mask"]),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "attention_mask": {0: "batch", 1: "sequence"},
            "logits": {0: "batch"},
        },
        opset_version=ONNX_OPSET_VERSION,
        dynamo=False,
    )


def copy_tokenizer_and_config(model_dir: Path, output_dir: Path) -> None:
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)
    tokenizer.save_pretrained(output_dir)
    shutil.copy2(model_dir / "config.json", output_dir / "config.json")


def quantize_classifier(model_dir: Path, output_dir: Path, keep_fp32: bool) -> None:
    validate_model_dir(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = output_dir / "model.onnx"
    quantized_path = output_dir / "model.quant.onnx"

    print(f"Exporting {model_dir} to ONNX...")
    export_to_onnx(model_dir, fp32_path)

    print("Applying dynamic INT8 weight quantization...")
    quantize_dynamic(
        model_input=str(fp32_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QUInt8,
    )

    copy_tokenizer_and_config(model_dir, output_dir)

    if not keep_fp32:
        fp32_path.unlink()

    fp32_size_mb = fp32_path.stat().st_size / (1024 * 1024) if keep_fp32 else None
    quant_size_mb = quantized_path.stat().st_size / (1024 * 1024)

    print(f"Saved quantized model to {output_dir}")
    print(f"  model.quant.onnx: {quant_size_mb:.1f} MB")
    if fp32_size_mb is not None:
        print(f"  model.onnx: {fp32_size_mb:.1f} MB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fine-tuned DistilBERT classifier to quantized ONNX."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Folder containing the fine-tuned Hugging Face model",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder for the quantized ONNX model (default: <input>-quantized)",
    )
    parser.add_argument(
        "--keep-fp32",
        action="store_true",
        help="Keep the intermediate FP32 ONNX file alongside the quantized model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output or Path(f"{args.input}-quantized")
    quantize_classifier(args.input, output_dir, keep_fp32=args.keep_fp32)


if __name__ == "__main__":
    main()
