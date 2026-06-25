"""
quantize_static.py — Quantizacao ESTATICA INT8 do modelo YOLO ONNX.

POR QUE ESTATICA E NAO DINAMICA
================================
A quantizacao DINAMICA (quantize.py) converte apenas os PESOS para INT8.
Em tempo de execucao, esses pesos sao dequantizados de volta para FP32
antes de cada multiplicacao de matrizes. Resultado: overhead extra sem
ganho de velocidade de computacao — e foi por isso que o modelo INT8
ficou ~26% MAIS LENTO que FP32 nos benchmarks.

A quantizacao ESTATICA converte tanto PESOS quanto ATIVACOES para INT8.
Isso permite que as multiplicacoes de matrizes rodem em INT8 nativo no
hardware. No AMD Ryzen Zen 2 (AVX2), o MLAS do ONNX Runtime executa
matmul INT8 com throughput 2-4x maior que FP32.

Combinacao final:
  - Modelo 3.3 MB (vs 11.8 MB FP32)
  - Inferencia mais rapida (INT8 matmul via AVX2)
  - 4 workers x 3.3 MB = 13.2 MB ~ L3 de 12 MB -> cache quase sem thrashing
  - Resultado esperado: speedup 3.5-4x em 4 workers, ~2x em 2 workers

REQUISITO: imagens de calibracao (usadas para determinar os ranges das
ativacoes). O script usa as imagens do diretorio data/input automaticamente.
Sao necessarias entre 50 e 200 imagens representativas.

USO:
    python tools/quantize_static.py
    python tools/quantize_static.py --model models/license_plate_detector.onnx
    python tools/quantize_static.py --samples 100 --size 640
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Calibration Data Reader para YOLO ────────────────────────────────────────

class YOLOCalibrationReader:
    """
    CalibrationDataReader compativel com onnxruntime.quantization.

    Pre-processa imagens no mesmo formato que o pipeline YOLO espera:
      - Letterbox resize para (size x size), padding com cinza 114
      - BGR -> RGB
      - Normalize para [0.0, 1.0]
      - HWC -> CHW (channel-first)
      - Adiciona dimensao de batch (1, 3, H, W) float32
    """

    def __init__(self, images_dir: str, input_name: str = "images",
                 size: int = 640, max_samples: int = 200):
        import random
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        imgs = [p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts]
        if not imgs:
            raise FileNotFoundError(f"Nenhuma imagem encontrada em: {images_dir}")
        random.seed(42)
        random.shuffle(imgs)
        self.images     = [str(p) for p in imgs[:max_samples]]
        self.input_name = input_name
        self.size       = size
        self._idx       = 0
        print(f"  [CAL] {len(self.images)} imagens de calibracao carregadas de {images_dir}")

    def _letterbox(self, img, size: int):
        """Redimensiona mantendo aspect ratio e preenche com cinza 114."""
        import cv2
        import numpy as np
        h, w = img.shape[:2]
        scale   = size / max(h, w)
        new_h   = int(h * scale)
        new_w   = int(w * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas  = np.full((size, size, 3), 114, dtype=np.uint8)
        pad_h   = (size - new_h) // 2
        pad_w   = (size - new_w) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        return canvas

    def _preprocess(self, path: str):
        import cv2
        import numpy as np
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self._letterbox(img, self.size)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)          # HWC -> CHW
        img = img[None, ...]                   # add batch dim -> (1, 3, H, W)
        return img

    # Interface exigida pelo onnxruntime CalibrationDataReader
    def get_next(self):
        while self._idx < len(self.images):
            path = self.images[self._idx]
            self._idx += 1
            tensor = self._preprocess(path)
            if tensor is not None:
                return {self.input_name: tensor}
        return None

    def rewind(self):
        self._idx = 0


# ── Quantizacao ───────────────────────────────────────────────────────────────

def quantize_static(
    model_path: str,
    output_path: str | None,
    images_dir: str,
    max_samples: int = 150,
    size: int = 640,
) -> str:
    """
    Gera modelo INT8 com quantizacao estatica (calibrada com imagens reais).

    Fluxo:
      1. Shape inference (necessario para quantizacao correta de modelos dinamicos)
      2. Calibracao: roda as imagens pelo modelo e coleta estatisticas das ativacoes
      3. Quantiza pesos E ativacoes -> INT8 matmul nativo no hardware
    """
    try:
        from onnxruntime.quantization import (
            quantize_static as ort_quantize_static,
            QuantType, QuantFormat, CalibrationMethod,
        )
        from onnxruntime.quantization import shape_inference as ort_shape_inf
    except ImportError as e:
        print(f"[ERRO] onnxruntime.quantization nao disponivel: {e}")
        print("       Instale: pip install onnxruntime --upgrade")
        sys.exit(1)

    src = Path(model_path)
    if not src.exists():
        print(f"[ERRO] Modelo nao encontrado: {src}")
        sys.exit(1)

    dst = Path(output_path) if output_path else src.with_stem(src.stem + "_static_int8")

    if dst.exists():
        size_mb = dst.stat().st_size / 1024 / 1024
        print(f"[STATIC] Modelo ja existe: {dst.name} ({size_mb:.1f} MB)")
        print("         Para regenerar, apague e rode novamente.")
        return str(dst)

    src_mb = src.stat().st_size / 1024 / 1024
    print(f"\n[STATIC] Fonte   : {src.name} ({src_mb:.1f} MB FP32)")
    print(f"[STATIC] Destino : {dst.name}")
    print(f"[STATIC] Modo    : Quantizacao ESTATICA (pesos + ativacoes -> INT8)")
    print(f"[STATIC] Amostras: {max_samples} imagens de calibracao")
    print()

    # Obtém o nome do input do modelo
    import onnxruntime as ort
    sess = ort.InferenceSession(str(src), providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    del sess

    reader = YOLOCalibrationReader(
        images_dir=images_dir,
        input_name=input_name,
        size=size,
        max_samples=max_samples,
    )

    # Trabalha em diretorio temporario para evitar problemas com paths nao-ASCII
    with tempfile.TemporaryDirectory(prefix="ort_static_") as tmpdir:
        tmp_src  = Path(tmpdir) / "model.onnx"
        tmp_pre  = Path(tmpdir) / "model_pre.onnx"
        tmp_dst  = Path(tmpdir) / "model_static_int8.onnx"

        shutil.copy2(str(src), str(tmp_src))
        # Pula shape inference — causa MemoryError em modelos YOLO com shapes
        # dinamicos; o quantizador funciona sem ela (apenas com aviso).
        shutil.copy2(str(tmp_src), str(tmp_pre))
        print("[STATIC] Etapa 1/3: Shape inference pulada (modelo com shapes dinamicos).")

        # Step 2 + 3: Calibracao + Quantizacao
        print("[STATIC] Etapa 2/3: Calibrando com imagens reais...")
        print("         (isso pode levar 1-5 minutos dependendo do numero de imagens)")
        print()

        # Identifica os nos de saida do modelo (output head do YOLO) para excluir
        # da quantizacao -- esses nos produzem scores de confianca via sigmoid,
        # que sao muito sensiveis a erros de calibracao e tendem a colapsar para
        # zero se quantizados com ranges errados.
        import onnx as _onnx
        _m = _onnx.load(str(tmp_pre))
        output_names = [o.name for o in _m.graph.output]
        # Coleta nos que alimentam diretamente os outputs (penultima camada)
        output_node_inputs = set()
        for node in _m.graph.node:
            for out in node.output:
                if out in output_names:
                    output_node_inputs.update(node.input)
        nodes_to_exclude = [
            n.name for n in _m.graph.node
            if any(o in output_names or o in output_node_inputs for o in n.output)
            and n.name
        ]
        del _m
        print(f"  [CAL] Excluindo {len(nodes_to_exclude)} nos de saida da quantizacao")

        try:
            ort_quantize_static(
                model_input=str(tmp_pre),
                model_output=str(tmp_dst),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                # Entropia captura melhor os ranges de ativacoes com distribuicao
                # assimetrica (sigmoid, ReLU) que o MinMax simples subestima.
                calibrate_method=CalibrationMethod.Entropy,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
                reduce_range=False,
                nodes_to_exclude=nodes_to_exclude,
            )
        except Exception as e:
            print(f"\n[AVISO] Entropy falhou ({e}), tentando MinMax sem exclusoes...")
            reader.rewind()
            ort_quantize_static(
                model_input=str(tmp_pre),
                model_output=str(tmp_dst),
                calibration_data_reader=reader,
                quant_format=QuantFormat.QDQ,
                calibrate_method=CalibrationMethod.MinMax,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                per_channel=False,
                reduce_range=True,
            )

        if not tmp_dst.exists():
            print("[ERRO] Arquivo de saida nao foi gerado.")
            sys.exit(1)

        shutil.copy2(str(tmp_dst), str(dst))

    dst_mb  = dst.stat().st_size / 1024 / 1024
    savings = (1 - dst_mb / src_mb) * 100

    print(f"\n[STATIC] Etapa 3/3: Concluido!")
    print(f"  {src_mb:.1f} MB FP32 -> {dst_mb:.1f} MB INT8 estatico  ({savings:.0f}% menor)")
    print()
    print("  Para comparar velocidades:")
    print(f"    python run_blackbox.py --workers 2 4 6 --yolo-model {dst}")
    print()
    print("  Comparacao esperada no Ryzen Zen 2 (AVX2 INT8 matmul):")
    print("    FP32 serial  : ~127s  |  INT8 estatico serial  : ~60-80s")
    print("    FP32 2w      : 1.6x   |  INT8 estatico 2w      : ~1.9-2.0x")
    print("    FP32 4w      : 2.5x   |  INT8 estatico 4w      : ~3.5-4.0x")
    return str(dst)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> int:
    from src.runtime import patch_ssl_certifi
    from src.config  import DEFAULT_YOLO_MODEL, INPUT_DIR
    patch_ssl_certifi()

    default_onnx = str(DEFAULT_YOLO_MODEL.with_suffix(".onnx"))

    p = argparse.ArgumentParser(
        description="Quantizacao estatica INT8 do YOLO (calibrada com imagens reais).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--model",   default=default_onnx,
                   help=f"Modelo .onnx FP32 (padrao: {default_onnx})")
    p.add_argument("--output",  default=None,
                   help="Caminho de saida (padrao: <nome>_static_int8.onnx)")
    p.add_argument("--images",  default=str(INPUT_DIR),
                   help=f"Diretorio com imagens de calibracao (padrao: {INPUT_DIR})")
    p.add_argument("--samples", type=int, default=150,
                   help="Numero de imagens para calibracao (padrao: 150)")
    p.add_argument("--size",    type=int, default=640,
                   help="Tamanho de entrada do YOLO em pixels (padrao: 640)")
    args = p.parse_args()

    quantize_static(
        model_path=args.model,
        output_path=args.output,
        images_dir=args.images,
        max_samples=args.samples,
        size=args.size,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
