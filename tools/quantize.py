"""
quantize.py — Quantização dinâmica INT8 do modelo YOLO ONNX.

USO:
    python tools/quantize.py
    python tools/quantize.py --model models/license_plate_detector.onnx
    python tools/quantize.py --model models/meu_modelo.onnx --output models/meu_modelo_int8.onnx

Após gerar o modelo INT8, use com:
    python main.py --yolo-model models/license_plate_detector_int8.onnx --execution parallel --workers 4 --benchmark

POR QUE ISSO RESOLVE O GARGALO DE 4 WORKERS
============================================
O principal limite de speedup no modo parallel com N >= 4 workers é contenção
de cache L3 ("cache thrashing"): cada worker carrega sua própria cópia FP32 do
modelo YOLO (~22 MB), e com N cópias simultâneas o L3 da CPU (~6-12 MB) fica
saturado — cada inferência lê os pesos da DRAM (10-50× mais lento que L3) em
vez do cache. Resultado: YOLO fica 2× mais lento por imagem com 4 workers do
que com 2 workers, zerando o ganho de paralelismo.

A quantização dinâmica converte os pesos FP32 -> INT8 (4× menor). Com o modelo
em ~5-6 MB por cópia, 4 cópias = ~24 MB. Ainda não cabe inteiramente em 6 MB
de L3, mas reduz dramaticamente a pressão: mais operadores quentes ficam em
cache, menos acessos a DRAM. Resultado esperado: YOLO com 4 workers de
~144 ms/img cai para ~70-90 ms/img.

Em CPUs com AVX2/VNNI (Intel Haswell+, AMD Zen 2+) as operações INT8 também
têm throughput 2-4× maior que FP32 no nível de silício. O ORT aproveita isso
automaticamente nas operações de MatMul e Conv.

TRADE-OFF
=========
  - Precisão: perda de ~0.5-2% mAP em modelos de detecção (imperceptível na
    prática para detecção de placas, que o modelo já executa com alta confiança).
  - Velocidade de conversão: 30-120 s (feito UMA VEZ).
  - Não requer dados de calibração ("dinâmica" = só quantiza pesos, não ativações).
"""

import argparse
import sys
from pathlib import Path

# Garante que src/ seja importável independente do CWD
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _check_ort_quantization() -> None:
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType  # noqa: F401
    except ImportError:
        print(
            "[ERRO] onnxruntime.quantization não disponível.\n"
            "       Atualize: pip install onnxruntime --upgrade"
        )
        sys.exit(1)


def quantize(model_path: str, output_path: str | None = None) -> str:
    """
    Aplica quantização dinâmica INT8 ao modelo ONNX.
    Retorna o caminho do arquivo gerado.
    """
    import shutil
    import tempfile
    from onnxruntime.quantization import quantize_dynamic, QuantType

    src = Path(model_path)
    if not src.exists():
        print(f"[ERRO] Modelo não encontrado: {src}")
        sys.exit(1)
    if src.suffix.lower() != ".onnx":
        print(f"[ERRO] Esperado arquivo .onnx, recebido: {src.suffix!r}")
        sys.exit(1)

    dst = Path(output_path) if output_path else src.with_stem(src.stem + "_int8")

    if dst.exists():
        size_mb = dst.stat().st_size / (1024 ** 2)
        print(f"[QUANTIZE] Modelo já existe: {dst.name} ({size_mb:.1f} MB)")
        print(f"           Para regenerar, apague o arquivo e rode novamente.")
        return str(dst)

    src_mb = src.stat().st_size / (1024 ** 2)
    print(f"[QUANTIZE] Fonte   : {src.name} ({src_mb:.1f} MB FP32)")
    print(f"[QUANTIZE] Destino : {dst.name}")
    print("           Quantizando pesos FP32 -> INT8 (sem calibracao)...")
    print("           Isso pode levar 30-120 s na primeira execucao.\n")

    # onnxruntime cria um arquivo intermediário "*-inferred.onnx" no mesmo
    # diretório do modelo. Se o caminho contém caracteres não-ASCII (ex.:
    # "Usuário"), a camada C++ do onnxruntime falha silenciosamente ao
    # escrever o arquivo, causando FileNotFoundError na leitura seguinte.
    # Solução: copiar o modelo para um diretório temporário ASCII-seguro,
    # quantizar lá, copiar o resultado de volta.
    try:
        with tempfile.TemporaryDirectory(prefix="ort_quant_") as tmpdir:
            tmp_src = Path(tmpdir) / "model.onnx"
            tmp_dst = Path(tmpdir) / "model_int8.onnx"
            shutil.copy2(str(src), str(tmp_src))

            quantize_dynamic(
                model_input=str(tmp_src),
                model_output=str(tmp_dst),
                weight_type=QuantType.QUInt8,
                per_channel=False,
            )

            shutil.copy2(str(tmp_dst), str(dst))
    except Exception as exc:
        print(f"[ERRO] Quantização falhou: {exc}")
        if dst.exists():
            dst.unlink()
        sys.exit(1)

    dst_mb  = dst.stat().st_size / (1024 ** 2)
    ratio   = dst_mb / src_mb
    savings = (1 - ratio) * 100

    print(f"[QUANTIZE] Concluido!")
    print(f"           {src_mb:.1f} MB FP32 -> {dst_mb:.1f} MB INT8  ({savings:.0f}% menor)")
    print(f"\n  Para testar a velocidade:")
    print(f"    python main.py --yolo-model {dst} --execution serial --benchmark")
    print(f"    python main.py --yolo-model {dst} --execution parallel --workers 4 --benchmark")
    return str(dst)


def main() -> int:
    from src.runtime import patch_ssl_certifi
    patch_ssl_certifi()

    _check_ort_quantization()

    from src.config import DEFAULT_YOLO_MODEL
    default_onnx = str(DEFAULT_YOLO_MODEL.with_suffix(".onnx"))

    parser = argparse.ArgumentParser(
        description="Quantiza o modelo YOLO ONNX de FP32 para INT8 (dinâmica).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", default=default_onnx,
        help=f"Modelo .onnx a quantizar (padrão: {default_onnx})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Caminho de saída (padrão: <nome>_int8.onnx no mesmo diretório)",
    )
    args = parser.parse_args()
    quantize(args.model, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
