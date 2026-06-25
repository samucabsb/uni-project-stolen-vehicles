"""
yolo_infer.py — Inferência YOLO direta via ONNX Runtime, sem ultralytics.

SharedYOLO é thread-safe: uma única InferenceSession pode ser chamada
concorrentemente por N threads. ORT garante thread-safety em session.run().

Vantagem de cache:
  Processos (ProcessPoolExecutor): N × 3.4 MB de pesos no L3 (N cópias).
  Threads   (ThreadPoolExecutor) : 1 × 3.4 MB de pesos no L3 (1 cópia compartilhada).
  Com 12 threads: 40.8 MB → 3.4 MB — cabe inteiro no L3 de 12 MB.
"""

import cv2
import numpy as np


# ── Pré-processamento ─────────────────────────────────────────────────────────

def _letterbox(img: np.ndarray, target: int = 640) -> tuple:
    """Resize + pad cinza para quadrado target×target (sem distorção)."""
    h, w = img.shape[:2]
    scale = min(target / h, target / w)
    nh = int(round(h * scale))
    nw = int(round(w * scale))

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    dy = (target - nh) // 2
    dx = (target - nw) // 2

    canvas = np.full((target, target, 3), 114, dtype=np.uint8)
    canvas[dy:dy + nh, dx:dx + nw] = resized

    return canvas, scale, dy, dx


def _preprocess_batch(images: list, target: int = 640) -> tuple:
    """Pré-processa lista de imagens em um tensor BCHW float32."""
    tensors = []
    metas   = []   # (scale, dy, dx) por imagem

    for img in images:
        canvas, scale, dy, dx = _letterbox(img, target)
        t = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensors.append(t.transpose(2, 0, 1))   # HWC → CHW
        metas.append((scale, dy, dx))

    return np.stack(tensors, axis=0), metas   # [N, 3, H, W]


# ── Pós-processamento ─────────────────────────────────────────────────────────

def _nms_and_decode(raw: np.ndarray, meta: tuple,
                    conf_thresh: float, iou_thresh: float) -> list:
    """
    Decodifica uma imagem do tensor de saída [5, anchors] e aplica NMS.

    Retorna lista de dicts {'xyxy': [x1,y1,x2,y2], 'conf': float}
    em coordenadas da imagem original.
    """
    scale, dy, dx = meta

    confs = raw[4]               # [anchors]
    mask  = confs > conf_thresh
    if not mask.any():
        return []

    boxes_cxywh = raw[:4, mask]  # [4, N]
    scores      = confs[mask]    # [N]

    cx, cy, bw, bh = boxes_cxywh

    # Centro + tamanho → cantos (espaço do modelo, 0–640)
    x1m = cx - bw / 2
    y1m = cy - bh / 2
    x2m = cx + bw / 2
    y2m = cy + bh / 2

    # Inverso do letterbox → coordenadas da imagem original
    x1 = (x1m - dx) / scale
    y1 = (y1m - dy) / scale
    x2 = (x2m - dx) / scale
    y2 = (y2m - dy) / scale

    # NMS
    boxes_xywh = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
    idxs = cv2.dnn.NMSBoxes(
        boxes_xywh, scores.tolist(), conf_thresh, iou_thresh
    )

    result = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            result.append({
                "xyxy": [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                "conf": float(scores[i]),
            })

    return result


def _decode_output(output: np.ndarray, metas: list,
                   conf_thresh: float = 0.25,
                   iou_thresh:  float = 0.45) -> list:
    """output shape: [batch, 5, anchors] → list[list[dict]] por imagem."""
    return [
        _nms_and_decode(output[i], metas[i], conf_thresh, iou_thresh)
        for i in range(len(metas))
    ]


# ── SharedYOLO ────────────────────────────────────────────────────────────────

class SharedYOLO:
    """
    Wrapper thread-safe em torno de uma única ort.InferenceSession.

    Uma única instância pode ser chamada concorrentemente por N threads —
    ORT garante que session.run() é re-entrante (os pesos são somente-leitura
    e ficam em cache no L3 como uma única cópia, não N cópias).

    Uso:
        yolo = SharedYOLO("model_int8.onnx")
        results = yolo.infer_batch([img1, img2, ...])
        # results[i] é uma lista de dicts {'xyxy': [...], 'conf': float}
    """

    def __init__(self, model_path: str, conf_thresh: float = 0.25,
                 iou_thresh: float = 0.45):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads  = 1
        opts.inter_op_num_threads  = 1
        opts.execution_mode        = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
        except Exception:
            pass

        self._session     = ort.InferenceSession(
            model_path, sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name  = self._session.get_inputs()[0].name
        self._conf_thresh = conf_thresh
        self._iou_thresh  = iou_thresh

        # Warmup: aquece kernels ORT antes de qualquer chamada real
        dummy = np.zeros((1, 3, 640, 640), dtype=np.float32)
        self._session.run(None, {self._input_name: dummy})

    def infer_batch(self, images: list) -> list:
        """
        Infere um lote de imagens BGR.
        Thread-safe: chamável concorrentemente por múltiplas threads.
        Retorna list[list[dict]] — detecções por imagem.
        """
        if not images:
            return []

        batch, metas = _preprocess_batch(images)
        output = self._session.run(None, {self._input_name: batch})[0]
        return _decode_output(output, metas, self._conf_thresh, self._iou_thresh)
