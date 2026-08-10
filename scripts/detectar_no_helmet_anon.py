"""
Inferencia con anonimización de rostros
===============================================================================
Realiza detección sobre imágenes, carpetas o videos usando el
modelo_fewshot.pt por defecto. Aplica blur gaussiano sobre rostros
detectados con InsightFace antes de guardar el resultado.

Uso:
    # Detectar ambas clases (helmet y no_helmet) — por defecto
    python scripts/detectar_no_helmet.py --source imagen.jpg

    # Detectar solo no_helmet (infractores)
    python scripts/detectar_no_helmet.py --source imagen.jpg --class_id 1

    # Detectar solo helmet
    python scripts/detectar_no_helmet.py --source imagen.jpg --class_id 0

    # Usar un modelo diferente
    python scripts/detectar_no_helmet.py --source imagen.jpg --model models/modelo_base.pt

    # Sobre una carpeta
    python scripts/detectar_no_helmet.py --source datasets/validacion/

    # Con umbral de confianza personalizado
    python scripts/detectar_no_helmet.py --source imagen.jpg --conf 0.5

Requisitos:
    pip install insightface onnxruntime opencv-python
    (con GPU: pip install onnxruntime-gpu en lugar de onnxruntime)
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
from ultralytics import YOLO

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_OK = True
except ImportError:
    INSIGHTFACE_OK = False
    print("⚠️  InsightFace no instalado — sin anonimización de rostros.")
    print("   Instalar con: pip install insightface onnxruntime")


# ── Anonimización con InsightFace ─────────────────────────────────────────────

def init_face_detector():
    """Inicializa el detector de rostros InsightFace."""
    if not INSIGHTFACE_OK:
        return None
    app = FaceAnalysis(
        name      = "buffalo_sc",   # modelo liviano, bueno para rostros pequeños
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def anonimizar_rostros(img_bgr, detector, expansion=0.15, blur_kernel=99):
    """
    Detecta rostros con InsightFace y aplica blur gaussiano.
    Devuelve la imagen con rostros difuminados.
    """
    if detector is None:
        return img_bgr

    h, w   = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    faces   = detector.get(img_rgb)
    img_out = img_bgr.copy()

    if not faces:
        return img_out

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        # Expandir bbox para cubrir mejor el rostro
        dx = int((x2 - x1) * expansion)
        dy = int((y2 - y1) * expansion)
        x1 = max(0,  x1 - dx)
        y1 = max(0,  y1 - dy)
        x2 = min(w,  x2 + dx)
        y2 = min(h,  y2 + dy)

        roi = img_out[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = blur_kernel | 1  # garantizar impar
        img_out[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

    return img_out


# ── Inferencia ────────────────────────────────────────────────────────────────

def procesar_imagen(img_path, model, face_detector, classes, conf, output_path):
    """Procesa una imagen: detecta cascos + anonimiza rostros + dibuja bboxes + guarda."""
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  ⚠️  No se pudo leer: {img_path.name}")
        return 0, 0

    # 1. Inferencia YOLO sobre la imagen original (sin distorsión)
    results = model.predict(
        source   = img,
        classes  = classes,
        conf     = conf,
        verbose  = False,
    )

    # 2. Anonimizar rostros sobre la imagen original
    img_anonimizada = anonimizar_rostros(img, face_detector)

    # 3. Dibujar bounding boxes sobre la imagen ya anonimizada
    #    Reemplazamos la imagen base del resultado por la anonimizada
    results[0].orig_img = img_anonimizada
    img_resultado = results[0].plot()

    # 4. Guardar
    dst = output_path / img_path.name
    cv2.imwrite(str(dst), img_resultado)

    # Contar detecciones
    n_helmet = n_no_helmet = 0
    if results[0].boxes is not None:
        for cls in results[0].boxes.cls.tolist():
            if int(cls) == 0:
                n_helmet    += 1
            elif int(cls) == 1:
                n_no_helmet += 1

    return n_helmet, n_no_helmet


def main():

    BASE = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Detección de cascos con anonimización de rostros (MediaPipe)"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Ruta de imagen, carpeta o video"
    )
    parser.add_argument(
        "--model", type=str,
        default=str(BASE / "models" / "modelo_fewshot.pt"),
        help="Ruta del modelo YOLO (.pt). Por defecto: models/modelo_fewshot.pt"
    )
    parser.add_argument(
        "--class_id", type=int, required=False, default=None,
        help="ID de clase: 0=helmet, 1=no_helmet. Sin valor: detecta ambas."
    )
    parser.add_argument(
        "--conf", type=float, default=0.45,
        help="Umbral de confianza mínimo (default: 0.45)"
    )
    args = parser.parse_args()

    CLASES = {0: "helmet", 1: "no_helmet"}

    # Validar modelo
    modelo_path = Path(args.model)
    if not modelo_path.exists():
        raise SystemExit(
            f"❌ No se encontró el modelo: {modelo_path}\n"
            f"   Ejecutar primero: python scripts/evaluar_fewshot.py"
        )

    # Clases a detectar
    if args.class_id is not None:
        if args.class_id not in CLASES:
            raise SystemExit(
                f"❌ class_id inválido: {args.class_id}. "
                f"Valores válidos: 0 (helmet) o 1 (no_helmet)"
            )
        classes    = [args.class_id]
        clase_desc = CLASES[args.class_id]
    else:
        classes    = None
        clase_desc = "helmet + no_helmet (ambas)"

    # Carpeta de salida
    output_path = BASE / "outputs"
    output_path.mkdir(parents=True, exist_ok=True)

    # Info
    print("=" * 60)
    print("INFERENCIA — Detección de cascos")
    print("=" * 60)
    print(f"  Modelo        : {modelo_path.name}")
    print(f"  Fuente        : {args.source}")
    print(f"  Clases        : {clase_desc}")
    print(f"  Confianza     : {args.conf}")
    print(f"  Anonimización : {'InsightFace (activa)' if INSIGHTFACE_OK else 'No disponible'}")
    print(f"  Salida        : {output_path}")
    print()

    # Cargar modelo y detector de rostros
    model         = YOLO(str(modelo_path))
    face_detector = init_face_detector()

    # Determinar fuente
    source_path = Path(args.source)
    ext_img     = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    total_helmet    = 0
    total_no_helmet = 0

    # ── Imagen individual ─────────────────────────────────────────────────────
    if source_path.is_file() and source_path.suffix.lower() in ext_img:
        n_h, n_nh = procesar_imagen(
            source_path, model, face_detector,
            classes, args.conf, output_path
        )
        total_helmet    += n_h
        total_no_helmet += n_nh
        print(f"  ✅ {source_path.name} — helmet: {n_h}  no_helmet: {n_nh}")

    # ── Carpeta de imágenes ───────────────────────────────────────────────────
    elif source_path.is_dir():
        imagenes = sorted([
            f for f in source_path.iterdir()
            if f.suffix.lower() in ext_img
        ])
        print(f"  {len(imagenes)} imágenes encontradas en {source_path.name}/")
        for i, img_path in enumerate(imagenes, 1):
            n_h, n_nh = procesar_imagen(
                img_path, model, face_detector,
                classes, args.conf, output_path
            )
            total_helmet    += n_h
            total_no_helmet += n_nh
            if i % 10 == 0 or i == len(imagenes):
                print(f"  [{i}/{len(imagenes)}] procesadas...")

    # ── Video ─────────────────────────────────────────────────────────────────
    elif source_path.is_file() and source_path.suffix.lower() in {".mp4",".avi",".mov",".mkv"}:
        cap = cv2.VideoCapture(str(source_path))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 25
        ancho = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        alto  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        dst_video = output_path / (source_path.stem + "_detectado.mp4")
        writer    = cv2.VideoWriter(
            str(dst_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps, (ancho, alto)
        )

        frame_n = 0
        print("  Procesando video...")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_n += 1

            # 1. Inferencia YOLO sobre el frame original
            results = model.predict(
                source  = frame,
                classes = classes,
                conf    = args.conf,
                verbose = False,
            )

            # 2. Anonimizar rostros sobre el frame original
            frame_anon = anonimizar_rostros(frame, face_detector)

            # 3. Dibujar bboxes sobre el frame anonimizado
            results[0].orig_img = frame_anon
            frame_out = results[0].plot()

            writer.write(frame_out)

            if results[0].boxes is not None:
                for cls in results[0].boxes.cls.tolist():
                    if int(cls) == 0: total_helmet    += 1
                    elif int(cls) == 1: total_no_helmet += 1

            if frame_n % 30 == 0:
                print(f"  Frame {frame_n} procesado...")

        cap.release()
        writer.release()
        print(f"  ✅ Video guardado en: {dst_video}")

    else:
        raise SystemExit(f"❌ Fuente no reconocida: {args.source}")

    # ── Resumen ───────────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("RESULTADO")
    print("=" * 60)
    print(f"  Con casco (helmet)    : {total_helmet}")
    print(f"  Sin casco (no_helmet) : {total_no_helmet}")
    print(f"  Total detecciones     : {total_helmet + total_no_helmet}")
    print(f"  Guardado en           : {output_path}")


if __name__ == "__main__":
    main()