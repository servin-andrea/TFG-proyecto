"""
SCRIPT 1 — Entrenamiento K-Shot (K=10, 20, 30)
================================================
Entrena 3 modelos Few-Shot partiendo siempre desde modelo_base.pt,
usando K ejemplos por clase (K=10, 20, 30).

Cambios respecto al entrenamiento base:
    - lr0     : 0.01 → 0.001  (10× menor, preserva conocimiento previo)
    - patience: 15   → 20     (más paciencia por convergencia irregular)
    - freeze  : —    → 10     (congela backbone, solo reentrena cabeza)
    - cos_lr  : True → True   (igual)
    - imgsz   : 640  → 640    (igual, garantiza comparabilidad)
    - mosaic  : 0.0  → 0.0    (igual)

Resultado:
    models/modelo_fewshot_K10.pt
    models/modelo_fewshot_K20.pt
    models/modelo_fewshot_K30.pt

Requisitos previos:
    - models/modelo_base.pt  (generado en Fase 4)
    - datasets/local_split_K/local_train/images/  (60 imgs: 30 helmet + 30 no_helmet)
    - datasets/local_split_K/local_train/labels/
    - datasets/local_split_K/local_train/data.yaml

Uso:
    python scripts/entrenar_modelos_fewshot.py
"""

import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# ── Configuración

BASE        = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE / "models"
RUNS_DIR    = BASE / "runs"
IMAGES_IN   = BASE / "datasets" / "local_split" / "local_train" / "images"
LABELS_IN   = BASE / "datasets" / "local_split" / "local_train" / "labels"

VALORES_K   = [10, 20, 30]   # ← K=5 eliminado, solo 10, 20 y 30
SEED        = 42

TRAIN_CONFIG = {
    "epochs"  : 50,      # igual al entrenamiento base
    "batch"   : 8,       # igual
    "imgsz"   : 640,     # igual → comparabilidad garantizada
    "device"  : 0,
    "workers" : 0,
    "lr0"     : 0.001,   # ← 10× menor que el base (0.01) → evita olvido catastrófico
    "lrf"     : 0.01,    # igual
    "patience": 20,      # ← mayor que el base (15) → más paciencia con pocos datos
    "freeze"  : 10,      # ← congela primeras 10 capas del backbone
    "mosaic"  : 0.0,     # igual
    "cos_lr"  : True,    # igual
    "seed"    : SEED,
    "exist_ok": True,
}

# ── Entry point

if __name__ == "__main__":

    # ── Clasificar imágenes por clase

    random.seed(SEED)
    ext = {".jpg", ".jpeg", ".png"}

    helmet_imgs, nohelmet_imgs = [], []

    for img in sorted(IMAGES_IN.iterdir()):
        if img.suffix.lower() not in ext:
            continue
        lbl = LABELS_IN / (img.stem + ".txt")
        if not lbl.exists():
            continue
        clases = set()
        for linea in lbl.read_text(encoding="utf-8").splitlines():
            partes = linea.strip().split()
            if partes:
                clases.add(int(partes[0]))
        if clases == {0}:
            helmet_imgs.append(img)
        elif clases == {1}:
            nohelmet_imgs.append(img)

    print(f"\nImágenes disponibles en local_train:")
    print(f"  Solo helmet    : {len(helmet_imgs)}")
    print(f"  Solo no_helmet : {len(nohelmet_imgs)}")

    k_max = min(len(helmet_imgs), len(nohelmet_imgs))
    for k in VALORES_K:
        if k > k_max:
            raise SystemExit(
                f"❌ K={k} supera el máximo disponible ({k_max} imgs por clase).\n"
                f"   Reducir VALORES_K o ampliar el dataset local."
            )

    # ── Entrenamiento por cada K

    modelos_entrenados = {}

    for K in VALORES_K:
        print(f"\n{'='*60}")
        print(f"ENTRENANDO  K={K}  ({K} helmet + {K} no_helmet = {K*2} imágenes)")
        print(f"{'='*60}")

        subset = helmet_imgs[:K] + nohelmet_imgs[:K]

        # Crear dataset temporal
        tmp     = BASE / "datasets" / f"tmp_K{K}"
        tmp_img = tmp / "images"
        tmp_lbl = tmp / "labels"
        tmp_img.mkdir(parents=True, exist_ok=True)
        tmp_lbl.mkdir(parents=True, exist_ok=True)

        for img in subset:
            shutil.copy2(img, tmp_img / img.name)
            lbl = LABELS_IN / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, tmp_lbl / lbl.name)

        yaml_content = (
            f"path: {tmp.resolve()}\n"
            f"train: images\n"
            f"val: images\n"
            f"nc: 2\n"
            f"names:\n"
            f"  0: helmet\n"
            f"  1: no_helmet\n"
        )
        (tmp / "data.yaml").write_text(yaml_content, encoding="utf-8")

        # Fine-tuning desde modelo_base (siempre desde el mismo punto)
        modelo = YOLO(str(MODELS_DIR / 'modelo_base.pt'))
        modelo.train(
            data    = str(tmp / "data.yaml"),
            name    = f"fewshot_K{K}",
            project = str(RUNS_DIR),
            **TRAIN_CONFIG,
        )

        # Buscar best.pt y copiar a models/
        candidatos = list(RUNS_DIR.rglob(f"fewshot_K{K}/weights/best.pt"))
        if not candidatos:
            print(f"  ⚠️  No se encontró best.pt para K={K}. Buscando en runs/...")
            for p in RUNS_DIR.rglob("best.pt"):
                print(f"      Encontrado: {p}")
            shutil.rmtree(tmp)
            continue

        best = candidatos[0]
        dst  = MODELS_DIR / f"modelo_fewshot_K{K}.pt"
        shutil.copy2(best, dst)
        modelos_entrenados[K] = str(dst)
        print(f"\n  ✅ modelo_fewshot_K{K}.pt guardado en: {dst}")

        # Limpiar dataset temporal
        shutil.rmtree(tmp)

    # ── Resumen final

    print(f"\n{'='*60}")
    print("ENTRENAMIENTO K-SHOT COMPLETADO")
    print(f"{'='*60}")
    for K, ruta in modelos_entrenados.items():
        print(f"  K={K:<3} → {ruta}")
