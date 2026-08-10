"""
FASE 2.1 — Entrenamiento de un modelo base
Entrena YOLOv8n sobre Dataset_X_YOLO
Resultado:
    models/modelo_X.pt  ← nuevo modelo base

Requisitos previos:
    - datasets/Dataset_X_YOLO/data.yaml  (dataset X normalizado)

Uso:
    python scripts/entrenar_modelos_base.py --dataset nombre_dataset_YOLO --name modelo_X
"""

import shutil
import subprocess
from pathlib import Path
import argparse
import pandas as pd
import torch
from ultralytics import YOLO

## Configuración

BASE = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE / "datasets"
MODELS_DIR = BASE / "models"
RESULTS_DIR = BASE / "results"
RUNS_DIR = BASE / "runs"

## Entry point

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Realiza el entrenamiento de un modelo YOLO sobre imagenes"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Nombre del dataset _YOLO a utilizar. Debe estar dentro de /datasets"
    )
    
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Nombre del modelo"
    )
    
    args = parser.parse_args()

    YAML = str(DATASETS_DIR / args.dataset / "data.yaml")
    
    CONFIG = {
        "model" : "yolov8n.pt",
        "epochs" : 50,
        "batch" : 8,
        "imgsz" : 640,
        "device" : 0,
        "workers" : 0,
        "lr0" : 0.01,
        "lrf" : 0.01,
        "patience" : 15,
        "mosaic" : 0.0,
        "cos_lr" : True,
        "seed" : 42,
        "name" : args.name,
        "exist_ok" : True,
    }

    ## Verificaciones
    print("=" * 60)
    print(f"ENTRENAMIENTO {args.name}")
    print("=" * 60)

    if not Path(YAML).exists():
        raise SystemExit(
            f"  No se encontró: {YAML}\n"
            f"  Verificar que {args.dataset} está normalizado."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset  : {YAML}")
    print(f"Configuración:")
    for k, v in CONFIG.items():
        print(f"  {k:<12}: {v}")

    print(f"   Los pesos se guardan automáticamente en runs/{args.name}/weights/")
    print(f"   Si se interrumpe, reanudar con resume=True sobre last.pt\n")

    ## Entrenamiento

    model = YOLO(CONFIG["model"])
    model.train(
        data = YAML,
        epochs = CONFIG["epochs"],
        batch = CONFIG["batch"],
        imgsz = CONFIG["imgsz"],
        device = CONFIG["device"],
        workers = CONFIG["workers"],
        lr0 = CONFIG["lr0"],
        lrf = CONFIG["lrf"],
        patience = CONFIG["patience"],
        mosaic = CONFIG["mosaic"],
        cos_lr = CONFIG["cos_lr"],
        seed = CONFIG["seed"],
        name = CONFIG["name"],
        project = str(RUNS_DIR),
        exist_ok = CONFIG["exist_ok"],
    )

    # Guardar modelo

    candidatos = list(RUNS_DIR.rglob(f"{CONFIG['name']}/weights/best.pt"))
    if not candidatos:
        print(" No se encontró best.pt. Verificar runs/")
        raise SystemExit(1)

    best = candidatos[0]
    dst = MODELS_DIR / f"{args.name}.pt"
    shutil.copy2(best, dst)
    print(f"\n  Modelo guardado en: {dst}")

    # Evaluación inmediata en su propio dataset

    print(f"\n{'='*60}")
    print(f"EVALUACIÓN {args.name} sobre {args.dataset}/val")
    print(f"{'='*60}")

    model_eval = YOLO(str(dst))
    metricas = model_eval.val(
        data = YAML,
        imgsz = CONFIG["imgsz"],
        device = CONFIG["device"],
        verbose = False,
    )

    resultados = []

    precision = round(float(metricas.box.mp), 4)
    recall = round(float(metricas.box.mr), 4)
    map50 = round(float(metricas.box.map50), 4)
    map5095 = round(float(metricas.box.map), 4)
    f1 = round(2*precision*recall/(precision+recall), 4) if (precision+recall) > 0 else 0.0
   
    # Guardar los resultados
    resultados.append({
            "Modelo" : f"{args.name}",
            "Dataset" : f"{args.dataset}",
            "Precision" : precision,
            "Recall" : recall,
            "F1-Score" : f1,
            "mAP@0.5" : map50,
            "mAP@0.5:95" : map5095,
    })
    df = pd.DataFrame(resultados)
    
    print(df.to_string(index=False))

    csv_path = RESULTS_DIR / 'eval' / f"eval_{args.name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV guardado en: {csv_path}")
    
    print(f"  Precision  : {precision}")
    print(f"  Recall     : {recall}")
    print(f"  F1-Score   : {f1}")
    print(f"  mAP@0.5    : {map50}")
    print(f"  mAP@0.5:95 : {map5095}")
