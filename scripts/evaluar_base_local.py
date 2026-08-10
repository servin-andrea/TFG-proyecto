"""
Evaluación del modelo base en local_test (Domain Shift)
===================================================================
Evalúa modelo_base.pt sobre el conjunto local_test para evidenciar
el fenómeno de domain shift.

Este script puede correrse ANTES o DESPUÉS del Script 1, ya que
es independiente del entrenamiento K-Shot.

Resultado:
    results/fase5_K_domain_shift.json

Requisitos previos:
    - models/modelo_base.pt
    - datasets/local_split/local_test/data.yaml
    - results/modelo_base_seleccionado.json  (para leer MAP_PUBLICO automáticamente)

Uso:
    python scripts/evaluar_base_local.py
"""

import json
from pathlib import Path
from ultralytics import YOLO

# ── Configuración

BASE        = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE / "models"
RESULTS_DIR = BASE / "results"
YAML_TEST   = str(BASE / "datasets" / "local_split" / "local_test" / "data.yaml")

# Leer mAP público del JSON de selección si existe,
# o definirlo manualmente si no está disponible
_json_sel = RESULTS_DIR / "modelo_base_seleccionado.json"
if _json_sel.exists():
    with open(_json_sel, encoding="utf-8") as _f:
        MAP_PUBLICO = json.load(_f)["mAP@0.5"]
    print(f"MAP_PUBLICO leído desde modelo_base_seleccionado.json: {MAP_PUBLICO}")
else:
    # ── AJUSTAR MANUALMENTE SI NO EXISTE EL JSON
    MAP_PUBLICO = 0.8885   # mAP@0.5 del modelo base en su propio dataset
    print(f"MAP_PUBLICO definido manualmente: {MAP_PUBLICO}")

# ── Entry point

if __name__ == "__main__":

    # ── Evaluación

    print(f"\n{'='*60}")
    print("FASE 5 — Domain Shift")
    print("Evaluando modelo_base.pt en local_test...")
    print(f"{'='*60}")

    modelo = YOLO(str(str(MODELS_DIR / f"modelo_base.pt")))
    m = modelo.val(
        data    = YAML_TEST,
        imgsz   = 640,
        device  = 0,
        verbose = True,
    )

    precision = round(float(m.box.mp),    4)
    recall    = round(float(m.box.mr),    4)
    map50     = round(float(m.box.map50), 4)
    map5095   = round(float(m.box.map),   4)
    f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) > 0 else 0.0

    caida_abs = round(map50 - MAP_PUBLICO, 4)
    caida_pct = round(caida_abs / MAP_PUBLICO * 100, 1)

    # ── Imprimir resultados

    print(f"\n{'='*60}")
    print("RESULTADOS — Domain Shift")
    print(f"{'='*60}")
    print(f"  Precision  : {precision}")
    print(f"  Recall     : {recall}")
    print(f"  F1-Score   : {f1}")
    print(f"  mAP@0.5    : {map50}")
    print(f"  mAP@0.5:95 : {map5095}")
    print()
    print(f"  mAP dominio público (Dataset base) : {MAP_PUBLICO}")
    print(f"  mAP dominio local   (local_test)   : {map50}")
    print(f"  Caída absoluta                     : {caida_abs:+.4f}")
    print(f"  Caída porcentual                   : {caida_pct:+.1f}%")

    if abs(caida_pct) > 20:
        print(f"\n  ✅ Domain shift significativo detectado ({caida_pct:.1f}%)")
        print(f"     Este resultado justifica la aplicación de Few-Shot Learning.")
    elif abs(caida_pct) > 5:
        print(f"\n  ⚠️  Domain shift leve detectado ({caida_pct:.1f}%).")
    else:
        print(f"\n  ℹ️  Caída mínima ({caida_pct:.1f}%). El dominio local es similar al público.")

    # ── Guardar JSON

    resultado = {
        "modelo"           : "modelo_base.pt",
        "evaluado_en"      : "local_test (90 imágenes)",
        "Precision"        : precision,
        "Recall"           : recall,
        "F1-Score"         : f1,
        "mAP@0.5"          : map50,
        "mAP@0.5:95"       : map5095,
        "mAP_publico"      : MAP_PUBLICO,
        "caida_absoluta"   : caida_abs,
        "caida_porcentual" : caida_pct,
    }

    json_path = RESULTS_DIR / "fase5_K_domain_shift.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Guardado en: {json_path}")

