"""
Comparación final: modelo_base vs. modelo_fewshot
==================================================
Evalúa modelo_fewshot.pt en local_test y lo compara contra
los resultados del modelo_base guardados en fase5_K_domain_shift.json.

Resultado:
    results/fase6_comparacion_final.csv
    results/fase6_datos_comparacion.json
    results/fase6_comparacion_final.png

Requisitos previos:
    - models/modelo_fewshot.pt
    - results/fase5_K_domain_shift.json   (evaluar_base_local.py)
    - results/fase4_Kshot_resultados.json (evaluar_fewshot.py)
    - datasets/local_split/local_test/data.yaml

Uso:
    python scripts/comparacion_final.py
"""

import json
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO

# ── Configuración ────────────────────────────────────────────────────────────

BASE        = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE / "models"
RESULTS_DIR = BASE / "results"
YAML_TEST   = str(BASE / "datasets" / "local_split" / "local_test" / "data.yaml")

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import torch
    if not torch.cuda.is_available():
        raise SystemExit("❌ CUDA no disponible.")
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")

    # ── Verificar archivos ────────────────────────────────────────────────────

    json_base = RESULTS_DIR / "fase5_K_domain_shift.json"
    modelo_fs = MODELS_DIR  / "modelo_fewshot.pt"

    print("\nVerificando archivos...")
    for archivo in [json_base, modelo_fs, Path(YAML_TEST)]:
        print(f"  {'✅' if Path(archivo).exists() else '❌'}  {archivo}")

    if not json_base.exists():
        raise SystemExit(
            "\n❌ Falta fase5_K_domain_shift.json.\n"
            "   Ejecutar primero: python scripts/evaluar_base_local.py"
        )
    if not modelo_fs.exists():
        raise SystemExit(
            "\n❌ Falta modelo_fewshot.pt.\n"
            "   Ejecutar primero: python scripts/evaluar_fewshot.py"
        )

    # ── Cargar resultado del modelo base ─────────────────────────────────────

    with open(json_base, encoding="utf-8") as f:
        datos_base = json.load(f)

    resultado_base = {
        "Modelo"     : "Base (sin adaptación)",
        "Precision"  : datos_base["Precision"],
        "Recall"     : datos_base["Recall"],
        "F1-Score"   : datos_base["F1-Score"],
        "mAP@0.5"    : datos_base["mAP@0.5"],
        "mAP@0.5:95" : datos_base["mAP@0.5:95"],
    }

    # ── Leer K seleccionado ───────────────────────────────────────────────────

    json_kshot = RESULTS_DIR / "fase4_Kshot_resultados.json"
    K_mejor = "?"
    if json_kshot.exists():
        with open(json_kshot, encoding="utf-8") as f:
            K_mejor = json.load(f)["mejor_K"]

    # ── Evaluar modelo_fewshot en local_test ──────────────────────────────────

    json_comp = RESULTS_DIR / "fase6_datos_comparacion.json"

    if json_comp.exists():
        print(f"\n✅ Evaluación ya existe — cargando desde {json_comp.name}")
        with open(json_comp, encoding="utf-8") as f:
            datos_comp = json.load(f)
        resultado_fs = datos_comp["resultado_fewshot"]
        K_mejor      = datos_comp["mejor_K"]
    else:
        print(f"\nEvaluando modelo_fewshot.pt (K={K_mejor}) en local_test...")
        m = YOLO(str(modelo_fs)).val(
            data    = YAML_TEST,
            imgsz   = 640,
            device  = 0,
            verbose = False,
        )

        precision = round(float(m.box.mp),    4)
        recall    = round(float(m.box.mr),    4)
        map50     = round(float(m.box.map50), 4)
        map5095   = round(float(m.box.map),   4)
        f1        = round(2*precision*recall / (precision+recall), 4) \
                    if (precision+recall) > 0 else 0.0

        resultado_fs = {
            "Modelo"     : f"Few-Shot K={K_mejor}",
            "Precision"  : precision,
            "Recall"     : recall,
            "F1-Score"   : f1,
            "mAP@0.5"    : map50,
            "mAP@0.5:95" : map5095,
        }

    # ── Tabla comparativa ─────────────────────────────────────────────────────

    df = pd.DataFrame([resultado_base, resultado_fs])

    print(f"\n{'='*65}")
    print("TABLA COMPARATIVA FINAL (evaluado en local_test)")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print()

    metricas = ["Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@0.5:95"]
    for col in metricas:
        delta = df.loc[1, col] - df.loc[0, col]
        pct   = delta / df.loc[0, col] * 100 if df.loc[0, col] > 0 else 0.0
        signo = "+" if delta >= 0 else ""
        print(f"  Δ {col:<15}: {signo}{delta:.4f}  ({signo}{pct:.1f}%)")

    # ── Guardar CSV y JSON ────────────────────────────────────────────────────

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = RESULTS_DIR / "fase6_comparacion_final.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV guardado en: {csv_path}")

    comparacion = {
        "mejor_K"           : K_mejor,
        "resultado_base"    : resultado_base,
        "resultado_fewshot" : resultado_fs,
        "deltas"            : {
            col: round(resultado_fs[col] - resultado_base[col], 4)
            for col in metricas
        },
        "mejoras_pct"       : {
            col: round(
                (resultado_fs[col] - resultado_base[col]) / resultado_base[col] * 100, 1
            )
            for col in metricas if resultado_base[col] > 0
        },
    }
    json_path = RESULTS_DIR / "fase6_datos_comparacion.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparacion, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON guardado en: {json_path}")

    # ── Gráfico comparativo de barras ─────────────────────────────────────────

    val_base = [resultado_base[m] for m in metricas]
    val_fs   = [resultado_fs[m]   for m in metricas]
    x        = range(len(metricas))
    width    = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor("#F8F7F4")
    ax.set_facecolor("#F8F7F4")

    bars1 = ax.bar([i - width/2 for i in x], val_base, width,
                   label="Modelo Base (sin adaptación)",
                   color="#94A3B8", edgecolor="white")
    bars2 = ax.bar([i + width/2 for i in x], val_fs, width,
                   label=f"Few-Shot K={K_mejor}",
                   color="#378ADD", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#0C447C")

    ax.set_xticks(list(x))
    ax.set_xticklabels(metricas, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Valor", fontsize=11)
    ax.set_title(
        f"Comparación final: Modelo Base vs. Few-Shot (K={K_mejor})\n"
        f"Evaluado en local_test (90 imágenes)",
        fontweight="bold", fontsize=12,
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    grafico_path = RESULTS_DIR / "fase6_comparacion_final.png"
    plt.savefig(grafico_path, dpi=150, bbox_inches="tight", facecolor="#F8F7F4")
    plt.close()
    print(f"✅ Gráfico guardado en: {grafico_path}")

    print(f"\n{'='*65}")
    print("COMPARACIÓN FINAL COMPLETADA")
    print(f"{'='*65}")
    print(f"  CSV     : results/fase6_comparacion_final.csv")
    print(f"  JSON    : results/fase6_datos_comparacion.json")
    print(f"  Gráfico : results/fase6_comparacion_final.png")