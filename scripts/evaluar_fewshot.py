"""
SCRIPT 2 — Evaluación de modelos K-Shot y selección del mejor
==============================================================
Evalúa cada modelo_fewshot_KX.pt sobre local_test y selecciona
el mejor K como modelo_fewshot.pt.

NO compara contra el modelo base — esa comparación va en el Script 4.

Resultado:
    results/fase4_Kshot_curva.csv
    results/fase4_Kshot_resultados.json
    results/fase4_Kshot_curva.png
    models/modelo_fewshot.pt   ← el mejor K

Requisitos previos:
    - models/modelo_fewshot_K10.pt, K20.pt, K30.pt
    - datasets/local_split/local_test/data.yaml

Uso:
    python scripts/evaluar_fewshot.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultralytics import YOLO

# ── Configuración

BASE        = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE / "models"
RESULTS_DIR = BASE / "results"
YAML_TEST   = str(BASE / "datasets" / "local_split" / "local_test" / "data.yaml")

VALORES_K = [10, 20, 30]

# ── Entry point

if __name__ == "__main__":

    # ── Verificar modelos

    print("\nVerificando modelos K-Shot:")
    for K in VALORES_K:
        p = MODELS_DIR / f"modelo_fewshot_K{K}.pt"
        print(f"  {'✅' if p.exists() else '❌'}  modelo_fewshot_K{K}.pt")

    faltantes = [K for K in VALORES_K
                 if not (MODELS_DIR / f"modelo_fewshot_K{K}.pt").exists()]
    if faltantes:
        raise SystemExit(
            f"\n❌ Faltan modelos para K={faltantes}.\n"
            f"   Ejecutar primero: python scripts/01_entrenar_kshot.py"
        )

    # ── Evaluación de cada modelo

    resultados = []

    for K in VALORES_K:
        print(f"\nEvaluando modelo_fewshot_K{K}.pt en local_test...")
        modelo = YOLO(str(MODELS_DIR / f"modelo_fewshot_K{K}.pt"))
        m = modelo.val(
            data    = YAML_TEST,
            imgsz   = 640,
            device  = 0,
            verbose = False,
        )

        precision = round(float(m.box.mp),    4)
        recall    = round(float(m.box.mr),    4)
        map50     = round(float(m.box.map50), 4)
        map5095   = round(float(m.box.map),   4)
        f1        = round(2*precision*recall / (precision+recall), 4) if (precision+recall) > 0 else 0.0

        resultados.append({
            "K"          : K,
            "N_total"    : K * 2,
            "Precision"  : precision,
            "Recall"     : recall,
            "F1-Score"   : f1,
            "mAP@0.5"    : map50,
            "mAP@0.5:95" : map5095,
        })
        print(f"  Precision={precision}  Recall={recall}  F1={f1}  mAP@0.5={map50}  mAP@0.5:95={map5095}")

    # ── Tabla comparativa

    df = pd.DataFrame(resultados)

    print(f"\n{'='*70}")
    print("TABLA COMPARATIVA — MODELOS K-SHOT")
    print(f"{'='*70}")
    print(df.to_string(index=False))

    csv_path = RESULTS_DIR / "fase4_Kshot_curva.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV guardado en: {csv_path}")

    # ── Selección del mejor modelo

    mejor_fila = df.loc[df["mAP@0.5"].idxmax()]
    K_mejor    = int(mejor_fila["K"])
    mAP_mejor  = mejor_fila["mAP@0.5"]

    print(f"\n🏆 Mejor modelo K-Shot: K={K_mejor}  (mAP@0.5 = {mAP_mejor:.4f})")

    origen  = MODELS_DIR / f"modelo_fewshot_K{K_mejor}.pt"
    destino = MODELS_DIR / "modelo_fewshot.pt"
    shutil.copy2(origen, destino)
    print(f"✅ Copiado como: {destino}")

    # ── Guardar JSON

    with open(RESULTS_DIR / "fase4_Kshot_resultados.json", "w", encoding="utf-8") as f:
        json.dump({
            "mejor_K"      : K_mejor,
            "mejor_mAP@0.5": float(mAP_mejor),
            "experimentos" : resultados,
        }, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON guardado en: results/fase4_Kshot_resultados.json")

    # ── Gráfico curva K-Shot

    ks   = df["K"].tolist()
    maps = df["mAP@0.5"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        ks, maps, color="#378ADD", linewidth=2,
        marker="o", markersize=9,
        markerfacecolor="white", markeredgewidth=2.5,
        label="mAP@0.5 por K",
    )
    for k, m_val in zip(ks, maps):
        ax.annotate(
            f"{m_val:.4f}",
            xy=(k, m_val), xytext=(0, 13),
            textcoords="offset points", ha="center",
            fontsize=10, fontweight="bold", color="#0C447C",
        )

    # Marcar el mejor K
    ax.scatter([K_mejor], [mAP_mejor], color="#E24B4A", s=120, zorder=5,
               label=f"Mejor: K={K_mejor} (mAP={mAP_mejor:.4f})")

    ax.set_xlabel("K (ejemplos por clase)", fontsize=11)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_title(
        "Comparación de modelos K-Shot\nmAP@0.5 en local_test (90 imágenes)",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(ks)
    ax.set_xticklabels([f"K={k}\n({k*2} imgs)" for k in ks])
    ax.set_ylim(0.20, 1.00)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    grafico_path = RESULTS_DIR / "fase4_Kshot_curva.png"
    plt.savefig(grafico_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gráfico guardado en: {grafico_path}")

    # ── Resumen final

    print(f"\n{'='*70}")
    print("EVALUACIÓN K-SHOT COMPLETADA")
    print(f"{'='*70}")
    print(f"  Mejor modelo : K={K_mejor}  (mAP@0.5 = {mAP_mejor:.4f})")
    print(f"  Guardado como: models/modelo_fewshot.pt")
    print(f"\n  Siguiente paso: python scripts/04_comparacion_final.py")
