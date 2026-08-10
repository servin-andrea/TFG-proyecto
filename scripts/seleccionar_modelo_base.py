"""
Selección del modelo base desde CSVs existentes
================================================
Lee los resultados de evaluación ya generados en results/eval/
y selecciona el mejor modelo como modelo_base.pt.

No re-evalúa ningún modelo — usa los resultados ya existentes.

Requisitos previos:
    - results/eval/eval_modelo_A.csv
    - results/eval/eval_modelo_B.csv
    - results/eval/eval_modelo_C.csv
    - models/modelo_A.pt, modelo_B.pt, modelo_C.pt

Uso:
    python scripts/seleccionar_modelo_base.py
"""

import json
import shutil
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuración

BASE        = Path(__file__).resolve().parent.parent
MODELS_DIR  = BASE / "models"
RESULTS_DIR = BASE / "results"
EVAL_DIR    = RESULTS_DIR / "eval"

LETRAS = ["A", "B", "C"]

# ── Entry point

if __name__ == "__main__":

    # ── Verificar que existen los archivos

    print("Verificando archivos...")
    todo_ok = True
    for letra in LETRAS:
        csv    = EVAL_DIR   / f"eval_modelo_{letra}.csv"
        modelo = MODELS_DIR / f"modelo_{letra}.pt"
        print(f"  {'✅' if csv.exists()    else '❌'}  {csv.relative_to(BASE)}")
        print(f"  {'✅' if modelo.exists() else '❌'}  {modelo.relative_to(BASE)}")
        if not csv.exists() or not modelo.exists():
            todo_ok = False

    if not todo_ok:
        raise SystemExit("\n❌ Faltan archivos. Verificar rutas.")

    # ── Leer y combinar los tres CSVs

    print(f"\n{'='*60}")
    print("TABLA COMPARATIVA — MODELOS BASE")
    print(f"{'='*60}")

    filas = []
    for letra in LETRAS:
        csv = EVAL_DIR / f"eval_modelo_{letra}.csv"
        df_tmp = pd.read_csv(csv)
        filas.append(df_tmp.iloc[0])

    df = pd.DataFrame(filas).reset_index(drop=True)
    print(df.to_string(index=False))

    # ── Guardar CSV combinado

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_out = RESULTS_DIR / "fase3_comparacion_base.csv"
    df.to_csv(csv_out, index=False)
    print(f"\n✅ CSV combinado guardado en: {csv_out}")

    # ── Seleccionar el mejor modelo

    idx_mejor  = df["mAP@0.5"].idxmax()
    mejor_fila = df.iloc[idx_mejor]
    nombre_mejor = mejor_fila["Modelo"]

    # Extraer letra del nombre (ej: "modelo_A" → "A")
    letra_mejor = nombre_mejor.split("_")[-1].upper()
    mapa_mejor  = mejor_fila["mAP@0.5"]

    print(f"\n🏆 Mejor modelo: {nombre_mejor}  (mAP@0.5 = {mapa_mejor:.4f})")

    origen  = MODELS_DIR / f"modelo_{letra_mejor}.pt"
    destino = MODELS_DIR / "modelo_base.pt"
    shutil.copy2(origen, destino)
    print(f"✅ Copiado como: {destino}")

    # ── Guardar JSON de selección

    seleccion = {
        "modelo_seleccionado": nombre_mejor,
        "archivo"            : "modelo_base.pt",
        "Precision"          : float(mejor_fila["Precision"]),
        "Recall"             : float(mejor_fila["Recall"]),
        "F1-Score"           : float(mejor_fila["F1-Score"]),
        "mAP@0.5"            : float(mapa_mejor),
        "mAP@0.5:95"         : float(mejor_fila["mAP@0.5:95"]),
    }
    json_path = RESULTS_DIR / "modelo_base_seleccionado.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(seleccion, f, indent=2, ensure_ascii=False)
    print(f"✅ Selección guardada en: {json_path}")

    # ── Gráfico comparativo

    metricas_plot  = ["Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@0.5:95"]
    modelos_labels = df["Modelo"].tolist()
    colores        = ["#378ADD", "#1D9E75", "#EF9F27"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 5))
    fig.suptitle("Comparación de Modelos Base", fontsize=14, fontweight="bold")

    for i, metrica in enumerate(metricas_plot):
        valores = df[metrica].tolist()
        bars = axes[i].bar(
            modelos_labels, valores,
            color=colores, edgecolor="white", linewidth=0.8,
        )
        axes[i].set_title(metrica, fontweight="bold")
        axes[i].set_ylim(0, 1.1)
        axes[i].set_ylabel("Valor")
        axes[i].grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, valores):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.tight_layout()
    grafico_path = RESULTS_DIR / "fase3_comparacion.png"
    plt.savefig(grafico_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"✅ Gráfico guardado en: {grafico_path}")

    # ── Resumen final

    print(f"\n{'='*60}")
    print("SELECCIÓN DEL MODELO BASE COMPLETADA")
    print(f"{'='*60}")
    print(f"  Modelo base : models/modelo_base.pt  ({nombre_mejor})")
    print(f"  CSV         : results/fase3_comparacion_base.csv")
    print(f"  JSON        : results/modelo_base_seleccionado.json")
    print(f"  Gráfico     : results/fase3_comparacion.png")
