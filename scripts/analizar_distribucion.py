"""
FASE 3 — Analizar distribución de un dataset
Analiza un dataset en formato YOLO y genera:
    - Distribución de clases (helmet / no_helmet)
    - Clasificación por tipo de imagen (solo helmet, solo no_helmet, mixtas)
    - Balance entre clases
    - Gráficos de distribución

Uso:
    python scripts/analizar_distribucion.py --path path/al/dataset

Requisitos previos:
    - /train/images/
    - /train/labels/

Resultado en results/analisis_<nombre_carpeta>/:
    distribucion.json
    distribucion_tipo_y_clases.png
    distribucion_areas_bboxes.png
"""

import argparse
import json
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
 
NOMBRES_CLASES = {0: "helmet", 1: "no_helmet"}
COLORES = {
    "helmet" : "#1D9E75",
    "no_helmet" : "#E24B4A",
    "mixta" : "#EF9F27",
    "vacia" : "#94A3B8",
}
 
## Funciones auxiliares
 
def analizar(dataset_path: Path, results_dir: Path):
 
    images_dir = dataset_path / "images"
    labels_dir = dataset_path / "labels"
 
    if not images_dir.exists():
        raise SystemExit(f" No se encontró: {images_dir}")
 
    ext = {".jpg", ".jpeg", ".png"}
    imagenes = sorted([f for f in images_dir.iterdir() if f.suffix.lower() in ext])
    total_imgs = len(imagenes)
 
    if total_imgs == 0:
        raise SystemExit(f" No hay imágenes en: {images_dir}")
 
    print("=" * 60)
    print(f"ANÁLISIS: {dataset_path}")
    print("=" * 60)
    print(f"\n  Total imágenes: {total_imgs}")
 
    # Recorrer imágenes
 
    solo_helmet   = []
    solo_nohelmet = []
    mixtas        = []
    vacias        = []
 
    total_bboxes    = 0
    bboxes_helmet   = 0
    bboxes_nohelmet = 0
    areas_helmet    = []
    areas_nohelmet  = []
 
    for img_path in imagenes:
        lbl_path = labels_dir / (img_path.stem + ".txt")
 
        if not lbl_path.exists():
            vacias.append(img_path.name)
            continue
 
        lineas = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
 
        if not lineas:
            vacias.append(img_path.name)
            continue
 
        clases_img = set()
        for linea in lineas:
            partes = linea.split()
            if len(partes) < 5:
                continue
            cls_id = int(partes[0])
            w = float(partes[3])
            h = float(partes[4])
            area = w * h
 
            clases_img.add(cls_id)
            total_bboxes += 1
 
            if cls_id == 0:
                bboxes_helmet += 1
                areas_helmet.append(area)
            elif cls_id == 1:
                bboxes_nohelmet += 1
                areas_nohelmet.append(area)
 
        if clases_img == {0}:
            solo_helmet.append(img_path.name)
        elif clases_img == {1}:
            solo_nohelmet.append(img_path.name)
        elif len(clases_img) > 1:
            mixtas.append(img_path.name)
 
    # Estadísticas
 
    ratio = (
        max(bboxes_helmet, bboxes_nohelmet) /
        max(min(bboxes_helmet, bboxes_nohelmet), 1)
    )
 
    print(f"\n{'─'*60}")
    print("DISTRIBUCIÓN POR TIPO DE IMAGEN")
    print(f"{'─'*60}")
    print(f"  Solo helmet    : {len(solo_helmet):>4}  ({len(solo_helmet)/total_imgs*100:.1f}%)")
    print(f"  Solo no_helmet : {len(solo_nohelmet):>4}  ({len(solo_nohelmet)/total_imgs*100:.1f}%)")
    print(f"  Mixtas         : {len(mixtas):>4}  ({len(mixtas)/total_imgs*100:.1f}%)")
    print(f"  Sin etiqueta   : {len(vacias):>4}  ({len(vacias)/total_imgs*100:.1f}%)")
    print(f"  Total          : {total_imgs:>4}")
 
    print(f"\n{'─'*60}")
    print("BOUNDING BOXES")
    print(f"{'─'*60}")
    print(f"  Total          : {total_bboxes}")
    print(f"  helmet         : {bboxes_helmet:>4}  ({bboxes_helmet/max(total_bboxes,1)*100:.1f}%)")
    print(f"  no_helmet      : {bboxes_nohelmet:>4}  ({bboxes_nohelmet/max(total_bboxes,1)*100:.1f}%)")
    print(f"  Ratio balance  : {ratio:.2f}x  ", end="")
    if ratio <= 1.5:
        print(" Bien balanceado")
    elif ratio <= 3.0:
        print(" Desbalance moderado")
    else:
        print(" Desbalanceado")
    print(f"  Promedio bboxes/img: {total_bboxes/total_imgs:.2f}")
 
    if areas_helmet or areas_nohelmet:
        print(f"\n{'─'*60}")
        print("TAMAÑO DE BBOXES (área normalizada w×h)")
        print(f"{'─'*60}")
        if areas_helmet:
            print(f"  helmet    media={np.mean(areas_helmet):.4f}  std={np.std(areas_helmet):.4f}  "
                  f"min={np.min(areas_helmet):.4f}  max={np.max(areas_helmet):.4f}")
        if areas_nohelmet:
            print(f"  no_helmet media={np.mean(areas_nohelmet):.4f}  std={np.std(areas_nohelmet):.4f}  "
                  f"min={np.min(areas_nohelmet):.4f}  max={np.max(areas_nohelmet):.4f}")
 
    # Guardar JSON
 
    results_dir.mkdir(parents=True, exist_ok=True)
 
    resumen = {
        "dataset" : str(dataset_path),
        "total_imagenes" : total_imgs,
        "solo_helmet" : len(solo_helmet),
        "solo_no_helmet" : len(solo_nohelmet),
        "mixtas" : len(mixtas),
        "vacias" : len(vacias),
        "total_bboxes" : total_bboxes,
        "bboxes_helmet" : bboxes_helmet,
        "bboxes_no_helmet" : bboxes_nohelmet,
        "ratio_balance" : round(ratio, 2),
        "promedio_bboxes_img" : round(total_bboxes / total_imgs, 2),
        "area_media_helmet" : round(float(np.mean(areas_helmet)),    4) if areas_helmet    else 0,
        "area_media_no_helmet" : round(float(np.mean(areas_nohelmet)),  4) if areas_nohelmet  else 0,
    }
    json_path = results_dir / "distribucion.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)
 
    # GRÁFICO 1: Pie + Barras
 
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    titulo = f"Distribución — {dataset_path.name} ({total_imgs} imágenes)"
    fig.suptitle(titulo, fontsize=12, fontweight="bold")
 
    tipos   = ["Solo helmet", "Solo no_helmet", "Mixtas", "Sin etiqueta"]
    valores = [len(solo_helmet), len(solo_nohelmet), len(mixtas), len(vacias)]
    colores_pie = [COLORES["helmet"], COLORES["no_helmet"], COLORES["mixta"], COLORES["vacia"]]
    filtrado = [(t, v, c) for t, v, c in zip(tipos, valores, colores_pie) if v > 0]
 
    if filtrado:
        t_f, v_f, c_f = zip(*filtrado)
        wedges, texts, autotexts = axes[0].pie(
            v_f, labels=t_f, colors=c_f,
            autopct="%1.1f%%", startangle=90, pctdistance=0.78,
        )
        for at in autotexts:
            at.set_fontsize(10)
            at.set_fontweight("bold")
    axes[0].set_title("Por tipo de imagen", fontweight="bold")
 
    bars = axes[1].bar(
        ["helmet", "no_helmet"],
        [bboxes_helmet, bboxes_nohelmet],
        color=[COLORES["helmet"], COLORES["no_helmet"]],
        edgecolor="white", linewidth=0.8, width=0.5,
    )
    max_val = max(bboxes_helmet, bboxes_nohelmet)
    for bar, val in zip(bars, [bboxes_helmet, bboxes_nohelmet]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_val * 0.01,
            str(val), ha="center", va="bottom",
            fontsize=12, fontweight="bold",
        )
    axes[1].set_title("Bounding boxes por clase", fontweight="bold")
    axes[1].set_ylabel("Cantidad")
    axes[1].set_ylim(0, max_val * 1.15)
    axes[1].grid(axis="y", alpha=0.3)
 
    plt.tight_layout()
    g1 = results_dir / "distribucion_tipo_y_clases.png"
    plt.savefig(g1, dpi=150, bbox_inches="tight")
    plt.close()
 
    # GRÁFICO 2: Histograma de áreas
 
    if areas_helmet or areas_nohelmet:
        fig, ax = plt.subplots(figsize=(9, 4))
        if areas_helmet:
            ax.hist(areas_helmet, bins=20, alpha=0.65,
                    color=COLORES["helmet"],
                    label=f"helmet ({len(areas_helmet)} bboxes)",
                    edgecolor="white")
        if areas_nohelmet:
            ax.hist(areas_nohelmet, bins=20, alpha=0.65,
                    color=COLORES["no_helmet"],
                    label=f"no_helmet ({len(areas_nohelmet)} bboxes)",
                    edgecolor="white")
        ax.set_xlabel("Área normalizada del bounding box (w × h)", fontsize=11)
        ax.set_ylabel("Frecuencia", fontsize=11)
        ax.set_title(f"Tamaño de bounding boxes — {dataset_path.name}",
                     fontweight="bold", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        g2 = results_dir / "distribucion_areas_bboxes.png"
        plt.savefig(g2, dpi=150, bbox_inches="tight")
        plt.close()
 
    print(f"Archivos en: {results_dir}")
    for a in sorted(results_dir.iterdir()):
        print(f"  {a.name}")
 
 
## Entry point
 
if __name__ == "__main__":
 
    BASE = Path(__file__).resolve().parent.parent
 
    parser = argparse.ArgumentParser(
        description="Análisis de distribución de cualquier dataset YOLO con /images y /labels"
    )
    parser.add_argument(
        "--path", required=True,
        help=(
            "Ruta a la carpeta que contiene images/ y labels/. Ejemplos:\n"
            "  datasets/local_raw/train\n"
            "  datasets/local_split_K/local_train\n"
            "  datasets/local_split_K/local_test\n"
            "  datasets/Dataset_A_YOLO/train"
        )
    )
    args = parser.parse_args()
 
    dataset_parent_path = Path(args.path).parent
    dataset_path = Path(args.path)
    if not dataset_path.is_absolute():
        dataset_path = BASE / dataset_path
 
    # Carpeta de resultados: results/analisis_<nombre_parent>_<nombre>/
    results_dir = BASE / "results" / f"analisis_{dataset_parent_path.name}_{dataset_path.name}"
 
    analizar(dataset_path, results_dir)