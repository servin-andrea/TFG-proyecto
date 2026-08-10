"""
Análisis inicial de un dataset YOLO
Uso: python analizar_dataset.py --path ruta/al/dataset
"""

import os
import argparse
from pathlib import Path
from collections import Counter

def analizar_dataset(dataset_path: str, nombres_clases: list = None):
    """
    Analiza un dataset en formato YOLO y muestra estadísticas.
    Espera la estructura:
        dataset/
            images/  (o train/images, val/images)
            labels/  (o train/labels, val/labels)
    """
    base = Path(dataset_path)

    # Buscar carpetas de etiquetas
    carpetas_labels = []
    for sub in ["labels", "train/labels", "local_train/labels", "val/labels", "test/labels", "local_test/labels"]:
        p = base / sub
        if p.exists():
            carpetas_labels.append(p)

    if not carpetas_labels:
        print(f"[ERROR] No se encontraron carpetas 'labels' en {dataset_path}")
        return

    total_imagenes = 0
    contador_clases = Counter()
    total_bboxes = 0

    for carpeta in carpetas_labels:
        archivos_txt = list(carpeta.glob("*.txt"))
        total_imagenes += len(archivos_txt)

        for txt in archivos_txt:
            with open(txt, "r") as f:
                lineas = [l.strip() for l in f.readlines() if l.strip()]
            for linea in lineas:
                partes = linea.split()
                if len(partes) >= 5:
                    clase_id = int(partes[0])
                    contador_clases[clase_id] += 1
                    total_bboxes += 1

    print("\n" + "="*50)
    print(f"ANÁLISIS: {base.name}")
    print("="*50)
    print(f"  Total imágenes (con etiqueta): {total_imagenes}")
    print(f"  Total bounding boxes:          {total_bboxes}")
    print(f"  Promedio BBoxes por imagen:    {total_bboxes/max(total_imagenes,1):.2f}")
    print()
    print("  Distribución de clases:")

    ids_ordenados = sorted(contador_clases.keys())
    for clase_id in ids_ordenados:
        count = contador_clases[clase_id]
        nombre = nombres_clases[clase_id] if nombres_clases and clase_id < len(nombres_clases) else f"clase_{clase_id}"
        porcentaje = count / total_bboxes * 100 if total_bboxes > 0 else 0
        barra = "█" * int(porcentaje / 2)
        print(f"    [{clase_id}] {nombre:<15} {count:>5} ({porcentaje:5.1f}%)  {barra}")

    print()

    # Advertencias de balance
    if len(contador_clases) >= 2:
        counts = list(contador_clases.values())
        ratio = max(counts) / max(min(counts), 1)
        if ratio > 3:
            print(f"  ⚠️  Dataset desbalanceado (ratio {ratio:.1f}x). Considerar oversampling.")
        else:
            print(f"  ✅ Balance aceptable entre clases (ratio {ratio:.1f}x)")

    return {
        "imagenes": total_imagenes,
        "bboxes": total_bboxes,
        "clases": dict(contador_clases)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza un dataset YOLO")
    parser.add_argument("--path", required=True, help="Ruta al dataset")
    parser.add_argument("--clases", nargs="+", default=["helmet", "no_helmet"],
                        help="Nombres de clases (por orden de ID)")
    args = parser.parse_args()

    analizar_dataset(args.path, args.clases)
