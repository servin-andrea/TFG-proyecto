"""
FASE 2.2 — Normalización de clases y estructura YOLO
Convierte IDs de clases originales al estándar del proyecto:
    0 → helmet
    1 → no_helmet

Uso:
    python normalizar_dataset.py --input ruta/dataset_original –-output 
    ruta/dataset_normalizado --mapa 2:0 3:1
        # Ejemplo: la clase 2 original → clase 0 (helmet)
        #          la clase 3 original → clase 1 (no_helmet)
"""

import os
import shutil
import argparse
from pathlib import Path

## Definicion de funciones auxiliares

def parsear_mapa(lista_mapa: list) -> dict:
    """
    Convierte lista ["2:0", "3:1"] en dict {2: 0, 3: 1}
    """
    mapa = {}
    for item in lista_mapa:
        partes = item.split(":")
        if len(partes) == 2:
            original = int(partes[0])
            nuevo = int(partes[1])
            mapa[original] = nuevo
    return mapa

def normalizar_etiqueta(txt_path: Path, mapa_clases: dict) -> list:
    """
    Lee un archivo .txt YOLO y remapea los IDs de clase.
    Devuelve las líneas transformadas (o vacío si la clase no está en el mapa).
    """
    lineas_nuevas = []
    with open(txt_path, "r") as f:
        for linea in f:
            linea = linea.strip()
            if not linea:
                continue
            partes = linea.split()
            clase_original = int(partes[0])
            if clase_original in mapa_clases:
                clase_nueva = mapa_clases[clase_original]
                partes[0] = str(clase_nueva)
                lineas_nuevas.append(" ".join(partes))
            # Si la clase no está en el mapa → se descarta
    return lineas_nuevas

def normalizar_split(input_split: Path, output_split: Path, mapa_clases: dict):
    """
    Procesa un split (train/val/test) copiando imágenes y normalizando etiquetas.
    """
    input_images = input_split / "images"
    input_labels = input_split / "labels"

    output_images = output_split / "images"
    output_labels = output_split / "labels"
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    # Extensiones de imagen soportadas
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    archivos_imagen = [f for f in input_images.iterdir()
                       if f.suffix.lower() in extensiones] if input_images.exists() else []

    copiadas = 0
    descartadas = 0

    for img_path in archivos_imagen:
        label_path = input_labels / (img_path.stem + ".txt")

        if not label_path.exists():
            # Sin etiqueta → copiar de todas formas (imagen vacía, sin objetos)
            shutil.copy2(img_path, output_images / img_path.name)
            (output_labels / (img_path.stem + ".txt")).write_text("")
            copiadas += 1
            continue

        lineas_nuevas = normalizar_etiqueta(label_path, mapa_clases)

        # Copiar imagen
        shutil.copy2(img_path, output_images / img_path.name)

        # Guardar etiqueta normalizada
        with open(output_labels / (img_path.stem + ".txt"), "w") as f:
            f.write("\n".join(lineas_nuevas))
            if lineas_nuevas:
                f.write("\n")

        copiadas += 1

    return copiadas, descartadas

def crear_yaml(output_path: Path, nombre: str):
    """
    Genera el archivo data.yaml necesario para entrenamiento YOLO.
    """
    yaml_content = (
        f"# Dataset: {nombre}\n"
        f"# Generado automáticamente por 02_normalizar_dataset.py\n\n"
        f"path: {output_path.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n\n"
        f"nc: 2\n"
        f"names:\n"
        f"  0: helmet\n"
        f"  1: no_helmet\n"
    )
    yaml_path = output_path / "data.yaml"
    yaml_path.write_text(yaml_content)
    print(f"    Generado: {yaml_path}")

def normalizar_dataset(input_path: str, output_path: str, mapa_clases: dict):
    base_in = Path(input_path)
    base_out = Path(output_path)
    base_out.mkdir(parents=True, exist_ok=True)

    nombre = base_out.name
    print(f"\n{'='*50}")
    print(f"NORMALIZANDO: {base_in.name} → {nombre}")
    print(f"Mapa de clases: {mapa_clases}")
    print("="*50)

    # Roboflow exporta "valid", YOLO estándar usa "val".
    # Mapeamos ambos nombres de entrada al nombre de salida correcto.
    SPLITS_BUSCAR = [
        ("train", "train"),   # (nombre en _raw, nombre en _YOLO)
        ("valid", "val"),     # Roboflow → estándar YOLO
        ("val",   "val"),     # ya estándar
        ("test",  "test"),
    ]

    splits_procesados = []
    for nombre_in, nombre_out in SPLITS_BUSCAR:
        carpeta_in = base_in / nombre_in
        if (carpeta_in / "images").exists():
            copiadas, _ = normalizar_split(carpeta_in, base_out / nombre_out, mapa_clases)
            print(f"  [{nombre_in} → {nombre_out}] {copiadas} imágenes procesadas")
            splits_procesados.append(nombre_out)

    if not splits_procesados:
        # Estructura plana: images/ y labels/ directamente en la raíz
        print("  Estructura detectada: plana (sin splits) → copiando como train")
        copiadas, _ = normalizar_split(base_in, base_out / "train", mapa_clases)
        print(f"  [train] {copiadas} imágenes procesadas")
        splits_procesados.append("train")

    crear_yaml(base_out, nombre)
    print(f"\n  Dataset normalizado en: {base_out}")

## Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normaliza clases de un dataset YOLO")
    parser.add_argument("--input",  required=True, help="Dataset original")
    parser.add_argument("--output", required=True, help="Dataset normalizado de salida")
    parser.add_argument(
        "--mapa", nargs="+", required=True,
        help="Mapeo original:nuevo. Ejemplo: --mapa 2:0 3:1"
    )
    args = parser.parse_args()

    mapa = parsear_mapa(args.mapa)
    normalizar_dataset(args.input, args.output, mapa)
