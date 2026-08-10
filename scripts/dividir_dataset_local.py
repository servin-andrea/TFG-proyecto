"""
División del dataset local
--path es opcional, por defecto es datasets/
Usos: python dividir_dataset_local.py --path ruta/que/contiene/al/dataset_raw
      python dividir_dataset_local.py
"""

import os
import argparse
import random, shutil
from pathlib import Path

def copiar(out_path, labels_path, imgs, split):
        dst_img = out_path / split / 'images'
        dst_lbl = out_path / split / 'labels'
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img in imgs:
            shutil.copy2(img, dst_img / img.name)
            lbl = labels_path / (img.stem + '.txt')
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)
            else:
                (dst_lbl / (img.stem + '.txt')).write_text('')

def dividir_dataset(dataset_path: str):
    BASE      = Path(dataset_path)
    RAW       = BASE / 'local_raw/train'
    OUT       = BASE / 'local_split'
    IMAGES_IN = RAW / 'images'
    LABELS_IN = RAW / 'labels'
    
    random.seed(42)
    
    # Cuenta la cantidad de imagenes locales
    ext   = {'.jpg', '.jpeg', '.png'}
    todas = [f for f in IMAGES_IN.iterdir() if f.suffix.lower() in ext]
    print(f'Total imagenes: {len(todas)}')

    # Cuenta la cantidad de imagenes por clase
    helmet, nohelmet, mixed = [], [], []
    for img in todas:
        lbl = LABELS_IN / (img.stem + '.txt')
        if not lbl.exists():
            mixed.append(img)
            continue
        clases = set()
        with open(lbl) as f:
            for linea in f:
                p = linea.strip().split()
                if p: clases.add(int(p[0]))
        if clases == {0}:   helmet.append(img)
        elif clases == {1}: nohelmet.append(img)
        else:               mixed.append(img)

    # Mezclamos...
    random.shuffle(helmet)
    random.shuffle(nohelmet)

    # Empezamos la particion
    # En local_train se necesitan 60 imagenes, K_MAX=30
    # El resto va a local_test
    K_MAX = 30
    N_TRAIN = K_MAX * 2

    train_imgs = helmet[:K_MAX] + nohelmet[:K_MAX]

    if len(train_imgs) < N_TRAIN:
        random.shuffle(mixed)
        train_imgs += mixed[:N_TRAIN - len(train_imgs)]

    train_set = {img.name for img in train_imgs}
    test_imgs  = [img for img in todas if img.name not in train_set]

    copiar(OUT, LABELS_IN, train_imgs, 'local_train')
    copiar(OUT, LABELS_IN, test_imgs,  'local_test')

    # Escribimos el data.yaml
    for split in ['local_train', 'local_test']:
        sp = OUT / split
        yaml = (
            f'path: {sp.resolve()}\n'
            f'train: images\nval: images\n'
            f'nc: 2\nnames:\n  0: helmet\n  1: no_helmet\n'
        )
        (sp / 'data.yaml').write_text(yaml, encoding='utf-8')

    print(f'\nlocal_train : {len(train_imgs)} imagenes (30 helmet + 30 no_helmet)')
    print(f'local_test  : {len(test_imgs)} imagenes')
    print(f'Guardado en : {OUT}')
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dividir dataset local")
    parser.add_argument("--path", default='datasets', required=False, help="Ruta que contiene al dataset")
    args = parser.parse_args()
    
    dividir_dataset(args.path)

