# Guía completa de reproducción del experimento

**Aplicación del Enfoque Few-Shot Learning para la Detección de Motociclistas sin Casco en Imágenes Adaptadas al Entorno Urbano Local**  
Carlos Cristobal Torres Carballo · Andrea Inés Servín Mendoza  
Tutor: Ing. Victor Andrés Zorrilla Villanueva

---

## Índice

1. [Contexto del experimento](#1-contexto-del-experimento)
2. [Entorno de desarrollo](#2-entorno-de-desarrollo)
3. [Estructura de carpetas](#3-estructura-de-carpetas)
4. [Etapa 1 — Preparación de los datos](#4-etapa-1--preparación-de-los-datos)
5. [Etapa 2 — Entrenamientos y adaptación](#5-etapa-2--entrenamientos-y-adaptación)
6. [Etapa 3 — Evaluaciones y conclusiones](#6-etapa-3--evaluaciones-y-conclusiones)
7. [Archivos generados](#7-archivos-generados)
8. [Troubleshooting](#8-troubleshooting)
9. [Decisiones metodológicas clave](#9-decisiones-metodológicas-clave)

---

## 1. Contexto del experimento

### Hipótesis

> *"Un modelo de detección entrenado con datasets generales no mantiene el mismo rendimiento cuando se aplica en imágenes de un entorno urbano local, pero puede mejorar su desempeño mediante la aplicación de Few-Shot Learning con un número reducido de imágenes representativas del entorno objetivo."*

### Paradigma utilizado

**2-way K-Shot Learning** donde:
- **N = 2** clases: `helmet` y `no_helmet`
- **K = 10, 20, 30** ejemplos por clase en el fine-tuning

### Resultados obtenidos

| Etapa | mAP@0.5 | F1-Score |
|-------|---------|----------|
| Modelo base — dominio público | 0.8885 | 0.8440 |
| Modelo base — dominio local (sin FSL) | 0.5887 | 0.6143 |
| Few-Shot K=10 | 0.6960 | 0.6842 |
| Few-Shot K=20 | 0.7861 | 0.7645 |
| Few-Shot K=30 | 0.8146 | 0.8013 |
| **Recuperación del domain shift** | **75.4%** | **80.6%** |

---

## 2. Entorno de desarrollo

### Hardware utilizado

| Componente | Especificación |
|------------|---------------|
| SO | Windows 11 |
| GPU | NVIDIA GeForce GTX 1660 Ti (6 GB VRAM) |
| RAM | 16 GB |
| Python | 3.12.6 |
| PyTorch | 2.5.1 + CUDA 12.1 |
| Ultralytics | YOLOv8 8.4.75 |

### Configuración del entorno

```powershell
# 1. Crear entorno virtual
python -m venv tfg-venv
tfg-venv\Scripts\activate

# 2. Instalar PyTorch con CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar
python -c "import torch; print(torch.cuda.is_available())"
```

### Notas importantes para Windows

- Todos los scripts usan `workers=0` en el DataLoader — es obligatorio en Windows para evitar errores de multiprocesamiento
- Los scripts deben ejecutarse con el bloque `if __name__ == '__main__':` — requerido por el sistema de multiprocesamiento de Windows
- Se desactivó AMP (Automatic Mixed Precision) por incompatibilidades con la GTX 1660 Ti

---

## 3. Estructura de carpetas

```
TFG_Proyecto/
├── scripts/                    ← scripts Python del pipeline
├── datasets/
│   ├── Dataset_A_raw/          ← dataset A descargado de Roboflow
│   ├── Dataset_B_raw/          ← dataset B descargado de Roboflow
│   ├── Dataset_C_raw/          ← dataset C descargado de Roboflow
│   ├── Dataset_A_YOLO/         ← dataset A normalizado
│   ├── Dataset_B_YOLO/         ← dataset B normalizado
│   ├── Dataset_C_YOLO/         ← dataset C normalizado
│   ├── local_raw/              ← imágenes locales originales
│   └── local_split/
│       ├── local_train/        ← 60 imgs para fine-tuning (30+30)
│       └── local_test/         ← 90 imgs para evaluación (NO TOCAR)
├── models/
│   ├── modelo_A.pt             ← modelo base entrenado con Dataset A
│   ├── modelo_B.pt             ← modelo base entrenado con Dataset B
│   ├── modelo_C.pt             ← modelo base entrenado con Dataset C
│   ├── modelo_base.pt          ← mejor modelo base (copia del mejor)
│   ├── modelo_fewshot_K10.pt   ← modelo K-Shot K=10
│   ├── modelo_fewshot_K20.pt   ← modelo K-Shot K=20
│   ├── modelo_fewshot_K30.pt   ← modelo K-Shot K=30
│   └── modelo_fewshot.pt       ← mejor modelo K-Shot (copia del mejor)
├── results/                    ← CSVs, JSONs y gráficos generados
├── runs/                       ← logs de entrenamiento YOLOv8
├── outputs/                    ← imágenes de inferencia
├── TFG_Pipeline_Notebook.ipynb
├── requirements.txt
├── .gitignore
├── README.md
└── GUIA_COMPLETA.md
```

> **Importante:** `local_test` nunca debe usarse en entrenamiento. Es el conjunto de evaluación fijo para todas las fases del experimento.

---

## 4. Etapa 1 — Preparación de los datos

### Fase 1 — Análisis y normalización de datasets públicos

#### Datasets utilizados

| Dataset | Fuente Roboflow | Imágenes | Clases originales |
|---------|----------------|----------|-------------------|
| Dataset A | projectt/motorcycle-helmet-object-detection | 3 607 | With Helmet, Without Helmet |
| Dataset B | programa-delfn/helmet-no-helmet-detection-hjdvx | 1 909 | with_helmet, without_helmet |
| Dataset C | gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection | 1 533 | With Helmet, Without Helmet, licence |

#### Descarga

1. Ir a cada URL en Roboflow Universe
2. Seleccionar formato **YOLOv8**
3. Descomprimir en `datasets/Dataset_X_raw/`

#### Normalización

```powershell
python scripts/normalizar_dataset.py
```

**Qué hace:**
- Reasigna IDs de clase al esquema estándar: `0 = helmet`, `1 = no_helmet`
- Descarta clases irrelevantes (ej: `licence` del Dataset C)
- Detecta automáticamente el nombre del split de validación (`valid/` o `val/`)
- Genera `data.yaml` con rutas absolutas para cada dataset

**Resultado:** `datasets/Dataset_A_YOLO/`, `Dataset_B_YOLO/`, `Dataset_C_YOLO/`

#### Análisis de distribución (opcional)

```powershell
python scripts/analizar_dataset.py --path datasets/Dataset_A_raw
python scripts/analizar_dataset.py --path datasets/Dataset_B_raw
python scripts/analizar_dataset.py --path datasets/Dataset_C_raw
```

---

### Fase 2 — Construcción y división del dataset local

#### Estructura esperada del dataset local

```
datasets/local_raw/train/
├── images/   ← imágenes .jpg/.png capturadas en Encarnación
└── labels/   ← anotaciones .txt en formato YOLO (0=helmet, 1=no_helmet)
```

#### Análisis de distribución

```powershell
python scripts/analizar_distribucion.py --path datasets/local_raw/train
```

**Resultado:** `results/analisis_train/distribucion.json` y gráficos

#### División train/test

```powershell
python scripts/dividir_dataset_local.py
```

**Qué hace:**
- Divide las 150 imágenes en `local_train` (60 imgs: 30+30) y `local_test` (90 imgs)
- Usa `seed=42` para reproducibilidad
- Garantiza balance de clases en `local_train`

**Resultado:** `datasets/local_split/local_train/` y `datasets/local_split/local_test/`

> **Atención:** Una vez ejecutado este script, no volver a ejecutarlo ni mover imágenes entre carpetas manualmente. La separación debe permanecer fija durante todo el experimento.

---

## 5. Etapa 2 — Entrenamientos y adaptación

### Fase 3 — Entrenamiento y selección del modelo base

#### Hiperparámetros del entrenamiento base

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| model | yolov8n.pt | Preentrenado en COCO (80 clases, incluye motos y personas) |
| epochs | 50 | Valor estándar con early stopping |
| batch | 8 | Adaptado a 6 GB VRAM |
| imgsz | 640 | Resolución estándar YOLOv8 |
| lr0 | 0.01 | Valor por defecto documentado (Jocher et al., 2023) |
| lrf | 0.01 | LR final = lr0 × lrf = 0.0001 |
| patience | 15 | Early stopping |
| mosaic | 0.0 | Desactivado para igualar condiciones con fine-tuning |
| cos_lr | True | Scheduler coseno (Loshchilov & Hutter, 2017) |
| workers | 0 | Obligatorio en Windows |
| seed | 42 | Reproducibilidad |

#### Entrenamiento

```powershell
python scripts/entrenar_modelos_base.py
```

⏱️ **Tiempo estimado:** ~5h 46min total (GTX 1660 Ti)
- Modelo A: 2h 29min
- Modelo B: 2h 16min
- Modelo C: 1h 01min

**Resultado:** `models/modelo_A.pt`, `modelo_B.pt`, `modelo_C.pt`

#### Selección del modelo base

```powershell
python scripts/seleccionar_modelo_base.py
```

**Qué hace:**
- Lee los CSVs de `results/eval/eval_modelo_X.csv`
- Selecciona el modelo con mayor mAP@0.5
- Copia el mejor como `models/modelo_base.pt`
- Genera `results/modelo_base_seleccionado.json`
- Genera `results/fase4_comparacion_barras.png`

**Resultado:** `models/modelo_base.pt` (en el experimento: Modelo A con mAP@0.5 = 0.8885)

---

### Fase 4 — Entrenamiento K-Shot y selección del modelo Few-Shot

#### Hiperparámetros del fine-tuning K-Shot

| Parámetro | Base | Fine-tuning | Justificación del cambio |
|-----------|------|-------------|--------------------------|
| epochs | 50 | 50 | Igual |
| batch | 8 | 8 | Igual |
| imgsz | 640 | 640 | Igual — comparabilidad garantizada |
| lr0 | 0.01 | 0.001 | 10× menor — evita olvido catastrófico |
| patience | 15 | 20 | Mayor paciencia con pocos datos |
| freeze | — | 10 | Congela backbone, solo reentrena cabeza |
| mosaic | 0.0 | 0.0 | Igual |
| cos_lr | True | True | Igual |

#### Por qué freeze=10

YOLOv8n tiene 22 capas. Congelar las primeras 10 preserva:
- El conocimiento de COCO (personas, motos, contexto vial) — capas 1-5
- El conocimiento de helmet/no_helmet aprendido con datasets públicos — capas 6-10

Solo la cabeza de detección (capas 11-22) se adapta al dominio local.

#### Entrenamiento

```powershell
python scripts/entrenar_modelos_fewshot.py
```

⏱️ **Tiempo estimado:** ~30-40 min por K (3 experimentos = ~2h total)

**Resultado:** `models/modelo_fewshot_K10.pt`, `K20.pt`, `K30.pt`

#### Evaluación y selección

```powershell
python scripts/evaluar_fewshot.py
```

**Qué hace:**
- Evalúa cada modelo K-Shot sobre `local_test` (90 imgs)
- Genera `results/fase9_Kshot_curva.csv`
- Genera `results/fase9_Kshot_resultados.json`
- Genera `results/fase9_Kshot_curva.png`
- Copia el mejor como `models/modelo_fewshot.pt`

**Resultado:** `models/modelo_fewshot.pt` (en el experimento: K=30 con mAP@0.5 = 0.8146)

---

## 6. Etapa 3 — Evaluaciones y conclusiones

### Fase 5 — Evaluación del modelo base en entorno local

```powershell
python scripts/evaluar_base_local.py
```

**Qué hace:**
- Evalúa `modelo_base.pt` sobre `local_test` sin adaptación previa
- Cuantifica el domain shift
- Genera `results/fase8_K_domain_shift.json`

**Cómo calcular la recuperación del domain shift:**

```
Caída    = mAP_público − mAP_base_local = 0.8885 − 0.5887 = 0.2998
Mejora   = mAP_fewshot − mAP_base_local = 0.8146 − 0.5887 = 0.2259
Recuperación % = (Mejora / Caída) × 100 = (0.2259 / 0.2998) × 100 = 75.3%
```

---

### Fase 6 — Comparativa y análisis

#### Comparativa final

```powershell
python scripts/comparacion_final.py
```

**Genera:**
- `results/fase10_comparacion_final.csv`
- `results/fase10_datos_comparacion.json`
- `results/fase11_comparacion_final.png`

#### Conclusión

```powershell
python scripts/conclusion.py
```

**Genera:**
- `results/fase12_conclusion.json`
- `results/fase12_domain_shift_recuperacion.png`

#### Inferencia con el modelo final

```powershell
# Sobre una imagen
python scripts/detectar_no_helmet.py --source ruta/imagen.jpg

# Sobre una carpeta
python scripts/detectar_no_helmet.py --source datasets/validacion/

# Sobre un video
python scripts/detectar_no_helmet.py --source ruta/video.mp4

# Con umbral de confianza personalizado
python scripts/detectar_no_helmet.py --source ruta/ --conf 0.5
```

---

## 7. Archivos generados

### Modelos

| Archivo | Descripción |
|---------|-------------|
| `models/modelo_A.pt` | Modelo base entrenado con Dataset A |
| `models/modelo_B.pt` | Modelo base entrenado con Dataset B |
| `models/modelo_C.pt` | Modelo base entrenado con Dataset C |
| `models/modelo_base.pt` | Mejor modelo base seleccionado |
| `models/modelo_fewshot_K10.pt` | Few-Shot K=10 |
| `models/modelo_fewshot_K20.pt` | Few-Shot K=20 |
| `models/modelo_fewshot_K30.pt` | Few-Shot K=30 |
| `models/modelo_fewshot.pt` | Mejor modelo Few-Shot seleccionado |

### Resultados

| Archivo | Descripción |
|---------|-------------|
| `results/fase4_comparacion_base.csv` | Métricas comparativas de los 3 modelos base |
| `results/fase4_comparacion_barras.png` | Gráfico comparativo modelos base |
| `results/modelo_base_seleccionado.json` | Métricas del modelo base seleccionado |
| `results/fase8_K_domain_shift.json` | Evidencia cuantitativa del domain shift |
| `results/fase9_Kshot_curva.csv` | Métricas curva K-Shot (K=10,20,30) |
| `results/fase9_Kshot_curva.png` | Gráfico curva K-Shot |
| `results/fase9_Kshot_resultados.json` | Resultados detallados K-Shot |
| `results/fase10_comparacion_final.csv` | Comparativa Base vs. Few-Shot |
| `results/fase10_datos_comparacion.json` | Datos comparación con deltas |
| `results/fase11_comparacion_final.png` | Gráfico comparativo final |
| `results/fase12_conclusion.json` | Conclusión experimental |
| `results/fase12_domain_shift_recuperacion.png` | Gráfico domain shift y recuperación |

---

## 8. Troubleshooting

### Error: `RuntimeError: DataLoader worker ... exited unexpectedly`

**Causa:** `workers > 0` en Windows.  
**Solución:** Verificar que todos los scripts tienen `workers=0`.

### Error: `CUDA out of memory`

**Causa:** Memoria VRAM insuficiente.  
**Solución:** Reducir `batch` de 8 a 4 en el script correspondiente. No afecta las métricas finales, solo la velocidad.

### Error: `FileNotFoundError: data.yaml not found`

**Causa:** El dataset no fue normalizado o la ruta es incorrecta.  
**Solución:** Ejecutar `normalizar_dataset.py` primero y verificar que existen las carpetas `Dataset_X_YOLO/`.

### Warning: `FigureCanvasAgg is non-interactive`

**Causa:** `plt.show()` no funciona en Jupyter con backend Agg.  
**Solución:** Usar `display(Image(filename=str(path)))` en lugar de `plt.show()`. Ya corregido en el notebook.

### Error al entrenar: `assert num_workers == 0`

**Causa:** Conflicto de multiprocesamiento en Windows.  
**Solución:** Asegurarse de que el script tiene `if __name__ == '__main__':` como bloque principal.

### El modelo no detecta bien en imágenes nuevas

**Causa probable 1:** Umbral de confianza muy bajo — genera falsos positivos.  
**Solución:** Subir `--conf` de 0.45 a 0.55 o 0.60.

**Causa probable 2:** Las imágenes nuevas son de un dominio muy diferente al local_train.  
**Solución:** Agregar más imágenes representativas al local_train y repetir el fine-tuning.

---

## 9. Decisiones metodológicas clave

### Por qué YOLOv8n y no una arquitectura más grande

Con 6 GB de VRAM, las arquitecturas YOLOv8s (small) o YOLOv8m (medium) requieren reducir el batch a 4 o 2, lo que perjudica la convergencia. YOLOv8n con batch=8 e imgsz=640 usa ~4.5-5 GB y ofrece el mejor balance para el hardware disponible.

### Por qué imgsz=640 y no 416

640px es la resolución estándar de YOLOv8 y tiene impacto directo en mAP@0.5:95 (precisión geométrica de los bounding boxes). Se mantuvo igual en entrenamiento base y fine-tuning para garantizar comparabilidad.

### Por qué mosaic=0.0

Mosaic combina 4 imágenes en una. Con 60 imágenes en local_train, Mosaic recorta y distorsiona las imágenes locales más de lo que ayuda, perjudicando el aprendizaje de las características del dominio local. Se desactivó en ambas etapas para mantener condiciones igualadas.

### Por qué freeze=10 y no freeze=4

Con freeze=4 el backbone tiene más libertad para modificarse, pero con solo 60 imágenes locales el riesgo de olvido catastrófico es alto. freeze=10 preserva el conocimiento general (COCO + dominio público) y solo adapta la cabeza de detección al dominio local.

### Por qué lr0=0.001 en el fine-tuning

Un LR 10× menor que el del entrenamiento base (0.01) garantiza actualizaciones de pesos sutiles que adaptan el modelo sin destruir lo aprendido. Con cos_lr=True el LR decae suavemente hasta 0.001 × 0.01 = 0.00001 al final del entrenamiento.

### Por qué local_test son 90 imágenes y no más

La división 60/90 garantiza que hay suficientes imágenes para evaluar con significancia estadística (90 imgs) mientras que local_train tiene exactamente las imágenes necesarias para K=30 por clase (60 imgs = 30+30). Esta proporción fue fija desde el inicio del experimento.

---

## Referencias

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (v8.0.0). GitHub. https://github.com/ultralytics/ultralytics
- Ultralytics. (2023). *YOLOv8 Documentation — Train*. https://docs.ultralytics.com/modes/train/
- Lin, T.-Y., et al. (2014). Microsoft COCO: Common objects in context. *ECCV 2014*. https://doi.org/10.1007/978-3-319-10602-1_48
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR 2017*. https://doi.org/10.48550/arXiv.1608.03983
