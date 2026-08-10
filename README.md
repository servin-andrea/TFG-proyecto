# Aplicación del Enfoque Few-Shot Learning para la Detección de Motociclistas sin Casco en Imágenes Adaptadas al Entorno Urbano Local

**Trabajo Final de Grado**  
Carlos Cristobal Torres Carballo · Andrea Inés Servín Mendoza  
Tutor: Ing. Victor Andrés Zorrilla Villanueva

---

## Descripción

Pipeline de detección de no uso de cascos en motociclistas aplicando Few-Shot Learning (paradigma 2-way K-Shot) para adaptar un modelo YOLOv8n preentrenado con datasets públicos al entorno urbano local de Encarnación, Paraguay.

El experimento demuestra y cuantifica el fenómeno de **domain shift** (caída del 33.7% en mAP@0.5 al cambiar de dominio) y su recuperación mediante fine-tuning con congelamiento parcial del backbone (**75.4% de recuperación** con K=30).

---

## Resultados principales

| Etapa | mAP@0.5 | F1-Score |
|-------|---------|----------|
| Modelo base (dominio público) | 0.8885 | 0.8440 |
| Modelo base (dominio local, sin FSL) | 0.5887 | 0.6143 |
| Few-Shot K=30 (dominio local) | 0.8146 | 0.8013 |
| **Recuperación** | **75.4%** | **80.6%** |

---

## Estructura del repositorio

```
TFG_Proyecto/
├── datasets/
├── models/
├── results/
├── runs/
├── scripts/
│   ├── analizar_dataset.py          # análisis de un dataset
│   ├── normalizar_dataset.py        # normalización al formato YOLO
│   ├── analizar_distribucion.py     # distribución de un dataset
│   ├── dividir_dataset_local.py     # división train/test
│   ├── entrenar_modelos_base.py     # entrenamiento 3 modelos base
│   ├── seleccionar_modelo_base.py   # evaluación y selección de modelo base
│   ├── evaluar_base_local.py        # evaluación domain shift
│   ├── entrenar_modelos_fewshot.py  # entrenamiento K-Shot (K=10,20,30)
│   ├── evaluar_fewshot.py           # evaluación modelos K-Shot
│   ├── comparacion_final.py         # comparativa Base vs. Few-Shot
│   ├── conclusion.py                # conclusión experimental
│   └── detectar_no_helmet_anon.py   # inferencia con modelo final
├── TFG_Pipeline_Notebook.ipynb      # Notebook con el pipeline completo
├── requirements.txt                 # Dependencias del proyecto
├── .gitignore
└── README.md
```

> **Nota:** Las carpetas `models/`, `datasets/` y `results/` se incluyen en el repositorio como carpetas vacías. Los datos de éstos y de las demas carpetas se generan al ejecutar el pipeline.

---

## Requisitos de hardware

| Componente | Especificación mínima |
|------------|----------------------|
| GPU | NVIDIA con CUDA (recomendado ≥ 6 GB VRAM) |
| RAM | 16 GB |
| Almacenamiento | 10 GB libres |
| SO | Windows 10/11 o Linux |

---

## Instalación y configuración del entorno

### 1. Clonar el repositorio

```bash
git clone https://github.com/servin-andrea/TFG-no-helmet-detection.git
cd TFG-no-helmet-detection
```

### 2. Crear y activar el entorno virtual

**Windows:**
```powershell
python -m venv tfg-venv
tfg-venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv tfg-venv
source tfg-venv/bin/activate
```

### 3. Instalar PyTorch con CUDA

> Verificar la versión de CUDA instalada con `nvidia-smi` antes de ejecutar.

```bash
# CUDA 12.1 (GTX 1660 Ti / RTX serie 30xx)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Sin GPU (CPU solamente — muy lento para entrenamiento)
pip install torch torchvision torchaudio
```

### 4. Instalar el resto de dependencias

```bash
pip install -r requirements.txt
```

### 5. Verificar la instalación

```python
import torch
print("PyTorch:", torch.__version__)
print("CUDA disponible:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

from ultralytics import YOLO
print("Ultralytics OK")
```

---

## Compatibilidad con otros sistemas operativos

Este proyecto fue desarrollado y probado en **Windows 11**. Si reproducís el experimento en Linux o macOS, hay algunos ajustes necesarios:

### Linux / macOS

**Activación del entorno virtual:**
```bash
source tfg-venv/bin/activate
```

**`workers` en el DataLoader:**  
En Windows se usa `workers=0` para evitar errores de multiprocesamiento. En Linux podés aumentarlo para acelerar el entrenamiento:
```python
# En entrenar_modelos_base.py y entrenar_modelos_fewshot.py
# Cambiar workers=0 por:
workers=4   # o el número de núcleos disponibles
```

**Rutas de archivos:**  
Los scripts usan `Path()` de Python, que maneja automáticamente las barras (`/` vs `\`). No debería haber problemas de rutas entre sistemas operativos.

**CUDA en Linux:**  
En Linux con GPU NVIDIA, CUDA suele estar mejor soportado que en Windows. Verificar la versión instalada con `nvidia-smi` y elegir el comando de instalación de PyTorch correspondiente (ver sección de instalación).

**Jupyter Notebook:**  
En Linux/macOS el backend de matplotlib funciona distinto. Si aparecen warnings al mostrar imágenes en el notebook, agregar al inicio de la celda de configuración:
```python
import matplotlib
matplotlib.use('Agg')
```

### macOS con Apple Silicon (M1/M2/M3)

**No hay soporte CUDA** — las GPU Apple Silicon usan Metal, no CUDA. El entrenamiento corre en CPU y es significativamente más lento. Alternativa recomendada: usar Google Colab con GPU T4 gratuita.

Para instalar PyTorch sin CUDA:
```bash
pip install torch torchvision torchaudio
```

PyTorch tiene soporte experimental para Metal (MPS) desde la versión 1.12. Para habilitarlo en los scripts, cambiar `device=0` por `device='mps'` si está disponible:
```python
import torch
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
```

### Google Colab (cualquier SO sin GPU)

Podés correr el notebook en Colab sin GPU local. Subí los datasets a Google Drive y montá la unidad:
```python
from google.colab import drive
drive.mount('/content/drive')
BASE = Path('/content/drive/MyDrive/TFG-no-helmet-detection')
```

---

## Preparación de datos

### Datasets públicos

Descargar los tres datasets desde Roboflow Universe en formato **YOLOv8** y descomprimir en `datasets/`:

| Dataset | URL | Carpeta destino |
|---------|-----|-----------------|
| Dataset A | [motorcycle-helmet-object-detection](https://universe.roboflow.com/projectt/motorcycle-helmet-object-detection) | `datasets/Dataset_A_raw/` |
| Dataset B | [helmet-no-helmet-detection-hjdvx](https://universe.roboflow.com/programa-delfn/helmet-no-helmet-detection-hjdvx) | `datasets/Dataset_B_raw/` |
| Dataset C | [helmet-and-no-helmet-rider-detection](https://universe.roboflow.com/gw-khadatkar-and-sv-wasule/helmet-and-no-helmet-rider-detection) | `datasets/Dataset_C_raw/` |

### Dataset local

Colocar las imágenes capturadas y etiquetadas en:

```
datasets/
└── local_raw/
    └── train/
        ├── images/   ← imágenes .jpg/.png
        └── labels/   ← anotaciones .txt formato YOLO
```

---

## Ejecución del pipeline

### Opción A — Notebook (recomendado)

```bash
jupyter notebook TFG_Pipeline_Notebook.ipynb
```

Ejecutar las celdas en orden. El notebook detecta automáticamente qué pasos ya fueron completados y los omite.

### Opción B — Scripts individuales

Ejecutar en el siguiente orden:

```bash
# Fase 1 — Preparación de datasets públicos
python scripts/analizar_dataset.py --path datasets/Dataset_A_raw
python scripts/normalizar_dataset.py

# Fase 2 — Entrenamiento de modelos base (~5h 46min total)
python scripts/entrenar_modelos_base.py

# Fase 3 — Evaluación y selección del modelo base
python scripts/seleccionar_modelo_base.py

# Fase 4 — Dataset local
python scripts/analizar_distribucion.py --path datasets/local_raw/train
python scripts/dividir_dataset_local.py

# Fase 5 — Evaluación domain shift
python scripts/evaluar_base_local.py

# Fase 6 — Entrenamiento K-Shot (~30-40 min por K)
python scripts/entrenar_modelos_fewshot.py
python scripts/evaluar_fewshot.py

# Fase 7 — Comparativa final
python scripts/comparacion_final.py

# Fase 8 — Conclusión
python scripts/conclusion.py

# Fase 9 — Inferencia con modelo final
python scripts/detectar_no_helmet.py --source ruta/imagen_o_carpeta
```

---

## Parámetros clave del experimento

### Entrenamiento base

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| Arquitectura | YOLOv8n | Balance velocidad/precisión, compatible con 6 GB VRAM |
| `epochs` | 50 | Convergencia estable con early stopping |
| `imgsz` | 640 | Resolución estándar YOLOv8 |
| `batch` | 8 | Adaptado a 6 GB VRAM |
| `lr0` | 0.01 | Valor por defecto documentado (Jocher et al., 2023) |
| `cos_lr` | True | Scheduler coseno (Loshchilov & Hutter, 2017) |
| `mosaic` | 0.0 | Desactivado para igualar condiciones con fine-tuning |

### Fine-tuning K-Shot

| Parámetro | Valor | Cambio respecto al base |
|-----------|-------|------------------------|
| `lr0` | 0.001 | 10× menor — preserva conocimiento previo |
| `freeze` | 10 | Congela backbone — solo reentrena cabeza de detección |
| `patience` | 20 | Mayor paciencia con datasets pequeños |
| K evaluados | 10, 20, 30 | Ejemplos por clase (paradigma 2-way K-Shot) |

---

## Archivos generados

Al completar el pipeline, los siguientes archivos estarán disponibles en `results/`:

```
results/
├── fase3_comparacion_base.csv                      # Métricas de los 3 modelos base
├── fase3_comparacion.png                           # Gráfico comparativo
├── modelo_base_seleccionado.json                   # Modelo base seleccionado
├── fase5_K_domain_shift.json                       # Evidencia del domain shift
├── fase4_Kshot_curva.csv                           # Curva K-Shot (K=10,20,30)
├── fase4_Kshot_curva.png                           # Gráfico curva K-Shot
├── fase4_Kshot_resultados.json                     # Resultados detallados
├── fase6_comparacion_final.csv                     # Tabla Base vs. Few-Shot
├── fase6_comparacion_final.png                     # Gráfico comparativo final
├── fase6_conclusion.json                           # Conclusión experimental
├── fase6_datos_comparacion.json                    # Comparativa final
├── fase6_cfase6_domain_shift_recuperacion.png      # Gráfico caída del domain shift
└── domain_shift_recuperacion.png
```

---

## Referencias

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). *Ultralytics YOLO* (v8.0.0). GitHub. https://github.com/ultralytics/ultralytics
- Lin, T.-Y., et al. (2014). Microsoft COCO: Common objects in context. *ECCV 2014*. https://doi.org/10.1007/978-3-319-10602-1_48
- Loshchilov, I., & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *ICLR 2017*. https://doi.org/10.48550/arXiv.1608.03983
- Ultralytics. (2023). *YOLOv8 Documentation*. https://docs.ultralytics.com

---

## Licencia

Este repositorio contiene únicamente el código del pipeline experimental. Los datasets utilizados están sujetos a sus propias licencias (CC BY 4.0 para los datasets públicos de Roboflow).
