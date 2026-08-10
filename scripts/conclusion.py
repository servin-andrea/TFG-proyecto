"""
SCRIPT 5 — Conclusión experimental
====================================
Lee todos los JSONs generados y produce el resumen completo
del experimento con la evaluación de la hipótesis de investigación.

Resultado:
    results/fase6_conclusion.json
    results/fase6_domain_shift_recuperacion.png

Requisitos previos:
    - results/fase5_K_domain_shift.json
    - results/fase4_Kshot_resultados.json
    - results/fase6_datos_comparacion.json

Uso:
    python scripts/conclusion.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuración ────────────────────────────────────────────────────────────

BASE        = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"

# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Verificar y cargar JSONs ──────────────────────────────────────────────

    archivos = {
        "domain_shift" : RESULTS_DIR / "fase5_K_domain_shift.json",
        "kshot"        : RESULTS_DIR / "fase4_Kshot_resultados.json",
        "comparacion"  : RESULTS_DIR / "fase6_datos_comparacion.json",
    }

    for nombre, ruta in archivos.items():
        if not ruta.exists():
            raise SystemExit(
                f"❌ No se encontró: {ruta}\n"
                f"   Ejecutar los scripts anteriores en orden (03 → 04)."
            )

    with open(archivos["domain_shift"],  encoding="utf-8") as f:
        datos_base = json.load(f)
    with open(archivos["kshot"],         encoding="utf-8") as f:
        datos_kshot = json.load(f)
    with open(archivos["comparacion"],   encoding="utf-8") as f:
        datos_comp = json.load(f)

    # ── Extraer valores clave ─────────────────────────────────────────────────

    MAP_PUBLICO  = datos_base["mAP_publico"]
    MAP_BASE     = datos_base["mAP@0.5"]
    F1_BASE      = datos_base["F1-Score"]
    MAP_FEWSHOT  = datos_comp["resultado_fewshot"]["mAP@0.5"]
    F1_FEWSHOT   = datos_comp["resultado_fewshot"]["F1-Score"]
    K_MEJOR      = datos_comp["mejor_K"]

    CAIDA_ABS    = round(MAP_BASE    - MAP_PUBLICO, 4)
    CAIDA_PCT    = round(CAIDA_ABS   / MAP_PUBLICO * 100, 1)
    MEJORA_ABS   = round(MAP_FEWSHOT - MAP_BASE,    4)
    MEJORA_PCT   = round(MEJORA_ABS  / MAP_BASE * 100, 1) if MAP_BASE > 0 else 0.0
    BRECHA_ABS   = round(MAP_FEWSHOT - MAP_PUBLICO, 4)
    BRECHA_PCT   = round(BRECHA_ABS  / MAP_PUBLICO * 100, 1)
    RECUPERACION = round(MEJORA_ABS  / abs(CAIDA_ABS) * 100, 1) if CAIDA_ABS != 0 else 0.0

    CAIDA_F1     = round(F1_BASE    - datos_base.get("mAP_publico", F1_BASE), 4)
    MEJORA_F1    = round(F1_FEWSHOT - F1_BASE, 4)
    MEJORA_F1_PCT= round(MEJORA_F1  / F1_BASE * 100, 1) if F1_BASE > 0 else 0.0

    experimentos = datos_kshot.get("experimentos", [])

    # ── Imprimir conclusión ───────────────────────────────────────────────────

    sep = "=" * 65

    print(f"\n{sep}")
    print("CONCLUSIÓN EXPERIMENTAL")
    print(sep)

    print("""
  Hipótesis:
  "Un modelo de detección entrenado con datasets generales no mantiene
   el mismo rendimiento en imágenes de un entorno urbano local, pero
   puede mejorar mediante la aplicación de Few-Shot Learning con un
   número reducido de imágenes representativas del entorno objetivo."
""")

    print(f"  {'Etapa':<42} {'mAP@0.5':>8}  {'F1-Score':>9}")
    print(f"  {'-'*42} {'-'*8}  {'-'*9}")
    print(f"  {'Modelo base — dominio público':<42} {MAP_PUBLICO:>8.4f}  {'—':>9}")
    print(f"  {'Modelo base — dominio local (sin FSL)':<42} {MAP_BASE:>8.4f}  {F1_BASE:>9.4f}")
    print(f"  {f'Few-Shot K={K_MEJOR} — dominio local':<42} {MAP_FEWSHOT:>8.4f}  {F1_FEWSHOT:>9.4f}")
    print()

    # Tabla K-Shot
    if experimentos:
        print(f"  Curva K-Shot (local_test):")
        print(f"  {'K':<6} {'N total':<10} {'mAP@0.5':<10} {'F1-Score':<10} {'¿Mejora vs base?'}")
        print(f"  {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*16}")
        for exp in experimentos:
            mejora = "✅ sí" if exp["mAP@0.5"] > MAP_BASE else "❌ no"
            f1_exp = exp.get("F1-Score", "—")
            print(f"  K={exp['K']:<4} {exp['N_total']:<10} {exp['mAP@0.5']:<10} {f1_exp:<10} {mejora}")
        print()

    # Evaluación hipótesis
    print(f"  Evaluación de la hipótesis:")
    print()

    # Parte 1
    if abs(CAIDA_PCT) > 20:
        est1 = "✅ CONFIRMADA"
        txt1 = f"Caída de {abs(CAIDA_PCT):.1f}% en mAP@0.5 (de {MAP_PUBLICO} a {MAP_BASE}) — domain shift significativo."
    elif abs(CAIDA_PCT) > 5:
        est1 = "⚠️  PARCIAL"
        txt1 = f"Caída de {abs(CAIDA_PCT):.1f}% — domain shift leve pero presente."
    else:
        est1 = "❌ NO DETECTADO"
        txt1 = f"Caída de {abs(CAIDA_PCT):.1f}% — domain shift no significativo."

    print(f"  Parte 1 — Domain shift    : {est1}")
    print(f"    {txt1}")
    print()

    # Parte 2
    k_mejoran  = [e for e in experimentos if e["mAP@0.5"] > MAP_BASE]
    k_no_mejor = [e for e in experimentos if e["mAP@0.5"] <= MAP_BASE]

    if MEJORA_PCT > 10 and len(k_mejoran) >= 2:
        est2 = "✅ CONFIRMADA"
        txt2 = (f"Mejora de {MEJORA_PCT:+.1f}% en mAP@0.5 y {MEJORA_F1_PCT:+.1f}% en F1-Score con K={K_MEJOR}.\n"
                f"    {len(k_mejoran)}/{len(experimentos)} valores de K superan al modelo base.")
    elif MEJORA_PCT > 0:
        est2 = "⚠️  MARGINAL"
        txt2 = f"Mejora de {MEJORA_PCT:+.1f}% — {len(k_mejoran)}/{len(experimentos)} K superan al base."
    else:
        est2 = "❌ NO CONFIRMADA"
        txt2 = "El few-shot no mejoró el rendimiento. Revisar hiperparámetros."

    print(f"  Parte 2 — Mejora FSL      : {est2}")
    print(f"    {txt2}")
    print()

    # Resultado global
    if "CONFIRMADA" in est1 and "CONFIRMADA" in est2:
        resultado_global = "HIPÓTESIS COMPLETAMENTE CONFIRMADA"
    elif "CONFIRMADA" in est1 or "CONFIRMADA" in est2:
        resultado_global = "HIPÓTESIS PARCIALMENTE CONFIRMADA"
    else:
        resultado_global = "HIPÓTESIS NO CONFIRMADA"

    print(f"  {'─'*60}")
    print(f"  RESULTADO: {resultado_global}")
    print(f"  {'─'*60}")
    print()
    print(f"  Recuperación del rendimiento: {RECUPERACION:.1f}%")
    print(f"  Brecha residual vs. público : {BRECHA_ABS:+.4f} ({BRECHA_PCT:+.1f}%)")

    # ── Guardar JSON conclusión ───────────────────────────────────────────────

    conclusion = {
        "hipotesis"          : "Un modelo entrenado con datasets generales pierde rendimiento en un entorno local, pero puede recuperarlo mediante Few-Shot Learning.",
        "MAP_publico"        : MAP_PUBLICO,
        "MAP_base_local"     : MAP_BASE,
        "MAP_fewshot_local"  : MAP_FEWSHOT,
        "F1_base_local"      : F1_BASE,
        "F1_fewshot_local"   : F1_FEWSHOT,
        "mejor_K"            : K_MEJOR,
        "caida_domain_shift" : {"absoluta": CAIDA_ABS, "porcentual": CAIDA_PCT},
        "mejora_fewshot"     : {"absoluta": MEJORA_ABS, "porcentual": MEJORA_PCT},
        "mejora_f1"          : {"absoluta": MEJORA_F1,  "porcentual": MEJORA_F1_PCT},
        "brecha_residual"    : {"absoluta": BRECHA_ABS, "porcentual": BRECHA_PCT},
        "recuperacion_pct"   : RECUPERACION,
        "resultado_parte1"   : est1,
        "resultado_parte2"   : est2,
        "resultado_global"   : resultado_global,
    }

    with open(RESULTS_DIR / "fase6_conclusion.json", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Conclusión guardada en: results/fase6_conclusion.json")

    # ── Gráfico domain shift y recuperación ───────────────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5))

    puntos_x  = [0, 1, 2]
    puntos_y  = [MAP_PUBLICO, MAP_BASE, MAP_FEWSHOT]
    etiquetas = [
        f"Dominio público\n(Dataset base)",
        f"Dominio local\n(sin FSL)",
        f"Dominio local\n(FSL K={K_MEJOR})",
    ]
    colores_p = ["#1D9E75", "#E24B4A", "#1D9E75"]

    # Flecha caída
    ax.annotate("", xy=(1, MAP_BASE + 0.01), xytext=(0, MAP_PUBLICO - 0.01),
                arrowprops=dict(arrowstyle="-|>", color="#E24B4A", lw=2))
    ax.text(0.45, (MAP_PUBLICO + MAP_BASE) / 2,
            f"Domain shift\n{CAIDA_ABS:+.4f} ({CAIDA_PCT:+.1f}%)",
            ha="center", va="center", fontsize=9, color="#791F1F", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#E24B4A", lw=0.8))

    # Flecha recuperación
    ax.annotate("", xy=(2, MAP_FEWSHOT - 0.01), xytext=(1, MAP_BASE + 0.01),
                arrowprops=dict(arrowstyle="-|>", color="#1D9E75", lw=2))
    ax.text(1.55, (MAP_BASE + MAP_FEWSHOT) / 2,
            f"FSL K={K_MEJOR}\n{MEJORA_ABS:+.4f} ({MEJORA_PCT:+.1f}%)",
            ha="center", va="center", fontsize=9, color="#085041", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#1D9E75", lw=0.8))

    # Línea de referencia
    ax.axhline(y=MAP_PUBLICO, color="#B4B2A9", linewidth=1,
               linestyle="--", label=f"Techo dominio público ({MAP_PUBLICO})")

    # Puntos
    for px, py, col in zip(puntos_x, puntos_y, colores_p):
        ax.scatter(px, py, s=120, color=col, zorder=5)
        ax.annotate(f"{py:.4f}", xy=(px, py), xytext=(0, 14),
                    textcoords="offset points", ha="center",
                    fontsize=10, fontweight="bold", color=col)

    ax.set_xticks(puntos_x)
    ax.set_xticklabels(etiquetas, fontsize=10)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_ylim(min(puntos_y) - 0.10, max(puntos_y) + 0.15)
    ax.set_title("Domain Shift y recuperación mediante K-Shot Learning",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    grafico_path = RESULTS_DIR / "fase6_domain_shift_recuperacion.png"
    plt.savefig(grafico_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Gráfico guardado en: {grafico_path}")

    # ── Archivos generados ────────────────────────────────────────────────────

    print(f"\n{sep}")
    print(f"RESULTADO GLOBAL: {resultado_global}")
    print(f"  Recuperación del domain shift: {RECUPERACION:.1f}%")
    print(sep)
    print("\nArchivos del experimento completo:")
    archivos_finales = [
        "results/modelo_base_seleccionado.json",
        "results/fase3_comparacion_base.csv",
        "results/fase3_comparacion.png",
        "results/fase5_K_domain_shift.json",
        "results/fase4_Kshot_curva.csv",
        "results/fase4_Kshot_curva.png",
        "results/fase4_Kshot_resultados.json",
        "results/fase6_comparacion_final.csv",
        "results/fase6_datos_comparacion.json",
        "results/fase6_comparacion_final.png",
        "results/fase6_conclusion.json",
        "results/fase6_domain_shift_recuperacion.png",
        "models/modelo_base.pt",
        "models/modelo_fewshot.pt",
    ]
    for a in archivos_finales:
        ruta = BASE / a
        print(f"  {'✅' if ruta.exists() else '❌'}  {a}")
