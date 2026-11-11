# Actividad 04: Naturaleza de Modelos de Clustering Lineales y No Lineales y Métricas de Validación Externa e Interna

<div align="center">

**UNIVERSIDAD NACIONAL DEL ALTIPLANO — PUNO**  
**FACULTAD DE INGENIERÍA MECÁNICA ELÉCTRICA, ELECTRÓNICA Y SISTEMAS**  
**ESCUELA PROFESIONAL DE INGENIERÍA DE SISTEMAS**

<img src="reports/logo_unap_caratula.png" width="130" alt="Escudo UNAP"/>

### APRENDIZAJE DE MÁQUINA (IX CICLO — GRUPO B)
**Docente:** Ing. Mayenka Fernández Chambi  
**Fecha de Presentación y Evaluación:** Miércoles 11 de Noviembre del 2026  

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-0F172A.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4%2B-0D9488.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Jupyter](https://img.shields.io/badge/Notebooks-.ipynb-E11D48.svg?logo=jupyter&logoColor=white)](notebooks/)

</div>

---

## 👥 Integrantes del Equipo y Distribución de Responsabilidades

Cada integrante cuenta con su propio cuaderno Jupyter `.ipynb` independiente, pre-ejecutado y autónomo (entrena los modelos y calcula sus métricas y gráficos directamente):

| N.º | Integrante del Equipo | Área de Investigación y Responsabilidad Técnica | Cuaderno Jupyter (`.ipynb`) |
|:---:|:---|:---|:---|
| **1** | **Mamani Zapana Edward** | Comportamiento de `KMeans`, `DBSCAN` y `SpectralClustering` y análisis de validación externa e interna en **Iris** | [`notebooks/01_Integrante1_Clustering_Iris.ipynb`](notebooks/01_Integrante1_Clustering_Iris.ipynb) |
| **2** | **Ccapa Mamani Franklyn Antony** | Comportamiento de modelos y **Paradoja de Silhouette / Davies-Bouldin** en el dataset sintético **Moons** | [`notebooks/02_Integrante2_Clustering_Moons.ipynb`](notebooks/02_Integrante2_Clustering_Moons.ipynb) |
| **3** | **Choquepata Puma Renso Erik** | Comportamiento de modelos, colapso centroidal y análisis en espacio espectral (**Vector de Fiedler**) en **Circles** | [`notebooks/03_Integrante3_Clustering_Circles.ipynb`](notebooks/03_Integrante3_Clustering_Circles.ipynb) |
| **4** | **Gutierrez Arratia Kevin Diego** | Análisis de naturaleza **Lineal vs. No Lineal**, sensibilidad de **hiperparámetros críticos**, repositorio e Informe | [`notebooks/04_Integrante4_Lineal_vs_NoLineal_Hiperparametros.ipynb`](notebooks/04_Integrante4_Lineal_vs_NoLineal_Hiperparametros.ipynb) |


---

## 📊 Resumen de Resultados Experimentales (`scikit-learn`)

| Dataset | Modelo | Clusters | Ruido | ARI (Ext. ↑) | V-Measure (Ext. ↑) | Homogeneidad | Completitud | Silhouette (Int. ↑) | Davies-Bouldin (Int. ↓) | Inercia (WCSS) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Iris** | `K-Means (k=3)` | 3 | 0 (0.0%) | `0.6201` | `0.6595` | `0.6591` | `0.6598` | `0.4599` | `0.8336` | `139.82` |
| **Iris** | `DBSCAN (eps=0.50)` | 2 | 34 (22.7%) | `0.4421` | `0.5114` | `0.5005` | `0.5228` | `0.6559` | `0.4942` | — |
| **Iris** | `DBSCAN (eps=0.62)` | 2 | 21 (14.0%) | `0.4821` | `0.5566` | `0.5228` | `0.5950` | `0.6365` | `0.5143` | — |
| **Iris** | **`Spectral (k-NN=10)`** | **3** | **0 (0.0%)** | **`0.6465`** | **`0.6838`** | **`0.6737`** | **`0.6941`** | `0.4593` | `0.8224` | — |
| **Moons** | `K-Means (k=2)` | 2 | 0 (0.0%) | `0.4451` | `0.3515` | `0.3515` | `0.3515` | **`0.4932`** | **`0.8132`** | `419.12` |
| **Moons** | **`DBSCAN (eps=0.30)`** | **2** | **0 (0.0%)** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | `0.3836` | `1.0241` | — |
| **Moons** | **`Spectral (k-NN=15)`** | **2** | **0 (0.0%)** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | `0.3836` | `1.0241` | — |
| **Circles** | `K-Means (k=2)` | 2 | 0 (0.0%) | `-0.0020` | `0.0000` | `0.0000` | `0.0000` | **`0.3531`** | **`1.1825`** | `634.33` |
| **Circles** | **`DBSCAN (eps=0.25)`** | **2** | **0 (0.0%)** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | `0.1126` | `220.3668` | — |
| **Circles** | **`Spectral (RBF, γ=10)`** | **2** | **0 (0.0%)** | **`1.0000`** | **`1.0000`** | **`1.0000`** | **`1.0000`** | `0.1126` | `220.3668` | — |

---

## ⚙️ Estructura del Repositorio y Ejecución

```text
Clusting/
├── README.md                                                     # Documentación del repositorio GitHub
├── requirements.txt                                              # Dependencias de Python
├── caratula-trabajo.docx                                         # Plantilla base institucional UNAP
├── notebooks/                                                    # Cuadernos Jupyter (.ipynb) independientes y autónomos
│   ├── 01_Integrante1_Clustering_Iris.ipynb                      # Mamani Zapana Edward (Iris + Validación)
│   ├── 02_Integrante2_Clustering_Moons.ipynb                     # Ccapa Mamani Franklyn Antony (Moons + Paradoja Silhouette)
│   ├── 03_Integrante3_Clustering_Circles.ipynb                   # Choquepata Puma Renso Erik (Circles + Vector de Fiedler)
│   └── 04_Integrante4_Lineal_vs_NoLineal_Hiperparametros.ipynb   # Gutierrez Arratia Kevin Diego (Lineal/No Lineal + Hiperparámetros)
├── reports/                                                      # Informes en .docx y .pdf
│   ├── Informe_Actividad_04_V2.docx
│   └── Informe_Actividad_04_V2.pdf
├── figures/                                                      # Gráficos minimalistas en alta resolución
├── results/                                                      # Métricas de referencia
└── src/
    └── clustering_utils.py                                       # Funciones auxiliares de cálculo y formato
```
