# Airline Passenger Satisfaction — ML Classification

Modelo de clasificación binaria para predecir si un pasajero quedará satisfecho o no con el servicio de una aerolínea, basado en sus características de viaje y valoraciones de servicio.

**Stack:** Python · scikit-learn · pandas · matplotlib/seaborn · joblib

---

## El problema

La aerolínea Invistico recopila encuestas post-vuelo. Con 129,880 registros y 22 variables (demográficas, tipo de viaje y valoraciones de servicio), el objetivo es predecir `satisfaction` — una etiqueta binaria — con suficiente precisión para priorizar acciones de mejora en los puntos de dolor del pasajero.

---

## Estructura

```
airline-classification-ml/
├── data/raw/                   
├── notebooks/
│   └── 01_airline_classification_analysis.ipynb
├── src/
│   ├── data_preprocessing.py   
│   ├── train_model.py          
│   ├── evaluate_model.py       
│   └── utils.py                
├── models/                     
├── reports/figures/            
└── requirements.txt
```

---

## Flujo

```
Raw CSV → EDA → Preprocesamiento → Train/Test split (stratify) → 3 modelos → Evaluación → Export
```

El preprocesador (imputación + scaling + OHE) va dentro del pipeline de scikit-learn, sin data leakage.

---

## Modelos y resultados

| Modelo | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.877 | 0.851 | 0.910 |
| Random Forest | 0.958 | 0.955 | 0.992 |
| **Gradient Boosting** | **0.956** | **0.960** | **0.993** |

**Gradient Boosting** fue seleccionado como modelo final. Las features con mayor peso son `Online boarding`, `Inflight entertainment` e `Inflight wifi service` — la experiencia digital a bordo es el principal driver de satisfacción.

---

## Notas técnicas

- Dataset: 129,880 filas · 22 features · sin duplicados · 0.30% nulos en `Arrival Delay`
- Split: 80/20 estratificado, `random_state=42`
- El pipeline exportado acepta datos crudos directamente sin transformaciones previas

---

*End-to-End ML Project · Machine Learning · Classification · Data Science · scikit-learn*
