# 🚢 Titanic Survival Prediction (Random Forest + Streamlit)

Predict whether a Titanic passenger would have survived, and explore *why*.

## Files (all in one folder)
| File | Purpose |
|---|---|
| `app.py` | Interactive Streamlit app |
| `train_model.py` | Rebuilds `model.pkl`, `encoders.pkl`, `features.pkl` |
| `titanic_random_forest.ipynb` | Step-by-step notebook (EDA, training, evaluation) |
| `titanic.csv` | Dataset (bundled, works offline) |
| `model.pkl`, `encoders.pkl`, `features.pkl` | Trained model + encoders |
| `requirements.txt` | Dependencies |

## Run
```bash
pip install -r requirements.txt
python train_model.py        # optional - app retrains automatically if model files are missing/incompatible
streamlit run app.py
```

## App features
- **Live prediction** - no button; every sidebar change updates instantly
- **Presets / Random / Reset** passengers, adjustable **decision threshold**
- **Prediction tab** - gauge, verdict, per-tree vote histogram, CSV report download
- **Why this result** - feature effects vs. a typical passenger + global importance
- **What-if lab** - sweep any feature (split by sex/class/port) or 2-D heatmap
- **Data explorer** - filter the real data, survival-rate charts, scatter/histogram
- **Model report** - accuracy, precision, recall, F1, confusion matrix, ROC (threshold-tunable)
- **Compare** - save scenarios and compare them side by side

## Model
Random Forest (300 trees, max depth 6) on `pclass, sex, age, sibsp, parch, fare, embarked`.
Hold-out accuracy ≈ 80% (80/20 stratified split, `random_state=42`).
