# Homework 2 — Machine Learning (CS5710)
**University of Central Missouri — Fall 2025**

**Student:** SUNIL KUMAR VUTA
**Course:** CS5710 Machine Learning  
**Assignment:** Homework 2 — Part A (Q1–Q6: Theory) & Part B (Q7–Q9: Programming)

> Full repository with data, code, PDFs, and explanations:  
> https://github.com/Sunilvuta9/Home-Work-2/tree/main

---

## Repository Contents
- `Homework 2.docx`- A questions (theory).
- `homework2_7_to_9.py` — Part B programming (standalone script).  
- `HOMEWORK2_7_to_9.ipynb` — Jupyter/Colab notebook version.  
- `README.md` — this file (overview, instructions, and explanations).

---

## Running Part B (Programming)

### Option 1: Google Colab
1. Open Colab.  
2. Upload either `homework2_7_to_9.py` or `HOMEWORK2_7_to_9.ipynb`.  
3. Run all cells.

**Tip:** If using the `.py` in Colab, add a cell at the top:
```python
%run homework2_7_to_9.py
```

### Option 2: Local Python (>=3.9)
```bash
python -V
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install numpy pandas scikit-learn matplotlib
python homework2_7_to_9.py
```

**Outputs include:**
- Train/test accuracies of decision trees (Q7).  
- Decision boundary plots for kNN with k=1,3,5,10 (Q8).  
- Confusion matrix, precision/recall/F1, multiclass ROC & AUC (Q9).

---

## Part A (Theory: Q1–Q6)

### Q1. Decision Stump
- Rule: predict “+” if Sneezing=Yes else “−”.  
- Errors: 1/4 → **25% training error**.  
- Memorizer: **0% error** (no generalization).

### Q2. Splitting Criterion
- Age, Exercise, Diet all yield 1/6 error = **16.7%**.  
- Tie between features.

### Q3. Entropy & Info Gain (split on Exercise)
- Entropy(S)=1.0.  
- Weighted entropy=0.333.  
- **IG = 0.667** → good split.

### Q4. Confusion Matrix Metrics
TP=25, FN=5, FP=15, TN=55.  
- Accuracy=0.80, Precision=0.625, Recall=0.833, Specificity=0.786, F1=0.714.  
- In imbalance (80N/20P): rely on **F1/Precision-Recall**.

### Q5. kNN Distances
A(2,4,Red), B(4,4,Blue), C(4,6,Red); P(5,4).  
- Distances: 3, 1, ~2.24.  
- **1-NN:** Blue. **3-NN:** Red.

### Q6. Cross-Validation
Fold errors:  
- k=1 → mean=0.225.  
- k=3 → mean=0.1625.  
- k=5 → mean=0.1375.  
**Best:** k=5.

---

## Part B (Programming: Q7–Q9)

### Q7. Decision Trees on Iris
- Depth=1 underfits (low accuracy).  
- Depth=2–3 improves generalization.  
- Best chosen by highest test accuracy.

### Q8. kNN (Sepal length & width only)
- k=1: jagged boundary (overfit).  
- k=3,5: smoother, good generalization.  
- k=10: overly smooth (underfit).

### Q9. Evaluation of kNN (k=5)
- Confusion matrix + classification report.  
- ROC & AUC (per-class + micro/macro).  
- Low recall/F1 indicates harder-to-classify class.  
- Compare micro vs macro AUC for imbalance.

---

## Notes
- Random seeds fixed for reproducibility.  
- Code is fully documented.  
- This README + DOCX form the full submission.
