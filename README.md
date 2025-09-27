Homework 2 — Machine Learning (CS5710)

University of Central Missouri — Fall 2025

Student: SUNIL KUMAR VUTA
Course: CS5710 Machine Learning
Assignment: Home Assignment 2 — Part A (Q1–Q6, theory) & Part B (Q7–Q9, programming) ***Easily accuss the all data(codes, pdfs and explaination throw the Github link--https:(https://github.com/Sunilvuta9/Home-Work-2/tree/main)
Repo Structure

Homework 2 ML(1-6).pdf — assignment text for Part A (theory).
homework2_7_to_9.py — Colab/localscript for Part B (programming).
README.md — this file (full explanations + run instructions).
How to Run the Code (Part B)

Option 1: Google Colab

Open Colab and upload homework2_7_to_9.py.
Run all cells (they’re self-contained).
Option 2: Local Python

python -V            # Python 3.9+ recommended
pip install numpy pandas scikit-learn matplotlib
python homework2_7_to_9.py
Part A — Calculations (Q1–Q6)

Q1. Decision Stump Prediction

Rule: predict “+” if Sneezing=Yes, else “−`.
Data: (Yes,+), (No,−), (Yes,−), (No,−)

Step-by-step:

Predictions: +, −, +, −
Mistakes: only the 3rd item (Yes,−) → 1 error out of 4
Training error = 1/4 = 0.25 = 25%
Memorizer (perfect memorization) → 0% error (but low generalization).
Q2. Training Error as Splitting Criterion

Data (6 rows):
(Young, High, Poor, Yes)
(Young, Medium, Good, Yes)
(Mid, Low, Poor, No)
(Old, Medium, Poor, No)
(Old, High, Good, Yes)
(Mid, Low, Poor, No)

Compute training error for root split on each feature:

Age (x1): Young→Yes (0 err), Mid→No (0), Old→No (1 err) → 1/6 = 16.7%
Exercise (x2): High→Yes (0), Medium→Yes (1), Low→No (0) → 1/6 = 16.7%
Diet (x3): Poor→No (1), Good→Yes (0) → 1/6 = 16.7%
Best split: tie; any of x1/x2/x3 yields 16.7%.

Q3. Entropy & Information Gain (split on Exercise)

Label counts: Yes=3, No=3 →
Entropy(S) = −[0.5 log2 0.5 + 0.5 log2 0.5] = 1.0

After split on x2:
High: {Yes,Yes} Entropy=0
Medium: {Yes,No} Entropy=1
Low: {No,No} Entropy=0
Weighted entropy = (2/6)*0 + (2/6)*1 + (2/6)*0 = 0.333

Information Gain: 1.0 − 0.333 = 0.667 → Exercise is a good split.

Q4. Confusion Matrix Metrics

Confusion matrix (rows=true, cols=pred):
TP=25, FN=5, FP=15, TN=55 (total=100)

Accuracy = (TP+TN)/N = (25+55)/100 = 0.80
Precision = TP/(TP+FP) = 25/40 = 0.625
Recall (Sensitivity) = TP/(TP+FN) = 25/30 = 0.833
Specificity = TN/(TN+FP) = 55/70 = 0.786
F1 = 2PR/(P+R) ≈ 0.714
Imbalanced case (80N/20P): Accuracy can mislead; prefer F1 / Precision-Recall.

Q5. kNN Distance Calculations

Points: A(2,4, Red), B(4,4, Blue), C(4,6, Red); new point P(5,4).

d(P,A)=3, d(P,B)=1, d(P,C)=√5≈2.24
1-NN: nearest=B → Blue
3-NN: votes {Red,Blue,Red} → Red
Q6. 4-Fold Cross-Validation (k=1,3,5)

Fold errors:
k=1: 0.20,0.25,0.15,0.30 → mean 0.225
k=3: 0.15,0.20,0.10,0.20 → mean 0.1625
k=5: 0.10,0.15,0.10,0.20 → mean 0.1375

Best generalization: k=5 (lowest CV error).

Part B — Programming (Q7–Q9)

Q7. Decision Trees on Iris (max_depth = 1,2,3)

Steps implemented (see script):

Load Iris; stratified train/test split (70/30).
Train DecisionTreeClassifier with depths 1, 2, 3.
Record train/test accuracy in a table; show feature importances of best model.
Discussion:

Underfitting: depth=1 → low train & test accuracy, similar values.
Better fit: depth=2 or 3 → higher test accuracy.
Overfitting: if train ≫ test as depth increases; pick the depth with best test accuracy (usually 2–3 on Iris).
Code: see Q7 section in homework2_7_to_9.py.

Q8. kNN (2 features: sepal length, sepal width)

Steps:

Use only first two features X[:, :2].
Train KNeighborsClassifier with k=1,3,5,10.
Plot decision boundaries for each k.
Commentary:

k=1: jagged, complex boundaries → high variance (overfit risk).
k=3/5: smoother, better generalization.
k=10: overly smooth, may underfit; minority regions shrink.
Code & plots: see Q8 section in homework2_7_to_9.py.

Q9. Performance Evaluation (kNN, k=5)

Steps:

Train KNeighborsClassifier(n_neighbors=5).
Compute confusion matrix (confusion_matrix).
Print classification report (classification_report) → accuracy, precision, recall, F1 (per class + averages).
Plot multiclass ROC (OvR) using predict_proba; report AUC per class plus micro and macro AUC.
Interpretation guide:

Check which class has lower recall/F1 → harder to separate.
Micro-AUC (instance-weighted) vs Macro-AUC (class-averaged) for imbalance sensitivity.
Code & curves: see Q9 section in homework2_7_to_9.py.

Academic Integrity & Notes

Part A explanations follow the assignment’s data and questions; steps are shown explicitly.
Part B code is fully commented and reproducible; random state is fixed for comparability.
