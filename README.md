# MSCS-634 Lab 2: Classification Using KNN and RNN Algorithms

## Purpose
This lab explores how different neighborhood-based classifiers behave on the classic Wine dataset from `sklearn`. The goal is to compare **K-Nearest Neighbors (KNN)** and **Radius Neighbors (RNN)** by changing their key parameters (k and radius) and observing the impact on classification accuracy. This helps in understanding model sensitivity to hyperparameters and guides selection of optimal values.

## Steps Performed
1. Loaded the Wine dataset and reviewed feature names and class distribution.
2. Split the data into 80% training and 20% testing (with stratification).
3. Trained a **KNNClassifier** for multiple k values: `1, 5, 11, 15, 21`, and recorded test accuracy.
4. Trained a **RadiusNeighborsClassifier** for multiple radius values: `350, 400, 450, 500, 550, 600` (as specified), using standardized data.
5. Plotted accuracy trends for both KNN and RNN.
6. Compared results and discussed when each method may be preferable.

## Key Insights
- KNN showed that very small k (k=1) can perform well on this dataset, but increasing k may slightly stabilize or reduce accuracy depending on the data.
- RNN performance depends heavily on the chosen radius and on feature scaling. Very large radii tend to include many neighbors, which can flatten performance.
- Proper scaling (e.g., `StandardScaler`) is important for distance-based methods.
- KNN is generally easier to tune because k is an intuitive parameter; RNN can be useful for datasets with uneven density.
