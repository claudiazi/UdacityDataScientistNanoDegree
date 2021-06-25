## Motivation
Breast cancer is one of the most killing cancers for women worldwide. It is important that we could detect it earlier and diagnose it accurately.
The aim of this project is to build two simple Supervised Machine Learning Algorithms (Logistic regression and Random forest) to detect the malignancy of breast cancer based on the features of the patient's cell nucleus.

The medium article can be seen: https://claudiazi1244.medium.com/analysis-of-breast-cancer-diagnostic-data-using-logistic-regression-and-random-forest-algorithms-eb2eb705bbfd.

## Data Set
The [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) is used here. They were computed from a digitized image of a fine needle aspirate of a breast mass and describe characteristics of the cell nuclei present in the image.

Attributes:

- ID number
- Diagnosis (M = malignant, B = benign)

Ten real-valued features are computed for each cell nucleus:

- radius (mean of distances from center to points on the perimeter)
- texture (standard deviation of gray-scale values)
- perimeter
- area
- smoothness (local variation in radius lengths)
- compactness (perimeter² / area — 1.0)
- concavity (severity of concave portions of the contour)
- concave points (number of concave portions of the contour)
- symmetry
- fractal dimension (“coastline approximation” — 1)

## Used Libraries
- `pandas`
- `numpy`
- `seaborn`
- `sklearn`


## Key Results
- In the situation of malignant and benign breast cancer, the features of cell nucleus 
  are showing noticeable difference.
- Both LogisticRegression and RandomForest algorithms are performing more than 90% acccurancy to predict the diagnosis. The RandomForest is performing better than the LogisticRegression.