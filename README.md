# Student Performance Predictor

## Overview

This project is an end-to-end Machine Learning pipeline designed to predict a student's final academic score based on behavioral and educational inputs. The system uses polynomial regression with Ridge regularization to reverse-engineer relationships between student activities and their final performance.

---

## System Architecture

### 1. Data Generation (Synthetic Environment)

The model learns from synthetically generated data that follows a predefined scoring formula:

```python
final_score = (
    (study_hours * 0.8) + (attendance * 0.3) + 
    (previous_score * 0.4) + (assignment_score * 0.2) + 
    (sleep_hours * 1.5) + (internet_access * 5) + 
    np.random.normal(0, 4, n_students) 
).clip(0, 100)
```

**Logic:** This acts as the ground truth environment. The ML model does not know these weights exist; its sole purpose is to **reverse-engineer this exact equation** purely by observing the generated CSV output.

---

### 2. Feature Engineering (Polynomial Transformation)

```python
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
```

**Logic:** If a system only understands straight lines, it fails at the margins. `PolynomialFeatures` forces the machine to consider curves, dynamically creating new columns in the dataset where features interact with themselves and each other.

---

### 3. Model Training (Ridge Regression)

```python
ridge_model = Ridge(alpha=15.0)
ridge_model.fit(X_train_poly, y_train)
```

**Logic:** The `alpha=15.0` parameter acts as a **braking mechanism**. If the polynomial model attempts to assign a massive multiplier to a specific feature just to perfectly fit a noisy data point, Ridge regression penalizes that complexity, keeping the model generalized.

---

## Project Structure

- **Data Generation**: Creates synthetic student performance data
- **Feature Processing**: Applies polynomial transformation to capture non-linear relationships
- **Model Training**: Trains a Ridge regression model with regularization
- **Evaluation**: Measures prediction accuracy on test data

---

## Key Concepts

| Concept | Purpose |
|---------|---------|
| **Polynomial Features** | Capture non-linear relationships between features |
| **Ridge Regularization** | Prevent overfitting by penalizing large coefficients |
| **Synthetic Data** | Test the model's ability to learn the underlying equation |

---

## Getting Started

1. Clone the repository
2. Install required dependencies (`scikit-learn`, `numpy`, `pandas`)
3. Run the pipeline to generate data, train the model, and evaluate results

---

## Requirements

- Python 3.x
- scikit-learn
- numpy
- pandas

---

## License

[Add your license information here]
