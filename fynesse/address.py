"""
Address module for the fynesse framework.

This module handles question addressing functionality including:
- Statistical analysis
- Predictive modeling
- Data visualization for decision-making
- Dashboard creation
"""

from typing import Any, Union, Optional, List
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    roc_auc_score,
    accuracy_score,
)
from sklearn.naive_bayes import GaussianNB
from itertools import cycle

# Set up logging
logger = logging.getLogger(__name__)


def analyze_data(data: Union[pd.DataFrame, Any]) -> dict[str, Any]:
    """
    Address a particular question that arises from the data.

    IMPLEMENTATION GUIDE FOR STUDENTS:
    ==================================

    1. REPLACE THIS FUNCTION WITH YOUR ANALYSIS CODE:
       - Perform statistical analysis on the data
       - Create visualizations to explore patterns
       - Build models to answer specific questions
       - Generate insights and recommendations

    2. ADD ERROR HANDLING:
       - Check if input data is valid and sufficient
       - Handle analysis failures gracefully
       - Validate analysis results

    3. ADD BASIC LOGGING:
       - Log analysis steps and progress
       - Log key findings and insights
       - Log any issues encountered

    4. EXAMPLE IMPLEMENTATION:
       if data is None or len(data) == 0:
           print("Error: No data available for analysis")
           return {}

       print("Starting data analysis...")
       # Your analysis code here
       results = {"sample_size": len(data), "analysis_complete": True}
       return results
    """
    logger.info("Starting data analysis")

    # Validate input data
    if data is None:
        logger.error("No data provided for analysis")
        print("Error: No data available for analysis")
        return {"error": "No data provided"}

    if len(data) == 0:
        logger.error("Empty dataset provided for analysis")
        print("Error: Empty dataset provided for analysis")
        return {"error": "Empty dataset"}

    logger.info(f"Analyzing data with {len(data)} rows, {len(data.columns)} columns")

    try:
        # STUDENT IMPLEMENTATION: Add your analysis code here

        # Example: Basic data summary
        results = {
            "sample_size": len(data),
            "columns": list(data.columns),
            "data_types": data.dtypes.to_dict(),
            "missing_values": data.isnull().sum().to_dict(),
            "analysis_complete": True,
        }

        # Example: Basic statistics (students should customize this)
        numeric_columns = data.select_dtypes(include=["number"]).columns
        if len(numeric_columns) > 0:
            results["numeric_summary"] = data[numeric_columns].describe().to_dict()

        logger.info("Data analysis completed successfully")
        print(f"Analysis completed. Sample size: {len(data)}")

        return results

    except Exception as e:
        logger.error(f"Error during data analysis: {e}")
        print(f"Error analyzing data: {e}")
        return {"error": str(e)}


def train_underserved_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = 1000,
    class_weight: str = "balanced",
) -> Pipeline:
    """
    Trains a Logistic Regression model with scaling.

    Returns:
        Pipeline: Trained sklearn pipeline.
    """

    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=max_iter, class_weight=class_weight),
    )

    clf.fit(X_train, y_train)
    return clf


def evaluate_underserved_classifier(
    clf: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, zero_division: int = 0
) -> pd.Series:
    """
    Predicts and prints a classification report.

    Returns:
        pd.Series: Predicted labels.
    """
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, zero_division=zero_division))
    return pd.Series(y_pred)


def plot_underserved_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
) -> None:
    """
    Plots the confusion matrix for given true and predicted labels.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, values_format="d")  # type: ignore[attr-defined]
    plt.title(title)
    plt.show()


def train_naive_bayes(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Train a Gaussian Naive Bayes model.

    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training labels

    Returns:
        GaussianNB: trained model
    """
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    return nb_model


def evaluate_naive_bayes(
    model: GaussianNB, X_test: pd.DataFrame, y_test: pd.Series, label: int
):
    """
    Evaluate Naive Bayes model and return probability of class 'label'
    along with accuracy.
    """
    # Accuracy
    acc = accuracy_score(y_test, model.predict(X_test))

    # Predicted probabilities for each class
    y_proba = model.predict_proba(X_test)

    # Average probability of samples being 'label'
    prob_label = y_proba[:, list(model.classes_).index(label)].mean()

    return (
        f"Probability of county being {label}: {prob_label:.3f} | "
        f"Overall accuracy: {acc:.3f}"
    )


def plot_roc_curve(
    nb_model: GaussianNB,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Naive Bayes",
    lw: int = 2,
) -> None:
    """
    Plot ROC curve for binary or multiclass classification.

    Args:
        nb_model (GaussianNB): trained model
        X_test (pd.DataFrame): test features
        y_test (pd.Series): test labels
        model_name (str): label for the plot
        lw (int): line width
    """
    classes = np.unique(y_test)
    n_classes = len(classes)

    # Binarize labels
    y_test_bin = label_binarize(y_test, classes=classes)
    if n_classes == 2:
        y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))

    # Predict probabilities
    y_score = nb_model.predict_proba(X_test)

    # Compute global AUC
    if n_classes == 2:
        global_auc = roc_auc_score(y_test, y_score[:, 1])
    else:
        global_auc = roc_auc_score(y_test, y_score, multi_class="ovr")

    # Store fpr, tpr, auc for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "green", "red"])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=lw,
            label=f"ROC of class {classes[i]} (AUC = {roc_auc[i]:.2f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name} (Overall AUC = {global_auc:.2f})")
    plt.legend(loc="lower right")
    plt.show()
