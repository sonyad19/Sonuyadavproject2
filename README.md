# Gun Violence Regression Analysis

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import os

# Load and clean the dataset
def load_data(filepath):
    df = pd.read_csv(filepath)
    df_cleaned = df.iloc[1:].reset_index(drop=True)
    df_cleaned.rename(columns={"Unnamed: 0": "Statistic"}, inplace=True)
    for col in df_cleaned.columns[1:]:
        df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
    return df_cleaned

# Save plot helper
def save_plot(fig, filename):
    fig.savefig(filename, bbox_inches='tight')
    plt.close(fig)

# Perform regression analysis
def perform_regressions(df):
    df["Interaction"] = df["Mass Shooting"] * df["Deaths"]
    y = df["Injuries"]
    X_simple = df[["Mass Shooting"]]
    X_multiple = df[["Mass Shooting", "Deaths"]]
    X_interaction = df[["Mass Shooting", "Deaths", "Interaction"]]

    results = {}

    # Simple Linear Regression
    model_simple = LinearRegression().fit(X_simple, y)
    y_pred_simple = model_simple.predict(X_simple)
    results['r2_simple'] = r2_score(y, y_pred_simple)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X_simple, y, color='blue', label='Actual')
    plt.plot(X_simple, y_pred_simple, color='red', label='Predicted')
    plt.xlabel("Mass Shootings")
    plt.ylabel("Injuries")
    plt.title("Simple Linear Regression: Injuries ~ Mass Shootings")
    plt.legend()
    plt.grid(True)
    save_plot(fig, "simple_regression.png")

    # Multiple Regression
    model_multiple = LinearRegression().fit(X_multiple, y)
    y_pred_multiple = model_multiple.predict(X_multiple)
    results['r2_multiple'] = r2_score(y, y_pred_multiple)

    # Interaction Regression
    model_interaction = LinearRegression().fit(X_interaction, y)
    y_pred_interaction = model_interaction.predict(X_interaction)
    results['r2_interaction'] = r2_score(y, y_pred_interaction)

    # Polynomial Regression
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X_simple)
    model_poly = LinearRegression().fit(X_poly, y)
    y_pred_poly = model_poly.predict(X_poly)
    results['r2_poly'] = r2_score(y, y_pred_poly)

    fig = plt.figure(figsize=(10, 6))
    plt.scatter(X_simple, y, color='green', label='Actual')
    plt.plot(X_simple, y_pred_poly, color='orange', label='Polynomial Fit')
    plt.xlabel("Mass Shootings")
    plt.ylabel("Injuries")
    plt.title("Polynomial Regression (Degree 2): Injuries ~ Mass Shootings")
    plt.legend()
    plt.grid(True)
    save_plot(fig, "polynomial_regression.png")

    return results

# Perform EDA
def perform_eda(df):
    fig = plt.figure(figsize=(12, 6))
    sns.barplot(x=df["Statistic"], y=df["Deaths"], color="red", label="Deaths")
    sns.barplot(x=df["Statistic"], y=df["Injuries"], color="blue", alpha=0.5, label="Injuries")
    plt.xlabel("Statistic")
    plt.ylabel("Count")
    plt.title("Comparison of Deaths and Injuries")
    plt.legend()
    plt.xticks(rotation=45)
    save_plot(fig, "bar_chart.png")

    fig = plt.figure(figsize=(12, 6))
    sns.heatmap(df.iloc[:, 1:].corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    save_plot(fig, "heatmap.png")

    fig = plt.figure(figsize=(8, 5))
    sns.boxplot(x=df["Mass Shooting"], y=df["Deaths"], palette="Oranges")
    plt.xlabel("Mass Shootings")
    plt.ylabel("Total Deaths")
    plt.title("Boxplot of Mass Shootings vs. Deaths")
    save_plot(fig, "boxplot.png")

    fig = plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["Mass Shooting"], y=df["Injuries"], color="purple")
    plt.xlabel("Mass Shootings")
    plt.ylabel("Injuries")
    plt.title("Mass Shootings vs. Injuries")
    plt.grid(True)
    save_plot(fig, "scatter.png")

if __name__ == "__main__":
    filepath = "Summary_Statistics.csv"  # Ensure this file is in the same directory
    df = load_data(filepath)
    perform_eda(df)
    results = perform_regressions(df)

    for model, score in results.items():
        print(f"{model}: R^2 = {score:.4f}")

