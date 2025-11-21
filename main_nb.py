import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid")
    return np, pd, plt, sns


@app.cell
def _(pd):
    file_path = 'datasets/healthcare-dataset-stroke-data.csv'
    df = pd.read_csv(file_path)

    df_clean = df.dropna(subset=['bmi']).copy()

    Q1 = df_clean['bmi'].quantile(0.25)
    Q3 = df_clean['bmi'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df_clean = df_clean[(df_clean['bmi'] >= lower_bound) & (df_clean['bmi'] <= upper_bound)]

    print(f"Original size: {len(df)}")
    print(f"Cleaned size: {len(df_clean)}")
    df_clean.head()
    return df, df_clean


@app.cell
def _(df_clean, plt, sns):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.boxplot(data=df_clean, x='stroke', y='bmi', ax=axes[0])
    axes[0].set_title('Boxplot: BMI by Stroke Status')

    sns.violinplot(data=df_clean, x='stroke', y='bmi', ax=axes[1])
    axes[1].set_title('Violin Plot: BMI by Stroke Status')

    plt.show()
    return


@app.cell
def _(df_clean, plt, sns):
    plt.figure(figsize=(10, 6))

    sns.histplot(data=df_clean, x='avg_glucose_level', kde=True, color='purple', bins=30)
    plt.title('Histogram & Density Plot of Average Glucose Level')
    plt.xlabel('Average Glucose Level')
    plt.ylabel('Frequency')

    plt.show()
    return


@app.cell
def _(df_clean, plt, sns):
    plt.figure(figsize=(10, 6))

    sns.kdeplot(data=df_clean, x='age', y='bmi', fill=True, cmap="Blues", thresh=0.05)
    plt.title('Contour Plot (Density): Age vs BMI')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()
    return


@app.cell
def _(df_clean, plt, sns):
    cols = ['age', 'avg_glucose_level', 'bmi']
    corr_matrix = df_clean[cols].corr()

    print("Correlation between Age and BMI:")
    print(f"{corr_matrix.loc['age', 'bmi']:.4f}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.show()
    return


@app.cell
def _(df_clean, pd):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    features = ['age', 'avg_glucose_level', 'bmi']
    x = df_clean[features].values

    x_scaled = StandardScaler().fit_transform(x)

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x_scaled)

    pca_df = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])

    loadings = pd.DataFrame(pca.components_.T, columns=['PC1', 'PC2'], index=features)

    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print("\nLoadings (Interpretation):")
    print(loadings)
    return


@app.cell
def _(df_clean, np):
    mode_work = df_clean['work_type'].mode()[0]
    mode_age = df_clean['age'].mode()[0]

    bmi_std = df_clean['bmi'].std()
    bmi_n = len(df_clean)
    bmi_se = bmi_std / np.sqrt(bmi_n)

    print(f"Mode (Work Type): {mode_work}")
    print(f"Mode (Age): {mode_age}")
    print(f"Standard Error of BMI Mean: {bmi_se:.4f}")
    return


@app.cell
def _(df_clean, np, plt, sns):
    from scipy.stats import norm

    plt.figure(figsize=(10, 6))

    sns.histplot(df_clean['bmi'], kde=False, stat="density", color='green', label='Real Data')

    mu, std = norm.fit(df_clean['bmi'])
    xmin, xmax = plt.xlim()
    x_axis = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x_axis, mu, std)

    plt.plot(x_axis, p, 'k', linewidth=2, label='Theoretical Normal (Gaussian)')
    plt.title(f"BMI Distribution vs Normal Curve (mean={mu:.2f}, std={std:.2f})")
    plt.legend()

    plt.show()
    return


@app.cell
def _(df_clean, np, plt, sns):
    bmi_data = df_clean['bmi'].values
    n_iterations = 1000
    n_size = 1000  # Sample size for each iteration

    bootstrap_means = []

    # Run bootstrap
    for _ in range(n_iterations):
        sample = np.random.choice(bmi_data, size=n_size, replace=True)
        bootstrap_means.append(np.mean(sample))

    # Calculate Confidence Interval (95%)
    lower_ci = np.percentile(bootstrap_means, 2.5)
    upper_ci = np.percentile(bootstrap_means, 97.5)

    print(f"Bootstrap 95% Confidence Interval for Mean BMI: [{lower_ci:.2f}, {upper_ci:.2f}]")
    print(f"Actual Sample Mean: {np.mean(bmi_data):.2f}")

    plt.figure(figsize=(10, 6))
    sns.histplot(bootstrap_means, color='orange', kde=True)
    plt.axvline(lower_ci, color='red', linestyle='--', label='Lower CI')
    plt.axvline(upper_ci, color='red', linestyle='--', label='Upper CI')
    plt.title('Bootstrap Distribution of Mean BMI')
    plt.legend()
    plt.show()
    return


@app.cell
def _(df):
    df
    return


if __name__ == "__main__":
    app.run()
