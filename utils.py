import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    sorted_idx = importance.argsort()
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig, ax = plt.subplots(figsize=(10, 12))
    ax.barh(pos, importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(np.array(feature_names)[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance (MDI)')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df):
    corr = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Features')
    plt.tight_layout()
    plt.show()

def analyze_results(results_path):
    results = pd.read_csv(results_path)
    
    print("Strategy Distribution:")
    print(results['recommended_strategy'].value_counts(normalize=True))
    
    print("\nAverage Payment Probability by Strategy:")
    print(results.groupby('recommended_strategy')['payment_probability'].mean())
    
    print("\nAverage Debt Amount by Strategy:")
    print(results.groupby('recommended_strategy')['debt_amount'].mean())
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='recommended_strategy', y='payment_probability', data=results)
    plt.title('Payment Probability Distribution by Strategy')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_results('final_results.csv')