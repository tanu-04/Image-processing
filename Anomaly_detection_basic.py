import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def estimate_gaussian(X):
    mu = np.mean(X, axis=0)
    var = np.var(X, axis=0)
    return mu, var


def multivariate_gaussian(X, mu, sigma2):
    k = len(mu)
    
    # Ensure sigma2 is a diagonal matrix
    sigma2 = np.diag(sigma2)
    
    X = X - mu
    p = (2 * np.pi) ** (- k / 2) * np.linalg.det(sigma2) ** (-0.5) * \
        np.exp(-0.5 * np.sum(np.dot(X, np.linalg.inv(sigma2)) * X, axis=1))
    
    return p

# 3. Select Threshold for Anomaly Detection
def select_threshold(yval, pval):
    best_epsilon = 0
    best_F1 = 0
    F1 = 0
    if yval.ndim == 2:
        yval = yval[:, 1]
        print(yval)
    
    step_size = (max(pval) - min(pval)) / 1000
    
    for epsilon in np.arange(min(pval), max(pval), step_size):
        predictions = (pval < epsilon)
        
        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))
        
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
        
        if F1 > best_F1:
            best_F1 = F1
            best_epsilon = epsilon
    
    return best_epsilon, best_F1

# Load dataset
data = np.loadtxt("sample_project2_text2.txt", delimiter=',')

X = data[:, :-1] 

Xval =data[:, :-1]  # All columns except the last are features
yval = data[:, -1]  # Last column is the label

# 5. Estimate mean and variance of the features
mu, sigma2 = estimate_gaussian(X)

# 6. Compute probabilities for training and cross-validation sets
p = multivariate_gaussian(X, mu, sigma2)
pval = multivariate_gaussian(Xval, mu, sigma2)

# 7. Find the best threshold to classify anomalies
epsilon, F1 = select_threshold(yval, pval)

# Find outliers based on the epsilon value
outliers = np.where(p < epsilon)

print(f'Best epsilon found: {epsilon}')
print(f'Best F1 on cross-validation set: {F1}')
print(f'Outliers found: {len(outliers[0])}')

# Plot normal points and outliers
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], label='Normal', c='b')
plt.scatter(X[outliers[0], 0], X[outliers[0], 1], c='r', marker='x', label='Outliers')
plt.title('Anomaly Detection')
plt.legend()
plt.show()
