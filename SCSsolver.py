import cvxpy as cp
import numpy as np

# Example data (you would replace this with your actual data)
# Suppose we have two-dimensional feature vectors for a binary classification task
X = np.array([[1, 2], [2, 3], [3, 3], [2, 1], [3, 2]])  # Feature matrix
y = np.array([1, 1, 1, -1, -1])  # Class labels

# Parameters
n_samples, n_features = X.shape
C = 1.0  # Regularization parameter
w = np.array([0.5,0.5])
# Define the dual variables (alphas)
alpha = cp.Variable(n_samples)
wx = np.matmul(w, X.T)
# Define the objective function (Lagrangian dual)
K = np.dot(X, X.T)  # Kernel matrix (linear kernel)
objective = cp.Maximize(cp.sum(alpha) - cp.sum(0.5 * cp.multiply(wx,cp.multiply(y, alpha)))  )

# Constraints
constraints = [alpha >= 0, alpha <= C, cp.sum(cp.multiply(alpha, y)) == 0]

# Solve the problem using the SCS solver
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.SCS)

# Get the optimal alpha values
alpha_values = alpha.value

# Compute the weight vector w
w = np.sum((alpha_values * y)[:, None] * X, axis=0)

# Find support vectors
support_vectors = np.where((alpha_values > 1e-4) & (alpha_values < C))[0]

# Compute the bias term b using one support vector
b = y[support_vectors[0]] - np.dot(w, X[support_vectors[0]])

# Output results
print("Alpha values:", alpha_values)
print("Weight vector (w):", w)
print("Bias (b):", b)