import sys
import numpy as np
sys.path.append("src")
from gradient_descent import GradientDescent, StochasticGradientDescent

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X[:, 0] + 4 + np.random.randn(100)

def test_gradient_descent():
    model = GradientDescent(learning_rate=0.1, epochs=1000)
    model.fit(X, y)
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    assert mse < 1.0, f"GD Test failed with MSE: {mse}"
    print("Gradient Descent Test Passed!")

def test_stochastic_gradient_descent():
    model = StochasticGradientDescent(learning_rate=0.01, epochs=50)
    model.fit(X, y)
    predictions = model.predict(X)
    mse = np.mean((predictions - y) ** 2)
    assert mse < 1.5, f"SGD Test failed with MSE: {mse}"
    print("Stochastic Gradient Descent Test Passed!")

if __name__ == "__main__":
    test_gradient_descent()
    test_stochastic_gradient_descent()
