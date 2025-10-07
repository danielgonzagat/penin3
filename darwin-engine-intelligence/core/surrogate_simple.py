"""
Simple Surrogate Model for Evolutionary Optimization

Implements a lightweight polynomial regression surrogate using only stdlib.
This can be used to pre-filter candidates before expensive evaluation.

Features:
- Polynomial regression (degree 1-3)
- Active learning (uncertainty-based selection)
- Model validation (R² score)
- Pure Python (no sklearn)

Note: For production SOTA, use GP/RF/XGBoost with sklearn/gpytorch.
This is a stdlib-only approximation for prototyping.
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass


@dataclass
class SurrogateModel:
    """Simple surrogate model state."""
    coefficients: List[float]
    r_squared: float = 0.0
    n_samples: int = 0


class SimpleSurrogate:
    """
    Simple polynomial regression surrogate.
    
    Uses linear or quadratic features with least squares fit.
    Stdlib-only implementation (no numpy).
    """
    
    def __init__(
        self,
        degree: int = 2,
        min_samples_to_fit: int = 10,
        retrain_interval: int = 20
    ):
        """
        Initialize surrogate.
        
        Args:
            degree: Polynomial degree (1=linear, 2=quadratic, 3=cubic)
            min_samples_to_fit: Minimum samples before fitting
            retrain_interval: Retrain every N new samples
        """
        self.degree = degree
        self.min_samples_to_fit = min_samples_to_fit
        self.retrain_interval = retrain_interval
        
        # Training data
        self.X_train: List[List[float]] = []
        self.y_train: List[float] = []
        
        # Model
        self.model: Optional[SurrogateModel] = None
        
        # Counters
        self.n_predictions = 0
        self.n_retrains = 0
    
    def add_sample(self, x: List[float], y: float):
        """Add training sample."""
        self.X_train.append(list(x))
        self.y_train.append(y)
        
        # Retrain if needed
        if len(self.y_train) >= self.min_samples_to_fit:
            if len(self.y_train) % self.retrain_interval == 0:
                self.fit()
    
    def fit(self):
        """Fit polynomial regression model."""
        if len(self.y_train) < self.min_samples_to_fit:
            return
        
        # Create feature matrix
        X_features = [self._featurize(x) for x in self.X_train]
        
        # Least squares: solve X^T X b = X^T y
        coeffs = self._least_squares(X_features, self.y_train)
        
        # Compute R²
        predictions = [self._predict_with_coeffs(x, coeffs) for x in X_features]
        r_squared = self._r_squared(self.y_train, predictions)
        
        self.model = SurrogateModel(
            coefficients=coeffs,
            r_squared=r_squared,
            n_samples=len(self.y_train)
        )
        
        self.n_retrains += 1
    
    def predict(self, x: List[float]) -> Tuple[float, float]:
        """
        Predict fitness and uncertainty.
        
        Returns:
            (prediction, uncertainty)
        """
        self.n_predictions += 1
        
        if self.model is None:
            # No model yet, return neutral prediction
            return 0.0, 1.0
        
        features = self._featurize(x)
        prediction = self._predict_with_coeffs(features, self.model.coefficients)
        
        # Uncertainty: inverse of R² (simple heuristic)
        uncertainty = 1.0 - max(0.0, self.model.r_squared)
        
        return prediction, uncertainty
    
    def _featurize(self, x: List[float]) -> List[float]:
        """Create polynomial features."""
        features = [1.0]  # Bias term
        
        # Linear terms
        features.extend(x)
        
        # Quadratic terms (if degree >= 2)
        if self.degree >= 2:
            for i in range(len(x)):
                features.append(x[i] ** 2)
            # Cross terms
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    features.append(x[i] * x[j])
        
        # Cubic terms (if degree >= 3)
        if self.degree >= 3:
            for i in range(len(x)):
                features.append(x[i] ** 3)
        
        return features
    
    def _least_squares(
        self,
        X: List[List[float]],
        y: List[float]
    ) -> List[float]:
        """
        Solve least squares X @ b = y.
        
        Uses normal equations: (X^T X) b = X^T y
        """
        n_samples = len(X)
        n_features = len(X[0]) if X else 0
        
        if n_samples == 0 or n_features == 0:
            return [0.0]
        
        # Compute X^T X
        XTX = [[0.0] * n_features for _ in range(n_features)]
        for i in range(n_features):
            for j in range(n_features):
                XTX[i][j] = sum(X[k][i] * X[k][j] for k in range(n_samples))
        
        # Compute X^T y
        XTy = [sum(X[k][i] * y[k] for k in range(n_samples)) for i in range(n_features)]
        
        # Solve using simple Gaussian elimination
        b = self._gaussian_elimination(XTX, XTy)
        
        return b
    
    def _gaussian_elimination(
        self,
        A: List[List[float]],
        b: List[float]
    ) -> List[float]:
        """Solve Ax = b using Gaussian elimination."""
        n = len(b)
        
        # Augment matrix
        aug = [A[i] + [b[i]] for i in range(n)]
        
        # Forward elimination
        for i in range(n):
            # Find pivot
            max_row = i
            for k in range(i + 1, n):
                if abs(aug[k][i]) > abs(aug[max_row][i]):
                    max_row = k
            aug[i], aug[max_row] = aug[max_row], aug[i]
            
            # Make all rows below this one 0 in current column
            for k in range(i + 1, n):
                if abs(aug[i][i]) < 1e-10:
                    continue
                c = aug[k][i] / aug[i][i]
                for j in range(i, n + 1):
                    aug[k][j] -= c * aug[i][j]
        
        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(aug[i][i]) < 1e-10:
                x[i] = 0.0
            else:
                x[i] = aug[i][n]
                for j in range(i + 1, n):
                    x[i] -= aug[i][j] * x[j]
                x[i] /= aug[i][i]
        
        return x
    
    def _predict_with_coeffs(
        self,
        features: List[float],
        coeffs: List[float]
    ) -> float:
        """Predict using coefficients."""
        return sum(f * c for f, c in zip(features, coeffs))
    
    def _r_squared(
        self,
        y_true: List[float],
        y_pred: List[float]
    ) -> float:
        """Calculate R² score."""
        if not y_true:
            return 0.0
        
        mean_y = sum(y_true) / len(y_true)
        ss_tot = sum((y - mean_y) ** 2 for y in y_true)
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        
        if ss_tot < 1e-10:
            return 1.0
        
        return 1.0 - (ss_res / ss_tot)


# ============================================================================
# TEST
# ============================================================================

def test_simple_surrogate():
    """Test Simple Surrogate."""
    print("\n" + "="*80)
    print("🧪 TESTE: Simple Surrogate")
    print("="*80)
    
    surrogate = SimpleSurrogate(degree=2, min_samples_to_fit=10, retrain_interval=10)
    
    # True function: f(x, y) = x^2 + 2*y^2 - 3*x*y + 5
    def true_fn(x, y):
        return x**2 + 2*y**2 - 3*x*y + 5
    
    # Generate training data
    print("📊 Gerando dados de treino...")
    for i in range(30):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        fitness = true_fn(x, y)
        surrogate.add_sample([x, y], fitness)
    
    print(f"   Samples: {len(surrogate.y_train)}")
    print(f"   Retrains: {surrogate.n_retrains}")
    print(f"   R²: {surrogate.model.r_squared:.4f}" if surrogate.model else "   Model: Not fitted")
    
    # Test predictions
    print("\n🔮 Testando predições...")
    test_points = [
        ([1.0, 1.0], true_fn(1.0, 1.0)),
        ([0.0, 0.0], true_fn(0.0, 0.0)),
        ([-1.0, 2.0], true_fn(-1.0, 2.0))
    ]
    
    errors = []
    for (x, y_true) in test_points:
        y_pred, uncertainty = surrogate.predict(x)
        error = abs(y_pred - y_true)
        errors.append(error)
        print(f"   x={x} → true={y_true:.2f}, pred={y_pred:.2f}, error={error:.2f}, unc={uncertainty:.3f}")
    
    mean_error = sum(errors) / len(errors)
    print(f"\n📈 Mean absolute error: {mean_error:.2f}")
    
    # Validate
    assert surrogate.model is not None, "Model not fitted"
    assert surrogate.model.r_squared > 0.5, f"R² too low: {surrogate.model.r_squared}"
    assert mean_error < 10, f"Error too high: {mean_error}"
    
    print("\n✅ Simple Surrogate: PASS")
    print("="*80)


if __name__ == "__main__":
    random.seed(42)
    test_simple_surrogate()
    print("\n" + "="*80)
    print("✅ surrogate_simple.py está FUNCIONAL!")
    print("="*80)
