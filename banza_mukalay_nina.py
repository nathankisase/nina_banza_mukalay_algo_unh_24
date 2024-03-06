import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from sklearn.metrics import mean_squared_error

# Génération des données
np.random.seed(0)
x = np.random.rand(20) * 10
y = 2 * x ** 2 - 3 * x + 1 + np.random.randn(20) * 2

# Méthode de Gauss (Directe)
A = np.vstack([x**2, x, np.ones_like(x)]).T
m, c, _ = np.linalg.lstsq(A, y, rcond=None)[0]

# Méthode de Lagrange
poly = lagrange(x, y)

# Méthode de Newton
coeffs = np.polyfit(x, y, 2)

# Calcul de l'erreur
y_pred_gauss = m * x**2 + c * x + 1
y_pred_lagrange = poly(x)
y_pred_newton = np.polyval(coeffs, x)

rmse_gauss = np.sqrt(mean_squared_error(y, y_pred_gauss))
rmse_lagrange = np.sqrt(mean_squared_error(y, y_pred_lagrange))
rmse_newton = np.sqrt(mean_squared_error(y, y_pred_newton))

print(f"RMSE Gauss: {rmse_gauss}")
print(f"RMSE Lagrange: {rmse_lagrange}")
print(f"RMSE Newton: {rmse_newton}")

# Visualisation des données
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Données originales')
plt.plot(x, y_pred_gauss, label='Gauss (Directe)')
plt.plot(x, y_pred_lagrange, label='Lagrange')
plt.plot(x, y_pred_newton, label='Newton')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Approximation de la fonction')
plt.show()
