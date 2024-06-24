import numpy as np
import matplotlib.pyplot as plt
from lagrangian_polynomial import LagrangianPolynomial

t_list = [0.0, 1.0, 2.0, 3.0]
x_list = [0.0, 1.0, 4.0, 9.0]

poly = LagrangianPolynomial(t_list, x_list)

print(poly.denominator(1))

times = np.linspace(0.0, 3.0, 40)
traj = [poly.coefficient(t) for t in times]
plt.plot(t_list, x_list, 'o')
plt.plot(times, traj)
plt.show()