import numpy as np
from functools import reduce
import jax.numpy as jnp

class LagrangianPolynomial():
    def __init__(self, x_list, t_list) -> None:

        # dependent var x, independent var t
        # lists of coordinates of control points
        self.x_list = x_list
        self.t_list = t_list
    
    def numerator(self, index, t):
        '''
        interpolated points based on input control points
        P(t) = sum over j [product(t - t_i) (for i != j)/product(t_j-t_i) * x_j]
        '''
        num = reduce( lambda a, b: a * b ,
                    [(t - t_i) for t_index, t_i in enumerate(self.t_list) if t_index != index], 1.0)
        return num * self.x_list[index]
    
    def denominator(self, index):
    
        denom = np.multiply.reduce ( [(self.t_list[index] - t_i) for t_index, t_i in enumerate(self.t_list) if t_index != index])
        return denom
    
    
    def coefficient(self, t):
        coeff = 0.0

        for index in range(len(self.t_list)):
            coeff += self.numerator(index, t) / self.denominator(index)
        return coeff
