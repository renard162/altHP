import math as mt
import cmath as ct
import numpy as np
import scipy as sp
import scipy.linalg as spl
import pandas as pd
import matplotlib.pyplot as plt

from numpy import (
                   array,
                   sqrt,
                   exp,
                   log,
                   log10,
                   average,
                   mean,
                   std,
                   roots
                  )

#Config
pd.set_option('display.max_rows', 999,
              'display.max_columns', 999,
              'precision', 6)

#Objetos multi dimensionais
_ND_ = (
        np.ndarray,
        np.matrix,
        pd.Series,
        pd.DataFrame,
        list
        )

#Conversão de ângulos (capaz de converter ângulos complexos)
rad2deg = lambda theta: (theta * 180) / np.pi
deg2rad = lambda theta: (theta * np.pi) / 180


#Funções Complexas
_vect_c_ = lambda func, z: np.vectorize(func)(z) if isinstance(z, _ND_) \
                                                 else func(z)
_pol_ = lambda z: (abs(z), rad2deg(ct.phase(z)))
_ret_ = lambda z: (z.real, z.imag)
_phase_ = lambda z: rad2deg(ct.phase(z))
_c_ = lambda re, im: complex(re, im)

def c_(re, im):
    '''Complexo no formato z = re + j*im
    '''
    if isinstance(re, _ND_):
        re = np.asarray(re)
        im = np.asarray(im)
        assert re.shape == im.shape, 'Real and Imaginary shape mismatch!'
        return np.vectorize(_c_)(re, im)
    else:
        return _c_(re, im)

def cp_(r, theta):
    '''Complexo na forma z = r*exp(j*theta)
    '''
    re = r*np.cos(deg2rad(theta))
    im = r*np.sin(deg2rad(theta))
    return c_(re, im)

def pol(z):
    '''Retorna os valores de ("r", "theta") de complexos.
    '''
    return _vect_c_(_pol_, z)

def ret(z):
    '''Retorna os valores de ("Real", "Imag") de complexos.
    '''
    return _vect_c_(_ret_, z)

phase = lambda z: _vect_c_(_phase_, z)


#Álgebra Linear
matrix = np.array

def qr_(A):
    '''Decomposição QR de matrizes na forma A = Q@R.
    '''
    return spl.qr(A, mode='economic')

def lu_(A):
    '''Decomposição LU de matrizes na forma A = P@L@U.
    '''
    return spl.lu(A)


#Funções Trigonometricas
_deg2rad_fun = lambda func, O: func(deg2rad(O))
sin = lambda O: _deg2rad_fun(np.sin, O)
cos = lambda O: _deg2rad_fun(np.cos, O)
tan = lambda O: _deg2rad_fun(np.tan, O)

sinh = lambda O: _deg2rad_fun(np.sinh, O)
cosh = lambda O: _deg2rad_fun(np.cosh, O)
tanh = lambda O: _deg2rad_fun(np.tanh, O)

_rad2deg_fun = lambda func, val: rad2deg(func(val))
asin = lambda val: _rad2deg_fun(np.arcsin, val)
acos = lambda val: _rad2deg_fun(np.arccos, val)
atan = lambda val: _rad2deg_fun(np.arctan, val)

asinh = lambda val: _rad2deg_fun(np.arcsinh, val)
acosh = lambda val: _rad2deg_fun(np.arccosh, val)
atanh = lambda val: _rad2deg_fun(np.arctanh, val)


#Constantes
pi_ = np.pi
e_ = exp(1)
j_ = complex(0,1)
i_ = complex(0,1)


#Visualização dos dados
def view(data):
    '''Visualização de matrizes e vetores.
    '''
    if isinstance(data, tuple):
        for aux_data in data:
            view(data=aux_data)
        return

    if isinstance(data, list):
        data = np.asarray(data)

    if isinstance(data, (np.ndarray, np.matrix)):
        if (data.ndim < 2):
            data = [data]
            idx = ['']
        else:
            idx = range(data.shape[0])
        out = pd.DataFrame(data=data,
                           index=idx)
    else:
        out = data

    print(f'\n{out}')
    return

def plot(x, y=None, show=True):
    '''Plota gráficos de séries de dados.
    '''
    if (y is None):
        plt.plot(x)
    else:
        plt.plot(x, y)
    if show:
        plt.grid()
        plt.show()

