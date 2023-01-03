import math
import numpy as np
import pickle
import sympy as sp

# # see DOI 10.1016/0734-189X(84)90140-3 and 10.1112/plms/s2-7.1.49 for explanation

u, v, p00, p01, p02, p10, p11, p12, p20, p21, p22 = sp.symbols('u,v,p00,p01,p02,p10,p11,p12,p20,p21,p22')
control_points = sp.Matrix([
    [p00, p01, p02],
    [p10, p11, p12],
    [p20, p21, p22]
])
Puv = 0
## int division is okay here since both numerator and denominator are always 2 or 1
for i in range(3):
    i_binomial_coef = math.factorial(2)//(math.factorial(i)*math.factorial(2-i))
    i_basis_u = i_binomial_coef * (u**i) * ((1-u)**(2-i))
    for j in range(3):
        j_binomial_coef = math.factorial(2)//(math.factorial(j)*math.factorial(2-j))
        j_basis_v = j_binomial_coef * (v**j) * ((1-v)**(2-j))
        Puv += i_basis_u*j_basis_v*control_points[i,j]
# printing Puv here yields:
# p00*(1 - u)**2*(1 - v)**2 + 2*p01*v*(1 - u)**2*(1 - v) + p02*v**2*(1 - u)**2 + 2*p10*u*(1 - u)*(1 - v)**2 + 4*p11*u*v*(1 - u)*(1 - v) + 2*p12*u*v**2*(1 - u) + p20*u**2*(1 - v)**2 + 2*p21*u**2*v*(1 - v) + p22*u**2*v**2
# now construct first matrix
x,p00_x,p01_x,p02_x,p10_x,p11_x,p12_x,p20_x,p21_x,p22_x = sp.symbols('x,p00_x,p01_x,p02_x,p10_x,p11_x,p12_x,p20_x,p21_x,p22_x')
y,p00_y,p01_y,p02_y,p10_y,p11_y,p12_y,p20_y,p21_y,p22_y = sp.symbols('y,p00_y,p01_y,p02_y,p10_y,p11_y,p12_y,p20_y,p21_y,p22_y')
z,p00_z,p01_z,p02_z,p10_z,p11_z,p12_z,p20_z,p21_z,p22_z = sp.symbols('z,p00_z,p01_z,p02_z,p10_z,p11_z,p12_z,p20_z,p21_z,p22_z')
P_x = Puv.subs({'p00':p00_x, 'p01':p01_x,'p02':p02_x,'p10':p10_x,'p11':p11_x,'p12':p12_x,'p20':p20_x,'p21':p21_x,'p22':p22_x}) - x
P_y = Puv.subs({'p00':p00_y, 'p01':p01_y,'p02':p02_y,'p10':p10_y,'p11':p11_y,'p12':p12_y,'p20':p20_y,'p21':p21_y,'p22':p22_y}) - y
P_z = Puv.subs({'p00':p00_z, 'p01':p01_z,'p02':p02_z,'p10':p10_z,'p11':p11_z,'p12':p12_z,'p20':p20_z,'p21':p21_z,'p22':p22_z}) - z
alpha, beta = sp.symbols('α,β')
P_x_ub = P_x.subs(v, beta)
P_x_av = P_x.subs(u, alpha)
P_x_ab = P_x_av.subs(v, beta)
P_y_ub = P_y.subs(v, beta)
P_y_av = P_y.subs(u, alpha)
P_y_ab = P_y_av.subs(v, beta)
P_z_ub = P_z.subs(v, beta)
P_z_av = P_z.subs(u, alpha)
P_z_ab = P_z_av.subs(v, beta)
small_matrix = sp.Matrix([
    [P_x, P_y, P_z],
    [P_x_ub, P_y_ub, P_z_ub],
    [P_x_ab, P_y_ab, P_z_ab],
])
# now take determinant of first matrix to find cayley polynomial. this is slow,
# so leave commented out and load from storage after after first execution
det = small_matrix.det()
max_a = 0
max_b = 0
max_u = 0
max_v = 0
for term in det.args:
    max_u = max(max_u, sp.degree(term, u))
    max_v = max(max_v, sp.degree(term, v))
    max_a = max(max_a, sp.degree(term, alpha))
    max_b = max(max_b, sp.degree(term, beta))
denom = (u-alpha)*(v-beta)
cayley_polynomial = sp.expand(det/denom).simplify()
# factor polynomial to compute cayley matrix
alphabeta_rows = {}
for term in cayley_polynomial.args:
    alpha_degree = sp.degree(term, alpha)
    beta_degree = sp.degree(term, beta)
    alphabeta_coeff = (alpha**alpha_degree)*(beta**beta_degree)
    quotient = term / alphabeta_coeff
    if alphabeta_coeff not in alphabeta_rows:
        alphabeta_rows[alphabeta_coeff] = quotient
    else: alphabeta_rows[alphabeta_coeff] += quotient
m,n = 3,3
cayley_matrix = sp.zeros(2*m*n, 2*m*n)
alphabeta_coeff_to_row = {}
row_index = 0
for alphabeta_coeff in alphabeta_rows.keys():
        alphabeta_coeff_to_row[alphabeta_coeff] = row_index
        row_index += 1
uv_coeff_to_column = {}
column_index = 0
for alphabeta_coeff in alphabeta_rows.keys():
    row = alphabeta_rows[alphabeta_coeff]
    for term in row.args:
        u_degree = sp.degree(term, u)
        v_degree = sp.degree(term, v)
        uv_coeff = (u**u_degree)*(v**v_degree)
        if uv_coeff not in uv_coeff_to_column:
            uv_coeff_to_column[uv_coeff] = column_index
            column_index += 1
        quotient = term / uv_coeff
        cayley_matrix[alphabeta_coeff_to_row[alphabeta_coeff], uv_coeff_to_column[uv_coeff]] += quotient
rows, columns = cayley_matrix.shape
simplified_matrix = sp.zeros(rows, columns)
for row in range(rows):
    for column in range(columns):
        simplified_matrix[row, column] = cayley_matrix[row, column].simplify()
rows, columns = simplified_matrix.shape
zero_rows = []
for row in range(rows):
    zeros = True
    for term in simplified_matrix[row,:]:
        if term != 0:
            zeros = False
    if zeros:
        zero_rows.append(row)
zero_cols = []
for col in range(columns):
    zeros = True
    for term in simplified_matrix[:,col]:
        if term != 0:
            zeros = False
    if zeros:
        zero_cols.append(col)
nonzero_rows = simplified_matrix[[row for row in range(rows) if row not in zero_rows], :]
nonzero = nonzero_rows[:, [col for col in range(columns) if col not in zero_cols]]
with open('biquadratic_quadrilateral_cayley_matrix_mathematica.txt', 'w') as f:
    f.write(sp.printing.mathematica.mathematica_code(nonzero))
print("non-zero ab coeffs:", alphabeta_coeff_to_row.keys())
print("non-zero uv coeffs:", uv_coeff_to_column.keys())