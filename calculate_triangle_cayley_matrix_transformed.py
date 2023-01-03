import numpy as np
import pickle
import sympy as sp

# # see DOI 10.1016/0734-189X(84)90140-3 and 10.1112/plms/s2-7.1.49 for explanation

# calculate polynomial used to compute cayley matrix
x,y,z,u,v,w,x0,p200_x,p020_x,p002_x,p110_x,p101_x,p011_x,y0,p200_y,p020_y,p002_y,p110_y,p101_y,p011_y,z0,p200_z,p020_z,p002_z,p110_z,p101_z,p011_z = sp.symbols('x,y,z,u,v,w,x0,p200_x,p020_x,p002_x,p110_x,p101_x,p011_x,y0,p200_y,p020_y,p002_y,p110_y,p101_y,p011_y,z0,p200_z,p020_z,p002_z,p110_z,p101_z,p011_z')
# P(u,v) = p200*(u**2) + p020*(v**2) + p002*(w**2) + 2*p110*u*v + 2*p101*u*w + 2*p011*v*w
P_x = (u**2)*p200_x + (v**2)*p020_x + (w**2)*p002_x + 2*u*v*p110_x + 2*u*w*p101_x + 2*v*w*p011_x - x
P_y = (u**2)*p200_y + (v**2)*p020_y + (w**2)*p002_y + 2*u*v*p110_y + 2*u*w*p101_y + 2*v*w*p011_y - y
P_z = (u**2)*p200_z + (v**2)*p020_z + (w**2)*p002_z + 2*u*v*p110_z + 2*u*w*p101_z + 2*v*w*p011_z - z
P_x = P_x.subs(w,1-u-v)
P_y = P_y.subs(w,1-u-v)
P_z = P_z.subs(w,1-u-v)
P_x = P_x.subs(p200_x,0)
P_y = P_y.subs(p200_y,0)
P_z = P_z.subs(p200_z,0)
P_x = P_x.subs(p020_x,1)
P_y = P_y.subs(p020_y,0)
P_z = P_z.subs(p020_z,0)
P_z = P_z.subs(p002_z,0)
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
small_matrix = [
    [P_x, P_y, P_z],
    [P_x_ub, P_y_ub, P_z_ub],
    [P_x_ab, P_y_ab, P_z_ab],
]
small_matrix = sp.Matrix(small_matrix)
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
# now factor to compute cayley matrix
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
with open('biquadratic_triangle_cayley_matrix_transformed_mathematica.txt', 'w') as f:
    f.write(sp.printing.mathematica.mathematica_code(nonzero))
print("non-zero ab coeffs:", alphabeta_coeff_to_row.keys())
print("non-zero uv coeffs:", uv_coeff_to_column.keys())