import numpy as np

def transform(points, transforms):
    divisor, R1, subtrahend, R2 = transforms
    transformed_points = points/divisor
    for i in range(len(transformed_points)):
        transformed_points[i] = np.dot(R1, transformed_points[i])
    transformed_points -= subtrahend
    for i in range(len(transformed_points)):
        transformed_points[i] = np.dot(R2, transformed_points[i])
    return transformed_points

def get_transforms(vertices):
    vertices = np.copy(vertices)
    p200, p020, p002 = vertices
    # scale so that distance from p200 to p020 is 1
    divisor = np.linalg.norm(p200-p020)
    vertices /= divisor
    # now find rotation matrix into xy plane
    # from https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    triangle_normal = np.cross(p200-p020,p002-p020)
    triangle_normal /= np.linalg.norm(triangle_normal)
    xy_plane_normal = np.array([0,0,1])
    R1 = np.identity(3)
    if np.array_equal(-1*triangle_normal, xy_plane_normal):
        R1 = np.array([
            [-1,0,0],
            [0,-1,0],
            [0,0,1]
        ])
        p200= np.dot(R1, p200)
        p020= np.dot(R1, p020)
        p002= np.dot(R1, p002)
        vertices = p200, p020, p002
    elif not np.array_equal(triangle_normal, xy_plane_normal):
        v = np.cross(triangle_normal, xy_plane_normal)
        c = np.dot(triangle_normal, xy_plane_normal)
        v_x = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R1 = np.identity(3) + v_x + np.dot(v_x,v_x)/(1+c)
        p200= np.dot(R1, p200)
        p020= np.dot(R1, p020)
        p002= np.dot(R1, p002)
        vertices = p200, p020, p002
    # now pin p200 to origin
    subtrahend = p200
    vertices -= subtrahend
    p200, p020, p002 = vertices
    # now align p020 to x axis
    theta = -np.arctan2(p020[1],p020[0])
    R2 = np.array([
        [np.cos(theta), -np.sin(theta),0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    return divisor, R1, subtrahend, R2