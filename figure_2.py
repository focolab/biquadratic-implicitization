import matplotlib.pyplot as plt
import numpy as np
from evaluate_point_triangle import evaluate_point_triangle, evaluate_point_triangle_matrix, evaluate_point_triangle_transform

# from https://stackoverflow.com/questions/10481990/matplotlib-axis-with-two-scales-shared-origin
def align_yaxis(ax1, ax2):
    _, y1 = ax1.transData.transform((0, 0))
    _, y2 = ax2.transData.transform((0, 0))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1-y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny+dy, maxy+dy)

def get_surface_points(controls):
    p200, p020, p002, p110, p101, p011 = controls
    points = []
    u_s = np.linspace(0., 1., 200)
    for u_ in u_s:
        for v_ in u_s:
            if u_ + v_ <= 1:
                w_ = 1-(u_+v_)
                p = p200*(u_**2) + p020*(v_**2) + p002*(w_**2) + \
                    2*p110*u_*v_ + 2*p101*u_*w_ + 2*p011*v_*w_
                points.append(p)
    return np.array(points)

def plot_evaluations(controls):
    origin = np.array([0., 0., 0.])
    u_ = .33
    v_ = .33
    w_ = 1-(u_+v_)
    addends = [0, 1e4, 6.5e4, 1e6, 1e13]
    figure, axis = plt.subplots(2, 3)
    plt.axis('off')
    for position in ['top','bottom','left','right']:
        axis[1,2].spines[position].set_color('gray')
    for i in range(len(addends)):
        addend = addends[i]
        control_points = controls + addend
        p200, p020, p002, p110, p101, p011 = control_points
        surface_point = p200*(u_**2) + p020*(v_**2) + p002 * \
            (w_**2) + 2*p110*u_*v_ + 2*p101*u_*w_ + 2*p011*v_*w_
        ray = surface_point - origin
        ray /= np.linalg.norm(ray)
        results = []
        transform_results = []
        for t in np.arange(-1, 1.01, 0.01):
            p = surface_point + t*ray
            result = evaluate_point_triangle(
                p, p200, p020, p002, p110, p101, p011)
            transform_result = evaluate_point_triangle_transform(
                p, p200, p020, p002, p110, p101, p011)
            results.append((t, result))
            transform_results.append((t, transform_result))
        determinants = []
        for t in np.arange(-1, 1, .1):
            p = surface_point + t*ray
            det = evaluate_point_triangle_matrix(
                p, p200, p020, p002, p110, p101, p011)
            determinants.append((t, det))
        patch_planar_center = (p200+p020+p002)/3
        patch_radius = np.max(np.linalg.norm(
            np.array([p200, p020, p002])-patch_planar_center, axis=1))
        t = [entry[0] for entry in results]
        result = [entry[1] for entry in results]
        transform_result = [entry[1] for entry in transform_results]
        determinant_t = [entry[0] for entry in determinants]
        determinant = [entry[1] for entry in determinants]
        j = i // 3
        k = i % 3
        axis[j, k].plot(t, result, zorder=-1, color='C0', linewidth=2, label='Implicit')
        axis[j, k].scatter(determinant_t, determinant, color='C1', s=24,
                           zorder=1, label='Numerical det')
        ax2 = axis[j, k].twinx()
        ax2.plot(t, transform_result, zorder=0, color='C2', dashes=[3, 3],
                 linewidth=3, label='Transformed implicit')
        axis_lines, axis_labels = axis[j, k].get_legend_handles_labels()
        ax2_lines, ax2_labels = ax2.get_legend_handles_labels()
        axis[j, k].set_xlabel('t', fontsize=12)
        plt.xticks(fontsize=12)
        axis[j,k].set_ylabel('Value', fontsize=12)
        ax2.set_ylabel('Value', fontsize=12, color='C2')
        plt.yticks(fontsize=12)
        ax2.tick_params(labelsize=10)
        if j==0 and k==0:
            axis[j, k].set_title("Offset: 0", fontsize=12)
        elif j==0 and k==2:
            axis[j, k].set_title("Offset: {:.1E}".format(addend), fontsize=12)
        else:
            axis[j, k].set_title("Offset: {:.0E}".format(int(addend)), fontsize=12)
        plt.axhline(0, color='black', linestyle=':')
        plt.axvline(0, color='black', linestyle=':')
        axis[j, k].spines['top'].set_visible(False)
        axis[j, k].spines['right'].set_visible(False)
        axis[j, k].spines['bottom'].set_visible(False)
        axis[j, k].spines['left'].set_visible(False)
        align_yaxis(ax2, axis[j, k])
    last_lines, last_labels = axis[1,2].get_legend_handles_labels()
    axis[1,2].legend(axis_lines + ax2_lines + last_lines, axis_labels +
                  ax2_labels + last_labels, fontsize=12, borderaxespad=0, borderpad=2, frameon=True, loc='center')
    plt.subplots_adjust(wspace=.4, hspace=.4)
    plt.show()

control_points = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [.65, .65, 0.0],
    [.65, 0.0, .65],
    [0.0, .65, .65]
])
plot_evaluations(control_points)
