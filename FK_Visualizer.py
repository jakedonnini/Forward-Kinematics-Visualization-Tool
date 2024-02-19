import numpy as np
from math import pi
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_3d_homogeneous_coordinates(T_list, colors=None, labels=None):
    """
    This function plots the postions and rotaions of each frame and draws a line in between to show the robot postion
    """

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each transformed point
    for i, T in enumerate(T_list):
        #
        # Homogeneous coordinate of the point (4x1 column vector)
        homogeneous_point = np.array([[0], [0], [0], [1]])

        # Apply the matrix transformation to the poin
        transformed_point = np.dot(T, homogeneous_point)

        # Extract the x, y, z coordinates from the transformed point
        x, y, z, _ = transformed_point.flatten()

        # Plot the transformed point
        print("point: ", i, x, y, z)
        ax.scatter(x, y, z, label=i)

        # Plot unit vectors for x, y, z axes
        axes_vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        transformed_axes = np.dot(T, axes_vectors.T).T

        # Update min and max values for x, y, z coordinates
        # min_val = min(min_val, x, y, z)
        # max_val = max(max_val, x, y, z)

        # Plot the unit vectors
        for axis_vector, axis_color in zip(transformed_axes, ['r', 'g', 'b']):
            ax.quiver(x, y, z, axis_vector[0], axis_vector[1], axis_vector[2], color=axis_color, length=0.1)

    for i in range(len(T_list) - 1):
        p1 = np.dot(T_list[i], np.array([[0], [0], [0], [1]])).flatten()[:3]
        p2 = np.dot(T_list[i + 1], np.array([[0], [0], [0], [1]])).flatten()[:3]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='gray')

    # ax.set_box_aspect([max_val - min_val] * 3)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(0, 1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Homogeneous Coordinates')

    # Add a legend
    ax.legend()

    # Show plot
    plt.show()

