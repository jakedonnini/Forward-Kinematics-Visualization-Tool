import numpy as np
from FK_Visualizer import plot_3d_homogeneous_coordinates
from math import pi

def Ai(a_i, alpha_i, d_i, theta_i):
    T_i = np.array([
        [np.cos(theta_i), -1 * np.sin(theta_i) * np.cos(alpha_i), np.sin(theta_i) * np.sin(alpha_i),
         a_i * np.cos(theta_i)],
        [np.sin(theta_i), np.cos(theta_i) * np.cos(alpha_i), -1 * np.cos(theta_i) * np.sin(alpha_i),
         a_i * np.sin(theta_i)],
        [0, np.sin(alpha_i), np.cos(alpha_i), d_i],
        [0, 0, 0, 1]])
    return T_i

def forward(q):
    """
    INPUT:
    q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

    OUTPUTS:
    jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
              Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
              The base of the robot is located at [0,0,0].
    T0e       - a 4 x 4 homogeneous transformation matrix,
              representing the end effector frame expressed in the
              world frame
    """

    jointPositions = np.zeros((8, 3))
    T0e = np.identity(4)

    l1 = 0.141
    l2 = 0.192
    l3 = 0.195
    l4 = 0.121
    l5 = 0.0825
    l6 = 0.0825
    l7 = 0.125
    l8 = 0.259
    l9 = 0.088
    l10 = 0.051
    l11 = 0.159
    l12 = 0.015

    T01 = Ai(0, np.pi / 2, l1 + l2, q[0])
    T12 = Ai(0, -1 * np.pi / 2, 0, -1 * q[1])
    T23 = Ai(l6, np.pi / 2, l3 + l4, q[2])
    T34 = Ai(-1 * l5, -1 * np.pi / 2, 0, q[3] + np.pi / 2 - np.pi / 2)
    T45 = Ai(0, np.pi / 2, l7 + l8, q[4])
    T56 = Ai(l9, np.pi / 2, 0, q[5] - np.pi / 2 + np.pi / 2)
    T6e = Ai(0, 0, l10 + l11, q[6] - np.pi / 4)

    T02 = np.dot(T01, T12)
    T03 = np.dot(T02, T23)
    T04 = np.dot(T03, T34)
    T05 = np.dot(T04, T45)
    T06 = np.dot(T05, T56)
    T0e = np.dot(T06, T6e)

    T_list = [np.identity(4), T01, T02, T03, T04, T05, T06, T0e]

    base = np.array([0, 0, 0, 1])

    T0_offset = Ai(0, 0, l1, 0)
    T02_offset = Ai(0, 0, l3, 0)
    T04_offset = Ai(0, 0, l7, 0)
    T05_offset = Ai(0, 0, -1 * l12, 0)
    T06_offset = Ai(0, 0, l10, 0)

    jointPositions[0] = np.dot(T0_offset, base)[:3]
    jointPositions[1] = np.dot(T01, base)[:3]
    jointPositions[2] = np.dot(np.dot(T02, T02_offset), base)[:3]
    jointPositions[3] = np.dot(T03, base)[:3]
    jointPositions[4] = np.dot(np.dot(T04, T04_offset), base)[:3]
    jointPositions[5] = np.dot(np.dot(T05, T05_offset), base)[:3]
    jointPositions[6] = np.dot(np.dot(T06, T06_offset), base)[:3]
    jointPositions[7] = np.dot(T0e, base)[:3]

    return jointPositions, T0e, T_list

if __name__ == "__main__":
    # matches figure in the handout
    # TEST 1
    q = np.array([0, 0, 0, -0.1, 0, pi, -3 * pi / 4])

    joint_positions, T0e, T_list = forward(q)

    plot_3d_homogeneous_coordinates(T_list)

    print("TEST 1")
    print("Joint Positions:\n", joint_positions)
    print("End Effector Pose:\n", T0e)

    # TEST 2
    q = np.array([0, -pi / 4, 0, -pi / 2, 0, pi / 2, 0])

    joint_positions, T0e, T_list = forward(q)

    plot_3d_homogeneous_coordinates(T_list)

    print("TEST 2")
    print("Joint Positions:\n", joint_positions)
    print("End Effector Pose:\n", T0e)

    # TEST 3
    q = np.array([pi / 4, -pi / 4, -pi / 4, -3 * pi / 4, pi / 2, pi / 2, 0])

    joint_positions, T0e, T_list = forward(q)

    plot_3d_homogeneous_coordinates(T_list)

    print("TEST 3")
    print("Joint Positions:\n", joint_positions)
    print("End Effector Pose:\n", T0e)