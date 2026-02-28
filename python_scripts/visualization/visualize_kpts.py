import numpy as np
import matplotlib.pyplot as plt
import pandas
import pdb

def visualize_kpt52():
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]

    data = np.load("../../HumanML3D/pose_data/ACCAD/Female1General_c3d/A1 - Stand_poses.npy")
    pose = data[0]
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(pose[:22,0],pose[:22,1],pose[:22,2])
    ax.scatter(pose[left_chain,0],pose[left_chain,1],pose[left_chain,2],marker="^")
    #ax.scatter(pose[right_hand_chain,0],pose[right_hand_chain,1],pose[right_hand_chain,2],marker="^")
    ax.set_xlim(-1,1)
    ax.set_ylim(0,2)
    ax.set_zlim(-1,1)
    plt.show()
    return

def check_csv():
    import pandas as pd
    index_path = "../../HumanML3D/index.csv"
    index_file = pd.read_csv(index_path)
    print(index_file.shape)

def visualize_kpts(source_fmt):
    if source_fmt=="humanml3d":
        visualize_kpt52()
if __name__ == "__main__":
    visualize_kpts(source_fmt="humanml3d")
