""" Module that plots the geometrical positioning of the sensors inside the detector
"""
import matplotlib.pyplot as plt

from datasets.sample_loader import sample_loader

if __name__ == '__main__':
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    df=sample_loader(flag='geometry')

    ax.scatter(df["x"], df["y"], df["z"], s=1.5, linewidths=0)
    plt.show()
    plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/geometry.png")
