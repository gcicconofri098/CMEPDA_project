import sys
import pandas as pd
import matplotlib.pyplot as plt

try:
    df = pd.read_csv("datasets/sensor_geometry.csv")

except OSError as e:
    print(f"dataset not found: {e}")
    sys.exit(1)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

ax.scatter(df["x"], df["y"], df["z"], s=1.5, linewidths=0)
plt.show()
plt.savefig("/gpfs/ddn/cms/user/cicco/miniconda3/CMEPDA/geometry.png")
