#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(threshold=20)

def spherical_to_cartesian(azimuth, zenith, r=600.0):
    x = r * np.sin(zenith) * np.cos(azimuth)
    y = r * np.sin(zenith) * np.sin(azimuth)
    z = r * np.cos(zenith)
    return x, y, z

ev_id = 3266196

df = pd.read_parquet("/scratchnvme/cicco/cmepda/batch_1.parquet").reset_index()

geom = pd.read_csv("/scratchnvme/cicco/cmepda/sensor_geometry.csv").reset_index()

target = pd.read_parquet("/scratchnvme/cicco/cmepda/train_meta.parquet").reset_index()

target1 = target[target['event_id']== ev_id]

print(target1)



df = df[df['auxiliary'] == False]

df["event_id"].astype(np.int32)

df1 = df.merge(geom, how="left", on="sensor_id").reset_index(drop=True)

ev_ids = np.unique(df1['event_id'].values)
print(ev_ids)

a = df1[df1['event_id']== ev_id]

a1 = a['charge'].values

#print(len(a['charge'].values))
#print(len(a['z'].values))

plt.hist(df1["charge"],bins=70, range=[0,1000])
plt.savefig("charge.png")
plt.yscale('log')
plt.close()

#%%
target_x1, target_y1, target_z1 = spherical_to_cartesian(target1['azimuth'], target1['zenith'])
target_x2, target_y2, target_z2 = spherical_to_cartesian(target1['azimuth']+np.pi,target1['zenith'])



# print("x1", target_x1, target_y1, target_z1)
# print("x2", target_x2, target_y2, target_z2)


fig = plt.figure()
ax = fig.add_subplot(projection="3d")

size = a['charge']*50

ax.scatter(geom['x'], geom['y'], geom['z'], s = 1.1, linewidth= 0)
map = ax.scatter(a['x'].values, a['y'].values, a['z'].values, s = size.values, c = a['time'].values, cmap='nipy_spectral_r')
#ax.plot([target_x1, -target_x1], [target_y1, -target_y1], [target_z1, -target_z1], linewidth= 3, color="red")
#ax.legend()
plt.colorbar(map, ax=ax)
plt.show()
#%%
plt.savefig("event_display.png")
plt.close()






# %%
