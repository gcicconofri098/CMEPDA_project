""" Plots a visualisation of an event.
    The time propagation is shown with the color palette, the size of each hit is proportional
    to the deposited charge
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets.sample_loader import sample_loader

if __name__=='__main__':

    ev_id = 3266196

    df = sample_loader(flag= 'dataset')
    geom = sample_loader(flag='geometry')
    target = sample_loader(flag= 'targets')

    target1 = target[target['event_id']== ev_id]

    df = df[df['auxiliary'] == False]

    df["event_id"].astype(np.int32)

    df1 = df.merge(geom, how="left", on="sensor_id").reset_index(drop=True)

    ev_ids = np.unique(df1['event_id'].values)

    a = df1[df1['event_id']== ev_id]

    a1 = a['charge'].values

    plt.hist(df1["charge"],bins=70, range=[0,1000])
    plt.savefig("charge.png")
    plt.yscale('log')
    plt.close()


    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    size = a['charge']*50

    ax.scatter(geom['x'], geom['y'], geom['z'], s = 1.1, linewidth= 0)
    map = ax.scatter(a['x'].values, a['y'].values, a['z'].values, s = size.values, c = a['time'].values, cmap='nipy_spectral_r')

    plt.colorbar(map, ax=ax)
    plt.show()
    plt.savefig("event_display.png")
    plt.close()
