""" Plots a visualisation of an event and the features distribution.
    In the event visualisation, the time propagation is shown with the color palette, 
    the size of each hit is proportional to the deposited charge
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('neutrinos_icecube/')

from datasets.sample_loader import sample_loader

PATH = os.path.dirname(os.path.abspath(__file__))

if __name__=='__main__':

    #loads the datasets

    df = sample_loader(flag= 'dataset')
    geom = sample_loader(flag='geometry')
    target = sample_loader(flag= 'targets')

    df = df[df['auxiliary'] == False]


    df1 = df.merge(geom, how="left", on="sensor_id").reset_index(drop=True)

    plt.hist(df1["charge"],bins=100, range=[0,10])
    plt.xlabel("Charge [A.U.]")
    plt.title("Charge distribution")
    plt.ylabel("Events")
    plt.savefig(PATH + "/feature_plots/charge.png")
    plt.yscale('log')
    plt.close()

    df1["n_counter"] = df1.groupby("event_id").cumcount()
    maxima = df1.groupby("event_id")["n_counter"].max().values


    plt.hist(maxima,bins=70)
    plt.xlabel("Hits per event")
    plt.title("Distribution of the number of hits per event")
    plt.ylabel("Events")
    plt.yscale('log')
    #plt.xscale('log')
    plt.savefig(PATH + "/feature_plots/n_hits.png")
    plt.close()


    plt.hist(df1["time"],bins=100, range=[5500,80000])
    plt.xlabel("Time [ns]")
    plt.ylabel("Events")
    plt.yscale('log')

    plt.savefig(PATH + "/feature_plots/time.png")
    plt.close()


    ev_id = 2779302
    #ev_id = 3035168 #high charge
    #ev_id = 3266196
    df["event_id"].astype(np.int32)

    target1 = target[target['event_id']== ev_id]

    a = df1[df1['event_id']== ev_id]

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    size = a['charge']*50

    ax.scatter(geom['x'], geom['y'], geom['z'], s = 1.1, linewidth= 0)
    map = ax.scatter(a['x'].values, a['y'].values, a['z'].values, s = size.values, c = a['time'].values, cmap='nipy_spectral_r')

    plt.colorbar(map, ax=ax)
    plt.show()
    plt.savefig(PATH + "/feature_plots/new_2_event_display_no_auxiliary.png")
    plt.close()
