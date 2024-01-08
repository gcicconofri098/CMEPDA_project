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
    df2 = sample_loader(flag='dataset')
    geom = sample_loader(flag='geometry')
    target = sample_loader(flag= 'targets')

    df = df[df['auxiliary'] == False]


    df1 = df.merge(geom, how="left", on="sensor_id").reset_index(drop=True)
    df3 = df2.merge(geom, how="left", on="sensor_id").reset_index(drop=True)


    plt.hist(df1["charge"],bins=50, range=[0,5],histtype='step', alpha=0.8,label='no auxiliary hits')
    #plt.hist(df3["charge"],bins=100, range=[0,4],histtype='step', label='with auxiliary hits')
    plt.xlabel("Charge [A.U.]")
    plt.title("Charge distribution")
    plt.ylabel("Events")
    #plt.legend()
    plt.savefig(PATH + "/feature_plots/charge.png")
    plt.yscale('log')
    plt.close()

    df1["n_counter"] = df1.groupby("event_id").cumcount()
    maxima = df1.groupby("event_id")["n_counter"].max().values

    df3["n_counter"] = df3.groupby("event_id").cumcount()
    maxima_aux = df3.groupby("event_id")["n_counter"].max().values

    plt.hist(maxima_aux,bins=90, range = [0, 1000],histtype='step', label='with auxiliary hits')

    plt.hist(maxima,bins=90, range = [0, 1000],histtype='step' ,label='no auxiliary hits')

    plt.xlabel("Hits per event")
    plt.title("Distribution of the number of hits per event")
    plt.ylabel("Events")
    plt.yscale('log')
    plt.legend()
    #plt.xscale('log')
    plt.savefig(PATH + "/feature_plots/n_hits_all.png")
    plt.close()


    sorted_maxima = np.sort(maxima)
    sorted_maxima_aux = np.sort(maxima_aux)

    cumulative_probabilities = np.arange(1, len(sorted_maxima) + 1) / len(sorted_maxima)
    cumulative_probabilities_aux = np.arange(1, len(sorted_maxima_aux) + 1) / len(sorted_maxima_aux)

    plt.plot(sorted_maxima,cumulative_probabilities, label='no auxiliary hits')
    #plt.plot(sorted_maxima_aux,cumulative_probabilities_aux, label='with auxiliary hits')

    plt.yscale("log")
    plt.title("Cumulative distribution of the number of hits per event")
    plt.xlabel("Number of hits")
    plt.ylabel("Cumulative distribution")
    #plt.legend()
    plt.xlim(0, 50)
    plt.grid()
    plt.savefig(PATH + "/feature_plots/cumulative_n_hits.png")
    plt.close()

    plt.hist(df1["time"],bins=50, range=[5500,80000],histtype='step',  label='no auxiliary hits')
    #plt.hist(df3["time"],bins=50, range=[5500,80000],histtype='step', label='with auxiliary hits')
    #plt.legend()
    plt.title("Time distribution")
    plt.xlabel("Time [ns]")
    plt.ylabel("Events")
    plt.yscale('log')

    plt.savefig(PATH + "/feature_plots/time.png")
    plt.close()


    ev_id = 563160
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
    #plt.savefig(PATH + "/feature_plots/new_2_event_display_w_auxiliary.png")
    plt.close()
