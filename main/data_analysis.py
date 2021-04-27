# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 15:13:50 2021

@author: Colin

Uses TSNE method for Visualizing data from here:
  - L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE.
    Journal of Machine Learning Research 9(Nov):2579-2605, 2008. 
"""

from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

prediction_strings = ["move_up","move_left","move_right","move_down","turn_right","turn_left","eat","cut","mine","build_wood","build_stone","build_bridge","destroy"]
extra_strings = ["pxcor", "pycor", "patch-ahead", "heading", "hunger", "carried-wood", "carried-stone", "advantage"]

def create_tsne_plot(run_dir):
    pred_vals = np.load(run_dir + "/tsne_pred_vals.npy")
    pred_in_states = np.load(run_dir + "/tsne_pred_states.npy")
    pred_in_extras = np.load(run_dir + "/tsne_pred_extras.npy")
    max_vals = pred_vals.max(axis=1).clip(0,0.1)
    
    print("shape of vals: {}, shape of states: {}, shape of extras: {}".format(pred_vals.shape, pred_in_states.shape, pred_in_extras.shape))
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pred_vals)
    df = pd.DataFrame({"tsne-2d-one" : tsne_results[:,0], "tsne-2d-two" : tsne_results[:,1], "max-vals" : max_vals})
    df.to_csv(run_dir + "/tsne_data.csv")
    
    
def plot_tsne_data(run_dir):
    df = pd.read_csv(run_dir + "/tsne_data.csv")
    df["max-vals"] = df["max-vals"].clip(0,0.05)
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="max-vals",
        palette=plt.get_cmap("cool"),
        data=df,
        alpha=0.5
    )
 
def show_actor_vision(state, values):
    colormap = ListedColormap(["black",
                               "lightgreen", "lightgreen",
                               "brown", "brown",
                               "lightblue", "lightblue", "lightblue", "lightblue",
                               "grey", "grey", "grey",
                               "pink", "crimson", "crimson", "crimson",
                               "red", "red", "red", "red",
                               "violet", "indigo", "indigo", "indigo", "purple", "gold"])
    state[15,15] = 25
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_aspect('equal')
    plt.imshow(np.flipud(state[:,:,-1]), interpolation='nearest', cmap=colormap, vmin=0, vmax=25)
    plt.colorbar()
    maxval = round(values.max(),3)
    plt.title("Max value is {:06.4f} for action {}".format(maxval, prediction_strings[values.argmax()]))
    plt.show()    

def show_related_states(run_dir):
    df = pd.read_csv(run_dir + "/tsne_data.csv")
    df.reset_index()
    cluster_df = df[(df['tsne-2d-one'] >= 2) & (df['tsne-2d-one'] <= 4) & (df['tsne-2d-two'] >= -8) & (df['tsne-2d-two'] <= -6)]
    # print(cluster_df)
    indexes = cluster_df.index
    pred_in_states = np.load(run_dir + "/tsne_pred_states.npy")
    pred_in_extras = np.load(run_dir + "/tsne_pred_extras.npy")
    pred_vals = np.load(run_dir + "/tsne_pred_vals.npy")
    print(extra_strings)
    for index in indexes[:25]:
        show_actor_vision(pred_in_states[index], pred_vals[index])
        print(pred_in_extras[index])
        
   
    
def test_tsne():
    myrng = np.random.default_rng()
    test_vals = myrng.random((1000,13))
    max_vals = test_vals.max(axis=1)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(test_vals)
    test_in = pd.DataFrame({"tsne-2d-one" : tsne_results[:,0], "tsne-2d-two" : tsne_results[:,1], "max-vals" : max_vals})
    print(test_in)
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="max-vals",
        palette=plt.get_cmap("YlOrRd"),
        data=test_in,
        alpha=0.5
    )
    
if __name__ == "__main__":
    np.set_printoptions(threshold=1000000, linewidth=1000000, precision=6, suppress=True)
    run_dir = "E:/TermProjectStorage/2021-04-26-2215"
    # create_tsne_plot(run_dir)
    # plot_tsne_data(run_dir)
    show_related_states(run_dir)
    # pred_in_states = np.load("E:/TermProjectStorage/2021-04-26-1644" + "/tsne_pred_vals.npy")
    # pred_in_states = pred_in_states.max(axis=1)
    # pred_in_states.sort()
    # print(pred_in_states[-100:])
    # print(pred_in_states.nbytes)
    # test_tsne()