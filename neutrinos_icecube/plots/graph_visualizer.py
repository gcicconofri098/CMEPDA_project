""" Module that plots a visualisation of the graph created with the module tensor_creation
"""
import matplotlib.pyplot as plt
import torch_geometric
import networkx as nx


def graph_visualisation(data, event_id, label):
    """Plots the graphs on the xy plane and yz plane.

    Args:
    
        data (torch_geometric.Data): Data containing both targets and features
    """
    g = torch_geometric.utils.to_networkx(data)
    x = data.x

    pos = {i: (x[i, 1].item(), x[i, 2].item()) for i in range(len(x))}
    subset_graph = g.subgraph(range(len(x)))

    #uses the charge of the hits as color scheme
    node_colors = x[:, 0].numpy()
    node_size = 40

    #! AZIMUTH

    fig, ax = plt.subplots()
    nx.draw(
        subset_graph,
        pos=pos,
        node_color=node_colors,
        node_size=node_size,
        cmap="viridis",
        with_labels=False,
        ax=fig.add_subplot(121),
    )
    plt.show()

    # plt.axhline(0, color='black', linestyle='--', linewidth=1)
    # plt.axvline(0, color='black', linestyle='--', linewidth=1)

    # x_min, x_max = x[:, 1].min().item(), x[:, 1].max().item()
    # y_min, y_max = x[:, 2].min().item(), x[:, 2].max().item()
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    # # Add axis labels
    # plt.text(x_min - 0.1 * (x_max - x_min), y_min - 0.1 * (y_max - y_min), 'x', ha='center')
    # plt.text(x_min - 0.15 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), 'y', va='center', rotation='vertical')

    # # Add tick values
    # plt.text(x_min, y_min - 0.05 * (y_max - y_min), f'{x_min:.2f}', ha='center')
    # plt.text(x_max, y_min - 0.05 * (y_max - y_min), f'{x_max:.2f}', ha='center')
    # plt.text(x_min - 0.08 * (x_max - x_min), y_min, f'{y_min:.2f}', va='center', rotation='vertical')
    # plt.text(x_min - 0.08 * (x_max - x_min), y_max, f'{y_max:.2f}', va='center', rotation='vertical')

    ax = plt.gca()
    ax.set(xlabel="x", ylabel="y")
    text_azimuth = data.y[:, 0]
    text_zenith = data.y[:, 1]

    scatter = plt.scatter(
        [], [], c=[], cmap="viridis", vmin=node_colors.min(), vmax=node_colors.max()
    )

    plt.colorbar(scatter, label="Charge")
    plt.text(
        0.5,
        1.05,
        "azimuth is: " + str(text_azimuth),
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="center",
    )

    plt.savefig("graph_proj_x_y_" + str(event_id) + "_" + str(label) + ".png")
    plt.close()

    #! ZENITH

    fig, ax = plt.subplots()

    pos1 = {i: (x[i, 2].item(), x[i, 3].item()) for i in range(len(x))}
    subset_graph1 = g.subgraph(range(len(x)))

    node_colors = x[:, 0].numpy()
    scatter = plt.scatter(
        [], [], c=[], cmap="viridis", vmin=node_colors.min(), vmax=node_colors.max()
    )

    nx.draw(
        subset_graph1,
        pos=pos1,
        node_color=node_colors,
        node_size=node_size,
        cmap="viridis",
        with_labels=False,
        ax=fig.add_subplot(121),
    )
    plt.show()
    plt.text(
        0.5,
        1.05,
        "zenith is:" + str(text_zenith),
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="center",
    )

    # x_min, x_max = x[:, 2].min().item(), x[:, 2].max().item()
    # y_min, y_max = x[:, 3].min().item(), x[:, 3].max().item()
    # plt.xlim(x_min, x_max)
    # plt.ylim(y_min, y_max)

    # # Add axis labels
    # plt.text(x_min - 0.1 * (x_max - x_min), y_min - 0.11 * (y_max - y_min), 'y', ha='center')
    # plt.text(x_min - 0.15 * (x_max - x_min), y_min + 0.5 * (y_max - y_min), 'z', va='center', rotation='vertical')

    # # Add tick values
    # plt.text(x_min, y_min - 0.05 * (y_max - y_min), f'{x_min:.2f}', ha='center')
    # plt.text(x_max, y_min - 0.05 * (y_max - y_min), f'{x_max:.2f}', ha='center')
    # plt.text(x_min - 0.08 * (x_max - x_min), y_min, f'{y_min:.2f}', va='center', rotation='vertical')
    # plt.text(x_min - 0.08 * (x_max - x_min), y_max, f'{y_max:.2f}', va='center', rotation='vertical')

    ax = plt.gca()
    ax.set(xlabel="x", ylabel="y")

    plt.colorbar(scatter, label="Charge")

    plt.savefig("graph_proj_y_z_" + str(event_id) + "_" + str(label) + ".png")