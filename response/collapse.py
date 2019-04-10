import numpy as np
from response import response_data
from spectrum import Spectrum
import matplotlib.pyplot as plt


def collapse_rfs(edge_indices):
    """Docstring"""

    # get response functions
    responses = response_data()

    # check edge_indices are a list
    assert type(edge_indices) is list

    # check that the indices are in increasing order
    pass

    # check that the indices are in the rfs
    pass

    # add a zero to the front of the edge_indices
    edge_indices.insert(0, 0)

    # loop through rfs
    for name, response in responses.items():

        # pull out the data
        edges, values, error = response.edges, response.int, response.int_error

        # create new structures for the values and errors
        new_edges = np.zeros(len(edge_indices))
        new_values, new_error = [np.zeros(len(edge_indices) - 1) for _ in range(2)]

        # calculate the weights of each bin
        weights = response.widths

        # sum over the bounds with the weighting
        for i in range(len(edge_indices) - 1):

            # put the edge in the new edges
            new_edges[i] = edges[edge_indices[i]]

            # find the fraction of the total response that makes up the bin
            total_fraction = np.sum(weights[edge_indices[i]:edge_indices[i + 1] + 1])

            # calculate the weighted average and store
            new_values[i] = np.sum(values[edge_indices[i]:edge_indices[i + 1] + 1] * weights[edge_indices[i]:edge_indices[i + 1] + 1]) / total_fraction

            # root sum squared of the errors
            new_error[i] = np.sqrt(np.sum((error[edge_indices[i]:edge_indices[i + 1] + 1] * weights[edge_indices[i]:edge_indices[i + 1] + 1])**2)) / total_fraction

        # add the final edge (missed by the looping)
        new_edges[-1] = edges[edge_indices[-1]]

        # store the collapsed data
        responses[name] = Spectrum(new_edges, new_values, new_error, form='int')

    # return the data
    return responses


def plot_collapsed_rfs(edge_indices):
    """Docstring"""

    # get the data
    responses = collapse_rfs(edge_indices)

    # plot response functions -------------------------------------------------
    fig = plt.figure(0, figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Energy $MeV$')
    ax.set_ylabel('Response Function $cm^2$')

    # establish colormap
    color = plt.cm.rainbow(np.linspace(0, 1, len(responses)))

    # loop through integral responses
    for i, item in enumerate(responses.items()):

        # parse out name and response
        name, response = item

        if 'ft_au' not in name:
            continue

        # plot the integral response
        ax.plot(*response.plot('plot', 'int'), label=name, color=color[i], lw=0.5)
        ax.errorbar(*response.plot('errorbar', 'int'), color=color[i], ls='None', lw=0.5)

    # add legend and save
    leg = ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=4, fancybox=True,
                    framealpha=1.0, shadow=True, edgecolor='k', facecolor='white')
    fig.savefig('plot/ft_au_collapsed.png', dpi=300, bbox_extra_artists=(leg,), bbox_inches='tight')
    fig.clear()

    return


if __name__ == '__main__':
    plot_collapsed_rfs([10, 50, 100, 150, 200, 252])
