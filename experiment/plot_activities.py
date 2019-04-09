import matplotlib.pyplot as plt
from process_activities import Au_Foil_Data
from theoretical_activities import Au_Foil_Theoretical


def plot_activities():
    """Plots a comparison of the activies found """

    # load in the two datasets
    experimental = Au_Foil_Data()
    theoretical = Au_Foil_Theoretical(experimental)

    print((experimental.a_sat_atom / theoretical.a_sat_atom)**-1)

    # setup plotting environment
    fig = plt.figure(14, figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Foil Position $In$')
    ax.set_ylabel(r'Activity per Atom $\frac{\mu Ci}{atom}$')
    ax.set_yscale('log')
    ax.set_xticks(range(experimental.n))
    ax.set_xticklabels(range(experimental.n))

    # plot the data
    style = {'color': 'red', 'marker': 'x', 'markerfacecolor': 'None',
             'markeredgecolor': 'red', 'linestyle': 'None', 'label': 'Experimental',
             'mew': 0.5, 'ms': 6}
    ax.plot(range(experimental.n), experimental.a_sat_atom, **style)
    style = {'color': 'blue', 'marker': 'o', 'markerfacecolor': 'None',
             'markeredgecolor': 'blue', 'linestyle': 'None', 'label': 'Theoretical',
             'mew': 0.5, 'ms': 6}
    ax.plot(range(experimental.n), theoretical.a_sat_atom, **style)

    # add legend and save
    ax.legend()
    plt.savefig('plot/compare_activities.png', dpi=300)

    return


if __name__ == '__main__':
    plot_activities()
