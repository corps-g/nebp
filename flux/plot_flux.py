import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from nebp_flux import extract_mcnp
from group_structures import energy_groups, cosine_groups, radial_groups
import sys
sys.path.insert(0, '../')
import paths
from spectrum import Spectrum, Spectrum2D


def plot_raw_data():
    """Used for plotting raw, extracted data, to study error, etc."""

    # first, grab the data
    flux_data = extract_mcnp('n', 1e3)

    # split into values and error
    flux = flux_data[:, :, :, 0]
    error = flux_data[:, :, :, 1]

    # -------------------------------------------------------------------------
    # first, make erg dependent plot
    # grab energy bins
    x = energy_groups('scale252')

    # sum over everything but energy
    y = np.sum(flux, axis=(0, 1))

    # handle error
    e = np.sqrt(np.sum(error**2, axis=(0, 1)))

    # convert to spectrum object
    flux_erg = Spectrum(x, y, e)

    # setup plotting environment
    fig = plt.figure(0)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Energy $MeV$')
    ax.set_ylabel('$\Phi$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # plot
    ax.plot(*flux_erg.plot('plot', 'diff'), c='k', lw=0.8)
    ax.errorbar(*flux_erg.plot('errorbar', 'diff'), ls='None', c='k', lw=0.8)

    # save
    fig.savefig('plot/flux_erg.png', dpi=300)

    # -------------------------------------------------------------------------
    # then, make cos dependent plot
    # grab cosine bins
    x = cosine_groups('fine')

    # sum over everything but energy
    y = np.sum(flux, axis=(0, 2))

    # handle error
    e = np.sqrt(np.sum(error**2, axis=(0, 2)))

    # convert to spectrum object
    flux_cos = Spectrum(x, y, e)

    # setup plotting environment
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Cosine')
    ax.set_ylabel('$\Phi$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # plot
    ax.plot(*flux_cos.plot('plot', 'diff'), c='k', lw=0.8)
    ax.errorbar(*flux_cos.plot('errorbar', 'diff'), ls='None', c='k', lw=0.8)

    # save
    fig.savefig('plot/flux_cos.png', dpi=300)

    # detail
    ax.set_xlim(0.9993, 1)
    ax.set_xscale('linear')
    fig.savefig('plot/flux_cos_detail.png', dpi=300)

    # -------------------------------------------------------------------------
    # finally, make spatially dependent plot
    # grab space bins
    x = radial_groups('nebp')

    # sum over everything but energy
    y = np.sum(flux, axis=(1, 2))[:-1]

    # handle error
    e = np.sqrt(np.sum(error**2, axis=(1, 2)))[:-1]

    # convert to spectrum object
    flux_rad = Spectrum(x, y, e)

    # setup plotting environment
    fig = plt.figure(2)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Radius $cm$')
    ax.set_ylabel('$\Phi$')
    ax.set_yscale('log')

    # plot
    ax.plot(*flux_rad.plot('plot', 'int'), c='k', lw=0.8)
    ax.errorbar(*flux_rad.plot('errorbar', 'int'), ls='None', c='k', lw=0.8)

    # save
    fig.savefig('plot/flux_rad.png', dpi=300)

    # -------------------------------------------------------------------------
    # now, let's plot energy dependence of each spatial region

    # setup plotting environement
    fig = plt.figure(3)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Energy $MeV$')
    ax.set_ylabel('$\Phi$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # grab energy groups
    x = energy_groups('scale252')

    # create some colors
    colors = cm.rainbow(np.linspace(0, 1, len(radial_groups('nebp') - 1)))

    # loop through each area
    for i, r in enumerate(radial_groups('nebp')[:-1]):

        # sum over everything but energy
        y = np.sum(flux[i, :, :], axis=0)

        # handle error
        e = np.sqrt(np.sum(error[i, :, :]**2, axis=0))

        # convert to spectrum object
        flux_erg = Spectrum(x, y, e)

        # plot it
        ax.plot(*flux_erg.plot('plot', 'diff'), label='Region ' + str(i), c=colors[i], lw=0.8)
        ax.errorbar(*flux_erg.plot('errorbar', 'diff'), c=colors[i], lw=0.8, ls='None')

    # legend and save the plot
    ax.legend()
    fig.savefig('plot/flux_rad_erg.png', dpi=300)

    # -------------------------------------------------------------------------
    # now, let's plot cosine dependence of each spatial region

    # setup plotting environement
    fig = plt.figure(4)
    ax = fig.add_subplot(111)
    ax.set_xlabel('Cosine')
    ax.set_ylabel('$\Phi$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    # grab cosine groups
    x = cosine_groups('fine')

    # create some colors
    colors = cm.rainbow(np.linspace(0, 1, len(radial_groups('nebp') - 1)))

    # loop through each area
    for i, r in enumerate(radial_groups('nebp')[:-1]):

        # sum over everything but energy
        y = np.sum(flux[i, :, :], axis=1)

        # handle error
        e = np.sqrt(np.sum(error[i, :, :]**2, axis=1))

        # convert to spectrum object
        flux_cos = Spectrum(x, y, e)

        # plot it
        ax.plot(*flux_cos.plot('plot', 'diff'), label='Region ' + str(i), c=colors[i], lw=0.8)
        ax.errorbar(*flux_cos.plot('errorbar', 'diff'), c=colors[i], lw=0.8, ls='None')

    # legend and save the plot
    ax.legend()
    fig.savefig('plot/flux_rad_cos.png', dpi=300)

    # detail
    ax.set_xlim(0.9993, 1)
    ax.set_xscale('linear')
    fig.savefig('plot/flux_rad_cos_detail.png', dpi=300)

    return


if __name__ == '__main__':
    plot_raw_data()
