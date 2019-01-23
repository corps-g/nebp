import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad


class Spectrum(object):

    def __init__(self, edges, values, error=False, S=1, dfde=False, label=None):
        assert len(edges) - 1 == len(values), '{} edges and {} values given.'.format(len(edges), len(values))
        self.label = label
        self.scaling_factor = S
        self.values = values
        self.num_bins, self.num_edges = self.count_bins()
        self.edges = edges
        self.region = self.edges[0], self.edges[-1]
        self.values = self.scale_values()
        if isinstance(error, bool):
            self.error = self.estimate_error()
        else:
            self.error = error * self.scaling_factor
        self.rel_error = self.error / self.values
        self.widths = self.edges[1:] - self.edges[:-1]
        self.midpoints = (self.edges[1:] + self.edges[:-1]) / 2
        if dfde:
            self.normalized_values = self.values
            self.values = self.values * self.widths
        else:
            self.normalized_values = self.values / self.widths
        self.step_x, self.step_y = self.make_step()
        self.step = self.step_x, self.step_y
        self.stepu_x, self.stepu_y = self.make_step_unnormalized()
        self.stepu = self.stepu_x, self.stepu_y
        self.total_flux = np.sum(self.values)

    def count_bins(self):
        '''Counts the number of bins and bin edges in the data'''
        l = len(self.values)
        return l, l + 1

    def scale_values(self):
        '''Normalizes values to given scaling factor'''
        return self.values * self.scaling_factor

    def estimate_error(self):
        '''If error is not given, '''
        return np.full(len(self.values), 0.5) * self.scaling_factor

    def make_step(self):
        '''Make the binned flux data able to be plotted with plt.plot'''
        assert len(self.edges) - 1 == len(self.normalized_values), 'x - 1 != y'
        Y = np.array([[yy, yy] for yy in np.array(self.normalized_values)]).flatten()
        X = np.array([[xx, xx] for xx in np.array(self.edges)]).flatten()[1:-1]
        return X, Y

    def make_step_unnormalized(self):
        '''Make the binned flux data able to be plotted with plt.plot'''
        assert len(self.edges) - 1 == len(self.values), 'x - 1 != y'
        Y = np.array([[yy, yy] for yy in np.array(self.values)]).flatten()
        X = np.array([[xx, xx] for xx in np.array(self.edges)]).flatten()[1:-1]
        return X, Y

    def functional_form(self, E):
        '''Produce a function from the step data (for use in integration)'''
        if type(E) == np.ndarray:
            val = np.empty(len(E))
            for j, e in enumerate(E):
                for i, v in enumerate(self.normalized_values):
                    if e >= self.edges[i] and e < self.edges[i+1]:
                        val[j] = v
        else:
            val = 0
            for i, v in enumerate(self.normalized_values):
                if E >= self.edges[i] and E < self.edges[i+1]:
                    val = v
        return val

    def change_bins(self, bins):
        '''Makes discrete energy groups from continuous function above, given a desired bin structure.
        TODO: Change method of integration.'''
        bin_values = []
        for i in range(len(bins) - 1):
            area, err = quad(self.functional_form, bins[i], bins[i+1])
            height = area / (bins[i+1] - bins[i])
            bin_values.append(height)
        Spectrum.__init__(self, bins, bin_values, dfde=True, label=self.label)
        return bin_values

    def findroot(self, percentile, func):
        l, r = self.region
        c = np.geomspace(l, r, 3)[1]
        while abs(func(c) - percentile) > 1e-12:
            c = np.geomspace(l, r, 3)[1]
            if percentile < func(c):
                r = c
            else:
                l = c
        return c

    def e_avg(self):
        '''Calculates the average x value of the spectrum.'''
        cumsum = 0
        values = self.values / self.total_flux
        for i, val in enumerate(values):
            if cumsum + val < 0.5:
                cumsum += val
            else:
                x1, x2 = self.edges[i], self.edges[i+1]
                next_val = values[i]
                break
        percent = (0.5 - cumsum) / (next_val)
        x_avg = (x2 - x1) * percent
        return x_avg

    def calc_r_tot_ratio(self, cutoff):
        '''Given a cutoff value, calculates the ratio of the integrated value
        above that cutoff to the total integrated value of the spectrum.'''
        top = quad(self.functional_form, cutoff, self.region[1])[0]
        return top / self.total_flux

    def plot(self, perMev=True):
        '''Plot flux data'''
        plt.figure(99)
        if perMev:
            plt.plot(self.step_x, self.step_y)
        else:
            plt.plot(self.stepu_x, self.stepu_y)
        plt.xlabel('Energy MeV')
        plt.ylabel('Flux $MeV^{-1}cm^{-2}s^{-1}$')
        plt.xlim(1E-8, 20)
        plt.xscale('log')
        plt.yscale('log')
        plt.legend()

    def info(self):
        s = '********************************************\n'
        s += 'Information on spectrum: {}\n'.format(self.label)
        s += '*******************************************\n'
        s += 'Number of Bins: {}\n'.format(self.num_bins)
        s += 'Number of Bin Edges: {}\n'.format(self.num_edges)
        s += 'Scaling Factor: {}\n'.format(self.scaling_factor)
        s += 'Total Flux: {:8.5e}\n'.format(self.total_flux)
        s += '*******************************************\n'
        s += ' |  Bin Edges  | Values     |\n'
        for i in range(len(self.edges)):
            s += ' |  {:8.5e}             |\n'.format(self.edges[i])
            if i != len(self.values - 1):
                s += ' |              {:8.5e} |\n'.format(self.values[i])
        print(s)

if __name__ == '__main__':
    bins = np.array([1E-11, 1E-5, 0.5, 1, 20])
    vals = np.array([0.1, 1, 10, 2])
    err = vals * 0.05
    s = Spectrum(bins, vals, err, 2, label='Test Spectrum')
    s.plot()
    s.change_bins(np.array([1E-8, 1E-4, 1E-2, 0.6, 1, 15, 20]))
    s.plot()
    s.info()
