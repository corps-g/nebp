import numpy as np
import re
from scipy.constants import N_A
from scipy.integrate import odeint
from scipy.interpolate import interp1d


class Au_Foil_Data(object):

    """This object is responsible for handling the postprocessing and storage
    of the gold foil activation data from the 4/5/19 irradiation in the ksu
    triga mark ii northeast beam port."""

    def __init__(self):
        """This handles all of the post processing on the gold foil responses
        upon initialization of the object, as well as the storing of any
        relevent constants used in the processing."""

        # decay data
        # source: https://www.nndc.bnl.gov/nudat2/decaysearchdirect.jsp?nuc=198AU&unc=nds
        self.halflife = 2.6941 * 24 * 3600
        self.decay_constant = np.log(2) / self.halflife
        self.intensity = 0.9562

        # molar mass (g/mol)
        # source: https://www-nds.iaea.org/amdc/ame2016/mass16.txt
        self.M = 196.966570114

        # density (g/cm3)
        # source: Brown, Theodore L. and Lemay Jr., H. Eugene,
        #         Chemistry: The Central Science. Prentice Hall Inc,
        #         Englewood, New Jersey, 1985: 10.
        self.rho = 19.32

        # the foils used in the experiment
        self.foil_ids = ('2', '13', '4', '5', '6', '7', '8', '9', '10')

        # number of foils used in the analysis
        self.n = len(self.foil_ids)

        # foil masses (g)
        self.masses = np.loadtxt('4_5_19/masses.txt', skiprows=1)

        # the nominal power level in W(th), 100 kW(th)
        self.P = 1E5

        # get activity info
        self.extract_RPT_data()

        # get irradiation power profile information
        self.extract_power_profile()

        # reset the absolute time so irradiation begins at 0s.
        self.shift_times()

        # calculate the saturation activities
        self.calc_a_sat()

        # calculate saturation per atom
        self.calc_a_sat_atom()

        return

    def convert_time(self, date, time, am_pm):
        """A useful utility for taking date time info in dd/mm/yy hh:mm:ss
        AM/PM and converting into seconds. In this function, it is assumed that
        the month and year remain constant and that everything is shifted back
        5 days, so the 5th is day zero."""

        # parse out the day
        d = float(date.split('/')[1])

        # parse out the times
        h, m, s = [float(t) for t in time.split(':')]

        # add 12 hours if pm
        pm_extra = 0 if am_pm is 'AM' else 12

        # return the absolute time in seconds
        return (d - 5) * 86400 + (h + pm_extra) * 3600 + m * 60 + s

    def extract_RPT_data(self):
        """A utility to read the .RPT files from the Genie2K software and
        extract the irradiation times and measured activities."""

        # a data structure to store the irradiation information
        # 4 is the number of values to be extracted from each file
        foil_activities = np.zeros((len(self.foil_ids), 4))

        # grab info for each foil
        for i, foil_id in enumerate(self.foil_ids):

            # create filename
            filename = '4_5_19/au' + foil_id + '.RPT'

            # read the file
            with open(filename, 'r') as F:
                output = F.read()

            # grab the activity and error data
            pattern = re.compile(r'AU-198     \d.\d\d\d      \d.\d\d\d\d\d\dE[+-]\d\d\d   \d.\d\d\d\d\d\dE[+-]\d\d\d')
            results = re.findall(pattern, output)

            # convert those values to floats and store
            act, err = [float(r) for r in results[-1].split()[2:]]

            # pull the detector live times
            pattern = re.compile(r'Live Time                       :\s+\d+.\d seconds')
            results = re.findall(pattern, output)

            # convert live times to floats
            live_time = float(results[0].split()[-2])

            # pull the counting times
            pattern = re.compile(r'Acquisition Started             : \d/\d?\d/\d\d\d\d\s+\d?\d:\d\d:\d\d [AP]M')
            results = re.findall(pattern, output)

            # split data into format accepted by self.convert_time
            t_c = results[0].split()[-3:]

            # convert times to absolute
            t_c = self.convert_time(*t_c)

            # store data in structure
            foil_activities[i] = act, err, live_time, t_c

        # unpack the columns into arrays stored by the object
        self.a_c, self.a_c_error, self.live, self.t_c = foil_activities.T

        return

    def extract_power_profile(self):
        """A utility that reads the strip chart data and pulls out the
        times and powers over which the irradiation occurred, then scales
        those values by the nominal power."""

        # read the file, slicing off the last 2000 lines, as they are unneeded
        with open('4_5_19/5APR2019.txt', 'r') as F:
            lines = F.readlines()[:-2000]

        # create structures to house the data
        self.times = np.empty(len(lines))
        self.powers = np.empty(len(lines))

        # loop through each line of data
        for i, line in enumerate(lines):

            # due the the format of the data, it's best to split at commas
            # and semicolons
            line = re.split('[,;]', line)

            # convert the times to absolute
            self.times[i] = self.convert_time(*line[:2], 'AM')

            # power is in column 6 in units of percent of MW(th), so
            # convert power from percent of a MW(th) to W(th)
            self.powers[i] = float(line[9]) * 0.01 * 1E6

        # data contains some random zeros, so let's fix that
        for i, power in enumerate(self.powers):

            # basically, if the power is a zero
            if not power:

                # set the power equal to the previous value in the array
                # this assumes the power doesn't deviate so much over a second
                self.powers[i] = self.powers[i - 1]

        # store the power profile by dividing by the nominal power
        self.power_profile = self.powers / self.P

        return

    def shift_times(self):
        """A utility that will shift all absolute times back by the first value
        in the irradiation times."""

        # pull the first value in the irradiation times
        shift = self.times[0]

        # shift every absolute temporal value
        self.times -= shift
        self.t_c -= shift

        return

    def calc_a_sat(self):
        """This utility backs out the saturation activities for each foil."""

        # we need an array of times that at least spans the entire time domain
        full_times = np.arange(0, self.t_c[-1] + 100)

        # convert the power profile to a callable function
        P = interp1d(self.times, self.power_profile, bounds_error=False, fill_value=0)

        # the differential form of the isotope production/decay balance
        def N_prime(N, t):
            return P(t) - self.decay_constant * N

        # solve the differential equation over the time domain
        N_t = odeint(N_prime, y0=0, t=full_times)[:, 0]

        # convert the number of atoms to activities
        self.activity_profile = N_t * self.decay_constant

        # divide each foil by the saturation ratio, which is the ratio of the
        # counting activity to the saturation activity
        self.a_sat = np.array([self.a_c[i] / self.activity_profile[int(self.t_c[i])] for i in range(self.n)])

        return

    def calc_a_sat_atom(self):
        """This utility normalizes the saturation activities to activity per
        sample atom."""

        # initialize a structure to house the number of atoms in the samples
        self.atoms = np.empty(len(self.foil_ids))

        # calculate for each foil
        for i, foil_id in enumerate(self.foil_ids):

            # calculate the number of atoms for the foil
            # the foil ids are one indexed, so they need to be shifted to
            # grab the mass
            self.atoms[i] = (self.masses[int(foil_id) - 1] * N_A) / self.M

        # divide the saturation activities by the number of atoms in the sample
        self.a_sat_atom = self.a_sat / self.atoms

        return


if __name__ == '__main__':
    fa = Au_Foil_Data()
