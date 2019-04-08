import numpy as np
import re
import matplotlib.pyplot as plt


class Au_Foil_Data(object):

    """ Au Foil Data """

    def __init__(self):
        """The init function."""

        # decay data
        self.halflife = 2.7 * 24 * 3600
        self.decay_constant = np.log(2) / self.halflife

        # number of foils used in the analysis
        self.n = 5

        # the nominal power level in W(th), 100 kW(th)
        self.P = 1E5

        # get activity info
        self.extract_RPT_data()

        # get power profile information
        self.extract_power_profile()

        return

    def convert_time(self, date, time, am_pm):
        """Converts the time."""

        # parse out the day
        d = float(date.split('/')[1])

        # parse out the times
        h, m, s = [float(t) for t in time.split(':')]

        # add 12 hours if pm
        pm_extra = 0 if am_pm is 'AM' else 12

        # return the absolute time in seconds
        return (d - 5) * 86400 + (h + pm_extra) * 3600 + m * 60 + s

    def extract_RPT_data(self):
        """Extracts foil data."""

        # name foils to be used
        foil_ids = ('2', '13', '4', '5', '6')
        foil_activities = np.zeros((len(foil_ids), 4))

        # grab info for each foil
        for i, foil_id in enumerate(foil_ids):

            # create filename
            filename = '4_5_19/au' + foil_id + '.RPT'

            with open(filename, 'r') as F:
                output = F.read()

            # grab all tally data
            pattern = re.compile(r'AU-198     \d.\d\d\d      \d.\d\d\d\d\d\dE[+-]\d\d\d   \d.\d\d\d\d\d\dE[+-]\d\d\d')
            results = re.findall(pattern, output)

            # activity and error
            act, err = [float(r) for r in results[0].split()[2:]]

            # grab all time data
            pattern = re.compile(r'Live Time                       :\s+\d+.\d seconds')
            results = re.findall(pattern, output)

            # convert live time
            live_time = float(results[0].split()[-2])

            # counting time
            pattern = re.compile(r'Acquisition Started             : \d/\d/\d\d\d\d\s+\d?\d:\d\d:\d\d [AP]M')
            results = re.findall(pattern, output)

            # pull out relevent time data
            t_c = results[0].split()[-3:]

            t_c = self.convert_time(*t_c)

            # store data
            foil_activities[i] = act, err, live_time, t_c

        return foil_activities

    def extract_power_profile(self):
        """Extract power profile."""

        # open the file
        with open('4_5_19/5APR2019.txt', 'r') as F:
            lines = F.readlines()[:-2000]

        # open structures
        times = np.empty(len(lines))
        powers = np.empty(len(lines))

        # loop and parse out times and power
        for i, line in enumerate(lines):

            #
            line = re.split('[,;]', line)

            #
            times[i] = self.convert_time(*line[:2], 'AM')

            # convert power from percent of a MW(th) to W(th)
            powers[i] = float(line[6]) * 0.01 * 1E6

        # convert all zero powers to the power before it
        for i, power in enumerate(powers):

            #
            if not power:

                #
                powers[i] = powers[i - 1]

        
        

if __name__ == '__main__':
    fa = Au_Foil_Data()
