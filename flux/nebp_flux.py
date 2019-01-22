import re
import numpy as np


def extract_mcnp(par):
    """Utility that grabs the data from a """

    assert par in ('n', 'g'), 'Incorrect particle type.'

    # for neutrons
    if par == 'n':

        # number of erg and cos groups
        data_shape = (57, 14)

        # open file w/ neutron data
        with open('mcnp/triga_finale_n.o') as F:
            mcnp_file = F.read()

        # split tally region from file
        mcnp_file = mcnp_file.split('1tally       21')[1].split('1tally       31')[0]

        # use regular expression to grab all data
        pattern = re.compile(r'    \d.\d\d\d\dE[+-]\d\d   \d.\d\d\d\d\dE[+-]\d\d \d.\d\d\d\d')
        results = re.findall(pattern, mcnp_file)

        # create empty array to house data
        data = np.empty((len(results), 2))

        # loop through
        for i, line in enumerate(results):

            # break line at spaces
            line = line.split()

            # put data into numpy array
            data[i] = float(line[1]), float(line[2])

        # get rid of totals and all the zeros
        data = data[:-data_shape[0]][data_shape[0]:]
        data = data[len(data)//2:]
        
        print(data)
        

    # gammas not currently implemented
    elif par == 'g':

        # raise an error
        assert False, 'Gamma data is not implemented yet.'

    return


if __name__ == '__main__':
    extract_mcnp('n')
