import numpy as np
import re
import matplotlib.pyplot as plt


def extract_foil(filename):

    #
    with open(filename, 'r') as F:
        output = F.read()

    # grab all tally data
    pattern = re.compile(r'AU-198     \d.\d\d\d      \d.\d\d\d\d\d\dE[+-]\d\d\d   \d.\d\d\d\d\d\dE[+-]\d\d\d')
    results = re.findall(pattern, output)

    # convert to np array
    return np.array([float(r) for r in results[0].split()[2:]])


def au_activities():
    """Extract the data from the gold foil measurements"""

    foil_ids = ('2', '13', '4', '5', '6')
    foil_activities = np.zeros((len(foil_ids), 2))

    for i, foil_id in enumerate(foil_ids):

        foil_activities[i] = extract_foil('4_5_19/au' + foil_id + '.RPT')

    return foil_activities


if __name__ == '__main__':
    fa = au_activities()
    print(fa)
