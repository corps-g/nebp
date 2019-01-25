from template import mcnp_template


def write_input(det, bonner_size=12):
    """Utility that writes two mcnp inputs (response function and
    integrated response) given a detector type."""

    # check input
    message = "Detector must be of type 'empty', 'bs, 'ft' or 'wt'."
    assert det in ('empty', 'bs', 'ft', 'wt'), message

    # first, grab the empty bp geometry template
    mcnp_input = mcnp_template

    # calculate some things
    # convert bonner size from diameter in inches to radius in cm
    bonner_size = (bonner_size / 2) * 2.54

    # select mcnp fill
    if det == 'empty':
        fill = ('      ', '      ')
    elif det == 'bs':
        fill = ('FILL=1', 'FILL=1')
    elif det == 'ft':
        raise NotImplementedError
    elif det == 'wt':
        raise NotImplementedError

    # format the mcnp
    mcnp_input = mcnp_input.format(*fill, bonner_size)

    # write to file
    with open(det + '.i', 'w+') as F:
        F.write(mcnp_input)

    return


def test_write_input():
    """A small utility used to test write_input()."""

    # test empty case
    write_input('empty')
    write_input('bs')


if __name__ == '__main__':
    test_write_input()
