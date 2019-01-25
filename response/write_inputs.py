from template import empty_template


def write_input(det):
    """Utility that writes two mcnp inputs (response function and
    integrated response) given a detector type."""

    # check input
    message = "Detector must be of type 'bs, 'ft' or 'wt'."
    assert det in ('empty', 'bs', 'ft', 'wt'), message

    # first, grab the empty bp geometry template
    mcnp_input = empty_template

    with open(det + '.i', 'w+') as F:
        F.write(mcnp_input)

    return


def test_write_input():
    """A small utility used to test write_input()."""

    # test empty case
    write_input('empty')


if __name__ == '__main__':
    test_write_input()
