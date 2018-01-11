import re
import os


def remove_whitespace(str):
    """
    Returns the string str with all whitespace removed.
    """

    p = re.compile(r'\s+')
    return p.sub('', str)


def get_environment():
    """
    Returns a string identifying the current environment.
    """

    try:
        hostname = os.environ['SHORT_HOSTNAME']

    except KeyError:

        try:
            hostname = os.environ['HOSTNAME']

        except KeyError:
            return 'scratch'

    if re.match('charles[0-9]{2,}|renown|anne', hostname):
        return 'cluster'

    else:
        return 'scratch'
