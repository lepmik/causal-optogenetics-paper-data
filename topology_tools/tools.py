import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

def prime_factors(n):
    result = []
    for i in itertools.chain([2], itertools.count(3, 2)):
        if n <= 1:
            break
        while n % i == 0:
            n //= i
            result.append(i)
    return result


def closest_product(n):
    primes = np.array(prime_factors(n))
    products = np.array([(np.prod(primes[:i]), np.prod(primes[i:]))
                         for i in range(len(primes))])
    idx = np.argmin(abs(np.diff(products, axis=1)))
    return products[idx]


def hasattrs(object, *strings):
    if isinstance(strings, str):
        return hasattr(object, strings)
    for string in strings:
        try:
            getattr(object, string)
        except:
            return False
    return True


def despine(ax=None, left=False, right=True, top=True, bottom=False,
            xticks=True, yticks=True, all_sides=False):
    """
    Removes axis lines
    """
    if all_sides:
        left, right, top, bottom = [True] * 4
    if ax is None:
        ax = plt.gca()
    if not isinstance(ax, list):
        ax = [ax]
    for a in ax:
        try:
            a.spines['top'].set_visible(not top)
            a.spines['right'].set_visible(not right)
            a.spines['left'].set_visible(not left)
            a.spines['bottom'].set_visible(not bottom)
        except AttributeError:
            raise
        except:
            raise
        if not xticks:
            a.get_xaxis().tick_bottom()
            plt.setp(a.get_xticklabels(), visible=False)
        if not yticks:
            a.get_yaxis().tick_left()
            plt.setp(a.get_yticklabels(), visible=False)


def set_style(style='article', sns_style='white', w=1, h=1):
    sdict = {
        'article': {
            # (11pt font = 360pt, 4.98) (10pt font = 345pt, 4.77)
            'figure.figsize' : (4.98 * w, 2 * h),
            'figure.autolayout': False,
            'lines.linewidth': 2,
            'font.size'      : 11,
            'legend.frameon' : False,
            'legend.fontsize': 11,
            'font.family'    : 'serif',
            'text.usetex'    : True
        },
        'notebook': {
            'figure.figsize' : (16, 9),
            'axes.labelsize' : 25,
            'lines.linewidth': 2,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'axes.titlesize' : 20,
            'font.size'      : 20,
            'legend.frameon' : False,
            'legend.fontsize': 20,
            'font.family'    : 'serif',
            'text.usetex'    : True
        }
    }
    rc = sdict[style]
    plt.rcParams.update(rc)
    sns.set(rc=rc, style=sns_style,
            color_codes=True)
