import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tlc.tlc import tlc


def test_plot_jv_shared_axis():
    """Two SQ instances can plot J-V on the same axes without error."""
    t1 = tlc(1.1, l_sq=True)
    t1.calculate()
    t2 = tlc(1.5, l_sq=True)
    t2.calculate()

    _, ax = plt.subplots()
    ax1 = t1.plot_jv(ax=ax)
    ax2 = t2.plot_jv(ax=ax)
    assert ax1 is ax
    assert ax2 is ax
    plt.close("all")
