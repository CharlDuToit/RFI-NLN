import matplotlib.pyplot as plt

def plot_array(uvd):

    # Get antennas positions and corresponding antenna numbers
    antpos, antnum = uvd.get_ENU_antpos()

    # Plot the EN antenna position.
    plt.scatter(antpos[:, 0], antpos[:, 1])
    for i, antnum in enumerate(uvd.antenna_numbers):
        plt.text(antpos[i, 0], antpos[i, 1], antnum)
    plt.show()
