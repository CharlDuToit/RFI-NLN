import matplotlib.pyplot as plt


def plot_sky_model(sky_model):
    print('Type of sky model:', sky_model.component_type)
    print('Number of sources:', sky_model.Ncomponents)

    # Extract the source positions and fluxes
    ra = sky_model.ra
    dec = sky_model.dec
    flux = sky_model.stokes[0, 0, :]

    # Plot the source positions with color and size showing source fluxes.
    plt.scatter(ra, dec, s=flux, c=flux)
    plt.colorbar(label='Flux Density [Jy]')
    plt.xlabel('RA [degree]')
    plt.ylabel('Dec [degree]')
    plt.title('50 Brightest GLEAM Sources')
    plt.show()
