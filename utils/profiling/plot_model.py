import keras

from utils.common import model_plot_file


def plot_model_to_file(model, file):
    if not file:
        file = 'plot.png'
    if len(file) > 4:
        if '.png' != file[-4:]:
            file += '.png'

    keras.utils.plot_model(model, file, show_shapes=True)


def plot_model_to_file_kwargs(model, **kwargs):
    file = model_plot_file(**kwargs)
    plot_model_to_file(model, file)

