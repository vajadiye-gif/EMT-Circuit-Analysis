"""
widgets.py
----------
Reusable ipywidgets patterns used across notebooks.

Usage
-----
    from src.utils.widgets import make_play_slider
    from ipywidgets import interact
    from IPython.display import display

    play, slider = make_play_slider(n_frames)
    interact(my_plot_fn, idx=slider)
    display(play)

Note: for Google Colab, call
    from google.colab import output
    output.enable_custom_widget_manager()
before using this module.
"""

import ipywidgets as widgets


def make_play_slider(n_frames: int, interval: int = 150):
    """
    Create a Play button and IntSlider that are jslinked.

    Parameters
    ----------
    n_frames : total number of frames (slider max = n_frames - 1)
    interval : animation interval in milliseconds

    Returns
    -------
    play   : widgets.Play
    slider : widgets.IntSlider  (pass this to interact())
    """
    play = widgets.Play(
        value=0, min=0, max=n_frames - 1,
        step=1, interval=interval, description="Play"
    )
    slider = widgets.IntSlider(
        value=0, min=0, max=n_frames - 1, step=1,
        description='Frame',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='500px')
    )
    widgets.jslink((play, 'value'), (slider, 'value'))
    return play, slider


def make_species_toggle(options=None, default="ZEB"):
    """
    ToggleButtons for selecting a species to visualise.

    Parameters
    ----------
    options : list of species names; defaults to the 6 dynamic species
    default : initially selected species

    Returns
    -------
    widgets.ToggleButtons
    """
    if options is None:
        options = ["miR200", "mZEB", "ZEB", "SNAIL", "mSNAIL", "miR34"]
    return widgets.ToggleButtons(
        options=options,
        value=default,
        description="Species:",
        button_style="info",
    )
