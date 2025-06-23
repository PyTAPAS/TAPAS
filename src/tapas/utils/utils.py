# -*- coding: utf-8 -*-
"""
Copyright © 2025, Philipp Frech

This file is part of TAPAS.

    TAPAS is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    TAPAS is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with TAPAS.  If not, see <http://www.gnu.org/licenses/>.
"""


# Standard Library Imports
import colorsys
import logging
import logging.config
import re
from decimal import Decimal

# Third‑Party Imports
import matplotlib.colors as mcolors
import matplotlib.ticker as tk
import numpy as np
from numpy.typing import NDArray
import scipy.constants as con
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# PyQt6 Imports
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QFrame,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget
)

logger = logging.getLogger(__name__)


@staticmethod
def _remove_widget(widget: QWidget) -> None:
    ''' helper, that removes widget from GUI '''
    widget.setParent(None)
    widget.deleteLater()


class SquareContainer(QWidget):
    '''A QWidget subclass that enforces a square layout for a single child widget. '''

    def __init__(self, child, parent=None, size=(600, 600)):
        super().__init__(parent)
        self.child = child
        self._min_size = size
        self.setMinimumSize(*size)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(self)
        layout.addWidget(child, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

    def resizeEvent(self, event):
        ''' Handle container resize events by resizing the child to a centered square. '''
        # Get the current size of the container
        w = self.width()
        h = self.height()
        # Determine the side length for a square
        side = min(w, h)
        # Center the child widget
        x = (w - side) // 2
        y = (h - side) // 2
        self.child.setGeometry(x, y, side, side)
        super().resizeEvent(event)


class ScrollableSquareContainer(QScrollArea):
    ''' A scrollable container that enforces a square layout for a single child widget. '''

    def __init__(self, child, parent=None):
        super().__init__(parent)
        # Allow widget to resize inside scroll area
        self.setWidgetResizable(True)
        # Wrap your child widget (your PlotCanvas) in a SquareContainer
        self.squareContainer = SquareContainer(child)
        # Enforce a minimum size of 600x600 for the square container
        self.squareContainer.setMinimumSize(600, 600)
        self.setWidget(self.squareContainer)


class ClickCursor():
    '''
    Interactive crosshair controller for 2D transient absorption data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The main image axes displaying Z vs (X, Y).
    ax_delA : matplotlib.axes.Axes
        Side axes for the spectral cut (ΔA vs X).
    ax_kin : matplotlib.axes.Axes
        Side axes for the kinetic cut (kinetics vs Y).
    X : np.ndarray, shape (M,)
        Wavelength (nm) grid corresponding to Z’s columns.
    Y : np.ndarray, shape (N,)
        Delay (s) grid corresponding to Z’s rows.
    Z : np.ndarray, shape (N, M)
        Data array of ΔA values at each (Y, X).
    formatter_x : callable
        Function to format X (wavelength) values into legend strings.
    formatter_y : callable
        Function to format Y (delay) values into legend strings.
    '''

    def __init__(self, ax, ax_delA, ax_kin, X, Y, Z, formatter_x, formatter_y):
        self.ax = ax
        self.ax_delA = ax_delA
        self.ax_kin = ax_kin
        self.X = X
        self.Y = Y
        self.Z = Z
        self.delay_formatter = formatter_y
        self.nm_formatter_legend = formatter_x

        # Compute the midpoint of X and Y.
        mid_x = (np.min(X) + np.max(X)) / 2
        mid_y = (np.min(Y) + np.max(Y)) / 2

        # Create crosshair lines in the main plot at the midpoints.
        self.horizontal_line_ax = ax.axhline(
            y=mid_y, color='k', lw=0.8, ls='--')
        self.vertical_line_ax = ax.axvline(x=mid_x, color='k', lw=0.8, ls='--')

        # Create vertical crosshair lines in the cut axes.
        # For ax_delA, we assume the cut is taken at a specific X position, so we set its vertical line at mid_x.
        self.vertical_line_ax_delA = ax_delA.axvline(
            x=mid_x, color='k', lw=0.8, ls='--')
        # For ax_kin, assuming the cut is taken along Y, we set the vertical line at mid_y.
        self.vertical_line_ax_kin = ax_kin.axvline(
            x=mid_y, color='k', lw=0.8, ls='--')

        # Determine the indices corresponding to the midpoints.
        idx_x = (np.abs(X - mid_x)).argmin()
        idx_y = (np.abs(Y - mid_y)).argmin()

        # Initialize the cut plot lines using the corresponding slice of Z.
        # For the delA cut, we take the row of Z at idx_y and plot it against X.
        self.interactive_delA_line, = ax_delA.plot(
            X, Z[idx_y, :], color="C0", label="")
        # For the kin cut, we take the column of Z at idx_x and plot it against Y.
        self.interactive_kin_line, = ax_kin.plot(
            Y, Z[:, idx_x], color="C0", label="")

        # Flag for MMB dragging.
        self.dragging = False
        # Store the latest x, y positions.
        self.last_x = None
        self.last_y = None

        # Setup a QTimer for throttling updates at roughly 10 Hz.
        self.update_timer = QTimer()
        self.update_timer.setInterval(100)  # ~10 Hz
        self.update_timer.timeout.connect(self.throttled_update)

        self.delay_formatter = formatter_y
        self.nm_formatter_legend = formatter_x

    def update_data(self, X, Y, Z):
        ''' replace the underlying data arrays and reset the cut‐line plots. '''
        self.X = X
        self.Y = Y
        self.Z = Z
        # Remove and re-create the interactive lines:
        if self.interactive_delA_line in self.ax_delA.lines:
            self.interactive_delA_line.remove()
        if self.interactive_kin_line in self.ax_kin.lines:
            self.interactive_kin_line.remove()
        self.set_cross_hair_visible(False)
        self.interactive_delA_line, = self.ax_delA.plot(
            self.X, np.zeros_like(self.X), color="C0", label="")

        self.interactive_kin_line, = self.ax_kin.plot(
            self.Y, np.zeros_like(self.Y), color="C0", label="")

    def set_cross_hair_visible(self, visible):
        ''' Show or hide all crosshair lines in the main and cut plots. '''

        self.horizontal_line_ax.set_visible(visible)
        self.vertical_line_ax.set_visible(visible)
        self.vertical_line_ax_delA.set_visible(visible)
        self.vertical_line_ax_kin.set_visible(visible)

    def on_mouse_press(self, event):
        ''' Begin dragging crosshairs on middle‐mouse-button press in the main axes. '''
        if event.inaxes != self.ax:
            return
        if event.button == 2:  # Middle Mouse Button pressed.
            self.dragging = True
            self.last_x, self.last_y = event.xdata, event.ydata
            self.set_cross_hair_visible(True)  # Show crosshair on press.
            self.update_timer.start()  # Start the timer when dragging begins.

    def on_mouse_move(self, event):
        ''' Record the latest mouse position while dragging. '''
        # Simply record the latest position if dragging.
        if not self.dragging or event.inaxes != self.ax:
            return
        self.last_x, self.last_y = event.xdata, event.ydata

    def throttled_update(self):
        ''' Perform a single throttled update of crosshairs and cut plots. '''
        # Only update if dragging and valid data exists.
        if not self.dragging or self.last_x is None or self.last_y is None:
            return

        x, y = self.last_x, self.last_y

        # Update crosshair positions in the main plot.
        self.horizontal_line_ax.set_ydata([y])
        self.vertical_line_ax.set_xdata([x])
        # Update the cut axes crosshairs.
        self.vertical_line_ax_delA.set_xdata([x])
        self.vertical_line_ax_kin.set_xdata([y])

        # Find the nearest indices.
        idx_X = (np.abs(self.X - x)).argmin()
        idx_Y = (np.abs(self.Y - y)).argmin()

        # Update cut plot data.
        new_delA_data = self.Z[idx_Y, :]
        new_kin_data = self.Z[:, idx_X]
        self.interactive_delA_line.set_ydata(new_delA_data)
        self.interactive_kin_line.set_ydata(new_kin_data)

        # Let Matplotlib autoscale the axes.
        self.ax_delA.relim()
        self.ax_delA.autoscale_view()
        self.ax_kin.relim()
        self.ax_kin.autoscale_view()

        # Update legend labels.
        label_delA = f"{self.delay_formatter(self.Y[idx_Y])} s"
        label_kin = f"{self.nm_formatter_legend(self.X[idx_X])} nm"
        self.interactive_delA_line.set_label(label_delA)
        self.interactive_kin_line.set_label(label_kin)
        self.ax_delA.legend()
        self.ax_kin.legend()

        # Redraw the cut axes.
        self.ax_delA.figure.canvas.draw_idle()
        self.ax_kin.figure.canvas.draw_idle()

    def on_mouse_release(self, event):
        ''' Finalize dragging on middle-mouse-button release. '''
        if event.inaxes != self.ax or event.button != 2:
            return

        self.dragging = False
        self.update_timer.stop()

        # Final update on release.
        idx_X = (np.abs(self.X - event.xdata)).argmin()
        idx_Y = (np.abs(self.Y - event.ydata)).argmin()

        final_delA_data = self.Z[idx_Y, :]
        final_kin_data = self.Z[:, idx_X]
        self.interactive_delA_line.set_ydata(final_delA_data)
        self.interactive_kin_line.set_ydata(final_kin_data)

        self.ax_delA.relim()
        self.ax_delA.autoscale_view()
        self.ax_kin.relim()
        self.ax_kin.autoscale_view()

        label_delA = f"{self.delay_formatter(self.Y[idx_Y])} s"
        label_kin = f"{self.nm_formatter_legend(self.X[idx_X])} nm"
        self.interactive_delA_line.set_label(label_delA)
        self.interactive_kin_line.set_label(label_kin)
        self.ax_delA.legend()
        self.ax_kin.legend()

        self.ax_delA.figure.canvas.draw_idle()
        self.ax_kin.figure.canvas.draw_idle()


class PlotCanvas(Canvas):
    ''' A Matplotlib FigureCanvas tailored for transient-absorption data with interactive zoom and formatting. '''

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.nm_formatter_ax = tk.FuncFormatter(self.nm_formatter_ax)

        self.nm_formatter_legend = tk.FuncFormatter(self.nm_formatter_legend)
        self._last_norms: dict[Axes, mcolors.TwoSlopeNorm] = {}
        self.delay_formatter0 = tk.EngFormatter(places=0, sep="\N{THIN SPACE}")
        self.delay_formatter1 = tk.EngFormatter(places=1, sep="\N{THIN SPACE}")
        self.emcee_formatter = tk.FuncFormatter(self.avoid_milli)
        self.fig = Figure(figsize=(width, height), dpi=dpi)

        super(PlotCanvas, self).__init__(self.fig)

    def _zoom_TA(self, event: MouseEvent,
                 axes_mapping: dict[Axes, tuple[object, Colorbar | None]]) -> None:
        """
        Update the normalization of a pcolormesh plot and its associated colorbar based on a scroll event.

        This function checks which Axes the event occurred in by comparing event.inaxes
        with the keys in the axes_mapping dictionary. If the event's Axes is registered,
        the function calculates new normalization limits based on an internal zoom factor,
        updates the normalization of the mappable (e.g., a QuadMesh returned by pcolormesh),
        and then updates the corresponding colorbar (if available).

        Parameters:
            event (MouseEvent): The Matplotlib scroll event.
            axes_mapping (Dict[Axes, Tuple[Any, Optional[Colorbar]]]): A dictionary mapping
                each Axes to a tuple containing:
                    - The mappable object (e.g., the QuadMesh from a pcolormesh plot)
                    - The associated Colorbar (or None if no colorbar is used for that Axes)

        Example of axes_mapping:
            {
                ax1: (pcolormesh_plot1, colorbar1),
                ax2: (pcolormesh_plot2, colorbar2)
            }

        Returns:
            None
        """
        # Check if event occurred in one of our registered axes
        if event.inaxes not in axes_mapping:
            return

        scale_rate = 1.5
        factor = 1/scale_rate if event.button == 'up' else scale_rate
        mappable, cb = self.axes_mapping[event.inaxes]
        norm = mappable.norm  # assume it's already a TwoSlopeNorm
        new_vmin = norm.vmin * factor
        new_vmax = norm.vmax * factor

        new_norm = mcolors.TwoSlopeNorm(vmin=new_vmin, vmax=new_vmax, vcenter=0)
        self._last_norms[event.inaxes] = new_norm
        # Retrieve the mappable and colorbar for the Axes that received the event.
        mappable, cb = axes_mapping[event.inaxes]
        mappable.set_norm(new_norm)
        if cb is not None:
            cb.update_normal(mappable)
        self.draw_idle()

    def nm_formatter_ax(self, tick_value, pos):
        val = float(tick_value) / 1e-9
        return f'{val:.0f}'

    def nm_formatter_legend(self, tick_value, pos):
        val = float(tick_value) / 1e-9
        return f'{val:.1f}'

    def avoid_milli(self, x, pos=None):
        if x == 0:
            return "0"
        abs_x = abs(x)
        if 1e-3 <= abs_x < 1:
            # For milli-range, just show a normal float
            # adjust the formatting (decimal places) to suit your needs
            return f"{x:.3g}"
        else:
            return self.delay_formatter1(x)  # fall back on EngFormatter


class Converter:
    def m_in_eV(wavelength: float) -> float:
        energy = (con.h * con.c) / (wavelength) * 1 / con.e
        return energy

    def eV_in_m(energy: float) -> float:
        wavelength = (con.h * con.c) / (energy) * 1 / con.e
        return wavelength

    def create_scrollable_widget(widget: QWidget, min_width: int = None, max_width: int = None,
                                 min_height: int = None, max_height: int = None,
                                 horizontal_scroll: bool = True, use_container: bool = True) -> QWidget:
        """
    Create a scrollable widget that wraps the given widget in a QScrollArea,
    applying fixed or flexible size constraints as specified.

    Parameters
    ----------
    widget : QWidget
        The widget to be displayed inside the scrollable area.
    min_width : int, optional
        The minimum width for the scroll area (and container, if used).
        If both min_width and max_width are provided and equal, a fixed width is enforced.
    max_width : int, optional
        The maximum width for the scroll area (and container, if used).
    min_height : int, optional
        The minimum height for the scroll area.
    max_height : int, optional
        The maximum height for the scroll area.
    horizontal_scroll : bool, optional
        If True, the horizontal scroll bar will appear as needed.
        If False, the horizontal scroll bar is always hidden.
        The default is True.
    use_container : bool, optional
        If True, the widget is wrapped in an extra container with a QVBoxLayout.
        This container can help with layout consistency when static content is expected.
        However, using the container may suppress dynamic changes of child widgets
        (e.g., in a Visualization Tab widget where dynamic updates are required).
        In that case, set this to False.
        The default is True.

    Returns
    -------
    scroll_area : QScrollArea
        The scroll area widget that wraps the given widget (or its container) with
        the specified size constraints and scroll bar policies.
    """
        if use_container:
            container = QWidget()
            layout = QVBoxLayout(container)
            layout.addWidget(widget)
        else:
            container = widget
        # If a fixed width is provided, enforce it on the container.
        if min_width is not None and max_width is not None and min_width == max_width:
            container.setFixedWidth(min_width)
        else:
            container.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)

        scroll_area = QScrollArea()
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        if horizontal_scroll:
            scroll_area.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            scroll_area.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        if use_container:
            scroll_area.setWidgetResizable(False)
        else:
            scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(container)

        if min_width:
            scroll_area.setMinimumWidth(min_width)
        if max_width:
            scroll_area.setMaximumWidth(max_width)
        if min_height:
            scroll_area.setMinimumHeight(min_height)
        if max_height:
            scroll_area.setMaximumHeight(max_height)
        return scroll_area

    def convert_str_input2float(string: str) -> float | bool | None:
        string = string.replace(",", ".")
        string = string.replace(" ", "")
        if string == "":
            return None
        elif string.lower() == "inf":
            return np.inf
        elif string.lower() == "-inf":
            return -np.inf
        elif string == "True" or string == "true":
            return True
        elif string == 'False' or string == "false":
            return False

        match = re.match(r"^([+-]?[0-9]*\.?[0-9]+(?:e[+-]?[0-9]+)?)([a-zA-Zµ\"]*)$", string)
        if not match:
            raise ValueError

        num_str, unit_str = match.groups()

        units = {
            "fs": 1e-15, "fm": 1e-15,
            "ps": 1e-12, "pm": 1e-12,
            "ns": 1e-9,  "nm": 1e-9,
            "us": 1e-6,  "um": 1e-6, "µs": 1e-6, "µm": 1e-6,
            "ms": 1e-3,  "mm": 1e-3,
            "cm": 1e-2,
            "in": 0.0254, "\"": 0.0254,
            "s": 1.0, "m": 1.0,
            "": 1.0
        }
        try:
            value = float(num_str)
        except ValueError:
            raise ValueError

        multiplier = units.get(unit_str.lower(), None)
        if multiplier is None:
            raise ValueError

        return value * multiplier

    def convert_str_input2list(string: str) -> list[float]:
        string = re.sub(r"\s|\(|\)|\[|\]", "", string)
        str_list = string.split(',')
        float_list = []
        for v in str_list:
            value = Converter.convert_str_input2float(v)
            if value is not None:
                float_list.append(value)

        return float_list

    def convert_nm_float2string(floating_number: float) -> str:
        return str(int(round(floating_number, 9) * 1e9)) + 'nm'

    def convert_str_input2colorlist(string: str) -> list[str]:
        if string == "":
            return []
        string = re.sub(r"\s|\(|\)|\[|\]|\"|\'", "", string)
        color_list = string.split(',')
        for color in color_list:
            if not mcolors.is_color_like(color):
                raise ValueError

        return color_list

    def fitting_colorlist(color_list: list[str]) -> list[str]:
        darkened_colors = []
        for color in color_list:

            try:
                color = mcolors.cnames[color]
            except KeyError:
                pass  # Assume color is already in hex or another valid format

            rgb = mcolors.to_rgb(color)
            # Convert RGB to HLS (Hue, Lightness, Saturation)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            # Reduce the lightness by the specified factor (ensuring it doesn't go below 0)
            new_l = max(0, l * 0.5)
            # Convert back to RGB and add to the list
            darkened_colors.append(colorsys.hls_to_rgb(h, new_l, s))

        return darkened_colors
    
    def components_colorlist(base_color : str, number : int) ->list[str]:
        variants = []
        try:
            color = mcolors.cnames[base_color]
        except KeyError:
            pass  # Assume color is already in hex or another valid format

        rgb = mcolors.to_rgb(color)
        # Convert RGB to HLS (Hue, Lightness, Saturation)
        h, l, s = colorsys.rgb_to_hls(*rgb)
        levels = np.linspace(l, 1.0, number + 1)[:-1]
        for li in levels:
            r, g, b = colorsys.hls_to_rgb(h, li, s)
            variants.append(mcolors.to_hex((r, g, b)))

        return variants

    def offset_corr(ydata: NDArray) -> NDArray:
        ydata = ydata - min(ydata)
        return ydata

    def plt_normalization(xdata: NDArray, ydata: NDArray, x_min: float, x_max: float) -> float:
        idx = np.where(ydata == np.max(
            ydata[(xdata > x_min) & (xdata < x_max)]))[0][0]
        # find index, where ydata is max in plotting window (inbetween xmin & xmax)
        return ydata[idx]   # max y value in plotting window normalized
