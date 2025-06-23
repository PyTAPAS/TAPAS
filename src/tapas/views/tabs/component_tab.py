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
import logging

# Third‑Party Imports

# PyQt6
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
)

# Matplotlib and Related Imports
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# Local Application Imports
from ...utils import utils
from ...configurations import exceptions as exc, messages as msg
from ...views.tabwidgets.component_tabwidgets import InputWidget, CanvasWidget

logger = logging.getLogger(__name__)


class ComponentTab(QWidget):
    def __init__(self, tabwidget, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3, controller, config):
        super().__init__()
        self.tab = tabwidget
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.component_controller = controller
        self.config = config
        self.ds = '1'
        self.need_GUI_update = False
        self.InitUI()
        self.update_config()

    def InitUI(self):
        # -------- create Widgets ------------------------------------------------------------------
        self.tw_input = InputWidget()
        self.tw_canvas = CanvasWidget()

        # -------- add Widgets to layout -----------------------------------------------------------
        self.components_layout = QHBoxLayout()
        self.setLayout(self.components_layout)
        self.components_layout.addWidget(self.tw_input, stretch=1)
        self.components_layout.addWidget(self.tw_canvas, stretch=5)

        # -------- connect Widgets to view / controller --------------------------------------------
        self.tw_input.rb_ds_1.toggled.connect(self.update_dataset)
        self.tw_input.rb_ds_2.toggled.connect(self.update_dataset)
        self.tw_input.rb_ds_3.toggled.connect(self.update_dataset)
        self.tw_input.sb_components.valueChanged.connect(self.plot_svd)
        self.tw_input.le_linlog.editingFinished.connect(self.plot_svd)

        # -------- listen to model event signals ---------------------------------------------------
        models = (self.ta_model, self.ta_model_ds1,
                  self.ta_model_ds2, self.ta_model_ds3)
        for i in models:
            i.rawdata_changed.connect(self.queue_update_GUI)
            i.data_changed.connect(self.queue_update_GUI)

    def queue_update_GUI(self) -> None:
        ''' called, if the raw or ds data is changed. GUI update waits till tab is selected '''
        self.need_GUI_update = True
        if self.tab.currentIndex() == 4:
            self.update_GUI()

    def update_GUI(self) -> None:
        ''' function called directly by the main window everytime the Tab is clicked
        or if the Tab is active and data was changed (handled by queue_update_GUI).
        Tab is updated if needed (handled by the need_GUI_update boolean). '''
        if not self.need_GUI_update:
            return
        self.component_controller.calculate_svd(ds=self.ds)
        self.plot_svd()
        self.need_GUI_update = False

    def update_config(self) -> None:
        '''updates configuration and QWidgets'''
        self.config.add_handler(
            'component_w_sb_components', self.tw_input.sb_components)
        self.config.add_handler('component_w_le_linlog',
                                self.tw_input.le_linlog)

    def update_dataset(self, state: bool) -> None:
        ''' triggred if dataset radiobutton is clicked. updates the cached fitting values and plots the data '''
        if state:
            if self.sender().objectName() == "ds1":
                self.ds = '1'

            elif self.sender().objectName() == "ds2":
                self.ds = '2'
            elif self.sender().objectName() == "ds3":
                self.ds = '3'
            self.component_controller.calculate_svd(ds=self.ds)
            self.plot_svd()

    def plot_svd(self) -> None:
        ''' gets calculated svd from the controller and plots the svd'''

        if not self.component_controller.verify_rawdata():
            utils._remove_widget(self.tw_canvas.w_canvas)
            self.tw_canvas.w_canvas = (QLabel(msg.Widgets.i13))
            self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)
            return

        components = self.tw_input.sb_components.value()  # Number of leading singular components to return
        x, y, Z = self.component_controller.get_data(self.ds)

        U, s, Vh, Z_calc = self.component_controller.get_svd(
            ds=self.ds, components=components)

        self.sc = utils.PlotCanvas()
        self.toolbar = NavigationToolbar2QT(self.sc, )
        self.fig = self.sc.fig
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sc)
        self.ax_U = self.sc.fig.add_subplot(2, 2, 4)
        self.ax_s = self.sc.fig.add_subplot(2, 2, 2)
        self.ax_res = self.sc.fig.add_subplot(2, 2, 3)
        self.ax_Vh = self.sc.fig.add_subplot(2, 2, 1)

        try:
            self.linthresh = utils.Converter.convert_str_input2float(
                self.tw_input.le_linlog.text())
            if self.linthresh is None:
                self.ax_res.set_yscale('linear')
                self.ax_U.set_xscale('linear')
            else:
                self.ax_res.set_yscale(
                    'symlog', linthresh=self.linthresh, linscale=1)

                self.ax_U.set_xscale(
                    'symlog', linthresh=self.linthresh, linscale=1)
        except AttributeError:
            self.component_controller.call_statusbar("error", msg.Error.e05)
            return
        except ValueError:
            self.component_controller.call_statusbar("error", msg.Error.e02)
            return

        self.ax_s.set_xlabel("Singular Value Index")
        self.ax_s.set_ylabel("normalized singular values")
        self.ax_U.axhline(y=0, color=self.ax_U.xaxis.label.get_color(
        ), linestyle="--", linewidth=1, zorder=0)
        self.ax_U.set_xlabel(msg.Labels.delay)
        self.ax_U.set_ylabel("norm. Intensity")
        self.ax_Vh.set_xlabel(msg.Labels.wavelength)
        self.ax_Vh.set_ylabel("norm. Intensity")
        self.ax_res.set_xlabel(msg.Labels.wavelength)
        self.ax_res.set_ylabel(msg.Labels.delay)
        self.ax_s.set_title("Components")
        self.ax_U.set_title("Temporal Vectors")
        self.ax_Vh.set_title("Spectral Vectors")
        self.ax_res.set_title("Residuals")
        self.ax_Vh.axhline(y=0, color=self.ax_Vh.xaxis.label.get_color(
        ), linestyle="--", linewidth=1, zorder=0)

        self.ax_Vh.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
        self.ax_U.xaxis.set_major_formatter(self.sc.delay_formatter0)
        self.ax_res.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
        self.ax_res.yaxis.set_major_formatter(self.sc.delay_formatter0)
        self.ax_s.locator_params(axis='x', nbins=components)
        self.ax_s.set_ylim(-0.05, 1.05)
        try:
            for idx, s1 in enumerate(s):
                self.ax_s.scatter(idx + 1, s1, marker='x')
                self.ax_Vh.plot(x, Vh[idx, :] * (-s1))
                self.ax_U.plot(y, U[:, idx] * (-s1))
        except ValueError:
            return
        self.ax_s.plot(np.arange(1, len(
            s)+1), s, color=self.ax_s.xaxis.label.get_color(), zorder=0, linewidth=0.5, alpha=0.8)
        normalization = colors.TwoSlopeNorm(vmin=-5, vmax=5, vcenter=0)
        X, Y = np.meshgrid(x, y)
        self.pcolormesh_plot = self.ax_res.pcolormesh(
            X, Y, Z - Z_calc, shading='auto', norm=normalization)
        divider = make_axes_locatable(self.ax_res)
        cax = divider.append_axes('right', size='5%', pad=0.05)

        self.cb = self.fig.colorbar(mappable=self.pcolormesh_plot, cax=cax, location="right",
                                    shrink=0.6, label=msg.Labels.delA_error)
        self.cb.minorticks_on()
        self.sc.axes_mapping = {
            self.ax_res: (self.pcolormesh_plot, self.cb)}  # mapping needed for applying scroll zooming

        self.sc.mpl_connect('scroll_event', lambda event: self.sc._zoom_TA(
            event, self.sc.axes_mapping))

        layout.removeWidget(self.tw_canvas.w_canvas)
        utils._remove_widget(self.tw_canvas.w_canvas)
        self.tw_canvas.w_canvas = QWidget()
        self.tw_canvas.w_canvas.setLayout(layout)
        self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)
        self.component_controller.call_statusbar("info", msg.Status.s23)
