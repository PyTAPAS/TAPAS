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
import datetime
import logging
import warnings
from pathlib import Path

# Third‑Party Imports

# PyQt6 Imports
from PyQt6.QtCore import Qt, QRegularExpression, QSignalBlocker
from PyQt6.QtGui import QRegularExpressionValidator, QFontDatabase, QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QCompleter,
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QGridLayout,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

# Matplotlib Imports
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as tk
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends import backend_svg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.colors import ListedColormap
from matplotlib.patches import PathPatch
from matplotlib.axes import Axes

# Other Third‑Party Imports
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.typing import NDArray
import corner

# Local Application Imports
from ...configurations import exceptions as exc, messages as msg
from ...utils import utils
from ...views.tabwidgets.visualization_tabwidgets import (
    SelectPlotWidget,
    CanvasWidget,
    PropertiesWidget,
    SurfPlotProperties,
    DelAPlotProperties,
    KinTraceProperties,
    LocalFitProperties,
    GlobalFitProperties_2D,
    GlobalFitProperties_EASDAS,
    GlobalFitProperties_conc,
    GlobalFitProperties_DelA,
    GlobalFitProperties_KinTrace,
    GlobalFitProperties_emcee,
)

logger = logging.getLogger(__name__)


class VisualizeTab(QWidget):
    def __init__(self, tabwidget, abs_model, em_model, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3, controller, config):
        super().__init__()

        self.tab = tabwidget
        self.abs_model = abs_model
        self.em_model = em_model
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.visualize_controller = controller
        self.config = config
        self.ds = '1'
        self.need_GUI_update = False
        self.need_data_update = False
        self.project_path = None
        self.results_dir = None

        self.font_property = fm.FontProperties(["DejaVu Sans", "sans-serif"])
        self.fig_width, self.fig_height = None, None
        self.x_min, self.x_max = None, None
        self.y_min, self.y_linlog, self.y_max = None, None, None
        self.z_min, self.z_max, self.z_center = None, None, 0
        self.xmin_hide, self.xmin_hide2, self.xmax_hide, self.xmax_hide2 = None, None, None, None
        self.delay_cut_list = []
        self.wavelength_trace_list = []
        self.custom_color_list = []
        self.norm_min, self.norm_max = None, None
        self.checkboxes = []
        self.InitUI()

    def InitUI(self):
        # -------- create Widgets ------------------------------------------------------------------
        self.tw_select_plot = SelectPlotWidget()
        self.tw_canvas = CanvasWidget()
        self.tw_properties = PropertiesWidget()
        self.tw_surf_plot_properties = SurfPlotProperties()
        self.tw_delA_plot_properties = DelAPlotProperties()
        self.tw_kin_trace_properties = KinTraceProperties()
        self.tw_local_fit_properties = [LocalFitProperties(), GlobalFitProperties_emcee()]
        self.tw_global_fit_properties = [
            GlobalFitProperties_2D(), GlobalFitProperties_EASDAS(), GlobalFitProperties_conc(),
            GlobalFitProperties_DelA(), GlobalFitProperties_KinTrace(), GlobalFitProperties_emcee()]
        self.update_config()

        # -------- add Widgets to layout -----------------------------------------------------------
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(utils.Converter.create_scrollable_widget(
            self.tw_select_plot, min_width=310, max_width=310, horizontal_scroll=False), 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.tw_canvas, 0, 1,)
        layout.addWidget(utils.Converter.create_scrollable_widget(
            self.tw_properties, min_width=370, max_width=370, horizontal_scroll=False,
            use_container=False), 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

        # -------- connect Widgets to view / controller --------------------------------------------
        self.tw_select_plot.rb_ds_1.toggled.connect(self.update_dataset)
        self.tw_select_plot.rb_ds_2.toggled.connect(self.update_dataset)
        self.tw_select_plot.rb_ds_3.toggled.connect(self.update_dataset)

        self.tw_select_plot.rb_2d_plot.toggled.connect(
            self.update_properties_gui)
        self.tw_select_plot.rb_delA_plot.toggled.connect(
            self.update_properties_gui)
        self.tw_select_plot.rb_kin_trace.toggled.connect(
            self.update_properties_gui)
        self.tw_select_plot.rb_local_fit.toggled.connect(
            self.update_properties_gui)
        self.tw_select_plot.rb_global_fit.toggled.connect(
            self.update_properties_gui)

        self.tw_select_plot.pb_add_rcParam.clicked.connect(
            self.show_rcParams_dialog)
        self.tw_select_plot.le_fig_size_w.editingFinished.connect(
            self.update_current_style)
        self.tw_select_plot.le_fig_size_h.editingFinished.connect(
            self.update_current_style)
        self.tw_select_plot.check_display_size.stateChanged.connect(
            self.plot_data)
        self.tw_select_plot.sb_label_size.valueChanged.connect(
            self.plot_data)
        self.tw_select_plot.sb_tick_size.valueChanged.connect(
            self.plot_data)
        self.tw_select_plot.font_cb.currentFontChanged.connect(
            lambda _: self.update_current_style(update_all=False)
        )
        self.tw_select_plot.cb_font_style.currentTextChanged.connect(
            lambda _: self.update_current_style(update_all=False)
        )
        self.tw_select_plot.pb_save_fig.clicked.connect(self.save_fig)
        self.tw_select_plot.pb_save_as_fig.pressed.connect(
            lambda: self.save_fig(save_as=True))

        tabwidgets = [self.tw_surf_plot_properties,
                      self.tw_global_fit_properties[0]]
        for tw in tabwidgets:
            tw.w_trimm.le_xmin.editingFinished.connect(
                self.check_wavelength_input)
            tw.w_trimm.le_xmax.editingFinished.connect(
                self.check_wavelength_input)
            tw.w_trimm.le_ymin.editingFinished.connect(
                self.check_delay_input)
            tw.w_trimm.le_ymax.editingFinished.connect(
                self.check_delay_input)
            tw.w_view_manipulations.le_linlog.editingFinished.connect(
                self.check_delay_input)
            tw.w_trimm.le_zmin.editingFinished.connect(
                self.check_z_input)
            tw.w_trimm.le_zmax.editingFinished.connect(
                self.check_z_input)
            tw.w_trimm.sb_zcenter.valueChanged.connect(
                lambda _: self.check_z_input())
            tw.w_hide_area.le_xmax_hide.editingFinished.connect(
                self.check_hide_input)
            tw.w_hide_area.le_xmax_hide2.editingFinished.connect(
                self.check_hide_input)
            tw.w_colormap.cb_select_cmap.currentIndexChanged.connect(
                self.plot_data)
            tw.w_colormap.check_cmap.stateChanged.connect(
                self.plot_data)

            tw.w_show_second_ax.check_show_second_ax.stateChanged.connect(
                self.plot_data)
            tw.w_colormap.cb_pos_cmap.currentIndexChanged.connect(
                self.plot_data)

            tw.w_show_info.check_show_info.stateChanged.connect(
                self.plot_data)
            tw.w_show_pump.check_show_pump.stateChanged.connect(
                self.plot_data)

            tw.w_view_manipulations.sb_lin_ratio.valueChanged.connect(
                self.plot_data)
            tw.w_view_manipulations.sb_log_ratio.valueChanged.connect(
                self.plot_data)

        self.tw_surf_plot_properties.w_show_ss.check_show_ss.stateChanged.connect(
            self.update_ss_ratio)
        self.tw_surf_plot_properties.w_view_manipulations.sb_ss_ratio.valueChanged.connect(
            self.plot_data)
        self.tw_surf_plot_properties.w_show_delA_cuts.check_show_delA_cuts.stateChanged.connect(
            self.plot_data)
        self.tw_surf_plot_properties.w_show_kin_cuts.check_show_kin_cuts.stateChanged.connect(
            self.plot_data)

        self.tw_delA_plot_properties.w_select_data_color.le_delay_list.editingFinished.connect(
            self.check_delay_list_input)
        self.tw_delA_plot_properties.w_trimm.le_xmin.editingFinished.connect(
            self.check_wavelength_input)
        self.tw_delA_plot_properties.w_trimm.le_xmax.editingFinished.connect(
            self.check_wavelength_input)
        self.tw_delA_plot_properties.w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_delA_plot_properties.w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)
        self.tw_delA_plot_properties.w_trimm.le_linlog_z.editingFinished.connect(
            self.plot_data)

        self.tw_delA_plot_properties.w_trimm.cb_zscale.currentIndexChanged.connect(
            self.plot_data)

        self.tw_delA_plot_properties.w_hide_area.le_xmax_hide.editingFinished.connect(
            self.check_hide_input)
        self.tw_delA_plot_properties.w_hide_area.le_xmax_hide2.editingFinished.connect(
            self.check_hide_input)
        self.tw_delA_plot_properties.w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_delA_plot_properties.w_select_data_color.cb_select_cmap.currentIndexChanged.connect(
            self.plot_data)
        self.tw_delA_plot_properties.w_show_second_ax.check_show_second_ax.stateChanged.connect(
            self.plot_data)
        self.tw_delA_plot_properties.w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        self.tw_delA_plot_properties.w_show_pump.check_show_pump.stateChanged.connect(
            self.plot_data)
        self.tw_delA_plot_properties.w_show_legend.check_show_legend.stateChanged.connect(
            self.plot_data)
        self.tw_delA_plot_properties.w_show_legend.cb_legend_loc.currentIndexChanged.connect(
            self.plot_data)

        self.tw_kin_trace_properties.w_select_data_color.le_wavelength_list.editingFinished.connect(
            self.check_wavelength_list_input)
        self.tw_kin_trace_properties.w_trimm.le_ymin.editingFinished.connect(
            self.check_delay_input)
        self.tw_kin_trace_properties.w_trimm.le_ymax.editingFinished.connect(
            self.check_delay_input)
        self.tw_kin_trace_properties.w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_kin_trace_properties.w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)

        self.tw_kin_trace_properties.w_trimm.le_linlog_y.editingFinished.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_trimm.cb_yscale.currentIndexChanged.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_trimm.le_linlog_z.editingFinished.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_trimm.cb_zscale.currentIndexChanged.connect(
            self.plot_data)

        self.tw_kin_trace_properties.w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_kin_trace_properties.w_select_data_color.cb_select_cmap.currentIndexChanged.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_normalize.le_norm_min.editingFinished.connect(
            self.check_norm_input)
        self.tw_kin_trace_properties.w_normalize.le_norm_max.editingFinished.connect(
            self.check_norm_input)
        self.tw_kin_trace_properties.w_normalize.check_normalize.stateChanged.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_normalize.check_abs_value.stateChanged.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_show_legend.check_show_legend.stateChanged.connect(
            self.plot_data)
        self.tw_kin_trace_properties.w_show_legend.cb_legend_loc.currentIndexChanged.connect(
            self.plot_data)

        self.tw_local_fit_properties[0].w_trimm.le_ymin.editingFinished.connect(
            self.check_delay_input)
        self.tw_local_fit_properties[0].w_trimm.le_ymax.editingFinished.connect(
            self.check_delay_input)
        self.tw_local_fit_properties[0].w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_local_fit_properties[0].w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)

        self.tw_local_fit_properties[0].w_trimm.le_linlog_y.editingFinished.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_trimm.cb_yscale.currentIndexChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_trimm.le_linlog_z.editingFinished.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_trimm.cb_zscale.currentIndexChanged.connect(
            self.plot_data)

        self.tw_local_fit_properties[0].w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_local_fit_properties[0].w_select_data_color.cb_select_cmap.currentIndexChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_normalize.le_norm_min.editingFinished.connect(
            self.check_norm_input)
        self.tw_local_fit_properties[0].w_normalize.le_norm_max.editingFinished.connect(
            self.check_norm_input)
        self.tw_local_fit_properties[0].w_normalize.check_normalize.stateChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_normalize.check_abs_value.stateChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_show_data.check_show_data.stateChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_show_conc.check_show_conc.stateChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_show_tau.check_show_tau.stateChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_show_residuals.check_show_residuals.stateChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        for checkbox in self.tw_local_fit_properties[0].w_select_fit.check_fits:
            checkbox.stateChanged.connect(self.plot_data)
        self.tw_local_fit_properties[0].w_show_legend.check_show_legend.stateChanged.connect(
            self.plot_data)
        self.tw_local_fit_properties[0].w_show_legend.cb_legend_loc.currentIndexChanged.connect(
            self.plot_data)

        for global_plot in self.tw_global_fit_properties+[self.tw_local_fit_properties[1]]:
            for checkbox in global_plot.w_select_fit.check_fits:
                checkbox.stateChanged.connect(self.plot_data)
        self.tw_global_fit_properties[0].w_select_2D_plot.cb_2D_plot.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[0].w_select_2D_plot.le_merge_thresh.editingFinished.connect(
            self.plot_data)

        self.tw_global_fit_properties[1].w_trimm.le_xmin.editingFinished.connect(
            self.check_wavelength_input)
        self.tw_global_fit_properties[1].w_trimm.le_xmax.editingFinished.connect(
            self.check_wavelength_input)
        self.tw_global_fit_properties[1].w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[1].w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[1].w_trimm.le_linlog_z.editingFinished.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_trimm.cb_zscale.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_hide_area.le_xmax_hide.editingFinished.connect(
            self.check_hide_input)
        self.tw_global_fit_properties[1].w_hide_area.le_xmax_hide2.editingFinished.connect(
            self.check_hide_input)
        self.tw_global_fit_properties[1].w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_global_fit_properties[1].w_select_data_color.cb_select_cmap.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_show_second_ax.check_show_second_ax.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_show_pump.check_show_pump.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_normalize.le_norm_min.editingFinished.connect(
            self.check_norm_input)
        self.tw_global_fit_properties[1].w_normalize.le_norm_max.editingFinished.connect(
            self.check_norm_input)
        self.tw_global_fit_properties[1].w_normalize.check_normalize.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_show_legend.check_show_legend.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[1].w_show_legend.cb_legend_loc.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_trimm.le_ymax.editingFinished.connect(
            self.check_delay_input)
        self.tw_global_fit_properties[2].w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[2].w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)

        self.tw_global_fit_properties[2].w_trimm.le_linlog_y.editingFinished.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_trimm.cb_yscale.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_trimm.le_linlog_z.editingFinished.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_trimm.cb_zscale.currentIndexChanged.connect(
            self.plot_data)

        self.tw_global_fit_properties[2].w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_global_fit_properties[2].w_select_data_color.cb_select_cmap.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_trimm.le_ymin.editingFinished.connect(
            self.check_delay_input)
        self.tw_global_fit_properties[2].w_trimm.le_ymax.editingFinished.connect(
            self.check_delay_input)
        self.tw_global_fit_properties[2].w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[2].w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[2].w_show_legend.check_show_legend.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_show_legend.cb_legend_loc.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[2].w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_global_fit_properties[2].w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_select_data_color.le_delay_list.editingFinished.connect(
            self.check_delay_list_input)
        self.tw_global_fit_properties[3].w_select_data_color.cb_select_cmap.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_trimm.le_xmin.editingFinished.connect(
            self.check_wavelength_input)
        self.tw_global_fit_properties[3].w_trimm.le_xmax.editingFinished.connect(
            self.check_wavelength_input)
        self.tw_global_fit_properties[3].w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[3].w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[3].w_trimm.le_linlog_z.editingFinished.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_trimm.cb_zscale.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_hide_area.le_xmax_hide.editingFinished.connect(
            self.check_hide_input)
        self.tw_global_fit_properties[3].w_hide_area.le_xmax_hide2.editingFinished.connect(
            self.check_hide_input)
        self.tw_global_fit_properties[3].w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_global_fit_properties[3].w_show_data.check_show_data.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_show_second_ax.check_show_second_ax.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_show_residuals.check_show_residuals.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_show_pump.check_show_pump.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_show_legend.check_show_legend.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[3].w_show_legend.cb_legend_loc.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_select_data_color.le_wavelength_list.editingFinished.connect(
            self.check_wavelength_list_input)
        self.tw_global_fit_properties[4].w_select_data_color.cb_select_cmap.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_trimm.le_ymin.editingFinished.connect(
            self.check_delay_input)
        self.tw_global_fit_properties[4].w_trimm.le_ymax.editingFinished.connect(
            self.check_delay_input)
        self.tw_global_fit_properties[4].w_trimm.le_zmin.editingFinished.connect(
            self.check_z_input)
        self.tw_global_fit_properties[4].w_trimm.le_zmax.editingFinished.connect(
            self.check_z_input)

        self.tw_global_fit_properties[4].w_trimm.le_linlog_y.editingFinished.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_trimm.cb_yscale.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_trimm.le_linlog_z.editingFinished.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_trimm.cb_zscale.currentIndexChanged.connect(
            self.plot_data)

        self.tw_global_fit_properties[4].w_select_data_color.le_custom_colors.editingFinished.connect(
            self.check_colorlist_input)
        self.tw_global_fit_properties[4].w_normalize.le_norm_min.editingFinished.connect(
            self.check_norm_input)
        self.tw_global_fit_properties[4].w_normalize.le_norm_max.editingFinished.connect(
            self.check_norm_input)
        self.tw_global_fit_properties[4].w_normalize.check_normalize.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_normalize.check_abs_value.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_show_info.check_show_info.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_show_legend.check_show_legend.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_show_legend.cb_legend_loc.currentIndexChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_show_residuals.check_show_residuals.stateChanged.connect(
            self.plot_data)
        self.tw_global_fit_properties[4].w_show_data.check_show_data.stateChanged.connect(
            self.plot_data)

        tw_list = [self.tw_global_fit_properties[5].w_corner_manipulation,
                   self.tw_local_fit_properties[1].w_corner_manipulation]
        for tw in tw_list:

            for sb in (
                tw.sb_bins,
                tw.sb_label_pad,
                tw.sb_subplot_pad,
                tw.sb_tick_num
            ):
                sb.valueChanged.connect(self.plot_data)

            # checkboxes
            for cb in (
                tw.check_truth,
                tw.check_show_titles,
                tw.check_show_quantiles,
                tw.check_plot_datapoints,
                tw.check_plot_contours,
                tw.check_plot_density
            ):
                cb.stateChanged.connect(self.plot_data)

        # -------- set default properties widget ---------------------------------------------------
        self.tw_properties.w_current_properties = self.tw_surf_plot_properties
        self.tw_properties.view_layout.addWidget(
            self.tw_properties.w_current_properties)

        # -------- listen to model event signals ---------------------------------------------------
        models = (self.abs_model, self.em_model, self.ta_model,
                  self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        for i in models:
            i.rawdata_changed.connect(self.queue_update_GUI)
            i.data_changed.connect(self.queue_update_GUI)
            i.local_fit_changed.connect(
                lambda: self.queue_update_GUI(only_fitting_list=True))
            i.global_fit_changed.connect(
                lambda: self.queue_update_GUI(only_fitting_list=True))

    def export_data(self) -> None:
        '''
        exports the selected plot to csv and adds metadata. The results directory
        is calculated based on the metadata in the input tab.

        Returns
        -------
        None
        '''

        # -------- get results directory -----------------------------------------------------------
        if self.tw_select_plot.le_results_dir.text() != "":
            self.results_dir = Path(self.tw_select_plot.le_results_dir.text())
        elif self.project_path is not None:
            self.results_dir = Path(self.project_path.parent)
        else:
            self._change_results_dir()
        if self.results_dir is None:
            self.visualize_controller.call_statusbar("error", msg.Error.e17)
            return

        # -------- lcreate folder ------------------------------------------------------------------
        experiment, sample = self.visualize_controller.get_exp_names()
        if experiment != '':
            self.results_dir = self.results_dir.joinpath(
                'Exports_' + experiment + '/')
            if sample != '':
                self.results_dir = self.results_dir.joinpath(sample + '/')
        elif sample != '':
            self.results_dir = self.results_dir.joinpath(
                'Exports_' + sample + '/')
        else:
            self.results_dir = self.results_dir.joinpath('Exports/')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # -------- export data ---------------------------------------------------------------------
        if self.tw_select_plot.rb_2d_plot.isChecked():
            metadata, data_array = self.visualize_controller.get_hyperspectrum(
                ds=self.ds)
            self.visualize_controller.save_data(ds=self.ds, datatype='hyperspectrum',
                                                metadata=metadata, data_array=data_array,
                                                results_dir=self.results_dir)

        elif self.tw_select_plot.rb_delA_plot.isChecked():
            metadata, data_array = self.visualize_controller.get_delA_plot(
                ds=self.ds, delay_cut_list=self.delay_cut_list)
            self.visualize_controller.save_data(
                ds=self.ds, datatype='delAcuts', metadata=metadata, data_array=data_array,
                results_dir=self.results_dir)

        elif self.tw_select_plot.rb_kin_trace.isChecked():
            metadata, data_array = self.visualize_controller.get_kin_trace(
                ds=self.ds, wavelength_trace_list=self.wavelength_trace_list)
            self.visualize_controller.save_data(
                ds=self.ds, datatype='kinTrace', metadata=metadata, data_array=data_array,
                results_dir=self.results_dir)

        elif self.tw_select_plot.rb_local_fit.isChecked():
            checkboxes = self.tw_local_fit_properties[1].w_select_fit.check_fits  # list
            checked_boxes = [cb.text() for cb in checkboxes if cb.isChecked()]
            if self.tw_select_plot.rb_local_fit.menuSelected() == 'kin_trace':
                metadata, data_array = self.visualize_controller.get_local_fit(
                    ds=self.ds, selected_ukeys=checked_boxes)
                self.visualize_controller.save_data(
                    ds=self.ds, datatype='localFit', metadata=metadata, data_array=data_array,
                    results_dir=self.results_dir)
            elif self.tw_select_plot.rb_local_fit.menuSelected() == 'emcee':
                ukey = checked_boxes[0]
                metadata, data_array = self.visualize_controller.get_emcee_flatchain(
                    ds=self.ds,  ukey=ukey, local=True)
                self.visualize_controller.save_data(
                    ds=self.ds, datatype=f'localfit_{ukey}_emcee_flatchain', metadata=metadata,
                    data_array=data_array, results_dir=self.results_dir)

        elif self.tw_select_plot.rb_global_fit.isChecked():
            try:
                ukey = self.visualize_controller.get_global_ukey(
                    checkboxes=self.tw_properties.w_current_properties.w_select_fit.check_fits)
            except exc.NoSelectionError:
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e22)
                return
            if self.tw_select_plot.rb_global_fit.menuSelected() == 'plot_2d':
                data_type = self.tw_global_fit_properties[0].w_select_2D_plot.cb_2D_plot.currentText(
                )
                if data_type == 'merge':
                    data_type = 'simulated'  # both give the same export
                metadata, data_array = self.visualize_controller.get_hyperspectrum(
                    ds=self.ds, ukey=ukey, datatype=data_type)

                self.visualize_controller.save_data(
                    ds=self.ds, datatype=f'{ukey}__hyperspectrum_{data_type}', metadata=metadata,
                    data_array=data_array, results_dir=self.results_dir)
            elif self.tw_select_plot.rb_global_fit.menuSelected() == 'eas_das':

                datatype, metadata, data_array = self.visualize_controller.get_EASDAS(
                    ds=self.ds, ukey=ukey)
                self.visualize_controller.save_data(
                    ds=self.ds, datatype=f'{ukey}_{datatype}', metadata=metadata,
                    data_array=data_array, results_dir=self.results_dir)
            elif self.tw_select_plot.rb_global_fit.menuSelected() == 'conc':
                metadata, data_array = self.visualize_controller.get_conc(
                    ds=self.ds, ukey=ukey)
                self.visualize_controller.save_data(
                    ds=self.ds, datatype=f'{ukey}_concProfile', metadata=metadata,
                    data_array=data_array, results_dir=self.results_dir)
            elif self.tw_select_plot.rb_global_fit.menuSelected() == 'dela_plot':
                metadata, data_array = self.visualize_controller.get_delA_plot(
                    ds=self.ds, delay_cut_list=self.delay_cut_list, ukey=ukey)
                self.visualize_controller.save_data(
                    ds=self.ds, datatype=f'{ukey}_delAplot', metadata=metadata,
                    data_array=data_array, results_dir=self.results_dir)
            elif self.tw_select_plot.rb_global_fit.menuSelected() == 'kinetic_trace':
                metadata, data_array = self.visualize_controller.get_kin_trace(
                    ds=self.ds, wavelength_trace_list=self.wavelength_trace_list, ukey=ukey)
                self.visualize_controller.save_data(
                    ds=self.ds, datatype=f'{ukey}_kinTrace', metadata=metadata,
                    data_array=data_array, results_dir=self.results_dir)
            elif self.tw_select_plot.rb_global_fit.menuSelected() == 'emcee':
                metadata, data_array = self.visualize_controller.get_emcee_flatchain(
                    ds=self.ds,  ukey=ukey)
                self.visualize_controller.save_data(
                    ds=self.ds, datatype=f'{ukey}_emcee_flatchain', metadata=metadata,
                    data_array=data_array, results_dir=self.results_dir)

        self.visualize_controller.call_statusbar('info', msg.Status.s20)

    def queue_update_GUI(self, only_fitting_list: bool = False) -> None:
        ''' called, if the raw or ds data is changed. GUI update waits till tab is selected '''
        if only_fitting_list:
            self.need_GUI_update = True
        else:
            self.need_data_update = True
        if self.tab.currentIndex() == 3:
            self.update_GUI()

    def update_GUI(self) -> None:
        ''' function called directly by the main window everytime the Tab is clicked
        or if the Tab is active and data was changed (handled by queue_update_GUI).
        Tab is updated if needed (handled by the need_GUI_update boolean). '''
        if self.need_data_update:
            self._update_fitting_list()
            if not self.tw_select_plot.rb_2d_plot.isChecked():
                # triggeres update_properties_gui -> plot data
                self.tw_select_plot.rb_2d_plot.setChecked(True)
            else:
                self.tw_select_plot.rb_2d_plot.toggled.emit(
                    self.tw_select_plot.rb_2d_plot.isChecked())

            self.need_data_update = False
            self.need_GUI_update = False
        elif self.need_GUI_update:
            self._update_fitting_list()
            self._update_ss_toggle()
            self.need_GUI_update = False

    def show_rcParams_dialog(self) -> None:
        ''' triggred, when Pushbutton add rc Params is clicked. Opens a dialog
        and  retrives and sets rcParam key and value pairs and plots the data'''
        self.rcParams_dialog = QDialog(self)
        self.rcParams_dialog.setWindowTitle('Refine Default rcParams')

        le_rcParam_key = QLineEdit(self.rcParams_dialog)
        le_rcParam_key.setPlaceholderText("Enter rc Param")
        le_rcParam_value = QLineEdit(self.rcParams_dialog)
        le_rcParam_value.setPlaceholderText("Enter value")
        button = QPushButton("OK", self.rcParams_dialog)
        layout = QGridLayout(self.rcParams_dialog)

        rcParams_key_list = mpl.rcParams.keys()
        rcParams_key_completer = QCompleter(rcParams_key_list)
        rcParams_key_completer.setCaseSensitivity(
            Qt.CaseSensitivity.CaseInsensitive)
        le_rcParam_key.setCompleter(rcParams_key_completer)
        regex = QRegularExpression(f"^({'|'.join(map(QRegularExpression.escape, rcParams_key_list))})$",
                                   QRegularExpression.PatternOption.CaseInsensitiveOption)
        validator = QRegularExpressionValidator(regex, le_rcParam_key)
        le_rcParam_key.setValidator(validator)

        layout.addWidget(le_rcParam_key, 0, 0)
        layout.addWidget(le_rcParam_value, 0, 1)
        layout.addWidget(button, 1, 0)

        button.pressed.connect(lambda: self._update_rcParams(
            key=le_rcParam_key.text(), value=le_rcParam_value.text()))
        self.rcParams_dialog.show()

    def _update_rcParams(self, key: str, value: str):
        ''' helper called by show_rcParams_dialog. tries to apply rcParam key
        value pairs and plots the data '''
        try:
            mpl.rcParams[key] = value
        except KeyError:
            self.visualize_controller.call_statusbar("error", msg.Error.e27)
            return
        except ValueError as e:
            self.visualize_controller.call_statusbar("error", str(e))
            return
        self.plot_data()

    def _update_fitting_list(self) -> None:
        '''
        in local fit and global fits: sets the checkboxes visible und sets the name accordingly.
        Currently up to 9 fits are supported.
        Triggered when new data is added or when ds is changed

        Returns
        -------
        None.

        '''

        # -------- update local list ---------------------------------------------------------------
        fit_dict = self.visualize_controller.get_local_fit_list(ds=self.ds)
        fits = list(fit_dict) if fit_dict else []

        local_fit_checkboxes = [
            tab_widget.w_select_fit.check_fits for tab_widget in self.tw_local_fit_properties]

        for checkboxes in local_fit_checkboxes:
            # Reset each checkbox in the current tab widget
            for checkbox in checkboxes:
                checkbox.setVisible(False)
                checkbox.setChecked(False)

            # Update checkboxes using the fits list
            for idx, fit in enumerate(fits):
                if idx < len(checkboxes):
                    checkboxes[idx].setVisible(True)
                    checkboxes[idx].setText(fit)
                else:
                    self.visualize_controller.call_statusbar(
                        "error", msg.Error.e25)
                    break

        # -------- update global list --------------------------------------------------------------
        fit_dict = self.visualize_controller.get_global_fit_list(ds=self.ds)
        fits = list(fit_dict) if fit_dict else []

        global_fit_checkboxes = [
            tab_widget.w_select_fit.check_fits for tab_widget in self.tw_global_fit_properties]

        for checkboxes in global_fit_checkboxes:
            # Reset each checkbox in the current tab widget
            for checkbox in checkboxes:
                checkbox.setVisible(False)
                checkbox.setChecked(False)

            # Update checkboxes using the fits list
            for idx, fit in enumerate(fits):
                if idx < len(checkboxes):
                    checkboxes[idx].setVisible(True)
                    checkboxes[idx].setText(fit)
                else:
                    self.visualize_controller.call_statusbar(
                        "error", msg.Error.e25)
                    break

    def _change_results_dir(self) -> None:
        '''
        called by save_fig if save as is pressed. Opens a file dialog to change self.results_dir

        '''

        dir_name = QFileDialog.getExistingDirectory(self, "Select a Directory")
        if not dir_name:
            return
        self.results_dir = Path(dir_name)

        self.tw_select_plot.le_results_dir.setText(str(self.results_dir))

    def save_fig(self, save_as: bool = False) -> None:
        '''
        creates a results folder at GUI lineedit or project-path or file-dialog(save_as = True) and saves figure to it.

        Parameters
        ----------
        save_as : Boolean, optional
            DESCRIPTION. if True, results directory is set by che _change_results_dir method. The default is False.

        Returns
        -------
        None.

        '''
        if save_as:
            self._change_results_dir()
        else:
            if self.tw_select_plot.le_results_dir.text() != "":
                self.results_dir = Path(
                    self.tw_select_plot.le_results_dir.text())
            elif self.project_path is not None:
                self.results_dir = Path(self.project_path.parent)
            else:
                self._change_results_dir()
        if self.results_dir is None:
            self.visualize_controller.call_statusbar("error", msg.Error.e17)
            return
        experiment, sample = self.visualize_controller.get_exp_names()
        if experiment != '':
            self.results_dir = self.results_dir.joinpath(
                'Plots_' + experiment + '/')
            if sample != '':
                self.results_dir = self.results_dir.joinpath(sample + '/')
        elif sample != '':
            self.results_dir = self.results_dir.joinpath(
                'Plots_' + sample + '/')
        else:
            self.results_dir = self.results_dir.joinpath('Plots/')
        self.results_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.plot_data(save_fig=True)
        except FileNotFoundError:
            self.visualize_controller.call_statusbar("error", msg.Error.e17)
            return

    def check_norm_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches intervalls for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all
        else:
            sender_w_current_properties = self.sender().parent().parent()

        self.norm_min, self.norm_max = self._read_range(
            sender_w_current_properties.w_normalize, 'le_norm_min', 'le_norm_max')

        if sender_w_current_properties.w_normalize.check_normalize.isChecked():
            self.plot_data()

    def check_colorlist_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches selected colors for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all
        else:
            sender_w_current_properties = self.sender().parent().parent()

        try:
            self.custom_color_list = utils.Converter.convert_str_input2colorlist(
                sender_w_current_properties.w_select_data_color.le_custom_colors.text())

        except ValueError:
            self.visualize_controller.call_statusbar("error", msg.Error.e02)
            self.custom_color_list = []
            return
        if not update_all:
            self.plot_data()

    def check_delay_list_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches selected delay times for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all
        else:
            sender_w_current_properties = self.sender().parent().parent()
        try:
            self.delay_cut_list = utils.Converter.convert_str_input2list(
                sender_w_current_properties.w_select_data_color.le_delay_list.text())
            if not update_all:
                self.plot_data()

        except ValueError:
            self.visualize_controller.call_statusbar("error", msg.Error.e02)
            self.delay_cut_list = []

    def check_wavelength_list_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches selected wavelengths for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all
        else:
            sender_w_current_properties = self.sender().parent().parent()
        try:
            self.wavelength_trace_list = utils.Converter.convert_str_input2list(
                sender_w_current_properties.w_select_data_color.le_wavelength_list.text())
            if not update_all:
                self.plot_data()

        except ValueError:
            self.visualize_controller.call_statusbar("error", msg.Error.e02)
            self.wavelength_trace_list = []

    def check_wavelength_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches intervalls for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all

        else:
            sender_w_current_properties = self.sender().parent().parent()

        self.x_min, self.x_max = self._read_range(
            sender_w_current_properties.w_trimm, 'le_xmin', 'le_xmax')

        if not update_all:
            self.plot_data()

    def check_delay_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches intervalls for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all
        else:
            # tw_surf_plot_properties, tw_delA_plot_properties etc.
            sender_w_current_properties = self.sender().parent().parent()

        self.y_min, self.y_max = self._read_range(
            sender_w_current_properties.w_trimm, 'le_ymin', 'le_ymax')
        try:
            self.y_linlog = utils.Converter.convert_str_input2float(
                sender_w_current_properties.w_view_manipulations.le_linlog.text())

        except ValueError:
            self.visualize_controller.call_statusbar("error", msg.Error.e02)
            return
        # function also used by objects which dont have le_linlog (e.g. kin_plot_prop)
        except AttributeError:
            pass

        if not update_all:
            self.plot_data()

    def check_z_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches intervalls for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all
        else:
            sender_w_current_properties = self.sender().parent().parent()

        self.z_min, self.z_max = self._read_range(
            sender_w_current_properties.w_trimm, 'le_zmin', 'le_zmax')

        if hasattr(sender_w_current_properties.w_trimm, 'sb_zcenter'):
            self.z_center = sender_w_current_properties.w_trimm.sb_zcenter.value()

        if not update_all:
            self.plot_data()

    def _read_range(self, sender: QWidget, le_min: str, le_max: str) -> (float, float):
        '''
        Helper for check input functions. Extracts the string inputs and returns
        values.

        Parameters
        ----------
        sender : QWidget
            Parent Widget.
        le_min : str
            lineedit object of parent widget.
        le_max : str
            lineedit object of parent widget.

        Returns
        -------
        (float, float)
            holds boundaries for intervalls.

        '''
        try:
            vmin = utils.Converter.convert_str_input2float(
                getattr(sender, le_min).text()
            )
            vmax = utils.Converter.convert_str_input2float(
                getattr(sender, le_max).text()
            )
        except ValueError:
            self.visualize_controller.call_statusbar("error", msg.Error.e02)
            return None, None

        if vmin is not None and vmax is not None and vmin >= vmax:
            self.visualize_controller.call_statusbar("error", msg.Error.e06)
            return None, None

        return vmin, vmax

    def check_hide_input(self, update_all: bool | QWidget = False) -> None:
        '''
        triggered when input is changed (update_all=False) or when a new plot
        is selected (update_all=sender property tabwidget). Checks and caches intervalls for plotting

        Parameters
        ----------
        update_all : bool, QWidget
            Holds the Property Tabwidget. The default is False.

        Returns
        -------
        None
        '''
        if update_all:
            sender_w_current_properties = update_all
        else:
            sender_w_current_properties = self.sender().parent().parent()

        self.xmin_hide, self.xmax_hide = self._read_range(
            sender_w_current_properties.w_hide_area, 'le_xmin_hide', 'le_xmax_hide')

        self.xmin_hide2, self.xmax_hide2 = self._read_range(
            sender_w_current_properties.w_hide_area, 'le_xmin_hide2', 'le_xmax_hide2')

        if not update_all:
            self.plot_data()

    def update_dataset(self, state: bool):
        ''' triggred if dataset radiobutton is clicked. updates the cached fitting values and plots the data '''
        if state:

            if self.sender().objectName() == "ds1":
                self.ds = '1'

            elif self.sender().objectName() == "ds2":
                self.ds = '2'
            elif self.sender().objectName() == "ds3":
                self.ds = '3'
            self._update_fitting_list()
            self.plot_data()

    def _clear_canvas(self) -> None:
        ''' helper to clean old widgets '''
        self.tw_canvas.w_canvas.hide()
        self.tw_canvas.w_canvas.setParent(
            None)  # remove widget from layout
        self.tw_canvas.w_canvas.deleteLater()
        self.tw_canvas.w_canvas = QLabel(msg.Widgets.i13, self.tw_canvas)
        self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)

    def _update_ss_toggle(self) -> None:
        '''updates the show steady-state toggle in the GUI if data is given'''
        if self.visualize_controller.verify_steadystate():
            self.tw_surf_plot_properties.w_show_ss.setEnabled(True)

        else:
            self.tw_surf_plot_properties.w_show_ss.check_show_ss.setChecked(
                False)
            self.tw_surf_plot_properties.w_show_ss.setEnabled(False)
            self.tw_surf_plot_properties.w_view_manipulations.sb_ss_ratio.setEnabled(
                False)

    def update_ss_ratio(self) -> None:
        '''updates the steady-state ratio spinbox in the GUI if selected'''
        if self.tw_surf_plot_properties.w_show_ss.check_show_ss.isChecked():

            self.tw_surf_plot_properties.w_view_manipulations.sb_ss_ratio.setEnabled(
                True)
        else:

            self.tw_surf_plot_properties.w_view_manipulations.sb_ss_ratio.setEnabled(
                False)
        self.plot_data()

    @staticmethod
    def _qt_weight_to_mpl_weight(css: int) -> str:
        """100-900 → keyword Matplotlib’s weight_dict knows."""
        if css < 250:
            return "ultralight"
        elif css < 350:
            return "light"
        elif css < 450:
            return "normal"      # 400
        elif css < 550:
            return "medium"
        elif css < 650:
            return "semibold"
        elif css < 750:
            return "bold"        # 700
        elif css < 850:
            return "heavy"
        else:
            return "black"       # 900

    @staticmethod
    def _qt_style_to_mpl_stretch(style_txt: str) -> str:
        s = style_txt.lower()

        if "ultra" in s and "condensed" in s:
            return "ultra-condensed"
        if "extra" in s and "condensed" in s:
            return "extra-condensed"
        if ("semi" in s and "condensed" in s) or "compressed" in s:
            return "semi-condensed"
        if "condensed" in s or "narrow" in s:
            return "condensed"

        if "ultra" in s and "expanded" in s:
            return "ultra-expanded"
        if "extra" in s and "expanded" in s:
            return "extra-expanded"
        if "semi" in s and "expanded" in s:
            return "semi-expanded"
        if "expanded" in s:
            return "expanded"

        return "normal"

    def update_current_style(self, update_all: bool | QWidget = False) -> None:
        '''updates the figure dimensions and style without tuouching rc Params'''

        if self.sender().objectName() == 'fig_size_w' or update_all:
            try:
                self.fig_width = utils.Converter.convert_str_input2float(
                    self.tw_select_plot.le_fig_size_w.text())

            except ValueError:
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e02)

        if self.sender().objectName() == 'fig_size_h' or update_all:
            try:
                self.fig_height = utils.Converter.convert_str_input2float(
                    self.tw_select_plot.le_fig_size_h.text())

            except ValueError:
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e02)

        if self.sender().objectName() in ['fig_font', 'fig_font_style'] or update_all:

            fam = self.tw_select_plot.font_cb.currentFont().family()
            if fam:
                style = self.tw_select_plot.cb_font_style.currentText() or "Normal"
                qf = QFontDatabase.font(fam, style, -1)  # Qt font obj

                if qf.style() == QFont.Style.StyleItalic:
                    mpl_style = "italic"
                elif qf.style() == QFont.Style.StyleOblique:
                    mpl_style = "oblique"
                else:
                    mpl_style = "normal"

                mpl_stretch = self._qt_style_to_mpl_stretch(style)  # condensed / expanded …
                mpl_weight = self._qt_weight_to_mpl_weight(qf.weight())

                # Convert to Matplotlib
                fp = fm.FontProperties(
                    family=[fam] + ["DejaVu Sans", "sans-serif"],
                    style=mpl_style,  # "italic", ...
                    weight=mpl_weight,  # 'light', ...
                    stretch=mpl_stretch)  # condensed / expanded …
                self.font_property = fp

        if not update_all:
            self.plot_data()

    def update_config(self) -> None:
        '''updates configuration and standard values of QWidgets'''

        self.config.add_handler(
            'visualize_w_le_fig_size_w', self.tw_select_plot.le_fig_size_w)
        self.config.add_handler(
            'visualize_w_le_fig_size_h', self.tw_select_plot.le_fig_size_h)
        self.config.add_handler('visualize_w_le_fig_dpi',
                                self.tw_select_plot.le_fig_dpi)
        self.config.add_handler(
            'visualize_w_sb_label_size', self.tw_select_plot.sb_label_size)
        self.config.add_handler(
            'visualize_w_sb_tick_size', self.tw_select_plot.sb_tick_size)
        self.config.add_handler(
            'visualize_w_cb_fig_format', self.tw_select_plot.cb_fig_format)

        tw = self.tw_surf_plot_properties
        widgets = [tw.w_trimm.le_xmin, tw.w_trimm.le_xmax, tw.w_trimm.le_ymin, tw.w_trimm.le_ymax,
                   tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,  tw.w_trimm.sb_zcenter,
                   tw.w_view_manipulations.le_linlog, tw.w_view_manipulations.sb_lin_ratio,
                   tw.w_view_manipulations.sb_log_ratio, tw.w_view_manipulations.sb_ss_ratio,
                   tw.w_colormap.check_cmap, tw.w_colormap.cb_select_cmap, tw.w_colormap.cb_pos_cmap,
                   tw.w_hide_area.le_xmin_hide, tw.w_hide_area.le_xmax_hide,
                   tw.w_hide_area.le_xmin_hide2, tw.w_hide_area.le_xmax_hide2,
                   tw.w_show_delA_cuts.check_show_delA_cuts,  tw.w_show_kin_cuts.check_show_kin_cuts,
                   tw.w_show_second_ax.check_show_second_ax, tw.w_show_pump.check_show_pump,
                   tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_2D_{w.objectName()}', w)
            del blocker

        tw = self.tw_delA_plot_properties
        widgets = [tw.w_trimm.le_xmin, tw.w_trimm.le_xmax,  tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,
                   tw.w_trimm.cb_zscale, tw.w_show_legend.check_show_legend, tw.w_show_legend.cb_legend_loc,
                   tw.w_select_data_color.le_delay_list, tw.w_select_data_color.cb_select_cmap,
                   tw.w_select_data_color.le_custom_colors, tw.w_hide_area.le_xmin_hide, tw.w_hide_area.le_xmax_hide,
                   tw.w_hide_area.le_xmin_hide2, tw.w_hide_area.le_xmax_hide2,
                   tw.w_show_second_ax.check_show_second_ax, tw.w_show_pump.check_show_pump,
                   tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_delA_{w.objectName()}', w)
            del blocker

        tw = self.tw_kin_trace_properties
        widgets = [tw.w_trimm.le_ymin, tw.w_trimm.le_ymax,  tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,
                   tw.w_trimm.cb_yscale, tw.w_trimm.cb_zscale, tw.w_show_legend.check_show_legend,
                   tw.w_show_legend.cb_legend_loc, tw.w_select_data_color.le_wavelength_list,
                   tw.w_select_data_color.cb_select_cmap, tw.w_select_data_color.le_custom_colors,
                   tw.w_normalize.check_normalize, tw.w_normalize.le_norm_min, tw.w_normalize.le_norm_max,
                   tw.w_normalize.check_abs_value, tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_kinTrace_{w.objectName()}', w)
            del blocker

        tw = self.tw_local_fit_properties[0]
        widgets = [tw.w_trimm.le_ymin, tw.w_trimm.le_ymax,  tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,
                   tw.w_trimm.cb_yscale, tw.w_trimm.cb_zscale, tw.w_show_legend.check_show_legend,
                   tw.w_show_legend.cb_legend_loc, tw.w_normalize.check_normalize, tw.w_normalize.le_norm_min,
                   tw.w_normalize.le_norm_max, tw.w_normalize.check_abs_value, tw.w_select_data_color.cb_select_cmap,
                   tw.w_select_data_color.le_custom_colors, tw.w_show_data.check_show_data,
                   tw.w_show_conc.check_show_conc, tw.w_show_tau.check_show_tau,
                   tw.w_show_residuals.check_show_residuals, tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_local_fit_{w.objectName()}', w)
            del blocker

        tw = self.tw_local_fit_properties[1].w_corner_manipulation
        widgets = [tw.sb_bins, tw.sb_label_pad, tw.sb_subplot_pad, tw.sb_tick_num,
                   tw.check_truth, tw.check_show_titles, tw.check_show_quantiles,
                   tw.check_plot_datapoints, tw.check_plot_contours, tw.check_plot_density]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_local_fit_emcee_{w.objectName()}', w)
            del blocker

        tw = self.tw_global_fit_properties[0]
        widgets = [tw.w_select_2D_plot.cb_2D_plot, tw.w_select_2D_plot.cb_2D_plot, tw.w_trimm.le_xmin,
                   tw.w_trimm.le_xmax, tw.w_trimm.le_ymin, tw.w_trimm.le_ymax, tw.w_trimm.le_zmin,
                   tw.w_trimm.le_zmax,  tw.w_trimm.sb_zcenter, tw.w_view_manipulations.le_linlog,
                   tw.w_view_manipulations.sb_log_ratio, tw.w_view_manipulations.sb_ss_ratio,
                   tw.w_colormap.check_cmap, tw.w_colormap.cb_select_cmap, tw.w_colormap.cb_pos_cmap,
                   tw.w_hide_area.le_xmin_hide, tw.w_hide_area.le_xmax_hide, tw.w_hide_area.le_xmin_hide2,
                   tw.w_hide_area.le_xmax_hide2, tw.w_show_second_ax.check_show_second_ax,
                   tw.w_show_pump.check_show_pump, tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_global_fit_2D_{w.objectName()}', w)
            del blocker

        tw = self.tw_global_fit_properties[1]
        widgets = [tw.w_trimm.le_xmin, tw.w_trimm.le_xmax,  tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,
                   tw.w_trimm.cb_zscale, tw.w_show_legend.check_show_legend, tw.w_show_legend.cb_legend_loc,
                   tw.w_normalize.check_normalize, tw.w_normalize.le_norm_min, tw.w_normalize.le_norm_max,
                   tw.w_select_data_color.cb_select_cmap, tw.w_select_data_color.le_custom_colors,
                   tw.w_hide_area.le_xmin_hide, tw.w_hide_area.le_xmax_hide, tw.w_hide_area.le_xmin_hide2,
                   tw.w_hide_area.le_xmax_hide2, tw.w_show_second_ax.check_show_second_ax,
                   tw.w_show_pump.check_show_pump, tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_global_fit_EASDAS_{w.objectName()}', w)
            del blocker

        tw = self.tw_global_fit_properties[2]
        widgets = [tw.w_trimm.le_ymin, tw.w_trimm.le_ymax,  tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,
                   tw.w_trimm.cb_yscale, tw.w_trimm.cb_zscale, tw.w_show_legend.check_show_legend,
                   tw.w_show_legend.cb_legend_loc, tw.w_select_data_color.cb_select_cmap,
                   tw.w_select_data_color.le_custom_colors, tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_global_fit_conc_{w.objectName()}', w)
            del blocker

        tw = self.tw_global_fit_properties[3]
        widgets = [tw.w_trimm.le_xmin, tw.w_trimm.le_xmax,  tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,
                   tw.w_trimm.cb_zscale, tw.w_show_legend.check_show_legend, tw.w_show_legend.cb_legend_loc,
                   tw.w_select_data_color.le_delay_list, tw.w_select_data_color.cb_select_cmap,
                   tw.w_select_data_color.le_custom_colors, tw.w_hide_area.le_xmin_hide,
                   tw.w_hide_area.le_xmax_hide, tw.w_hide_area.le_xmin_hide2, tw.w_hide_area.le_xmax_hide2,
                   tw.w_show_data.check_show_data, tw.w_show_residuals.check_show_residuals,
                   tw.w_show_second_ax.check_show_second_ax, tw.w_show_pump.check_show_pump,
                   tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_global_fit_delA_{w.objectName()}', w)
            del blocker

        tw = self.tw_global_fit_properties[4]
        widgets = [tw.w_trimm.le_ymin, tw.w_trimm.le_ymax,  tw.w_trimm.le_zmin, tw.w_trimm.le_zmax,
                   tw.w_trimm.cb_yscale, tw.w_trimm.cb_zscale, tw.w_show_legend.check_show_legend,
                   tw.w_show_legend.cb_legend_loc, tw.w_select_data_color.le_wavelength_list,
                   tw.w_select_data_color.cb_select_cmap, tw.w_select_data_color.le_custom_colors,
                   tw.w_normalize.check_normalize, tw.w_normalize.le_norm_min, tw.w_normalize.le_norm_max,
                   tw.w_normalize.check_abs_value, tw.w_show_data.check_show_data,
                   tw.w_show_residuals.check_show_residuals, tw.w_show_info.check_show_info]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_global_fit_kinTrace_{w.objectName()}', w)
            del blocker

        tw = self.tw_global_fit_properties[5].w_corner_manipulation
        widgets = [tw.sb_bins, tw.sb_label_pad, tw.sb_subplot_pad, tw.sb_tick_num,
                   tw.check_truth, tw.check_show_titles, tw.check_show_quantiles,
                   tw.check_plot_datapoints, tw.check_plot_contours, tw.check_plot_density]

        for w in widgets:
            blocker = QSignalBlocker(w)
            self.config.add_handler(
                f'visualize_w_global_emcee_{w.objectName()}', w)
            del blocker

    def update_properties_gui(self, state: bool) -> None:
        '''
        triggered when new plot type is selected (eg rb_2d_plot.toggled.)
        Loads the new properties widget and calls functions to reinitialize the cached inputs

        Parameters
        ----------
        state : bool
            the selected radio button.

        Returns
        -------
        None.

        '''
        if state:
            self.tw_properties.w_current_properties.setParent(None)

            if self.sender().objectName() == "2D_plot":
                properties2update = self.tw_surf_plot_properties
                self.tw_properties.w_current_properties = properties2update
                self.check_wavelength_input(update_all=properties2update)
                self.check_delay_input(update_all=properties2update)
                self.check_z_input(update_all=properties2update)
                self.check_hide_input(update_all=properties2update)
                self._update_ss_toggle()

            elif self.sender().objectName() == "delA_plot":
                properties2update = self.tw_delA_plot_properties
                self.tw_properties.w_current_properties = properties2update
                self.check_wavelength_input(update_all=properties2update)
                self.check_z_input(update_all=properties2update)
                self.check_delay_list_input(update_all=properties2update)
                self.check_hide_input(update_all=properties2update)
                self.check_colorlist_input(update_all=properties2update)

            elif self.sender().objectName() == "kin_trace":
                properties2update = self.tw_kin_trace_properties
                self.tw_properties.w_current_properties = properties2update
                self.check_delay_input(update_all=properties2update)
                self.check_z_input(update_all=properties2update)
                self.check_colorlist_input(update_all=properties2update)
                self.check_norm_input(update_all=properties2update)
                self.check_wavelength_list_input(update_all=properties2update)

            elif self.sender().objectName() == "local_fit":
                if self.sender().menuSelected() == 'kin_trace':
                    properties2update = self.tw_local_fit_properties[0]
                    self.tw_properties.w_current_properties = properties2update
                    self.check_delay_input(update_all=properties2update)
                    self.check_z_input(update_all=properties2update)
                    self.check_colorlist_input(update_all=properties2update)
                    self.check_norm_input(update_all=properties2update)
                elif self.sender().menuSelected() == 'emcee':
                    self.tw_properties.w_current_properties = self.tw_local_fit_properties[1]

            elif self.sender().objectName() == "global_fit":
                if self.sender().menuSelected() == 'plot_2d':
                    self.tw_properties.w_current_properties = self.tw_global_fit_properties[0]
                    self.check_wavelength_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_delay_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_z_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_hide_input(
                        update_all=self.tw_properties.w_current_properties)
                elif self.sender().menuSelected() == 'eas_das':
                    self.tw_properties.w_current_properties = self.tw_global_fit_properties[1]
                    self.check_wavelength_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_z_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_hide_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_colorlist_input(
                        update_all=self.tw_properties.w_current_properties)
                elif self.sender().menuSelected() == 'conc':
                    self.tw_properties.w_current_properties = self.tw_global_fit_properties[2]
                    self.check_delay_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_z_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_colorlist_input(
                        update_all=self.tw_properties.w_current_properties)
                elif self.sender().menuSelected() == 'dela_plot':
                    self.tw_properties.w_current_properties = self.tw_global_fit_properties[3]
                    self.check_wavelength_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_z_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_delay_list_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_hide_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_colorlist_input(
                        update_all=self.tw_properties.w_current_properties)
                elif self.sender().menuSelected() == 'kinetic_trace':
                    self.tw_properties.w_current_properties = self.tw_global_fit_properties[4]
                    self.check_delay_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_z_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_wavelength_list_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_norm_input(
                        update_all=self.tw_properties.w_current_properties)
                    self.check_colorlist_input(
                        update_all=self.tw_properties.w_current_properties)
                elif self.sender().menuSelected() == 'emcee':
                    self.tw_properties.w_current_properties = self.tw_global_fit_properties[5]

            self.tw_properties.view_layout.addWidget(
                self.tw_properties.w_current_properties)
            self.update_current_style(update_all=True)
            self.plot_data()

    def plot_data(self, *args, save_fig: bool = False) -> None:
        '''plotting function handling all the different plot types to display.
        If save_fig, figure will also be saved '''

        # -------- helper functions ----------------------------------------------------------------

        def _set_axes_scale(axes: Axes, xlim: tuple, ylim: tuple, x_formatter=None, y_formatter=None) -> None:
            ''' sets scales, formatters adn limits of mpl axes object '''
            axes.set(xlim=xlim)
            axes.set(ylim=ylim)
            trim_widget = self.tw_properties.w_current_properties.w_trimm

            if trim_widget.findChild(QComboBox, "cb_yscale"):
                if trim_widget.cb_yscale.currentText() == 'lin':
                    axes.set_xscale('linear')
                    axes.xaxis.set_minor_locator(tk.AutoMinorLocator())
                if trim_widget.cb_yscale.currentText() == 'log':
                    axes.set_xscale('log')
                    axes.xaxis.set_minor_locator(tk.LogLocator(
                        base=10.0, subs="all", numticks=10))
                if trim_widget.cb_yscale.currentText() == 'linlog':
                    try:
                        linthreshy = utils.Converter.convert_str_input2float(
                            trim_widget.le_linlog_y.text())
                    except ValueError:
                        linthreshy = None
                    if linthreshy is None or linthreshy <= 0:
                        linthreshy = 1e-12

                    axes.set_xscale('symlog', linthresh=linthreshy)
                    axes.xaxis.set_minor_locator(tk.SymmetricalLogLocator(
                        base=10.0, linthresh=linthreshy,
                        subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))

            if trim_widget.cb_zscale.currentText() == 'lin':
                axes.set_yscale('linear')
                axes.yaxis.set_minor_locator(tk.AutoMinorLocator())
            if trim_widget.cb_zscale.currentText() == 'log':
                axes.set_yscale('log')
                axes.yaxis.set_minor_locator(tk.LogLocator(
                    base=10.0, subs="all", numticks=10))
            if trim_widget.cb_zscale.currentText() == 'linlog':
                try:
                    linthreshz = utils.Converter.convert_str_input2float(
                        trim_widget.le_linlog_z.text())
                except ValueError:
                    linthreshz = None
                if linthreshz is None or linthreshz <= 0:
                    linthreshz = 1

                axes.set_yscale('symlog', linthresh=linthreshz)
                axes.yaxis.set_minor_locator(tk.SymmetricalLogLocator(
                    base=10.0, linthresh=linthreshz,
                    subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))
            if x_formatter:
                axes.xaxis.set_major_formatter(x_formatter)
            if y_formatter:
                axes.xaxis.set_major_formatter(y_formatter)

        def _get_norm_intervall(data: NDArray) -> slice:
            ''' returns valid interval which is used for normalization '''
            if self.norm_min is None or self.norm_min <= np.min(data):
                idx_norm_min = 0
            else:
                idx_norm_min = abs(
                    data[:] - self.norm_min).argmin()
            if self.norm_max is None or self.norm_max >= np.max(data):
                idx_norm_max = len(data)

            else:
                idx_norm_max = abs(
                    data[:] - self.norm_max).argmin()

            return slice(idx_norm_min, idx_norm_max)

        def _set_title(axes: Axes) -> None:
            ''' sets title of a given axes using meta information of the project '''
            title = self.visualize_controller.get_title()
            axes.set_title(
                title, fontproperties=self.font_property, size=self.label_size,  loc='left', )

        def _show_pump(axes: Axes, annotate: bool = True) -> None:
            ''' adds vertical line to axes to indicate pump wavelength '''
            try:
                pump = self.visualize_controller.get_pump()

                if pump is not None:
                    if self.x_min <= pump <= self.x_max:
                        axes.axvline(x=pump, color="black", linewidth=2 *
                                     float(mpl.rcParams['lines.linewidth']))
                        axes.axvline(x=pump, color="black", linewidth=2 *
                                     float(mpl.rcParams['lines.linewidth']))
                        if annotate:
                            axes.annotate('Pump', xy=(pump, axes.get_ylim()[1]),
                                          xytext=(3, -3), textcoords='offset points',
                                          ha='right', va='top', color='black', size=self.tick_size,
                                          fontproperties=self.font_property, rotation=90,
                                          rotation_mode='anchor')
                    else:
                        self.visualize_controller.call_statusbar(
                            "error", msg.Error.e28)
                else:
                    self.visualize_controller.call_statusbar(
                        "error", msg.Error.e26)
            except ValueError:
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e02)

        def _get_colorlist(plot_number: int, sender: QWidget | bool = False) -> list:
            ''' returns a list of valid colors depending on the user input '''
            colorlist = []
            if sender:
                color_widget = sender.w_select_data_color.cb_select_cmap
            else:
                color_widget = self.tw_properties.w_current_properties.w_select_data_color.cb_select_cmap
            if len(self.custom_color_list) >= plot_number:
                colorlist = self.custom_color_list
            elif color_widget.currentText() != '-':
                colorgrad = plt.colormaps[(
                    color_widget.currentText())]
                for i in range(plot_number):
                    colorlist.append(colorgrad((1 / plot_number) * i))
            else:  # use rc params color prop cycle
                color_cycles = plt.rcParams["axes.prop_cycle"].by_key()["color"]
                colorlist = list(color_cycles)
                while len(colorlist) < plot_number:
                    colorlist.extend(color_cycles)
            return colorlist

        def _show_2nd_axis(axes: Axes) -> None:
            ''' calculates an energy eV axis from the wavelength axis and attaches it to the axes '''
            _ax2 = axes.twiny()
            _ax2.spines['bottom'].set_visible(False)
            axes.spines['top'].set_visible(False)
            majEnTicks = np.arange(np.ceil(utils.Converter.m_in_eV(axes.get_xlim()[0]) * 2) / 2,
                                   np.floor(utils.Converter.m_in_eV(axes.get_xlim()[1]) * 2) / 2, -0.5)
            _ax2.set_xlabel(
                "Energy (eV)", fontproperties=self.font_property, fontsize=self.label_size)
            _ax2.set_xticks(utils.Converter.eV_in_m(majEnTicks))
            _ax2.set_xlim(axes.get_xlim())
            _ax2.set_xticklabels(majEnTicks)
            minEnTicks = np.arange(round(utils.Converter.m_in_eV(
                axes.get_xlim()[0]) * 5) / 5, round(utils.Converter.m_in_eV(axes.get_xlim()[1]) * 5) / 5, -0.1)
            _ax2.xaxis.set_minor_locator(tk.FixedLocator(
                utils.Converter.eV_in_m(minEnTicks)))
            _ax2.tick_params(
                labelleft=False, labelsize=self.tick_size)
            plt.setp(_ax2.xaxis.get_majorticklabels() + _ax2.yaxis.get_majorticklabels(),
                     fontproperties=self.font_property, fontsize=self.tick_size)

        def _add_hide_patches(axes: Axes) -> PathPatch:
            ''' calculates and returns the drawing area depending on axes limits and hide inputs '''
            allowed_rects = []

            # sort input
            if self.xmin_hide is None and self.xmin_hide2 is not None:
                xmin_hide, xmax_hide, xmin_hide2, xmax_hide2 = self.xmin_hide2, self.xmax_hide2, None, None
            elif (self.xmin_hide2 is not None and self.xmin_hide is not None and
                  self.xmin_hide2 < self.xmin_hide):
                xmin_hide, xmax_hide, xmin_hide2, xmax_hide2 = self.xmin_hide2, self.xmax_hide2, self.xmin_hide, self.xmax_hide
            else:
                xmin_hide, xmax_hide, xmin_hide2, xmax_hide2 = self.xmin_hide, self.xmax_hide, self.xmin_hide2, self.xmax_hide2

            # define patches
            if xmin_hide is not None and xmax_hide is not None:
                # Left allowed area
                if xmin_hide > self.x_min:
                    allowed_rects.append((self.x_min, self.z_min,
                                          xmin_hide - self.x_min,
                                          self.z_max - self.z_min))

                # middle allowed area
                if xmin_hide2 is not None and xmax_hide2 is not None:
                    if xmax_hide < xmin_hide2:
                        allowed_rects.append((xmax_hide, self.z_min,
                                              xmin_hide2 - xmax_hide,
                                              self.z_max - self.z_min))
                    # Right allowed area
                    if xmax_hide2 < self.x_max:
                        allowed_rects.append((xmax_hide2, self.z_min,
                                              self.x_max - xmax_hide2,
                                              self.z_max - self.z_min))
                else:
                    if xmax_hide < self.x_max:
                        allowed_rects.append((xmax_hide, self.z_min,
                                              self.x_max - xmax_hide,
                                              self.z_max - self.z_min))
            else:
                # entire region
                allowed_rects.append((self.x_min, self.z_min,
                                      self.x_max - self.x_min,
                                      self.z_max - self.z_min))

            vertices = []
            codes = []
            for rect in allowed_rects:
                x, y, w, h = rect

                vertices.extend([
                    (x, y),
                    (x + w, y),
                    (x + w, y + h),
                    (x, y + h),
                    (x, y)
                ])
                codes.extend([
                    mpl.path.Path.MOVETO,
                    mpl.path.Path.LINETO,
                    mpl.path.Path.LINETO,
                    mpl.path.Path.LINETO,
                    mpl.path.Path.CLOSEPOLY
                ])

            compound_path = mpl.path.Path(vertices, codes)
            clip_patch = PathPatch(compound_path, transform=axes.transData)
            return clip_patch

        def _init_bounds(normalized=False, absolute=False) -> None:
            ''' sets the axes bounds to default values if not set by user '''
            if self.x_min is None:
                self.x_min = self.buffer_dataX[0]
            if self.x_max is None:
                self.x_max = self.buffer_dataX[-1]
            if self.y_min is None:
                self.y_min = self.buffer_dataY[0]
            if self.y_linlog is None:
                self.y_linlog = abs(self.y_min)
            if self.y_max is None:
                self.y_max = self.buffer_dataY[-1]
            if self.z_min is None:
                if normalized and absolute:
                    self.z_min = -0.05
                elif normalized and not absolute:
                    self.z_min = -1.05
                else:
                    self.z_min = np.nanmin(self.buffer_dataZ)
            if self.z_max is None:
                if normalized:
                    self.z_max = 1.05
                else:
                    self.z_max = np.nanmax(self.buffer_dataZ)

        def _show_legend(axes, loc: str, title: str | None = None, handles=None, labels=None) -> None:
            ''' sets axes legend '''
            fp_labels = self.font_property.copy()
            fp_labels.set_size(self.tick_size)
            fp_title = self.font_property.copy()
            fp_title.set_size(self.label_size)
            kwargs = {
                'borderaxespad': 0,
                'title': title,
                'prop': fp_labels,
                'title_fontproperties': fp_title
            }
            if loc == 'outside':
                # For an outside legend, update with additional settings
                kwargs.update({
                    'bbox_to_anchor': (1.02, 1),
                    'borderpad': 0.0,
                    'loc': 'upper left'
                })
            else:
                kwargs.update({'loc': loc})
            if handles is not None and labels is not None:
                axes.legend(handles, labels, **kwargs)
            else:
                axes.legend(**kwargs)

    # -------- initialize canvas -------------------------------------------------------------------
        if self.visualize_controller.verify_rawdata() is False:
            self.visualize_controller.call_statusbar("error", msg.Error.e05)
            self._clear_canvas()
            return

        self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ = self.visualize_controller.get_ds_data(
            ds=self.ds)

        self.label_size = self.tw_select_plot.sb_label_size.value()
        self.tick_size = self.tw_select_plot.sb_tick_size.value()

        if self.fig_width is None:
            self.fig_width = 0.13  # SI
        if self.fig_height is None:
            self.fig_height = 0.13  # SI

        self.sc = utils.PlotCanvas(
            self, width=self.fig_width * 39.37008, height=self.fig_height * 39.37008, dpi=100)

        self.toolbar = NavigationToolbar2QT(self.sc, )

        self.toolbar.addWidget(QLabel('   '))
        self.pb_export_csv = QPushButton('export data')
        self.toolbar.addWidget(self.pb_export_csv)
        self.pb_export_csv.pressed.connect(self.export_data)
        self.fig = self.sc.fig

    # -------- 2D Surface --------------------------------------------------------------------------
        if self.tw_select_plot.rb_2d_plot.isChecked():
            plottype = "2d_plot"

            try:
                normalization = colors.TwoSlopeNorm(
                    vmin=self.z_min, vmax=self.z_max, vcenter=self.z_center)
            except ValueError:  # vmin, vcenter, and vmax must be in ascending order
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e14)
                return

            if not self.tw_surf_plot_properties.w_show_ss.check_show_ss.isChecked():
                height_ratios = [self.tw_surf_plot_properties.w_view_manipulations.sb_log_ratio.value(
                ), self.tw_surf_plot_properties.w_view_manipulations.sb_lin_ratio.value()]
                gs = self.fig.add_gridspec(
                    2, hspace=0, height_ratios=height_ratios, )
                axs = gs.subplots(sharex=True)
                axLin = axs[1]
                axLog = axs[0]

            else:
                height_ratios = [self.tw_surf_plot_properties.w_view_manipulations.sb_ss_ratio.value(
                ), self.tw_surf_plot_properties.w_view_manipulations.sb_log_ratio.value(),
                    self.tw_surf_plot_properties.w_view_manipulations.sb_lin_ratio.value()]

                gs = self.fig.add_gridspec(
                    3, hspace=0, height_ratios=height_ratios, )
                axs = gs.subplots(sharex=True)
                axLin = axs[2]
                axLog = axs[1]
                ax_Abs = axs[0]

            _init_bounds()
            axLin.set_xlim(self.x_min, self.x_max)
            axLin.set_xscale('linear')
            axLin.xaxis.set_minor_locator(tk.AutoMinorLocator())
            axLin.xaxis.set_major_formatter(self.sc.nm_formatter_ax)

            axLin.set_ylim((self.y_min, self.y_linlog))
            axLin.yaxis.set_minor_locator(tk.AutoMinorLocator())
            axLin.yaxis.set_major_formatter(self.sc.delay_formatter0)

            clip_patch = _add_hide_patches(axes=axLin)

            pcolormesh_plot_lin = axLin.pcolormesh(
                self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ, shading='auto',
                norm=normalization, rasterized=True)

            pcolormesh_plot_lin.set_clip_path(clip_patch)

            axLin.spines['top'].set_visible(False)

            axLin.tick_params(
                which='both',
                top=False,
                labeltop=False,)
            plt.setp(axLin.xaxis.get_majorticklabels() + axLin.yaxis.get_majorticklabels(),
                     fontproperties=self.font_property, fontsize=self.tick_size)

            axLin.grid(False)
            axLog.grid(False)
            axLog.tick_params(labelbottom=False)

            axLog.set_yscale('log')
            axLog.set_ylim((self.y_linlog, self.y_max))
            axLog.yaxis.set_major_formatter(self.sc.delay_formatter0)

            clip_patch = _add_hide_patches(axes=axLog)
            pcolormesh_plot_log = axLog.pcolormesh(
                self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ, shading='auto',
                norm=normalization, rasterized=True)

            pcolormesh_plot_log.set_clip_path(clip_patch)

            axLog.spines['bottom'].set_visible(False)

            axLog.yaxis.set_ticks_position("both")
            axLog.tick_params(which='both', bottom=False,)
            plt.setp(axLog.xaxis.get_majorticklabels() + axLog.yaxis.get_majorticklabels(),
                     fontproperties=self.font_property, fontsize=self.tick_size)

            axLin.set_xlabel(msg.Labels.wavelength,
                             fontsize=self.label_size, fontproperties=self.font_property)
            self.fig.supylabel(
                msg.Labels.delay, fontsize=self.label_size, fontproperties=self.font_property, )

            if self.tw_surf_plot_properties.w_show_delA_cuts.check_show_delA_cuts.isChecked():
                colorlist = _get_colorlist(
                    plot_number=len(self.delay_cut_list), sender=self.tw_delA_plot_properties)

                for i, v in enumerate(self.delay_cut_list):
                    axLin.axhline(y=v, color=colorlist[i],)
                    axLog.axhline(y=v, color=colorlist[i],)

            if self.tw_surf_plot_properties.w_show_kin_cuts.check_show_kin_cuts.isChecked():
                colorlist = _get_colorlist(
                    plot_number=len(self.wavelength_trace_list), sender=self.tw_kin_trace_properties)

                for i, v in enumerate(self.wavelength_trace_list):
                    axLin.axvline(x=v, color=colorlist[i])
                    axLog.axvline(x=v, color=colorlist[i])

            if self.tw_surf_plot_properties.w_show_second_ax.check_show_second_ax.isChecked():
                _show_2nd_axis(axes=axLog)

                axLog.tick_params(which='both', labeltop=False,
                                  labelbottom=False, top=False, bottom=False)

            if self.tw_surf_plot_properties.w_show_ss.check_show_ss.isChecked():

                ax_Abs.grid(False)
                for spine in ax_Abs.spines.values():
                    spine.set_visible(False)
                tick_cfg = dict(which='both', labelright=False, labelleft=False,
                                right=False, left=False, top=False, bottom=False)

                abs_before, abs_after = self.visualize_controller.get_ss_data('abs')
                em_before, em_after = self.visualize_controller.get_ss_data('em')

                if abs_before is not None:
                    # Normalize and plot absorption data (before)
                    Abs_offset_normal_before = utils.Converter.offset_corr(
                        abs_before[:, 1]) / utils.Converter.plt_normalization(abs_before[:, 0],
                                                                              utils.Converter.offset_corr(abs_before[:, 1]), self.x_min, self.x_max)
                    ax_Abs.plot(
                        abs_before[:, 0], Abs_offset_normal_before, color="red")

                    if abs_after is not None:
                        # Normalize and plot absorption data (after)
                        Abs_offset_normal_after = utils.Converter.offset_corr(
                            abs_after[:, 1]) / utils.Converter.plt_normalization(abs_after[:, 0],
                                                                                 utils.Converter.offset_corr(abs_after[:, 1]), self.x_min, self.x_max)

                        ax_Abs.plot(abs_after[:, 0], Abs_offset_normal_after,
                                    color="maroon", linestyle="dotted")

                    ax_Abs.set_ylabel(
                        "A (a.u.)", color="red", fontsize=self.label_size, fontproperties=self.font_property)
                    ax_Abs.fill_between(
                        abs_before[:, 0], Abs_offset_normal_before, 0, color='red', alpha=0.25)

                    # max at 1.0 due to normalization
                    ax_Abs.set(ylim=(0, 1), xlim=(self.x_min, self.x_max))
                    ax_Abs.yaxis.set_label_coords(-0.05, 0.5)

                if em_before is not None:
                    Fl_norm_before = em_before[:, 1] / utils.Converter.plt_normalization(
                        em_before[:, 0], em_before[:, 1], self.x_min, self.x_max)
                    ax_Fl = ax_Abs.twinx()
                    ax_Fl.grid(False)
                    ax_Fl.set_ylabel(
                        "PL (a.u.)", color="blue", fontsize=self.label_size, fontproperties=self.font_property)
                    ax_Fl.yaxis.set_label_coords(1.05, 0.5)
                    ax_Fl.plot(em_before[:, 0], Fl_norm_before, "b")
                    if em_after is not None:
                        Fl_norm_after = em_after[:, 1] / utils.Converter.plt_normalization(
                            em_after[:, 0], em_after[:, 1], self.x_min, self.x_max)
                        ax_Fl.plot(em_after[:, 0], Fl_norm_after,
                                   "navy", linestyle="dotted")
                    ax_Fl.fill_between(em_before[:, 0], Fl_norm_before,
                                       0, color='blue', alpha=0.25)
                    ax_Fl.set(ylim=(0, 1.1))
                    for spine in ax_Fl.spines.values():
                        spine.set_visible(False)
                    ax_Fl.tick_params(**tick_cfg)

                ax_Abs.tick_params(**tick_cfg)

            cmap = self.tw_surf_plot_properties.w_colormap.cb_select_cmap.currentText()
            if cmap != '-':  # otherwise use rc_param
                pcolormesh_plot_lin.set_cmap(cmap)
                pcolormesh_plot_log.set_cmap(cmap)

            if self.tw_surf_plot_properties.w_colormap.check_cmap.isChecked():

                if self.tw_surf_plot_properties.w_colormap.cb_pos_cmap.currentText() == 'right':
                    cax = axLin.inset_axes(
                        [1.02, 0, 0.025, height_ratios[-2] / height_ratios[-1] + 1])
                    cbar = self.fig.colorbar(
                        mappable=pcolormesh_plot_lin, cax=cax, location="right", shrink=0.6, use_gridspec=True)

                elif self.tw_surf_plot_properties.w_colormap.cb_pos_cmap.currentText() == 'bottom':
                    divider = make_axes_locatable(axLin)
                    cax = divider.new_vertical(
                        size="15%", pad=0.5, pack_start=True)
                    self.fig.add_axes(cax)
                    cbar = self.fig.colorbar(
                        pcolormesh_plot_lin, cax=cax, orientation="horizontal", use_gridspec=True)

                cbar.minorticks_on()
                plt.setp(cbar.ax.xaxis.get_majorticklabels() + cbar.ax.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)

                cbar.set_label(
                    label=msg.Labels.delA, size=self.label_size, fontproperties=self.font_property)

            if self.tw_surf_plot_properties.w_show_info.check_show_info.isChecked():
                if self.tw_surf_plot_properties.w_show_ss.check_show_ss.isChecked():
                    _set_title(axes=ax_Abs)
                else:
                    _set_title(axes=axLog)

            if self.tw_surf_plot_properties.w_show_pump.check_show_pump.isChecked():
                _show_pump(axes=axLin, annotate=False)
                _show_pump(axes=axLog, annotate=True)

    # -------- delA Plot ---------------------------------------------------------------------------
        elif self.tw_select_plot.rb_delA_plot.isChecked():
            plottype = "delA_plot"
            self.delay_cut_list = np.asarray(self.delay_cut_list)
            ind_delay_found = []

            for v in self.delay_cut_list:
                idx = (abs(self.buffer_dataY - v)).argmin()
                ind_delay_found.append(idx)

            ax1 = self.fig.add_subplot(111)
            ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)
            ax1.set_xlabel(msg.Labels.wavelength,
                           fontsize=self.label_size, fontproperties=self.font_property)
            ax1.set_ylabel(msg.Labels.delA, fontsize=self.label_size,
                           fontproperties=self.font_property)
            ax1.tick_params(labelleft=True, right=True)
            plt.setp(ax1.xaxis.get_majorticklabels() + ax1.yaxis.get_majorticklabels(),
                     fontproperties=self.font_property, fontsize=self.tick_size)
            _init_bounds()
            ax1.xaxis.set_minor_locator(tk.AutoMinorLocator())
            _set_axes_scale(axes=ax1, xlim=(self.x_min, self.x_max), ylim=(
                self.z_min, self.z_max), x_formatter=self.sc.nm_formatter_ax)
            colorlist = _get_colorlist(plot_number=len(self.delay_cut_list))
            clip_patch = _add_hide_patches(axes=ax1)

            for i, ind in enumerate(ind_delay_found):
                label = f"{self.sc.delay_formatter0(np.round(self.buffer_dataY[ind], decimals=14))}s"
                line, = ax1.plot(
                    self.buffer_dataX, self.buffer_dataZ[ind, :], label=label, color=colorlist[i], zorder=0)
                line.set_clip_path(clip_patch)
            if self.tw_delA_plot_properties.w_show_legend.check_show_legend.isChecked():
                _show_legend(axes=ax1, title='Delay times',
                             loc=self.tw_delA_plot_properties.w_show_legend.cb_legend_loc.currentText())

            if self.tw_delA_plot_properties.w_show_second_ax.check_show_second_ax.isChecked():
                _show_2nd_axis(axes=ax1)

            if self.tw_delA_plot_properties.w_show_info.check_show_info.isChecked():
                _set_title(axes=ax1)

            if self.tw_delA_plot_properties.w_show_pump.check_show_pump.isChecked():
                _show_pump(axes=ax1)

    # -------- kin trace Plot ----------------------------------------------------------------------
        elif self.tw_select_plot.rb_kin_trace.isChecked():
            plottype = "kin_trace"
            self.wavelength_trace_list = np.asarray(self.wavelength_trace_list)

            ind_wavelengths_found = []
            for v in self.wavelength_trace_list:
                idx = (abs(self.buffer_dataX - v)).argmin()
                ind_wavelengths_found.append(idx)

            normalize = self.tw_kin_trace_properties.w_normalize.check_normalize.isChecked()
            absolute = self.tw_kin_trace_properties.w_normalize.check_abs_value.isChecked()

            ax1 = self.fig.add_subplot(111)
            ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)
            ax1.set_xlabel(msg.Labels.delay,
                           fontsize=self.label_size, fontproperties=self.font_property)
            ax1.set_ylabel(msg.Labels.delA, fontsize=self.label_size,
                           fontproperties=self.font_property)
            ax1.tick_params(labelleft=True, right=True)
            plt.setp(ax1.xaxis.get_majorticklabels() + ax1.yaxis.get_majorticklabels(),
                     fontproperties=self.font_property, fontsize=self.tick_size)
            ax1
            self.check_z_input(update_all=self.tw_kin_trace_properties)
            _init_bounds(normalized=normalize, absolute=absolute)

            if normalize:
                ax1.set_ylabel(msg.Labels.delA_norm,
                               fontsize=self.label_size, fontproperties=self.font_property)
                norm_intervall = _get_norm_intervall(data=self.buffer_dataY)

            if absolute:
                absolute_Z = abs(self.buffer_dataZ)
                self.buffer_dataZ = absolute_Z

            _set_axes_scale(axes=ax1, xlim=(self.y_min, self.y_max), ylim=(
                self.z_min, self.z_max), x_formatter=self.sc.delay_formatter0)

            colorlist = _get_colorlist(
                plot_number=len(self.wavelength_trace_list))

            if normalize:
                for i, ind in enumerate(ind_wavelengths_found):
                    label = f"{self.sc.nm_formatter_ax(self.buffer_dataX[ind])} nm"
                    ax1.plot(self.buffer_dataY, self.buffer_dataZ[:, ind] / np.amax(abs(self.buffer_dataZ[norm_intervall, ind])),
                             label=label, color=colorlist[i], zorder=0)

            else:
                for i, ind in enumerate(ind_wavelengths_found):
                    label = f"{self.sc.nm_formatter_ax(self.buffer_dataX[ind])} nm"
                    ax1.plot(self.buffer_dataY, self.buffer_dataZ[:, ind],
                             label=label, color=colorlist[i], zorder=0)

            if self.tw_kin_trace_properties.w_show_legend.check_show_legend.isChecked():
                _show_legend(axes=ax1, title='Wavelength',
                             loc=self.tw_kin_trace_properties.w_show_legend.cb_legend_loc.currentText())

            if self.tw_kin_trace_properties.w_show_info.check_show_info.isChecked():
                _set_title(axes=ax1)

    # -------- local Fit Plot ----------------------------------------------------------------------
        elif self.tw_select_plot.rb_local_fit.isChecked():
            plottype = "local_fit"
            fit_dict = self.visualize_controller.get_local_fit_list(ds=self.ds)
            if not fit_dict:
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e24)
                self._clear_canvas()
                return

            if self.tw_properties.w_current_properties == self.tw_local_fit_properties[0]:
                normalize = self.tw_local_fit_properties[0].w_normalize.check_normalize.isChecked()
                absolute = self.tw_local_fit_properties[0].w_normalize.check_abs_value.isChecked()
                checkboxes = self.tw_local_fit_properties[0].w_select_fit.check_fits  # list
                checked_boxes = [cb for cb in checkboxes if cb.isChecked()]
                if not checked_boxes:
                    self.visualize_controller.call_statusbar(
                        "error", msg.Error.e22)
                    self._clear_canvas()
                    return

                if self.tw_local_fit_properties[0].w_show_residuals.check_show_residuals.isChecked():
                    gs = self.fig.add_gridspec(2, 1, height_ratios=[5, 1])
                    axs = gs.subplots(sharex=True)

                    ax1 = axs[0]
                else:
                    ax1 = self.fig.add_subplot(111)

                ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)

                ax1.set_ylabel(
                    msg.Labels.delA, fontsize=self.label_size, fontproperties=self.font_property)
                ax1.tick_params(labelleft=True, right=True)
                plt.setp(ax1.xaxis.get_majorticklabels() + ax1.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)

                # ----- set axis scale -----
                self.check_z_input(update_all=self.tw_local_fit_properties[0])
                _init_bounds(normalized=normalize, absolute=absolute)

                _set_axes_scale(axes=ax1, xlim=(self.y_min, self.y_max), ylim=(
                    self.z_min, self.z_max), x_formatter=self.sc.delay_formatter0)

                # ----- set normalization params -----
                if normalize:
                    ax1.set_ylabel(msg.Labels.delA_norm,
                                   fontsize=self.label_size, fontproperties=self.font_property)
                    norm_intervall = _get_norm_intervall(data=self.buffer_dataY)

                # ----- get colorlist -----
                colorlist = _get_colorlist(plot_number=len(checked_boxes))
                fit_colorlist = utils.Converter.fitting_colorlist(colorlist)

                # ----- plotting -----
                for color_idx, checkbox in enumerate(checked_boxes):

                    ukey = checkbox.text()
                    delay = fit_dict[ukey]['delay']
                    conc = fit_dict[ukey]['conc']
                    opt_params = fit_dict[ukey]['opt_params']

                    if normalize:
                        normalization_factor = np.amax(
                            abs(fit_dict[ukey]['delA_calc'][norm_intervall]))
                        delA_calc = fit_dict[ukey]['delA_calc'] / normalization_factor
                        delA_exp = fit_dict[ukey]['delA'] / normalization_factor
                        amp = fit_dict[ukey]['Amp'] / normalization_factor
                    else:
                        delA_calc = fit_dict[ukey]['delA_calc']
                        delA_exp = fit_dict[ukey]['delA']
                        amp = fit_dict[ukey]['Amp']

                    if absolute:
                        delA_calc = abs(delA_calc)
                        delA_exp = abs(delA_exp)
                        amp = abs(amp)

                    ax1.plot(delay, delA_calc, color=colorlist[color_idx],
                             label=f"{self.sc.nm_formatter_ax(fit_dict[ukey]['wavelength'])} nm fit")
                    labels = fit_dict[ukey]['meta']['components']
                    labels_tex = [f'${label}$' for label in labels]
                    ca_order = fit_dict[ukey]['meta']['ca_order']
                    has_inf = fit_dict[ukey]['meta']['Ainf']
                    n_cols = conc.shape[1]
                    n_kin = n_cols - ca_order - (1 if has_inf else 0)
                    comp_colorlist = utils.Converter.components_colorlist(
                        colorlist[color_idx], n_cols)

                    if self.tw_local_fit_properties[0].w_show_data.check_show_data.isChecked():
                        ax1.plot(delay, delA_exp, 'x', color=fit_colorlist[color_idx], alpha=0.75,
                                 markersize=(self.tick_size / 3), zorder=1)

                    if self.tw_local_fit_properties[0].w_show_conc.check_show_conc.isChecked():

                        # plotting conc only makes sense if more than 1 comp
                        if conc.shape[1] >= 2:
                            for j in range(n_cols):
                                if j < ca_order:
                                    label = labels_tex[j]

                                elif j < ca_order + n_kin:
                                    tau_idx = j - ca_order + 1
                                    tau_val = opt_params[f'τ{tau_idx}']
                                    label = f"{labels_tex[j]} {self.sc.delay_formatter0(tau_val)}s"

                                else:
                                    label = labels_tex[j]

                                ax1.plot(delay, conc[:, j] * amp[j],
                                         '--', color=comp_colorlist[j],
                                         label=label)
                        else:
                            if self.tw_local_fit_properties[0].w_show_tau.check_show_tau.isChecked():
                                ax1.plot([], [], '', alpha=0, label='{labels_tex[0]}' + str(
                                    self.sc.delay_formatter0(opt_params['τ1']) + 's '))

                    if self.tw_local_fit_properties[0].w_show_tau.check_show_tau.isChecked() and not self.tw_local_fit_properties[0].w_show_conc.check_show_conc.isChecked():
                        if conc.shape[1] >= 2:
                            for j in range(n_cols):
                                if j < ca_order:
                                    label = labels_tex[j]

                                elif j < ca_order + n_kin:
                                    tau_idx = j - ca_order + 1
                                    tau_val = opt_params[f't{tau_idx}']
                                    label = f"{labels_tex[j]} {self.sc.delay_formatter0(tau_val)}s"

                                else:
                                    label = labels_tex[j]

                                ax1.plot([], [], '', alpha=0, label=label)
                        else:
                            label = f"{labels_tex[0]} {self.sc.delay_formatter0(opt_params['t1'])}s"
                            ax1.plot([], [], '', alpha=0, label=label)

                    if self.tw_local_fit_properties[0].w_show_residuals.check_show_residuals.isChecked():
                        ax2 = axs[1]
                        ax2.axhline(y=0, linestyle='dashed',
                                    color='black', alpha=0.5)
                        ax2.plot(delay, delA_calc - delA_exp,
                                 color=colorlist[color_idx], label='residuals')
                        ax2.tick_params(labelleft=True, right=True)
                        plt.setp(ax2.xaxis.get_majorticklabels() + ax2.yaxis.get_majorticklabels(),
                                 fontproperties=self.font_property, fontsize=self.tick_size)
                        ax2.set_xlabel(msg.Labels.delay,
                                       fontsize=self.label_size, fontproperties=self.font_property)
                    else:
                        ax1.set_xlabel(msg.Labels.delay,
                                       fontsize=self.label_size, fontproperties=self.font_property)

                if self.tw_local_fit_properties[0].w_show_legend.check_show_legend.isChecked():
                    _show_legend(axes=ax1, title='Wavelength',
                                 loc=self.tw_local_fit_properties[0].w_show_legend.cb_legend_loc.currentText())
                if self.tw_local_fit_properties[0].w_show_info.check_show_info.isChecked():
                    _set_title(axes=ax1)

    # -------- local Fit Corner Plot ---------------------------------------------------------------
            elif self.tw_properties.w_current_properties == self.tw_local_fit_properties[1]:
                checkboxes = self.tw_local_fit_properties[1].w_select_fit.check_fits  # list
                checked_box = [cb for cb in checkboxes if cb.isChecked()]
                if not checked_box:
                    self.visualize_controller.call_statusbar(
                        "error", msg.Error.e22)
                    self._clear_canvas()
                    return
                ukey = checked_box[0].text()
                if 'emcee' not in fit_dict[ukey]:
                    self.visualize_controller.call_statusbar(
                        "error", msg.Error.e24)
                    self._clear_canvas()
                    return
                plottype = 'local_fit_' + str(ukey) + "_emcee_corner"

                flatchain = fit_dict[ukey]['emcee']['flatchain']
                params = fit_dict[ukey]['emcee']['params']

                num_vary = sum(p.vary for p in params.values())
                varying_names, labels, truths = [], [], []
                for name, par in params.items():
                    if not par.vary:
                        continue

                    # keep the name for slicing flatchain
                    varying_names.append(name)
                    truths.append(par.value)

                    # --- 2. pretty‑print the label ----------------------------------------
                    if name == "t0":
                        labels.append(r"$t_0$ (s)")
                    elif name == "IRF":
                        labels.append("IRF (s)")
                    elif name.startswith("t") and name[1:].isdigit():
                        labels.append(fr"$τ_{name[1:]}$ (s)")
                    elif name.startswith("__ln"):
                        labels.append(r"ln(σ/mOD)")
                    else:
                        labels.append(name)        # fallback

                subplot_spacing = self.tw_properties.w_current_properties.w_corner_manipulation.sb_subplot_pad.value() / \
                    100
                bins = self.tw_properties.w_current_properties.w_corner_manipulation.sb_bins.value()
                label_pad = self.tw_properties.w_current_properties.w_corner_manipulation.sb_label_pad.value() / 100
                max_n_ticks = self.tw_properties.w_current_properties.w_corner_manipulation.sb_tick_num.value()
                truths = truths if self.tw_properties.w_current_properties.w_corner_manipulation.check_truth.isChecked() else None

                show_titles = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_show_titles.isChecked() else False
                plot_contours = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_plot_contours.isChecked() else False
                plot_datapoints = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_plot_datapoints.isChecked() else False
                plot_density = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_plot_density.isChecked() else False
                quantiles = [
                    0.16, 0.5, 0.84] if self.tw_properties.w_current_properties.w_corner_manipulation.check_show_quantiles.isChecked() else None

                f = corner.corner(flatchain, labels=labels, truths=truths, quantiles=quantiles,
                                  plot_contours=plot_contours, plot_density=plot_density,
                                  plot_datapoints=plot_datapoints, labelpad=label_pad, bins=bins,
                                  max_n_ticks=max_n_ticks, fig=self.fig)

                axes = np.array(f.axes).reshape((num_vary, num_vary))
                diagonal_axes = np.diag(axes)

                if show_titles:
                    fp_title = self.font_property.copy()
                    fp_title.set_size(self.label_size)
                    for i, ax in enumerate(diagonal_axes):
                        data = flatchain[:, i]
                        median = np.median(data)
                        qlow, qhigh = np.percentile(data, [16, 84])

                        title_str = r"${}$ $^{{+{}}}_{{-{}}}$ ".format(
                            self.sc.emcee_formatter(median),
                            self.sc.emcee_formatter(qhigh - median),
                            self.sc.emcee_formatter(median - qlow)
                        )
                        ax.set_title(title_str, fontproperties=fp_title)

                bottom_row = axes[-1, :]
                for ax in bottom_row:
                    ax.xaxis.set_major_formatter(self.sc.emcee_formatter)
                    # ax.tick_params(axis='x', labelsize=self.tick_size,
                    #                labelfontfamily=self.font_family)
                    plt.setp(ax.xaxis.get_majorticklabels(),
                             fontproperties=self.font_property, fontsize=self.tick_size)
                    ax.xaxis.label.set_fontproperties(self.font_property)
                    ax.xaxis.label.set_fontsize(self.label_size)

                left_column = axes[:, 0]
                for ax in left_column:
                    ax.yaxis.set_major_formatter(self.sc.emcee_formatter)
                    # ax.tick_params(axis='y', labelsize=self.tick_size,
                    #                labelfontfamily=self.font_family)
                    plt.setp(ax.yaxis.get_majorticklabels(),
                             fontproperties=self.font_property, fontsize=self.tick_size)
                    ax.yaxis.label.set_fontproperties(self.font_property)
                    ax.yaxis.label.set_fontsize(self.label_size)

                f.tight_layout()

                f.subplots_adjust(wspace=subplot_spacing,
                                  hspace=subplot_spacing)

    # -------- Global Plots ------------------------------------------------------------------------
        elif self.tw_select_plot.rb_global_fit.isChecked():

            fit_dict = self.visualize_controller.get_global_fit_list(
                ds=self.ds)
            if not fit_dict:
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e24)
                self._clear_canvas()
                return
            # list
            try:
                ukey = self.visualize_controller.get_global_ukey(
                    checkboxes=self.tw_properties.w_current_properties.w_select_fit.check_fits)
            except exc.NoSelectionError:
                self.visualize_controller.call_statusbar(
                    "error", msg.Error.e22)
                self._clear_canvas()
                return

            if ukey not in fit_dict:
                return

    # -------- Global 2D Plot ---------------------------------------------------------------
            if self.tw_properties.w_current_properties == self.tw_global_fit_properties[0]:
                delA_calc = fit_dict[ukey]['delA_calc'].copy()
                merge = False
                if self.tw_properties.w_current_properties.w_select_2D_plot.cb_2D_plot.currentText() == 'simulated':
                    plottype = ukey + "_hypershpere"
                    data = delA_calc
                    cbar_label = msg.Labels.delA
                elif self.tw_properties.w_current_properties.w_select_2D_plot.cb_2D_plot.currentText() == 'residuals':
                    plottype = ukey + "_residuals"
                    data = fit_dict[ukey]['residuals']
                    cbar_label = r'$\Delta A_{exp} -\Delta A_{calc} $ $(mO_{D}$)'
                else:
                    plottype = ukey + "_residuals_merge"
                    data = delA_calc
                    residuals = fit_dict[ukey]['residuals']
                    merge = True
                    cbar_label = msg.Labels.delA
                try:
                    normalization = colors.TwoSlopeNorm(
                        vmin=self.z_min, vmax=self.z_max, vcenter=self.z_center)
                except ValueError:  # vmin, vcenter, and vmax must be in ascending order
                    self.visualize_controller.call_statusbar(
                        "error", msg.Error.e14)
                    return

                height_ratios = [self.tw_properties.w_current_properties.w_view_manipulations.sb_log_ratio.value(
                ), self.tw_properties.w_current_properties.w_view_manipulations.sb_lin_ratio.value()]
                gs = self.fig.add_gridspec(
                    2, hspace=0, height_ratios=height_ratios, )
                axs = gs.subplots(sharex=True)
                axLin = axs[1]
                axLog = axs[0]

                _init_bounds()
                axLin.set_xlim(self.x_min, self.x_max)
                axLin.set_xscale('linear')
                axLin.xaxis.set_minor_locator(tk.AutoMinorLocator())
                axLin.xaxis.set_major_formatter(self.sc.nm_formatter_ax)

                axLin.set_ylim((self.y_min, self.y_linlog))
                axLin.yaxis.set_minor_locator(tk.AutoMinorLocator())
                axLin.yaxis.set_major_formatter(self.sc.delay_formatter0)

                clip_patch_Lin = _add_hide_patches(axes=axLin)

                pcolormesh_plot_lin = axLin.pcolormesh(
                    self.buffer_dataX, self.buffer_dataY, data, shading='auto', norm=normalization, rasterized=True)
                pcolormesh_plot_lin.set_clip_path(clip_patch_Lin)

                axLin.spines['top'].set_visible(False)
                axLin.tick_params(
                    which='both',
                    top=False,
                    labeltop=False)
                plt.setp(axLin.xaxis.get_majorticklabels() + axLin.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)

                axLin.grid(False)
                axLog.grid(False)
                axLog.tick_params(labelbottom=False)

                axLog.set_yscale('log')
                axLog.set_ylim((self.y_linlog, self.y_max))
                axLog.yaxis.set_major_formatter(self.sc.delay_formatter0)

                clip_patch_Log = _add_hide_patches(axes=axLog)
                pcolormesh_plot_log = axLog.pcolormesh(
                    self.buffer_dataX, self.buffer_dataY, data, shading='auto', norm=normalization, rasterized=True)

                if merge:
                    try:
                        thresh = utils.Converter.convert_str_input2float(
                            self.tw_properties.w_current_properties.w_select_2D_plot.le_merge_thresh.text())

                    except ValueError:
                        self.visualize_controller.call_statusbar(
                            "error",  msg.Error.e02)
                        thresh = None
                    if thresh is None:
                        thresh = 1
                    residuals_pos = np.where(
                        residuals >= thresh, residuals, np.nan)
                    # Mask for largely negative residuals (<= -0.5):
                    residuals_neg = np.where(
                        residuals <= -thresh, residuals, np.nan)

                    # Plot positive residuals in red:
                    pos_plot = axLin.pcolormesh(
                        self.buffer_dataX, self.buffer_dataY, residuals_pos,
                        cmap=ListedColormap(['white']),
                        shading='auto', rasterized=True)

                    # Plot negative residuals in blue:
                    neg_plot = axLin.pcolormesh(
                        self.buffer_dataX, self.buffer_dataY, residuals_neg,
                        cmap=ListedColormap(['black']),
                        shading='auto', rasterized=True)

                    pos_plot.set_clip_path(clip_patch_Lin)
                    neg_plot.set_clip_path(clip_patch_Lin)

                    pos_plot = axLog.pcolormesh(
                        self.buffer_dataX, self.buffer_dataY, residuals_pos,
                        cmap=ListedColormap(['white']),
                        shading='auto', rasterized=True)

                    # Plot negative residuals in blue:
                    neg_plot = axLog.pcolormesh(
                        self.buffer_dataX, self.buffer_dataY, residuals_neg,
                        cmap=ListedColormap(['black']),
                        shading='auto', rasterized=True)
                    pos_plot.set_clip_path(clip_patch_Log)
                    neg_plot.set_clip_path(clip_patch_Log)

                pcolormesh_plot_log.set_clip_path(clip_patch_Log)
                axLog.spines['bottom'].set_visible(False)
                axLog.yaxis.set_ticks_position("both")
                axLog.tick_params(which='both', bottom=False)
                plt.setp(axLog.xaxis.get_majorticklabels() + axLog.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)
                axLin.set_xlabel(msg.Labels.wavelength,
                                 fontsize=self.label_size, fontproperties=self.font_property)
                self.fig.supylabel(
                    msg.Labels.delay, fontsize=self.label_size, fontproperties=self.font_property, )

                if self.tw_properties.w_current_properties.w_show_second_ax.check_show_second_ax.isChecked():
                    _show_2nd_axis(axes=axLog)

                    axLog.tick_params(which='both', labeltop=False,
                                      labelbottom=False, top=False, bottom=False)

                cmap = self.tw_properties.w_current_properties.w_colormap.cb_select_cmap.currentText()
                if cmap != '-':  # otherwise use rc_param
                    pcolormesh_plot_lin.set_cmap(cmap)
                    pcolormesh_plot_log.set_cmap(cmap)

                if self.tw_properties.w_current_properties.w_colormap.check_cmap.isChecked():
                    if self.tw_properties.w_current_properties.w_colormap.cb_pos_cmap.currentText() == 'right':
                        cax = axLin.inset_axes(
                            [1.02, 0, 0.025, height_ratios[-2] / height_ratios[-1] + 1])
                        cbar = self.fig.colorbar(
                            mappable=pcolormesh_plot_lin, cax=cax, location="right", shrink=0.6, use_gridspec=True)

                    elif self.tw_properties.w_current_properties.w_colormap.cb_pos_cmap.currentText() == 'bottom':
                        divider = make_axes_locatable(axLin)
                        cax = divider.new_vertical(size="15%", pad=0.5, pack_start=True)
                        self.fig.add_axes(cax)
                        cbar = self.fig.colorbar(
                            pcolormesh_plot_lin, cax=cax, orientation="horizontal", use_gridspec=True)

                    cbar.minorticks_on()
                    plt.setp(cbar.ax.xaxis.get_majorticklabels() + cbar.ax.yaxis.get_majorticklabels(),
                             fontproperties=self.font_property, fontsize=self.tick_size)
                    cbar.set_label(label=cbar_label, size=self.label_size,
                                   fontproperties=self.font_property)

                if self.tw_properties.w_current_properties.w_show_info.check_show_info.isChecked():
                    _set_title(axes=axLog)

                if self.tw_properties.w_current_properties.w_show_pump.check_show_pump.isChecked():
                    _show_pump(axes=axLin, annotate=False)
                    _show_pump(axes=axLog, annotate=True)

    # -------- Global EAS/DAS/SAS Plot -------------------------------------------------------------
            if self.tw_properties.w_current_properties == self.tw_global_fit_properties[1]:
                if fit_dict[ukey]['meta']['model'] == 'parallel':
                    plottype = ukey + "_DAS"
                    data = fit_dict[ukey]['DAS'].copy()
                elif fit_dict[ukey]['meta']['model'] == 'sequential':
                    plottype = ukey + "_EAS"
                    data = fit_dict[ukey]['EAS']
                else:
                    plottype = ukey + "_SAS"
                    data = fit_dict[ukey]['SAS']

                labels = fit_dict[ukey]['meta']['components']
                labels_tex = [f'${label}$' for label in labels]
                normalize = self.tw_global_fit_properties[1].w_normalize.check_normalize.isChecked()
                ax1 = self.fig.add_subplot(111)
                ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)
                ax1.set_xlabel(msg.Labels.wavelength,
                               fontsize=self.label_size, fontproperties=self.font_property)
                ax1.set_ylabel(msg.Labels.delA, fontsize=self.label_size,
                               fontproperties=self.font_property)
                ax1.tick_params(labelleft=True, right=True)
                plt.setp(ax1.xaxis.get_majorticklabels() + ax1.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)
                ax1.xaxis.set_minor_locator(tk.AutoMinorLocator())

                # refresh z input if unnormalized default is used
                self.check_z_input(update_all=self.tw_global_fit_properties[1])
                _init_bounds(normalized=normalize, absolute=False)
                _set_axes_scale(axes=ax1, xlim=(self.x_min, self.x_max), ylim=(
                    self.z_min, self.z_max), x_formatter=self.sc.nm_formatter_ax)

                # ----- set normalization params -----
                if normalize:
                    ax1.set_ylabel(msg.Labels.delA_norm,
                                   fontsize=self.label_size, fontproperties=self.font_property)
                    norm_intervall = _get_norm_intervall(data=self.buffer_dataX)
                    data /= np.amax(abs(data[norm_intervall, :]), axis=0)

                colorlist = _get_colorlist(
                    plot_number=data.shape[1])
                clip_patch = _add_hide_patches(axes=ax1)
                ax1.set_prop_cycle(color=colorlist)
                lines = ax1.plot(self.buffer_dataX, data)
                mpl.artist.setp(lines, clip_path=clip_patch)

                if self.tw_global_fit_properties[1].w_show_legend.check_show_legend.isChecked():
                    _show_legend(axes=ax1, loc=self.tw_global_fit_properties[1].w_show_legend.cb_legend_loc.currentText(),
                                 handles=lines, labels=labels_tex)

                if self.tw_global_fit_properties[1].w_show_second_ax.check_show_second_ax.isChecked():
                    _show_2nd_axis(axes=ax1)

                if self.tw_global_fit_properties[1].w_show_info.check_show_info.isChecked():
                    _set_title(axes=ax1)

                if self.tw_global_fit_properties[1].w_show_pump.check_show_pump.isChecked():
                    _show_pump(axes=ax1)

                # ----- get colorlist -----
                conc = fit_dict[ukey]['conc']
                colorlist = _get_colorlist(plot_number=conc.shape[1])

                # ----- plotting -----
                ax1.set_prop_cycle(color=colorlist)
                ax1.plot(self.buffer_dataY, conc)

                if self.tw_global_fit_properties[1].w_show_info.check_show_info.isChecked():
                    _set_title(axes=ax1)

    # -------- Global Concentration Plot -----------------------------------------------------------
            if self.tw_properties.w_current_properties == self.tw_global_fit_properties[2]:
                plottype = ukey + "_conc"
                labels = fit_dict[ukey]['meta']['components']
                labels_tex = [f'${label}$' for label in labels]
                ax1 = self.fig.add_subplot(111)
                ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)
                ax1.set_xlabel(msg.Labels.delay, fontsize=self.label_size,
                               fontproperties=self.font_property)
                ax1.set_ylabel(r'Concentration', fontsize=self.label_size,
                               fontproperties=self.font_property)
                ax1.tick_params(labelleft=True, right=True)
                plt.setp(ax1.xaxis.get_majorticklabels() + ax1.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)

                # ----- set axis scale -----
                _init_bounds(normalized=True, absolute=True)
                _set_axes_scale(axes=ax1, xlim=(self.y_min, self.y_max), ylim=(
                    self.z_min, self.z_max), x_formatter=self.sc.delay_formatter0)

                # ----- get colorlist -----
                conc = fit_dict[ukey]['conc']
                colorlist = _get_colorlist(plot_number=conc.shape[1])

                # ----- plotting -----
                ax1.set_prop_cycle(color=colorlist)
                lines = ax1.plot(self.buffer_dataY, conc)
                if self.tw_global_fit_properties[2].w_show_legend.check_show_legend.isChecked():
                    _show_legend(axes=ax1, loc=self.tw_global_fit_properties[2].w_show_legend.cb_legend_loc.currentText(),
                                 handles=lines, labels=labels_tex)

                if self.tw_global_fit_properties[2].w_show_info.check_show_info.isChecked():
                    _set_title(axes=ax1)

    # -------- global delA Plot --------------------------------------------------------------------
            if self.tw_properties.w_current_properties == self.tw_global_fit_properties[3]:
                plottype = ukey + "_delA"
                self.delay_cut_list = np.asanyarray(self.delay_cut_list)
                delA_calc = fit_dict[ukey]['delA_calc']
                ind_delay_found = []
                for v in self.delay_cut_list:
                    idx = (abs(self.buffer_dataY - v)).argmin()
                    ind_delay_found.append(idx)

                if self.tw_global_fit_properties[3].w_show_residuals.check_show_residuals.isChecked():
                    gs = self.fig.add_gridspec(2, 1, height_ratios=[5, 1])
                    axs = gs.subplots(sharex=True)
                    ax1 = axs[0]
                else:
                    ax1 = self.fig.add_subplot(111)

                ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)
                ax1.set_xlabel(msg.Labels.wavelength,
                               fontsize=self.label_size, fontproperties=self.font_property)
                ax1.set_ylabel(msg.Labels.delA, fontsize=self.label_size,
                               fontproperties=self.font_property)
                ax1.tick_params(labelleft=True, right=True)
                plt.setp(ax1.xaxis.get_majorticklabels() + ax1.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)
                _init_bounds()
                ax1.xaxis.set_minor_locator(tk.AutoMinorLocator())
                _set_axes_scale(axes=ax1, xlim=(self.x_min, self.x_max), ylim=(
                    self.z_min, self.z_max), x_formatter=self.sc.nm_formatter_ax)

                colorlist = _get_colorlist(plot_number=len(self.delay_cut_list))
                fit_colorlist = utils.Converter.fitting_colorlist(colorlist)

                clip_patch = _add_hide_patches(axes=ax1)

                for i, ind in enumerate(ind_delay_found):
                    label = f"{self.sc.delay_formatter0(np.round(self.buffer_dataY[ind], decimals=14))}s"
                    line, = ax1.plot(self.buffer_dataX,
                                     delA_calc[ind, :], label=label, color=colorlist[i])
                    line.set_clip_path(clip_patch)

                    if self.tw_global_fit_properties[3].w_show_data.check_show_data.isChecked():
                        cross, = ax1.plot(self.buffer_dataX, self.buffer_dataZ[ind, :], 'x', color=fit_colorlist[i], alpha=0.75, markersize=(
                            self.tick_size / 3), zorder=1)
                        cross.set_clip_path(clip_patch)

                if self.tw_global_fit_properties[3].w_show_legend.check_show_legend.isChecked():
                    _show_legend(axes=ax1, title='Delay times',
                                 loc=self.tw_global_fit_properties[3].w_show_legend.cb_legend_loc.currentText())

                if self.tw_global_fit_properties[3].w_show_residuals.check_show_residuals.isChecked():
                    ax2 = axs[1]
                    ax2.axhline(y=0, linestyle='dashed',
                                color='black', alpha=0.5)
                    for i, ind in enumerate(ind_delay_found):
                        ax2.plot(self.buffer_dataX, delA_calc[ind, :] - self.buffer_dataZ[ind, :],
                                 color=colorlist[i], )

                if self.tw_global_fit_properties[3].w_show_second_ax.check_show_second_ax.isChecked():
                    _show_2nd_axis(axes=ax1)

                if self.tw_global_fit_properties[3].w_show_info.check_show_info.isChecked():
                    _set_title(axes=ax1)

                if self.tw_global_fit_properties[3].w_show_pump.check_show_pump.isChecked():
                    _show_pump(axes=ax1)

    # -------- global kinetic trace Plot -----------------------------------------------------------
            if self.tw_properties.w_current_properties == self.tw_global_fit_properties[4]:
                plottype = ukey + "_kin_trace"
                self.wavelength_trace_list = np.asarray(self.wavelength_trace_list)
                ind_wavelengths_found = []
                for v in self.wavelength_trace_list:
                    idx = (abs(self.buffer_dataX - v)).argmin()
                    ind_wavelengths_found.append(idx)

                normalize = self.tw_global_fit_properties[4].w_normalize.check_normalize.isChecked()
                absolute = self.tw_global_fit_properties[4].w_normalize.check_abs_value.isChecked()

                if self.tw_global_fit_properties[4].w_show_residuals.check_show_residuals.isChecked():
                    gs = self.fig.add_gridspec(2, 1, height_ratios=[5, 1])
                    axs = gs.subplots(sharex=True)
                    ax1 = axs[0]
                else:
                    ax1 = self.fig.add_subplot(111)

                ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)

                ax1.set_ylabel(msg.Labels.delA, fontsize=self.label_size,
                               fontproperties=self.font_property)
                ax1.tick_params(labelleft=True, right=True)
                plt.setp(ax1.xaxis.get_majorticklabels() + ax1.yaxis.get_majorticklabels(),
                         fontproperties=self.font_property, fontsize=self.tick_size)

                # ----- set axis scale -----
                self.check_z_input(update_all=self.tw_global_fit_properties[4])
                _init_bounds(normalized=normalize, absolute=absolute)

                _set_axes_scale(axes=ax1, xlim=(self.y_min, self.y_max), ylim=(
                    self.z_min, self.z_max), x_formatter=self.sc.delay_formatter0)

                # ----- set normalization params -----
                if normalize:
                    ax1.set_ylabel(msg.Labels.delA_norm,
                                   fontsize=self.label_size, fontproperties=self.font_property)
                    norm_intervall = _get_norm_intervall(data=self.buffer_dataY)

                    delA_calc = fit_dict[ukey]['delA_calc'].copy()
                    delA_calc /= np.amax(
                        abs(delA_calc[norm_intervall, :]), axis=0)
                    self.buffer_dataZ = self.buffer_dataZ / np.amax(
                        abs(fit_dict[ukey]['delA_calc'][norm_intervall, :]), axis=0)
                else:
                    delA_calc = fit_dict[ukey]['delA_calc'].copy()
                if absolute:
                    np.abs(delA_calc, out=delA_calc)
                    self.buffer_dataZ = np.abs(self.buffer_dataZ)

                # ----- get colorlist -----

                colorlist = _get_colorlist(
                    plot_number=len(self.wavelength_trace_list))
                fit_colorlist = utils.Converter.fitting_colorlist(colorlist)

                # ----- plotting -----
                for i, ind in enumerate(ind_wavelengths_found):
                    label = f"{self.sc.nm_formatter_ax(self.buffer_dataX[ind])} nm"
                    ax1.plot(self.buffer_dataY, delA_calc[:, ind], label=label, color=colorlist[i])

                    if self.tw_global_fit_properties[4].w_show_data.check_show_data.isChecked():
                        ax1.plot(self.buffer_dataY, self.buffer_dataZ[:, ind], 'x', color=fit_colorlist[i], alpha=0.75, markersize=(
                            self.tick_size / 3), zorder=1)

                if self.tw_global_fit_properties[4].w_show_residuals.check_show_residuals.isChecked():
                    ax2 = axs[1]
                    ax2.axhline(y=0, linestyle='dashed',
                                color='black', alpha=0.5)
                    for i, ind in enumerate(ind_wavelengths_found):
                        ax2.plot(self.buffer_dataY, delA_calc[:, ind] - self.buffer_dataZ[:, ind],
                                 color=colorlist[i], )
                    ax2.tick_params(labelleft=True, right=True)
                    plt.setp(ax2.xaxis.get_majorticklabels() + ax2.yaxis.get_majorticklabels(),
                             fontproperties=self.font_property, fontsize=self.tick_size)
                    ax2.set_xlabel(msg.Labels.delay,
                                   fontsize=self.label_size, fontproperties=self.font_property)
                else:
                    ax1.set_xlabel(msg.Labels.delay,
                                   fontsize=self.label_size, fontproperties=self.font_property)

                if self.tw_global_fit_properties[4].w_show_legend.check_show_legend.isChecked():
                    _show_legend(axes=ax1, title='Wavelength',
                                 loc=self.tw_global_fit_properties[4].w_show_legend.cb_legend_loc.currentText())

                if self.tw_global_fit_properties[4].w_show_info.check_show_info.isChecked():
                    _set_title(axes=ax1)

    # -------- global fit corner Plot --------------------------------------------------------------
            if self.tw_properties.w_current_properties == self.tw_global_fit_properties[5]:
                plottype = ukey + "_emcee_corner"

                if 'emcee' not in fit_dict[ukey]:
                    self.visualize_controller.call_statusbar("error", msg.Error.e24)
                    self._clear_canvas()
                    return
                flatchain = fit_dict[ukey]['emcee']['flatchain']
                params = fit_dict[ukey]['emcee']['params']

                num_vary = sum(p.vary for p in params.values())
                varying_names, labels, truths = [], [], []
                for name, par in params.items():
                    if not par.vary:
                        continue

                    # keep the name for slicing flatchain
                    varying_names.append(name)
                    truths.append(par.value)

                    # --- 2. pretty‑print the label ----------------------------------------
                    if name == "t0":
                        labels.append(r"$t_0$ (s)")
                    elif name == "IRF":
                        labels.append("IRF (s)")
                    elif name.startswith("t") and name[1:].isdigit():
                        labels.append(fr"$τ_{name[1:]}$ (s)")
                    elif name.startswith("__ln"):
                        labels.append(r"ln(σ/mOD)")
                    else:
                        labels.append(name)        # fallback

                subplot_spacing = self.tw_properties.w_current_properties.w_corner_manipulation.sb_subplot_pad.value() / \
                    100
                bins = self.tw_properties.w_current_properties.w_corner_manipulation.sb_bins.value()
                label_pad = self.tw_properties.w_current_properties.w_corner_manipulation.sb_label_pad.value() / 100
                max_n_ticks = self.tw_properties.w_current_properties.w_corner_manipulation.sb_tick_num.value()
                truths = truths if self.tw_properties.w_current_properties.w_corner_manipulation.check_truth.isChecked() else None

                show_titles = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_show_titles.isChecked() else False
                plot_contours = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_plot_contours.isChecked() else False
                plot_datapoints = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_plot_datapoints.isChecked() else False
                plot_density = True if self.tw_properties.w_current_properties.w_corner_manipulation.check_plot_density.isChecked() else False
                quantiles = [
                    0.16, 0.5, 0.84] if self.tw_properties.w_current_properties.w_corner_manipulation.check_show_quantiles.isChecked() else None

                f = corner.corner(flatchain, labels=labels, truths=truths, quantiles=quantiles,
                                  plot_contours=plot_contours, plot_density=plot_density,
                                  plot_datapoints=plot_datapoints, labelpad=label_pad, bins=bins,
                                  max_n_ticks=max_n_ticks, fig=self.fig)

                axes = np.array(f.axes).reshape((num_vary, num_vary))
                diagonal_axes = np.diag(axes)

                if show_titles:
                    fp_title = self.font_property.copy()
                    fp_title.set_size(self.label_size)
                    for i, ax in enumerate(diagonal_axes):
                        data = flatchain[:, i]
                        median = np.median(data)
                        qlow, qhigh = np.percentile(data, [16, 84])
                        title_str = r"${}$ $^{{+{}}}_{{-{}}}$ ".format(
                            self.sc.emcee_formatter(median),
                            self.sc.emcee_formatter(qhigh - median),
                            self.sc.emcee_formatter(median - qlow))
                        ax.set_title(title_str, fontproperties=fp_title)

                bottom_row = axes[-1, :]
                for ax in bottom_row:
                    ax.xaxis.set_major_formatter(self.sc.emcee_formatter)
                    plt.setp(ax.xaxis.get_majorticklabels(),
                             fontproperties=self.font_property, fontsize=self.tick_size)
                    ax.xaxis.label.set_fontproperties(self.font_property)
                    ax.xaxis.label.set_fontsize(self.label_size)

                left_column = axes[:, 0]
                for ax in left_column:
                    ax.yaxis.set_major_formatter(self.sc.emcee_formatter)
                    plt.setp(ax.yaxis.get_majorticklabels(),
                             fontproperties=self.font_property, fontsize=self.tick_size)
                    ax.yaxis.label.set_fontproperties(self.font_property)
                    ax.yaxis.label.set_fontsize(self.label_size)

                f.tight_layout()

                f.subplots_adjust(wspace=subplot_spacing,
                                  hspace=subplot_spacing)

    # -------- update Canvas -----------------------------------------------------------------------

        # remove old widget from layout

        if self.tw_select_plot.check_display_size.isChecked():
            self.sc.draw()  # Ensure figure updates first
            width, height = self.sc.get_width_height()
            self.sc.setFixedSize(width, height)
            layout = QGridLayout()
            layout.addWidget(self.toolbar, 0, 0,
                             alignment=Qt.AlignmentFlag.AlignTop)
            layout.addWidget(
                self.sc, 1, 0, alignment=Qt.AlignmentFlag.AlignJustify)
            # void to align canvas
            layout.addWidget(
                QWidget(), 2, 0, alignment=Qt.AlignmentFlag.AlignBottom)
        else:
            self.sc.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.sc.updateGeometry()

            layout = QVBoxLayout()
            layout.addWidget(self.toolbar)
            layout.addWidget(self.sc)

        self.tw_canvas.w_canvas.hide()
        self.tw_canvas.w_canvas.setParent(None)
        self.tw_canvas.w_canvas.deleteLater()
        self.tw_canvas.w_canvas = QWidget(self.tw_canvas)
        self.tw_canvas.w_canvas.setLayout(layout)
        self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)

    # -------- saving -----------------------------------------------------------
        if save_fig:
            timestamp = datetime.datetime.now().strftime("%y%m%d_")
            fig_format = self.tw_select_plot.cb_fig_format.currentText()
            dpi = utils.Converter.convert_str_input2float(
                self.tw_select_plot.le_fig_dpi.text())
            if self.tw_select_plot.check_svg_text2path.isChecked():
                mpl.rcParams['svg.fonttype'] = "path"

            else:
                mpl.rcParams['svg.fonttype'] = "none"

            if dpi is None:
                dpi = 300
            save_path = self.results_dir / f"{timestamp}{plottype}_ds{self.ds}.{fig_format}"
            save_path2 = self.results_dir / f"{timestamp}{plottype}_ds{self.ds}_resize.{fig_format}"

            warning_detected = False
            # if problems with tight layout
            with warnings.catch_warnings(record=True) as warning:
                self.fig.savefig(save_path, format=fig_format, dpi=dpi,)
                for i in warning:

                    self.visualize_controller.call_statusbar("error", msg.Error.e35)
                    warning_detected = True
            if warning_detected:
                self.fig.savefig(save_path2, format=fig_format,
                                 dpi=dpi, bbox_inches='tight', pad_inches=0.1)
            else:
                self.visualize_controller.call_statusbar("info", msg.Status.s19)
            warnings.resetwarnings()
