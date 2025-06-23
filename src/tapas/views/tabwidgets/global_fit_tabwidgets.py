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

from PyQt6.QtWidgets import QWidget, QLabel, QLayout, QProgressBar,  QLineEdit, QSizePolicy, QPushButton, QHeaderView, QTextEdit, QComboBox,  QVBoxLayout,   QSpinBox, QGridLayout,  QGroupBox, QTableWidget,  QRadioButton,   QCheckBox
from PyQt6.QtSvgWidgets import QSvgWidget
from PyQt6.QtCore import Qt
from pathlib import Path
import pkg_resources
import sys
import logging
import logging.config
from ...configurations import messages as msg


def resource_path(*parts: str) -> Path:
    ''' find where that resource lives at runtime—
    whether in your source tree, an installed wheel, or inside a PyInstaller bundle—and returns
    a `pathlib.Path` pointing to it on disk '''
    if getattr(sys, "frozen", False):
        # PyInstaller: files were collected into _MEIPASS under the same subfolders
        return Path(sys._MEIPASS, "tapas", *parts)
    resource_subpath = "/".join(parts)
    filename = pkg_resources.resource_filename("tapas", resource_subpath)
    return Path(filename)


class InputWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_ds = QGroupBox("Dataset")
        self.w_ds.setToolTip(msg.ToolTips.t18)
        self.rb_ds_1 = QRadioButton("Dataset 1", self, objectName='ds1')
        self.rb_ds_1.setChecked(True)
        self.rb_ds_2 = QRadioButton("Dataset 2", self, objectName='ds2')
        self.rb_ds_3 = QRadioButton("Dataset 3", self, objectName='ds3')
        self.w_ds_layout = QVBoxLayout()
        self.w_ds_layout.addWidget(self.rb_ds_1)
        self.w_ds_layout.addWidget(self.rb_ds_2)
        self.w_ds_layout.addWidget(self.rb_ds_3)
        self.w_ds.setLayout(self.w_ds_layout)

        self.w_view_manipulations = QGroupBox("View Manipulations")
        self.le_linlog = QLineEdit(self, placeholderText=msg.Widgets.i07)
        self.le_linlog.setToolTip(msg.ToolTips.t22)
        self.le_linlog.setMaximumWidth(50)
        self.check_normalize = QCheckBox('Normalize DAS/EAS/SAS')
        self.check_normalize.setToolTip(msg.ToolTips.t116)
        self.check_normalized_resid = QCheckBox('Percent Residuals')
        self.check_normalized_resid.setToolTip(msg.ToolTips.t117)

        self.w_view_manipulations_layout = QGridLayout()
        self.w_view_manipulations_layout.addWidget(
            QLabel("Lin/Log Transition"), 0, 0)
        self.w_view_manipulations_layout.addWidget(self.le_linlog, 0, 1)
        self.w_view_manipulations_layout.addWidget(
            self.check_normalize, 2, 0, 1, 2)
        self.w_view_manipulations_layout.addWidget(self.check_normalized_resid, 3, 0)
        self.w_view_manipulations.setLayout(self.w_view_manipulations_layout)

        self.w_fit_params = QGroupBox("Fitting Parameters")
        self.sb_components = QSpinBox(minimum=1, maximum=9, value=1)
        self.check_infinte = QCheckBox("Infinite Component?")
        self.check_gs = QCheckBox("Explicit GSB")
        self.gs_use_ss_abs = QCheckBox("Use Steady-State")
        self.gs_use_ss_abs.setEnabled(False)
        self.label_ca = QLabel("Fit CA")
        self.label_ca.setToolTip(msg.ToolTips.t139)
        self.cb_ca_order = QComboBox()
        self.cb_ca_order.addItems(['false','zero order', 'zero + 1st order'])
        self.cb_ca_order.setToolTip(msg.ToolTips.t140)

        self.cb_model = QComboBox()
        self.cb_model.addItems(
            ['parallel', 'sequential', '2C_3k_1', '3C_5k_1', '3C_4k_1', '4C_6k_1', ])
        
        self.sp_substeps = QSpinBox(minimum=1, maximum=100, value=6)
        self.cb_t0_def = QComboBox()
        self.cb_t0_def.addItems(['5% Threshold', 'Gaussian Center'])
        self.cb_method = QComboBox()
        self.cb_method.addItems(['nelder', 'leastsq', 'diff-evol'])
        self.pb_fit = QPushButton("Fit")
        self.pb_show_guess = QPushButton('Show Initial Guess')
        self.w_fit_params_layout = QGridLayout()

        self.w_model_preview = QGroupBox("Model Preview")
        model = self.cb_model.currentText()
        svg_path = resource_path() / "assets" / f"{model}.svg"
        self.svg_widget = QSvgWidget(str(svg_path))
        self.svg_widget.setFixedSize(250, 250)
        model_preview_layout = QVBoxLayout(self.w_model_preview)
        model_preview_layout.addWidget(self.svg_widget)
        self.w_model_preview .setLayout(model_preview_layout)

        self.cb_model.currentIndexChanged.connect(self.update_model_preview)
        self.label_microsteps = QLabel("Microsteps")
        self.sp_substeps.setVisible(False)
        self.label_microsteps.setVisible(False)
        self.w_fit_params_layout.addWidget(QLabel("# Components"), 1, 0, )
        self.w_fit_params_layout.addWidget(self.sb_components, 1, 1, )
        self.w_fit_params_layout.addWidget(self.check_infinte, 2, 0, 1, 2)
        self.w_fit_params_layout.addWidget(self.check_gs, 3, 0)
        self.w_fit_params_layout.addWidget(self.gs_use_ss_abs, 3, 1)
        self.w_fit_params_layout.addWidget(self.label_ca, 4, 0, )
        self.w_fit_params_layout.addWidget(self.cb_ca_order, 4, 1, )
        self.w_fit_params_layout.addWidget(QLabel("Model"), 5, 0, )
        self.w_fit_params_layout.addWidget(self.cb_model, 5, 1, )
        self.w_fit_params_layout.addWidget(self.label_microsteps, 6, 0, )
        self.w_fit_params_layout.addWidget(self.sp_substeps, 6, 1, )
        self.w_fit_params_layout.addWidget(QLabel("Time Zero"), 7, 0, )
        self.w_fit_params_layout.addWidget(self.cb_t0_def, 7, 1, )
        self.w_fit_params_layout.addWidget(QLabel("Method"), 8, 0, )
        self.w_fit_params_layout.addWidget(self.cb_method, 8, 1, )

        self.sb_weights1_min = QSpinBox(suffix=' nm')
        self.sb_weights1_min.setEnabled(False)
        self.sb_weights1_max = QSpinBox(suffix=' nm')
        self.sb_weights1_value = QSpinBox(
            minimum=0, maximum=100, value=100, suffix=' %')
        self.sb_weights2_min = QSpinBox(suffix=' nm')
        self.sb_weights2_min.setEnabled(False)
        self.sb_weights2_max = QSpinBox(suffix=' nm')
        self.sb_weights2_value = QSpinBox(
            minimum=0, maximum=100, value=100, suffix=' %')
        self.sb_weights3_min = QSpinBox(suffix=' nm')
        self.sb_weights3_min.setEnabled(False)
        self.sb_weights3_max = QSpinBox(suffix=' nm')
        self.sb_weights3_value = QSpinBox(
            minimum=0, maximum=100, value=100, suffix=' %')
        self.sb_weights4_min = QSpinBox(suffix=' nm')
        self.sb_weights4_min.setEnabled(False)
        self.sb_weights4_max = QSpinBox(suffix=' nm')
        self.sb_weights4_value = QSpinBox(
            minimum=0, maximum=100, value=100, suffix=' %')
        self.sb_weights5_min = QSpinBox(suffix=' nm')
        self.sb_weights5_min.setEnabled(False)
        self.sb_weights5_max = QSpinBox(suffix=' nm')
        self.sb_weights5_max.setEnabled(False)
        self.sb_weights5_value = QSpinBox(
            minimum=0, maximum=100, value=100, suffix=' %')

        self.container1 = QWidget()
        h_layout = QGridLayout(self.container1)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(self.sb_weights1_min, 0, 0)
        h_layout.addWidget(self.sb_weights1_max, 0, 1)
        h_layout.addWidget(self.sb_weights1_value, 0, 2)
        self.container2 = QWidget()
        h_layout = QGridLayout(self.container2)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(self.sb_weights2_min, 0, 0)
        h_layout.addWidget(self.sb_weights2_max, 0, 1)
        h_layout.addWidget(self.sb_weights2_value, 0, 2)
        self.container3 = QWidget()
        h_layout = QGridLayout(self.container3)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(self.sb_weights3_min, 0, 0)
        h_layout.addWidget(self.sb_weights3_max, 0, 1)
        h_layout.addWidget(self.sb_weights3_value, 0, 2)
        self.container4 = QWidget()
        h_layout = QGridLayout(self.container4)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(self.sb_weights4_min, 0, 0)
        h_layout.addWidget(self.sb_weights4_max, 0, 1)
        h_layout.addWidget(self.sb_weights4_value, 0, 2)
        self.container5 = QWidget()
        h_layout = QGridLayout(self.container5)
        h_layout.setContentsMargins(0, 0, 0, 0)
        h_layout.addWidget(self.sb_weights5_min, 0, 0)
        h_layout.addWidget(self.sb_weights5_max, 0, 1)
        h_layout.addWidget(self.sb_weights5_value, 0, 2)
        self.container1.setEnabled(False)
        self.container2.hide()
        self.container3.hide()
        self.container4.hide()
        self.container5.hide()

        self.w_fit_params_layout.addWidget(QLabel("Weights"), 9, 0, )
        self.w_fit_params_layout.addWidget(self.container1, 10, 0, 1, 2)
        self.w_fit_params_layout.addWidget(self.container2, 11, 0, 1, 2)
        self.w_fit_params_layout.addWidget(self.container3, 12, 0, 1, 2)
        self.w_fit_params_layout.addWidget(self.container4, 13, 0, 1, 2)
        self.w_fit_params_layout.addWidget(self.container5, 14, 0, 1, 2)

        self.w_fit_params_layout.addWidget(self.pb_show_guess, 16, 0, )
        self.w_fit_params_layout.addWidget(self.pb_fit, 16, 1, 1, 2)

        self.w_fit_params.setLayout(self.w_fit_params_layout)
        self.sb_components.valueChanged.connect(self.update_tab_fit_params)

        self.tab_fit_params = QTableWidget(self)

        self.component_labels = ['t0', 'IRF', 'τ1',
                                 'τ2', 'τ3', 'τ4', 'τ5', 'τ6', 'τ7', 'τ8', 'τ9']
        self.tab_fit_params.setColumnCount(4)
        self.tab_fit_params.setMinimumHeight(150)
        header = self.tab_fit_params.horizontalHeader()
        self.tab_fit_params.setHorizontalHeaderLabels(
            ['value', 'min', 'max', 'vary'])
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.tab_fit_params.setRowCount(3)
        self.tab_fit_params.setVerticalHeaderLabels(self.component_labels[:])

        self.w_fit_params_layout.addWidget(self.tab_fit_params, 15, 0, 1, 2)

        self.sb_components.setToolTip(msg.ToolTips.t105)
        self.check_infinte.setToolTip(msg.ToolTips.t106)
        self.cb_model.setToolTip(msg.ToolTips.t107)
        self.cb_t0_def.setToolTip(msg.ToolTips.t108)
        self.cb_method.setToolTip(msg.ToolTips.t109)
        self.pb_show_guess.setToolTip(msg.ToolTips.t111)
        self.pb_fit.setToolTip(msg.ToolTips.t112)
        self.tab_fit_params.setToolTip(msg.ToolTips.t110)
        self.check_gs.setToolTip(msg.ToolTips.t113)
        self.gs_use_ss_abs.setToolTip(msg.ToolTips.t114)
        self.container1.setToolTip(msg.ToolTips.t115)
        self.sp_substeps.setToolTip(msg.ToolTips.t125)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_ds, )

        self.layout.addWidget(self.w_view_manipulations, )
        self.layout.addWidget(self.w_model_preview, )
        self.layout.addWidget(self.w_fit_params, )

        self.layout.addStretch()

    def update_model_preview(self):
        # Get current model text from combobox selection
        model = self.cb_model.currentText()
        svg_path = resource_path() / "assets" / f"{model}.svg"
        if model == 'parallel':
            self.sp_substeps.setVisible(False)
            self.label_microsteps.setVisible(False)
        else:
            self.sp_substeps.setVisible(True)
            self.label_microsteps.setVisible(True)
        # Reload the SVG into the widget by calling load() with the updated path
        self.svg_widget.load(str(svg_path))
        # Optionally, you might call update() to force a repaint:
        self.svg_widget.update()

    def update_tab_fit_params(self):
        row = self.tab_fit_params.rowCount()

        components = self.sb_components.value()

        while row > (components + 2):

            self.tab_fit_params.removeRow(row - 1)

            row -= 1
            self.tab_fit_params.setVerticalHeaderLabels(
                self.component_labels[:])

        while row < (components + 2):

            self.tab_fit_params.insertRow(row)

            row += 1
            self.tab_fit_params.setVerticalHeaderLabels(
                self.component_labels[:])

        self.w_fit_params_layout.update()
        self.w_fit_params.adjustSize()
        parent = self.parentWidget()


class CanvasWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.gb_canvas = QGroupBox('Fitting Preview')
        self.w_canvas = QLabel(msg.Widgets.i13)
        self.view_layout = QVBoxLayout(self)
        # self.layout.addWidget(self.group_box)
        self.view_layout.addWidget(self.w_canvas)
        self.gb_canvas.setLayout(self.view_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.gb_canvas)

    @staticmethod
    def create_emcee_canvas():
        widget = QWidget()
        widget.title = QLabel()
        widget.te_progress = QTextEdit()
        widget.te_progress.setFixedHeight(200)
        widget.progress_bar = QProgressBar()
        widget.progress_bar.setRange(0, 0)

        widget.label_results = QLabel()
        widget.te_results = QTextEdit()
        widget.te_results.setReadOnly(True)
        widget.te_results.setFixedHeight(200)
        widget.corner_canvas = QLabel()

        widget.title.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        widget.progress_bar.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        widget.label_results.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        widget.te_progress.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        widget.te_results.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        widget.corner_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        widget.layout = QGridLayout()
        widget.layout.addWidget(widget.title, 0, 0, 1, 2)
        widget.layout.addWidget(widget.te_progress, 1, 0,)
        widget.layout.addWidget(widget.te_results,  1, 1)
        widget.layout.addWidget(widget.progress_bar,  2,
                                0, 1, 2, alignment=Qt.AlignmentFlag.AlignVCenter)
        widget.layout.addWidget(widget.label_results, 3, 0, 1, 2)

        widget.layout.addWidget(widget.corner_canvas,  4, 0, 1, 2)

        widget.setLayout(widget.layout)
        return widget


class ResultsWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.w_results_output = QGroupBox("Fitting Results")
        self.te_results = QTextEdit()
        self.te_results.setReadOnly(True)
        self.te_results.setMinimumHeight(250)
        self.pb_save_fit = QPushButton('Save Fit to Dataset')

        self.w_results_output_layout = QGridLayout()
        self.w_results_output_layout.addWidget(self.te_results, 0, 0)
        self.w_results_output_layout.addWidget(self.pb_save_fit, 1, 0)
        self.w_results_output.setLayout(self.w_results_output_layout)

        self.w_fitting_list = QGroupBox("Fitting List")
        self.w_fitting_list_layout = QGridLayout()
        self.tab_ds_fits = QTableWidget(self)

        self.component_labels = ['model', 'r2', 'Ainf', 't0', 'IRF',
                                 'τ1', 'τ2', 'τ3', 'τ4', 'τ5', 'τ6', 'τ7', 'τ8', 'τ9']
        self.tab_ds_fits.setColumnCount(1)
        self.header = self.tab_ds_fits.horizontalHeader()
        self.tab_ds_fits.setHorizontalHeaderLabels(['Fit'])
        self.header.setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents)

        self.tab_ds_fits.setRowCount(5)
        self.tab_ds_fits.setVerticalHeaderLabels(self.component_labels)

        self.pb_delete_fit = QPushButton('Delete Selected Fit')
        self.w_fitting_list_layout.addWidget(self.tab_ds_fits, 0, 0)
        self.w_fitting_list_layout.addWidget(self.pb_delete_fit, 1, 0)
        self.w_fitting_list.setLayout(self.w_fitting_list_layout)

        self.w_results_posterior = QGroupBox("Explore Parameter Space")

        self.sb_burn = QSpinBox(minimum=0, maximum=9999,
                                value=300, suffix=' samples')
        self.sb_init = QSpinBox(minimum=100, maximum=99999,
                                value=500, suffix=' samples')
        self.sb_thin = QSpinBox(minimum=1,  value=1, suffix=' samples')
        self.sb_target_ratio = QSpinBox(minimum=2,  value=50, )
        self.pb_run_emcee = QPushButton('Perform Analysis')
        self.pb_cancel_emcee = QPushButton('Abort Analysis')
        self.pb_save_emcee = QPushButton('Save Analysis')

        self.w_results_posterior.setToolTip(msg.ToolTips.t118)
        self.sb_burn.setToolTip(msg.ToolTips.t119)
        self.sb_init.setToolTip(msg.ToolTips.t120)
        self.sb_thin.setToolTip(msg.ToolTips.t121)
        self.sb_target_ratio.setToolTip(msg.ToolTips.t122)
        self.pb_run_emcee.setToolTip(msg.ToolTips.t123)
        self.pb_cancel_emcee.setToolTip(msg.ToolTips.t124)

        self.w_results_posterior_layout = QGridLayout()
        self.w_results_posterior_layout.addWidget(
            QLabel('Discard the First'), 0, 0)
        self.w_results_posterior_layout.addWidget(self.sb_burn, 0, 1)
        self.w_results_posterior_layout.addWidget(QLabel('Initialize'), 1, 0)
        self.w_results_posterior_layout.addWidget(self.sb_init, 1, 1)
        self.w_results_posterior_layout.addWidget(
            QLabel('Accept one per '), 2, 0)
        self.w_results_posterior_layout.addWidget(self.sb_thin, 2, 1)
        self.w_results_posterior_layout.addWidget(QLabel('Target Ratio'), 3, 0)
        self.w_results_posterior_layout.addWidget(self.sb_target_ratio, 3, 1)
        self.w_results_posterior_layout.addWidget(self.pb_run_emcee, 4, 0)
        self.w_results_posterior_layout.addWidget(self.pb_cancel_emcee, 4, 1)
        self.w_results_posterior_layout.addWidget(self.pb_save_emcee, 5, 0)

        self.w_results_posterior.setLayout(self.w_results_posterior_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_results_output,
                              alignment=Qt.AlignmentFlag.AlignTop)
        self.layout.addWidget(self.w_fitting_list, )
        self.layout.addWidget(self.w_results_posterior, )
        self.layout.addStretch()
