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

from PyQt6.QtCore import Qt
import logging
import logging.config
from ...configurations import messages as msg
from PyQt6.QtWidgets import QWidget, QLabel, QProgressBar,  QLineEdit, QSizePolicy, QPushButton, QHeaderView, QTextEdit, QComboBox,  QVBoxLayout,   QSpinBox, QGridLayout,  QGroupBox, QTableWidget,  QRadioButton,   QCheckBox
from ...configurations import messages as msg


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

        self.w_plot_params = QGroupBox("Plot Manipulation")
        self.w_plot_params.setToolTip(msg.ToolTips.t102)
        self.le_ymin = QLineEdit(placeholderText=msg.Widgets.i10)
        self.le_ymax = QLineEdit(placeholderText=msg.Widgets.i11)
        self.le_zmin = QLineEdit(placeholderText=msg.Widgets.i10)
        self.le_zmax = QLineEdit(placeholderText=msg.Widgets.i11)
        self.cb_yscale = QComboBox(objectName='cb_yscale')
        self.cb_yscale.setToolTip(msg.ToolTips.t65)
        self.cb_yscale.addItems(['lin', 'log', 'linlog'])
        self.cb_zscale = QComboBox(objectName='cb_zscale')
        self.cb_zscale.setToolTip(msg.ToolTips.t65)
        self.cb_zscale.addItems(['lin', 'log', 'linlog'])
        self.le_linlog_y = QLineEdit(self, placeholderText=msg.Widgets.i12)
        self.le_linlog_z = QLineEdit(self, placeholderText=msg.Widgets.i12)
        self.le_linlog_y.setToolTip(msg.ToolTips.t22)
        self.le_linlog_z.setToolTip(msg.ToolTips.t22)
        self.le_linlog_y.setVisible(False)
        self.le_linlog_z.setVisible(False)
        self.w_plot_params_layout = QGridLayout()
        self.w_plot_params_layout.addWidget(QLabel("Delay"), 0, 0, )
        self.w_plot_params_layout.addWidget(self.le_ymin, 0, 1)
        self.w_plot_params_layout.addWidget(self.le_ymax, 0, 2)
        self.w_plot_params_layout.addWidget(self.cb_yscale, 0, 3)
        self.w_plot_params_layout.addWidget(self.le_linlog_y, 0, 4)
        self.w_plot_params_layout.addWidget(QLabel("ΔA"), 1, 0,)
        self.w_plot_params_layout.addWidget(self.le_zmin, 1, 1)
        self.w_plot_params_layout.addWidget(self.le_zmax, 1, 2)
        self.w_plot_params_layout.addWidget(self.cb_zscale, 1, 3)
        self.w_plot_params_layout.addWidget(self.le_linlog_z, 1, 4)
        self.w_plot_params.setLayout(self.w_plot_params_layout)

        self.cb_yscale.currentIndexChanged.connect(
            self.change_linlog_visibility)
        self.cb_zscale.currentIndexChanged.connect(
            self.change_linlog_visibility)

        self.w_fit_params = QGroupBox("Fitting Parameters")
        self.le_wavelength = QLineEdit()
        self.le_wavelength.setToolTip(msg.ToolTips.t103)
        self.sp_wavelength_area = QSpinBox(
            minimum=0,  value=0, suffix=' nm', prefix='± ')
        self.sp_wavelength_area.setToolTip(msg.ToolTips.t104)
        self.sb_components = QSpinBox(minimum=1, maximum=9, value=1)
        self.check_infinte = QCheckBox("Infinite Component?")
        self.label_ca = QLabel("Fit CA")
        self.label_ca.setToolTip(msg.ToolTips.t139)
        self.cb_ca_order = QComboBox()
        self.cb_ca_order.addItems(['false','zero order', 'zero + 1st order'])
        self.cb_ca_order.setToolTip(msg.ToolTips.t140)
        self.cb_model = QComboBox()
        self.cb_model.addItems(['parallel', 'sequential', ])
        self.cb_t0_def = QComboBox()
        self.cb_t0_def.addItems(['5% Threshold', 'Gaussian Center'])
        self.cb_method = QComboBox()
        self.cb_method.addItems(['nelder', 'leastsq', 'diff-evol'])
        self.pb_fit = QPushButton("Fit")
        self.pb_show_guess = QPushButton('Show Initial Guess')

        self.sb_components.setToolTip(msg.ToolTips.t105)
        self.check_infinte.setToolTip(msg.ToolTips.t106)
        self.cb_model.setToolTip(msg.ToolTips.t107)
        self.cb_t0_def.setToolTip(msg.ToolTips.t108)
        self.cb_method.setToolTip(msg.ToolTips.t109)
        self.pb_show_guess.setToolTip(msg.ToolTips.t111)
        self.pb_fit.setToolTip(msg.ToolTips.t112)

        self.w_fit_params_layout = QGridLayout()
        self.w_fit_params_layout.addWidget(QLabel("Wavelength"), 0, 0, )
        self.w_fit_params_layout.addWidget(self.le_wavelength, 0, 1, )
        self.w_fit_params_layout.addWidget(self.sp_wavelength_area, 0, 2, )
        self.w_fit_params_layout.addWidget(QLabel("# Components"), 1, 0, 1, 2)
        self.w_fit_params_layout.addWidget(self.sb_components, 1, 2, )
        self.w_fit_params_layout.addWidget(self.check_infinte, 2, 0, 1, 3)
        self.w_fit_params_layout.addWidget(self.label_ca, 3, 0, )
        self.w_fit_params_layout.addWidget(self.cb_ca_order, 3, 1, 1, 2)
        self.w_fit_params_layout.addWidget(QLabel("Model"), 4, 0, )
        self.w_fit_params_layout.addWidget(self.cb_model, 4, 1, 1, 2)
        self.w_fit_params_layout.addWidget(QLabel("Time Zero"), 5, 0, )
        self.w_fit_params_layout.addWidget(self.cb_t0_def, 5, 1, 1, 2)
        self.w_fit_params_layout.addWidget(QLabel("Method"), 6, 0, )
        self.w_fit_params_layout.addWidget(self.cb_method, 6, 1, 1, 2)
        self.w_fit_params_layout.addWidget(self.pb_show_guess, 8, 0, 1, 2)
        self.w_fit_params_layout.addWidget(self.pb_fit, 8, 2, 1, 2)

        self.w_fit_params.setLayout(self.w_fit_params_layout)
        self.sb_components.valueChanged.connect(self.update_tab_fit_params)

        self.tab_fit_params = QTableWidget(self)

        self.tab_fit_params.setToolTip(msg.ToolTips.t110)

        self.component_labels = ['t0', 'IRF', 'τ1',
                                 'τ2', 'τ3', 'τ4', 'τ5', 'τ6', 'τ7', 'τ8', 'τ9']
        self.tab_fit_params.setColumnCount(4)
        header = self.tab_fit_params.horizontalHeader()
        self.tab_fit_params.setHorizontalHeaderLabels(
            ['value', 'min', 'max', 'vary'])
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)

        self.tab_fit_params.setRowCount(3)
        self.tab_fit_params.setVerticalHeaderLabels(self.component_labels[:])

        self.w_fit_params_layout.addWidget(self.tab_fit_params, 7, 0, 1, 3)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_ds, )
        self.layout.addWidget(self.w_plot_params, )
        self.layout.addWidget(self.w_fit_params, )

        self.layout.addStretch()

    def change_linlog_visibility(self):
        if self.sender().objectName() == 'cb_yscale':
            if self.sender().currentText() == 'linlog':
                self.le_linlog_y.setVisible(True)
            else:
                self.le_linlog_y.setVisible(False)
        elif self.sender().objectName() == 'cb_zscale':
            if self.sender().currentText() == 'linlog':
                self.le_linlog_z.setVisible(True)
            else:
                self.le_linlog_z.setVisible(False)

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


class CanvasWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.gb_canvas = QGroupBox('Fitting Preview')
        self.w_canvas = QLabel(msg.Widgets.i13)
        self.view_layout = QVBoxLayout(self)

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
        self.tab_ds_fits.setHorizontalHeaderLabels(['Wavelength'])
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
