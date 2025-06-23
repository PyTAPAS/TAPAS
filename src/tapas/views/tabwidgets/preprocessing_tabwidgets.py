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

from PyQt6.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QFrame,  QVBoxLayout,  QSpinBox, QGridLayout,  QGroupBox,  QRadioButton
from PyQt6.QtCore import Qt

import logging
import logging.config
from ...configurations import messages as msg


@staticmethod
def separator():
    separator = QFrame()
    separator.setFrameShape(QFrame.Shape.HLine)
    separator.setFrameShadow(QFrame.Shadow.Sunken)
    return separator


class SelectViewWidget(QWidget):
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
        self.pb_clear_ds = QPushButton("Clear Selected Dataset")
        self.pb_clear_ds.setToolTip(msg.ToolTips.t19)
        self.pb_clear_ds.setMaximumWidth(200)
        self.w_ds_layout = QVBoxLayout()
        self.w_ds_layout.addWidget(self.rb_ds_1)
        self.w_ds_layout.addWidget(self.rb_ds_2)
        self.w_ds_layout.addWidget(self.rb_ds_3)
        self.w_ds_layout.addWidget(self.pb_clear_ds)
        self.w_ds.setLayout(self.w_ds_layout)

        self.gb_views = QGroupBox("Views")
        self.rb_full_view = QRadioButton(
            "Full View", self, objectName='full_view',)
        self.rb_full_view.setToolTip(msg.ToolTips.t20)
        self.rb_full_view.setChecked(True)
        self.rb_t0_view = QRadioButton(
            "Time Zero View", self, objectName='t0_view')
        self.rb_t0_view.setToolTip(msg.ToolTips.t21)
        self.rb_manual_view = QRadioButton(
            "Manual View", self, objectName='manual_view')
        self.w_view_layout = QVBoxLayout()
        self.w_view_layout.addWidget(self.rb_full_view)
        self.w_view_layout.addWidget(self.rb_t0_view)
        self.w_view_layout.addWidget(self.rb_manual_view)
        self.gb_views.setLayout(self.w_view_layout)

        self.w_view_manipulations = QGroupBox("View Manipulations")
        self.w_view_manipulations.setToolTip(msg.ToolTips.t22)
        self.le_linlog = QLineEdit(self, placeholderText=msg.Widgets.i07)
        self.le_linlog.setMaximumWidth(50)

        self.w_view_manipulations_layout = QGridLayout()
        self.w_view_manipulations_layout.addWidget(
            QLabel("Lin/Log Transition"), 0, 0)
        self.w_view_manipulations_layout.addWidget(self.le_linlog, 0, 1)

        self.w_view_manipulations.setLayout(self.w_view_manipulations_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.layout.addWidget(self.w_ds)
        self.layout.addWidget(self.gb_views)
        self.layout.addWidget(self.w_view_manipulations)
        self.layout.addStretch()


class CanvasWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.gb_canvas = QGroupBox('Data Preview')
        self.w_canvas = QLabel(msg.Widgets.i06)
        self.view_layout = QVBoxLayout(self)
        self.view_layout.addWidget(self.w_canvas)
        self.gb_canvas.setLayout(self.view_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.gb_canvas)


class ProcessWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_trimm = QGroupBox('Trim Data')
        self.w_trimm.setToolTip(msg.ToolTips.t23)
        self.le_xmin = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50)

        self.le_xmax = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50)
        self.le_ymin = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50)
        self.le_ymax = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50)
        self.pb_apply_trimm = QPushButton("Apply")
        self.pb_apply_trimm.setToolTip(msg.ToolTips.t24)
        self.w_trimm_layout = QGridLayout()

        self.w_trimm_layout.addWidget(
            QLabel("Wavelength", maximumWidth=90), 0, 0, )
        self.w_trimm_layout.addWidget(self.le_xmin, 0, 1)
        self.w_trimm_layout.addWidget(self.le_xmax, 0, 2)
        self.w_trimm_layout.addWidget(QLabel("Delay", maximumWidth=90), 1, 0)
        self.w_trimm_layout.addWidget(self.le_ymin, 1, 1)
        self.w_trimm_layout.addWidget(self.le_ymax, 1, 2)
        self.w_trimm_layout.addWidget(separator(), 2, 0, 1, 3)
        self.w_trimm_layout.addWidget(self.pb_apply_trimm, 3, 2)
        self.w_trimm.setLayout(self.w_trimm_layout)

        self.w_resample = QGroupBox('Resampling')
        self.pb_resample = QPushButton("Resample", maximumWidth=130)
        self.pb_resample.setToolTip(msg.ToolTips.t25)
        self.le_min_resample = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50)
        self.le_min_resample.setToolTip(msg.ToolTips.t26)
        self.le_max_resample = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50)
        self.le_max_resample.setToolTip(msg.ToolTips.t26)
        self.le_resample_factor = QLineEdit(maximumWidth=50, placeholderText=msg.Widgets.i21)
        self.le_resample_factor.setToolTip(msg.ToolTips.t27)
        self.cb_resample_method = QComboBox(self)
        self.cb_resample_method.setToolTip(msg.ToolTips.t28)
        self.cb_resample_method.addItems(
            ['linear', 'quadratic', 'cubic', 'pchip', 'akima', 'makima'])
        self.cb_resample_axis = QComboBox(self)
        self.cb_resample_axis.addItems(['timepoints', 'wavelength'])
        self.cb_resample_axis.setToolTip(msg.ToolTips.t29)

        self.pb_regularize = QPushButton("Regularize", maximumWidth=130)
        self.pb_regularize.setToolTip(msg.ToolTips.t30)
        self.sb_regularize_points = QSpinBox(self, minimum=2, maximum=10000, value=3)
        self.sb_regularize_points.setToolTip(msg.ToolTips.t31)
        self.cb_regularize_method = QComboBox(self)
        self.cb_regularize_method.addItems(
            ['linear', 'quadratic', 'cubic', 'pchip', 'akima', 'makima'])
        self.cb_regularize_method.setToolTip(msg.ToolTips.t28)
        self.cb_regularize_axis = QComboBox(self)
        self.cb_regularize_axis.addItems(['timepoints', 'wavelength'])
        self.cb_regularize_axis.setToolTip(msg.ToolTips.t29)
        self.pb_delete_resample = QPushButton("Delete Resampling", maximumWidth=130)
        self.pb_delete_resample.setToolTip(msg.ToolTips.t41)
        self.pb_apply_resample = QPushButton(
            "Apply", maximumWidth=130, objectName="apply_resampling")
        self.pb_apply_resample.setToolTip(msg.ToolTips.t24)

        self.w_resample_layout = QGridLayout()
        self.w_resample_layout.addWidget(self.pb_resample, 0, 0)
        self.w_resample_layout.addWidget(self.le_min_resample, 0, 1)
        self.w_resample_layout.addWidget(self.le_max_resample, 0, 2)
        self.w_resample_layout.addWidget(QLabel("Factor"), 1, 1)
        self.w_resample_layout.addWidget(self.le_resample_factor, 1, 2)
        self.w_resample_layout.addWidget(QLabel("Axis"), 2, 1)
        self.w_resample_layout.addWidget(self.cb_resample_axis, 2, 2)
        self.w_resample_layout.addWidget(QLabel("Method"), 3, 1)
        self.w_resample_layout.addWidget(self.cb_resample_method, 3, 2)
        self.w_resample_layout.addWidget(separator(), 4, 0, 1, 3)
        self.w_resample_layout.addWidget(self.pb_regularize, 5, 0)
        self.w_resample_layout.addWidget(QLabel('Datapoints'), 5, 1)
        self.w_resample_layout.addWidget(self.sb_regularize_points, 5, 2)
        self.w_resample_layout.addWidget(QLabel("Axis"), 6, 1)
        self.w_resample_layout.addWidget(self.cb_regularize_axis, 6, 2)
        self.w_resample_layout.addWidget(QLabel("Method"), 7, 1)
        self.w_resample_layout.addWidget(self.cb_regularize_method, 7, 2)
        self.w_resample_layout.addWidget(separator(), 8, 0, 1, 3)
        self.w_resample_layout.addWidget(self.pb_delete_resample, 9, 1)
        self.w_resample_layout.addWidget(self.pb_apply_resample, 9, 2)
        self.w_resample.setLayout(self.w_resample_layout)

        self.w_chirp = QGroupBox('Chirp Correction')
        self.pb_autocorr_chirp = QPushButton("Autocorrect", maximumWidth=130)
        self.pb_autocorr_chirp.setToolTip(msg.ToolTips.t32)
        self.le_threshold_chirp = QLineEdit(
            placeholderText=msg.Widgets.i12, maximumWidth=50)
        self.le_threshold_chirp.setToolTip(msg.ToolTips.t33)
        self.cb_threshold_unit = QComboBox(maximumWidth=50)
        self.cb_threshold_unit.addItems(('mOD', '%', 'max'))
        self.cb_threshold_unit.setToolTip(msg.ToolTips.t33)

        self.pb_manually_chirp = QPushButton(
            "Manually Correct ", objectName='manual_chirp', maximumWidth=130)
        self.pb_manually_chirp.setToolTip(msg.ToolTips.t36)
        self.pb_manually_chirp.setCheckable(True)
        self.pb_fit_chirp = QPushButton("Fit ", maximumWidth=130)
        self.pb_fit_chirp.setToolTip(msg.ToolTips.t37)
        self.pb_del_chirp_fit = QPushButton("Delete ", maximumWidth=130)
        self.pb_del_chirp_fit.setToolTip(msg.ToolTips.t38)

        self.pb_fromfile_chirp = QPushButton(
            "Correct from Project", maximumWidth=130)
        self.pb_fromfile_chirp.setToolTip(msg.ToolTips.t34)
        self.cb_fromfile_ds_chirp = QComboBox()
        self.cb_fromfile_ds_chirp.addItems(('ds 1', 'ds 2', 'ds 3'))
        self.cb_fromfile_ds_chirp.setToolTip(msg.ToolTips.t35)

        self.pb_show_chirp_correction = QPushButton(
            "Show Correction", maximumWidth=130)
        self.pb_show_chirp_correction.setToolTip(msg.ToolTips.t39)
        self.pb_delete_chirp = QPushButton("Delete Correction", maximumWidth=130)
        self.pb_delete_chirp.setToolTip(msg.ToolTips.t40)
        self.pb_apply_chirp = QPushButton("Apply", maximumWidth=130)
        self.pb_apply_chirp.setToolTip(msg.ToolTips.t24)
        self.w_chirp_layout = QGridLayout()
        self.w_chirp_layout.addWidget(self.pb_autocorr_chirp, 0, 0)
        self.w_chirp_layout.addWidget(self.le_threshold_chirp, 0, 1)
        self.w_chirp_layout.addWidget(self.cb_threshold_unit, 0, 2)
        self.w_chirp_layout.addWidget(self.pb_fromfile_chirp, 1, 0)
        self.w_chirp_layout.addWidget(self.cb_fromfile_ds_chirp, 1, 1)
        self.w_chirp_layout.addWidget(self.pb_manually_chirp, 2, 0)

        self.w_chirp_layout.addWidget(separator(), 3, 0, 1, 3)
        self.w_chirp_layout.addWidget(self.pb_fit_chirp, 4, 0)
        self.w_chirp_layout.addWidget(self.pb_del_chirp_fit, 4, 1)
        self.w_chirp_layout.addWidget(self.pb_show_chirp_correction, 5, 0, )
        self.w_chirp_layout.addWidget(self.pb_delete_chirp, 5, 1, )
        self.w_chirp_layout.addWidget(self.pb_apply_chirp, 5, 2, )
        self.w_chirp.setLayout(self.w_chirp_layout)

        self.w_background = QGroupBox('Background Correction')
        self.pb_corr_background = QPushButton("Subtract Area", maximumWidth=130)
        self.pb_corr_background.setToolTip(msg.ToolTips.t42)
        self.le_ymin_background = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50)
        self.le_ymin_background.setToolTip(msg.ToolTips.t43)
        self.le_ymax_background = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50)
        self.le_ymax_background.setToolTip(msg.ToolTips.t43)
        self.le_corr_max_background = QLineEdit(
            placeholderText=msg.Widgets.i07, maximumWidth=50)
        self.le_corr_max_background.setToolTip(msg.ToolTips.t44)
        self.cb_background_method = QComboBox(self)
        self.cb_background_method.setToolTip(msg.ToolTips.t45)
        self.cb_background_method.addItem('mean')
        self.cb_background_method.addItem('median')

        self.pb_fromfile_background = QPushButton(
            "Subtract from Project")
        self.pb_fromfile_background.setToolTip(msg.ToolTips.t46)
        self.cb_fromfile_background = QComboBox()
        self.cb_fromfile_background.addItems(('raw', 'ds 1', 'ds 2', 'ds 3'))
        self.cb_fromfile_background.setToolTip(msg.ToolTips.t55)
        self.pb_corr_from_solvent = QPushButton('Subtract Solvent')
        self.pb_corr_from_solvent.setToolTip(msg.ToolTips.t47)

        self.pb_delete_background = QPushButton("Delete Correction", maximumWidth=130)
        self.pb_delete_background.setToolTip(msg.ToolTips.t48)
        self.pb_apply_background = QPushButton(
            "Apply", maximumWidth=130, objectName="apply_background")
        self.pb_apply_background.setToolTip(msg.ToolTips.t24)
        self.w_background_layout = QGridLayout()
        self.w_background_layout.addWidget(self.pb_corr_background, 0, 0)
        self.w_background_layout.addWidget(self.le_ymin_background, 0, 1)
        self.w_background_layout.addWidget(self.le_ymax_background, 0, 2)
        self.w_background_layout.addWidget(QLabel("Subtract up to"), 1, 1)
        self.w_background_layout.addWidget(self.le_corr_max_background, 1, 2)
        self.w_background_layout.addWidget(QLabel("Method"), 2, 1)
        self.w_background_layout.addWidget(self.cb_background_method, 2, 2)
        self.w_background_layout.addWidget(self.pb_fromfile_background, 3, 0)
        self.w_background_layout.addWidget(self.cb_fromfile_background, 3, 1)
        self.w_background_layout.addWidget(self.pb_corr_from_solvent, 4, 0)
        self.w_background_layout.addWidget(separator(), 5, 0, 1, 3)
        self.w_background_layout.addWidget(self.pb_delete_background, 6, 1)
        self.w_background_layout.addWidget(self.pb_apply_background, 6, 2)
        self.w_background.setLayout(self.w_background_layout)

        self.w_t0 = QGroupBox('Time Zero Correction')

        self.le_t0 = QLineEdit(maximumWidth=50, placeholderText=msg.Widgets.i07)
        self.le_t0.setToolTip(msg.ToolTips.t51)
        self.pb_delete_t0 = QPushButton("Delete Correction", maximumWidth=130)
        self.pb_delete_t0.setToolTip(msg.ToolTips.t48)
        self.pb_apply_t0 = QPushButton("Apply", maximumWidth=80)
        self.pb_apply_t0.setToolTip(msg.ToolTips.t24)
        self.w_t0_layout = QGridLayout()

        self.w_t0_layout.addWidget(QLabel("set to", maximumWidth=50), 0, 0)
        self.w_t0_layout.addWidget(self.le_t0, 0, 1)
        self.w_t0_layout.addWidget(self.pb_delete_t0, 0, 2)
        self.w_t0_layout.addWidget(self.pb_apply_t0, 0, 3)
        self.w_t0_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.w_t0.setLayout(self.w_t0_layout)

        self.w_filter = QGroupBox('Averaging and Filtering')
        self.cb_filter = QComboBox()
        self.cb_filter.setToolTip(msg.ToolTips.t52)
        self.cb_filter.addItems(['Savitzky–Golay', 'Moving Median', 'Moving Average'])

        self.sb_filter_window = QSpinBox(self, minimum=2, maximum=30, value=3)
        self.sb_filter_window.setToolTip(msg.ToolTips.t53)
        self.sb_savgol_order = QSpinBox(self, minimum=1, maximum=5, value=2)
        self.sb_savgol_order.setToolTip(msg.ToolTips.t54)

        self.cb_filter_axis = QComboBox(self, maximumWidth=80)
        self.cb_filter_axis.addItems(['timepoints', 'wavelength'])
        self.cb_filter_axis.setToolTip(msg.ToolTips.t29)

        self.pb_show_filter = QPushButton("Show Filter")
        self.pb_show_filter.setToolTip(msg.ToolTips.t45)
        self.pb_delete_filter = QPushButton("Delete Filter")
        self.pb_delete_filter.setToolTip(msg.ToolTips.t49)
        self.pb_apply_savgol = QPushButton("Apply", objectName="apply_savgol", maximumWidth=80)
        self.pb_apply_savgol.setToolTip(msg.ToolTips.t24)
        self.label_window = QLabel('Window')
        self.label_order = QLabel('Order')
        self.label_size = QLabel('Size')

        self.w_filter_layout = QGridLayout()
        self.w_filter.setLayout(self.w_filter_layout)
        self.w_filter_layout.addWidget(self.cb_filter, 0, 0)
        self.w_filter_layout.addWidget(self.label_window, 0, 1)
        self.w_filter_layout.addWidget(self.label_size, 0, 1)
        self.label_size.setVisible(False)
        self.w_filter_layout.addWidget(self.sb_filter_window, 0, 2)
        self.w_filter_layout.addWidget(self.label_order, 1, 1)

        self.w_filter_layout.addWidget(self.sb_savgol_order, 1, 2)
        self.w_filter_layout.addWidget(QLabel("Axis"), 2, 1)
        self.w_filter_layout.addWidget(self.cb_filter_axis, 2, 2)
        self.w_filter_layout.addWidget(separator(), 3, 0, 1, 3)
        self.w_filter_layout.addWidget(self.pb_show_filter, 4, 0)
        self.w_filter_layout.addWidget(self.pb_delete_filter, 4, 1)
        self.w_filter_layout.addWidget(self.pb_apply_savgol, 4, 2)
        # self.w_filter_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_trimm, )
        self.layout.addWidget(self.w_resample, )
        self.layout.addWidget(self.w_chirp)
        self.layout.addWidget(self.w_background)

        self.layout.addWidget(self.w_t0)
        self.layout.addWidget(self.w_filter)
        self.layout.addStretch()

        self.cb_filter.currentIndexChanged.connect(self.change_filter_params)

    def change_filter_params(self):
        if self.sender().currentIndex() == 0:  # savgol
            self.label_size.setVisible(False)
            self.label_window.setVisible(True)
            self.label_order.setVisible(True)
            self.sb_savgol_order.setVisible(True)
        else:  # median or average
            self.label_window.setVisible(False)
            self.label_order.setVisible(False)
            self.sb_savgol_order.setVisible(False)
            self.label_size.setVisible(True)
