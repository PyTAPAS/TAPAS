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

from PyQt6.QtWidgets import QWidget, QLabel, QLineEdit, QVBoxLayout, QSpinBox, QGridLayout, QGroupBox, QRadioButton
import logging
import logging.config
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

        self.w_components_params = QGroupBox("Principal Components")
        self.sb_components = QSpinBox(
            minimum=1, maximum=20, value=1, minimumWidth=60, maximumWidth=60)
        self.w_components_params.setToolTip(msg.ToolTips.t101)

        self.w_components_params_layout = QGridLayout()
        self.w_components_params_layout.addWidget(
            QLabel("Number of Principal Components:"), 0, 0, )
        self.w_components_params_layout.addWidget(self.sb_components, 0, 1, )
        self.w_components_params.setLayout(self.w_components_params_layout)

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
        self.layout.addWidget(self.w_ds, )
        self.layout.addWidget(self.w_components_params, )
        self.layout.addWidget(self.w_view_manipulations, )

        self.layout.addStretch()


class CanvasWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.gb_canvas = QGroupBox('SVD Analysis')
        self.w_canvas = QLabel(msg.Widgets.i13)
        self.view_layout = QVBoxLayout(self)
        self.view_layout.addWidget(self.w_canvas)
        self.gb_canvas.setLayout(self.view_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.gb_canvas)
