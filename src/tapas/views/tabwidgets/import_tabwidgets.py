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

from PyQt6.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QVBoxLayout, QSpinBox, QGridLayout, QGroupBox,  QTextEdit
from PyQt6.QtCore import Qt
import logging
import logging.config
from ...configurations import messages as msg


class ImportWidget(QWidget):

    def __init__(self, label="Data", placeholder="Enter Data Path...",
                 time_unit=False, energy_unit=False, delA_unit=False, matrix_orientation=False):
        super().__init__()
        self.placeholder = placeholder
        self.label = label
        self.time_unit = time_unit
        self.energy_unit = energy_unit
        self.delA_unit = delA_unit
        self.matrix_orientation = matrix_orientation

        self.initUI()

    def initUI(self):
        # create Widgets
        self.label = QLabel(text=self.label)
        self.label.setMinimumWidth(100)
        self.le_path = QLineEdit(self, placeholderText=self.placeholder, clearButtonEnabled=True)
        self.le_path.setToolTip(msg.ToolTips.t08)
        self.le_path.setMaximumWidth(1000)
        self.setAcceptDrops(True)

        self.pb_browse = QPushButton('Browse')
        self.pb_browse.setToolTip(msg.ToolTips.t09)

        self.w_delimiter = QWidget()
        self.cb_delimiter = QComboBox(self)
        self.cb_delimiter.addItems((',', ';', 'tab'))
        self.w_delimiter.setToolTip(msg.ToolTips.t10)
        self.w_delimiter_layout = QVBoxLayout()
        self.w_delimiter.setLayout(self.w_delimiter_layout)
        self.w_delimiter_layout.addWidget(QLabel('Delimiter', self))
        self.w_delimiter_layout.addWidget(self.cb_delimiter)

        self.w_header = QWidget()
        self.w_header.setToolTip(msg.ToolTips.t11)
        self.sb_header = QSpinBox(self, minimum=0, maximum=100, value=2)
        self.w_header_layout = QVBoxLayout()
        self.w_header.setLayout(self.w_header_layout)
        self.w_header_layout.addWidget(QLabel('Ignore Header', self))
        self.w_header_layout.addWidget(self.sb_header)

        self.pb_load = QPushButton("Load Data")
        self.pb_load.setToolTip(msg.ToolTips.t15)
        self.pb_clear = QPushButton("Clear")
        self.pb_clear.setToolTip(msg.ToolTips.t16)

        # add Widgets to Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.label, 0, 0, )
        self.layout.addWidget(self.le_path, 0, 1, )
        self.layout.addWidget(self.pb_browse, 0, 2)
        self.layout.addWidget(self.w_delimiter, 0, 3)
        self.layout.addWidget(self.w_header, 0, 4)
        self.layout.addWidget(self.pb_load, 0, 9)
        self.layout.addWidget(self.pb_clear, 0, 10)

        self.layout.setColumnStretch(0, 0)
        self.layout.setColumnStretch(1, 2)
        self.layout.setColumnStretch(2, 0)
        self.layout.setColumnStretch(3, 0)
        self.layout.setColumnStretch(4, 0)
        self.layout.setColumnStretch(6, 0)
        self.layout.setColumnStretch(7, 0)
        self.layout.setColumnStretch(8, 0)

        if self.time_unit:
            self.w_time_unit = QWidget()
            self.w_time_unit.setToolTip(msg.ToolTips.t12)
            self.cb_time_unit = QComboBox(self)
            self.cb_time_unit.addItems(('ps', 'ns', 'us', 'ms', 's'))
            self.w_time_unit_layout = QVBoxLayout()
            self.w_time_unit.setLayout(self.w_time_unit_layout)
            self.w_time_unit_layout.addWidget(QLabel('Time Unit', self))
            self.w_time_unit_layout.addWidget(self.cb_time_unit)
            self.layout.addWidget(self.w_time_unit, 0, 5)

        if self.energy_unit:
            self.w_energy_unit = QWidget()
            self.w_energy_unit.setToolTip(msg.ToolTips.t13)
            self.cb_energy_unit = QComboBox(self)
            self.cb_energy_unit.addItems((['nm', 'm']))
            self.w_energy_unit_layout = QVBoxLayout()
            self.w_energy_unit.setLayout(self.w_energy_unit_layout)
            self.w_energy_unit_layout.addWidget(QLabel('Energy Unit', self))
            self.w_energy_unit_layout.addWidget(self.cb_energy_unit)
            self.layout.addWidget(self.w_energy_unit, 0, 6)

        if self.delA_unit:
            self.w_delA_unit = QWidget()
            self.w_delA_unit.setToolTip(msg.ToolTips.t14)
            self.cb_delA_unit = QComboBox(self)
            self.cb_delA_unit.addItems((['OD', 'mOD']))
            self.w_delA_unit_layout = QVBoxLayout()
            self.w_delA_unit.setLayout(self.w_delA_unit_layout)
            self.w_delA_unit_layout.addWidget(QLabel('ΔA Unit', self))
            self.w_delA_unit_layout.addWidget(self.cb_delA_unit)
            self.layout.addWidget(self.w_delA_unit, 0, 7)
        
        if self.matrix_orientation:
            self.w_matrix_orientation = QWidget()
            self.w_matrix_orientation.setToolTip(msg.ToolTips.t126)
            self.cb_matrix_orientation = QComboBox(self)
            self.cb_matrix_orientation.addItems((['λ in column', 'λ in row']))
            self.w_matrix_orientation_layout = QVBoxLayout()
            self.w_matrix_orientation.setLayout(self.w_matrix_orientation_layout)
            self.w_matrix_orientation_layout.addWidget(QLabel('Data Orientation', self))
            self.w_matrix_orientation_layout.addWidget(self.cb_matrix_orientation)
            self.layout.addWidget(self.w_matrix_orientation, 0, 8)

            

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        # saves list of paths in files list, converts list to easy to read line edit string
        files = [u.toLocalFile() for u in event.mimeData().urls()]

        self.le_path.setText(' , '.join(files))


class MetadataWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.gb_metadata = QGroupBox('Metadata')
        # self.gb_metadata.setFixedHeight(300)
        self.gb_metadata_layout = QGridLayout()
        self.gb_metadata.setLayout(self.gb_metadata_layout)
        self.le_exp_name = QLineEdit()
        self.le_exp_name.setMaximumWidth(200)
        self.le_sample_name = QLineEdit()
        self.le_sample_name.setMaximumWidth(200)
        self.le_exc_wavelen = QLineEdit()
        self.le_exc_wavelen.setMaximumWidth(200)
        self.le_exc_int = QLineEdit()
        self.le_exc_int.setMaximumWidth(200)
        self.le_solovent = QLineEdit()
        self.le_solovent.setMaximumWidth(200)
        self.te_notes = QTextEdit()
        self.te_notes.setMaximumWidth(200)
        # self.te_notes.setMaximumHeight(75)

        self.gb_metadata_layout.addWidget(QLabel("Experiment Name"), 0, 0)
        self.gb_metadata_layout.addWidget(self.le_exp_name, 0, 1)
        self.gb_metadata_layout.addWidget(QLabel("Sample Name"), 1, 0)
        self.gb_metadata_layout.addWidget(self.le_sample_name, 1, 1)
        self.gb_metadata_layout.addWidget(
            QLabel("Excitation Wavelength"), 2, 0)
        self.gb_metadata_layout.addWidget(self.le_exc_wavelen, 2, 1)
        self.gb_metadata_layout.addWidget(QLabel("Excitation Intensity"), 3, 0)
        self.gb_metadata_layout.addWidget(self.le_exc_int, 3, 1)
        self.gb_metadata_layout.addWidget(QLabel("Solvent"), 4, 0)
        self.gb_metadata_layout.addWidget(self.le_solovent, 4, 1)
        self.gb_metadata_layout.addWidget(QLabel("Notes"), 5, 0)
        self.gb_metadata_layout.addWidget(self.te_notes, 5, 1)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.gb_metadata)
        self.setLayout(self.layout)


class PreviewContainerWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.abs_window = QLabel(msg.Widgets.i04)
        self.em_window = QLabel(msg.Widgets.i05)
        self.ta_window = QLabel(msg.Widgets.i06)
        self.ta_window = QLabel(msg.Widgets.i06)
        self.solvent_window = QLabel(msg.Widgets.i06)

        self.group_box = QGroupBox('Data Preview')

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.group_box)
        self.setLayout(self.layout)

        self.inner_layout = QGridLayout()
        self.inner_layout.addWidget(
            self.ta_window, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        self.inner_layout.addWidget(
            self.solvent_window, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        self.inner_layout.addWidget(
            self.abs_window, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        self.inner_layout.addWidget(
            self.em_window, 0, 3, alignment=Qt.AlignmentFlag.AlignCenter)
        self.group_box.setLayout(self.inner_layout)
