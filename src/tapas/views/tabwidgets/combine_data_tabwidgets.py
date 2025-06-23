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

from PyQt6.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QComboBox, QVBoxLayout,   QGridLayout,  QGroupBox,    QHBoxLayout
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
import logging
import logging.config
from ...configurations import messages as msg


class FileDropLineEdit(QLineEdit):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if not urls:
            return super().dropEvent(event)

        local_path = urls[0].toLocalFile()
        if local_path:
            self.setText(local_path)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)


class ImportProjectsWidget(QWidget):
    def __init__(self, ):
        super().__init__()

        self.initUI()

    def initUI(self):

        # create Widgets
        self.gb_import_projects = QGroupBox("Load Projects")
        self.le_path_p1 = FileDropLineEdit(self, placeholderText=msg.Widgets.i14,
                                           clearButtonEnabled=True, )
        self.le_path_p1.setToolTip(msg.ToolTips.t08)

        self.le_path_p1.setMaximumWidth(1000)
        self.pb_browse_p1 = QPushButton('Browse', objectName='pb_browse_p1')
        self.pb_browse_p1.setToolTip(msg.ToolTips.t09)
        self.cb_ds_p1 = QComboBox(self)
        self.cb_ds_p1.addItems(('raw', 'ds 1', 'ds 2', 'ds 3'))
        self.pb_load_p1 = QPushButton("Load", objectName='pb_load_p1')
        self.pb_load_p1.setToolTip(msg.ToolTips.t15)
        self.le_x_range_p1 = QLineEdit(text="None", readOnly=True)
        self.le_x_range_p1.setToolTip(msg.ToolTips.t56)
        self.le_x_range_p1.setMaximumWidth(100)
        self.le_y_range_p1 = QLineEdit(text="None", readOnly=True)
        self.le_y_range_p1.setToolTip(msg.ToolTips.t56)
        self.le_y_range_p1.setMaximumWidth(100)
        self.pb_clear_p1 = QPushButton("Clear", objectName='pb_clear_p1')
        self.pb_clear_p1.setToolTip(msg.ToolTips.t57)

        self.le_path_p2 = FileDropLineEdit(self, placeholderText=msg.Widgets.i14,
                                           clearButtonEnabled=True, )
        self.le_path_p2.setToolTip(msg.ToolTips.t08)
        self.setAcceptDrops(False)
        self.le_path_p2.setMaximumWidth(1000)
        self.pb_browse_p2 = QPushButton('Browse', objectName='pb_browse_p2')
        self.pb_browse_p2.setToolTip(msg.ToolTips.t09)
        self.cb_ds_p2 = QComboBox(self)
        self.cb_ds_p2.addItems(('raw', 'ds 1', 'ds 2', 'ds 3'))
        self.pb_load_p2 = QPushButton("Load", objectName='pb_load_p2')
        self.pb_load_p2.setToolTip(msg.ToolTips.t15)
        self.le_x_range_p2 = QLineEdit(text="None", readOnly=True)
        self.le_x_range_p2.setToolTip(msg.ToolTips.t56)
        self.le_x_range_p2.setMaximumWidth(100)
        self.le_y_range_p2 = QLineEdit(text="None", readOnly=True)
        self.le_y_range_p2.setToolTip(msg.ToolTips.t56)
        self.le_y_range_p2.setMaximumWidth(100)
        self.pb_clear_p2 = QPushButton("Clear", objectName='pb_clear_p2')
        self.pb_clear_p2.setToolTip(msg.ToolTips.t57)

        # add Widgets to Layout
        self.gb_import_projects_layout = QGridLayout(self)
        self.gb_import_projects.setLayout(self.gb_import_projects_layout)
        self.gb_import_projects_layout.addWidget(QLabel("use"), 0, 4, )
        self.gb_import_projects_layout.addWidget(QLabel("x-range"), 0, 5, )
        self.gb_import_projects_layout.addWidget(QLabel("y-range"), 0, 6, )
        self.gb_import_projects_layout.addWidget(QLabel("Project 1"), 1, 0, )
        self.gb_import_projects_layout.addWidget(self.le_path_p1, 1, 1, )
        self.gb_import_projects_layout.addWidget(self.pb_browse_p1, 1, 2)
        self.gb_import_projects_layout.addWidget(self.pb_load_p1, 1, 3)
        self.gb_import_projects_layout.addWidget(self.cb_ds_p1, 1, 4)
        self.gb_import_projects_layout.addWidget(self.le_x_range_p1, 1, 5)
        self.gb_import_projects_layout.addWidget(self.le_y_range_p1, 1, 6)
        self.gb_import_projects_layout.addWidget(self.pb_clear_p1, 1, 7)
        self.gb_import_projects_layout.addWidget(QLabel("Project 2"), 2, 0, )
        self.gb_import_projects_layout.addWidget(self.le_path_p2, 2, 1, )
        self.gb_import_projects_layout.addWidget(self.pb_browse_p2, 2, 2)
        self.gb_import_projects_layout.addWidget(self.pb_load_p2, 2, 3)
        self.gb_import_projects_layout.addWidget(self.cb_ds_p2, 2, 4)
        self.gb_import_projects_layout.addWidget(self.le_x_range_p2, 2, 5)
        self.gb_import_projects_layout.addWidget(self.le_y_range_p2, 2, 6)
        self.gb_import_projects_layout.addWidget(self.pb_clear_p2, 2, 7)

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.gb_import_projects)


class CombineProjectsWidget(QWidget):
    def __init__(self, ):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.gb_combine_projects = QGroupBox("Combine Projects")
        self.cb_overlap_use = QComboBox()
        self.cb_overlap_use.addItems(('P1', 'P2', 'merge'))
        self.cb_overlap_use.setToolTip(msg.ToolTips.t58)
        self.cb_interpolation_use = QComboBox()
        self.cb_interpolation_use.addItems(
            ('nearest', 'linear', 'cubic', 'bicubic'))
        self.cb_interpolation_use.setToolTip(msg.ToolTips.t59)
        self.cb_extrapolation_use = QComboBox()
        self.cb_extrapolation_use.addItems(
            ('zero', 'linear',))
        self.cb_extrapolation_use.setToolTip(msg.ToolTips.t60)
        self.pb_preview_combine_projects = QPushButton('Preview')
        self.pb_preview_combine_projects.setToolTip(msg.ToolTips.t61)
        self.pb_apply_combine_projects = QPushButton(
            'Apply', objectName='pb_apply_combine_projects')
        self.pb_apply_combine_projects.setToolTip(msg.ToolTips.t62)

        self.gb_combine_projects_layout = QGridLayout(self)
        self.gb_combine_projects.setLayout(self.gb_combine_projects_layout)
        self.gb_combine_projects_layout.addWidget(
            QLabel('Overlapping Range use'), 0, 0)
        self.gb_combine_projects_layout.addWidget(self.cb_overlap_use, 0, 1)
        self.gb_combine_projects_layout.addWidget(
            QLabel('Interpolation Method'), 1, 0)
        self.gb_combine_projects_layout.addWidget(
            self.cb_interpolation_use, 1, 1)
        self.gb_combine_projects_layout.addWidget(
            QLabel('Extrapolation Method'), 2, 0)
        self.gb_combine_projects_layout.addWidget(
            self.cb_extrapolation_use, 2, 1)
        self.gb_combine_projects_layout.addWidget(
            self.pb_preview_combine_projects, 3, 0, )
        self.gb_combine_projects_layout.addWidget(
            self.pb_apply_combine_projects, 3, 1, )

        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.addWidget(self.gb_combine_projects, 0, 0)


class PreviewContainerWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.p1_window = QLabel(msg.Widgets.i15)
        self.p2_window = QLabel(msg.Widgets.i15)
        self.merge_window = QLabel(msg.Widgets.i16)

        self.group_box = QGroupBox('Data Preview')
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.group_box)
        self.setLayout(self.layout)

        self.inner_layout = QGridLayout(self)
        self.inner_layout.addWidget(
            self.p1_window, 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        self.inner_layout.addWidget(
            self.p2_window, 0, 1, alignment=Qt.AlignmentFlag.AlignHCenter)
        self.inner_layout.addWidget(
            self.merge_window, 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        self.group_box.setLayout(self.inner_layout)
