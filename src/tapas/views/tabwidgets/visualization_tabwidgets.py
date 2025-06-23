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

from PyQt6.QtWidgets import QButtonGroup,  QMenu, QFormLayout, QFontComboBox, QWidget, QLabel, QLineEdit, QPushButton, QFrame, QComboBox, QVBoxLayout,   QSpinBox, QGridLayout,  QGroupBox,  QRadioButton, QCompleter,  QCheckBox
from PyQt6.QtCore import Qt,  QRegularExpression
from PyQt6.QtGui import QIntValidator, QRegularExpressionValidator, QFontDatabase, QFont, QStandardItemModel, QStandardItem
import matplotlib.font_manager as fm
import logging
import logging.config
from ...configurations import messages as msg


class Widgets(QWidget):
    @staticmethod
    def show_info_widget():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t63)
        layout = QGridLayout(widget)
        check_show_info = QCheckBox(
            "Display Metadata", objectName='check_show_info')
        layout.addWidget(check_show_info, 0, 0)
        widget.setLayout(layout)
        widget.check_show_info = check_show_info
        return widget

    @staticmethod
    def separator():
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        return separator

    @staticmethod
    def hide_area():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t64)
        widget.le_xmin_hide = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50, objectName='le_xmin_hide')
        widget.le_xmax_hide = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50, objectName='le_xmax_hide')
        widget.le_xmin_hide2 = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50, objectName='le_xmin_hide2')
        widget.le_xmax_hide2 = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50, objectName='le_xmax_hide2')
        layout = QGridLayout()
        layout.addWidget(QLabel("Hide Area"), 0, 0)
        layout.addWidget(widget.le_xmin_hide, 0, 1)
        layout.addWidget(widget.le_xmax_hide, 0, 2)
        layout.addWidget(widget.le_xmin_hide2, 1, 1)
        layout.addWidget(widget.le_xmax_hide2, 1, 2)
        widget.setLayout(layout)

        return widget

    @staticmethod
    def trimm_widget(exclude=None, surf_plot=False):

        def change_linlog_visibility(sender):
            if sender.objectName() == 'cb_zscale':
                if sender.currentText() == 'linlog':
                    widget.le_linlog_z.setVisible(True)
                else:
                    widget.le_linlog_z.setVisible(False)
            elif sender.objectName() == 'cb_yscale':
                if sender.currentText() == 'linlog':
                    widget.le_linlog_y.setVisible(True)
                else:
                    widget.le_linlog_y.setVisible(False)

        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t102)
        widget.le_xmin = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50, objectName='le_xmin')
        widget.le_xmax = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50, objectName='le_xmax')
        widget.le_ymin = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50, objectName='le_ymin')
        widget.le_ymax = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50, objectName='le_ymax')
        widget.le_zmin = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50, objectName='le_zmin')
        widget.le_zmax = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50, objectName='le_zmax')

        widget.cb_yscale = QComboBox(objectName='cb_yscale')
        widget.cb_yscale.setToolTip(msg.ToolTips.t65)
        widget.cb_yscale.addItems(['lin', 'log', 'linlog'])
        widget.cb_zscale = QComboBox(objectName='cb_zscale')
        widget.cb_zscale.addItems(['lin', 'log', 'linlog'])
        widget.cb_zscale.setToolTip(msg.ToolTips.t65)
        widget.le_linlog_y = QLineEdit(
            placeholderText=msg.Widgets.i12, maximumWidth=50, objectName='le_linlog_y')
        widget.le_linlog_y.setToolTip(msg.ToolTips.t22)
        widget.le_linlog_z = QLineEdit(
            placeholderText=msg.Widgets.i12, maximumWidth=50, objectName='le_linlog_z')
        widget.le_linlog_z.setToolTip(msg.ToolTips.t22)
        widget.le_linlog_y.setVisible(False)
        widget.le_linlog_z.setVisible(False)
        layout = QGridLayout()

        if surf_plot:
            widget.sb_zcenter = QSpinBox(
                minimum=-100, maximum=100, value=0,  minimumWidth=60, maximumWidth=60, objectName='sb_zcenter')
            widget.sb_zcenter.setToolTip(msg.ToolTips.t66)
            layout.addWidget(QLabel("Wavelength"), 0, 0, 1, 2)
            layout.addWidget(widget.le_xmin, 0, 2)
            layout.addWidget(widget.le_xmax, 0, 3)
            layout.addWidget(QLabel("Delay"), 1, 0, 1, 2)
            layout.addWidget(widget.le_ymin, 1, 2)
            layout.addWidget(widget.le_ymax, 1, 3)
            layout.addWidget(QLabel("ΔA"), 2, 0)
            layout.addWidget(widget.sb_zcenter, 2, 1)
            layout.addWidget(widget.le_zmin, 2, 2)
            layout.addWidget(widget.le_zmax, 2, 3)
            widget.setLayout(layout)
            return widget

        if not exclude == 'Wavelength':
            layout.addWidget(QLabel("Wavelength"), 0, 0, )
            layout.addWidget(widget.le_xmin, 0, 1)
            layout.addWidget(widget.le_xmax, 0, 2)
        if not exclude == 'Delay':
            layout.addWidget(QLabel("Delay"), 1, 0, )
            layout.addWidget(widget.le_ymin, 1, 1)
            layout.addWidget(widget.le_ymax, 1, 2)
            layout.addWidget(widget.cb_yscale, 1, 3)
            layout.addWidget(widget.le_linlog_y, 1, 4)
        if not exclude == 'del A':
            widget.z_label = QLabel("ΔA")
            layout.addWidget(widget.z_label, 2, 0, )
            layout.addWidget(widget.le_zmin, 2, 1)
            layout.addWidget(widget.le_zmax, 2, 2)
            layout.addWidget(widget.cb_zscale, 2, 3)
            layout.addWidget(widget.le_linlog_z, 2, 4)
        widget.setLayout(layout)

        widget.cb_zscale.currentIndexChanged.connect(
            lambda index: change_linlog_visibility(widget.cb_zscale))
        widget.cb_yscale.currentIndexChanged.connect(
            lambda index: change_linlog_visibility(widget.cb_yscale))

        return widget

    @staticmethod
    def show_pump_widget():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t67)
        widget.check_show_pump = QCheckBox(
            "Display Pump", objectName='check_show_pump')
        layout = QGridLayout()
        layout.addWidget(widget.check_show_pump, 0, 0)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def show_2nd_ax_widget():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t68)
        widget.check_show_second_ax = QCheckBox(
            "Display 2nd Axis", objectName='check_show_second_ax')
        layout = QGridLayout()
        layout.addWidget(widget.check_show_second_ax, 0, 0)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def show_experimental():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t69)
        widget.check_show_data = QCheckBox(
            "Show Experimental Data", objectName='check_show_data')
        layout = QGridLayout()
        layout.addWidget(widget.check_show_data, 0, 0)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def show_residuals():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t70)
        widget.check_show_residuals = QCheckBox(
            "Show Residuals", objectName='check_show_residuals')
        layout = QGridLayout()
        layout.addWidget(widget.check_show_residuals, 0, 0)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def show_legend():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t71)
        widget.check_show_legend = QCheckBox(
            "Show Legend", objectName='check_show_legend')
        widget.cb_legend_loc = QComboBox(objectName='cb_legend_loc')
        widget.cb_legend_loc.addItems(
            ['outside', 'best', 'upper left', 'upper right', 'lower left', 'lower right'])
        layout = QGridLayout()
        layout.addWidget(widget.check_show_legend, 0, 0)
        layout.addWidget(widget.cb_legend_loc, 0, 1)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def select_data_color_widget(data):
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t72)
        widget.cb_select_cmap = QComboBox(objectName='cb_select_cmap')
        widget.cb_select_cmap.setToolTip(msg.ToolTips.t73)
        widget.cb_select_cmap.addItems(
            ['-', 'nipy_spectral', 'cool', 'plasma', 'rainbow', 'gist_rainbow', 'viridis', 'gray'])
        widget.le_custom_colors = QLineEdit(
            placeholderText=msg.Widgets.i19, objectName='le_custom_colors')
        widget.le_custom_colors.setToolTip(msg.ToolTips.t74)
        layout = QGridLayout()
        if data == 'Delay':
            widget.le_delay_list = QLineEdit(
                placeholderText=msg.Widgets.i17, objectName='le_delay_list')
            widget.le_delay_list.setToolTip(msg.ToolTips.t75)
            layout.addWidget(QLabel("Delay Times"), 0, 0, )
            layout.addWidget(widget.le_delay_list, 0, 1, 1, 2)
        elif data == 'Wavelength':
            widget.le_wavelength_list = QLineEdit(
                placeholderText=msg.Widgets.i18, objectName='le_wavelength_list')
            widget.le_wavelength_list.setToolTip(msg.ToolTips.t76)
            layout.addWidget(QLabel("Wavelength"), 0, 0, )
            layout.addWidget(widget.le_wavelength_list, 0, 1, 1, 2)

        layout.addWidget(QLabel("Colors"), 1, 0,)
        layout.addWidget(widget.cb_select_cmap, 1, 1)
        layout.addWidget(widget.le_custom_colors, 1, 2)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def normalize_widget(abs_values=True):
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t77)
        widget.check_normalize = QCheckBox("Normalize")
        widget.le_norm_min = QLineEdit(
            placeholderText=msg.Widgets.i10, maximumWidth=50, objectName='le_norm_min')
        widget.le_norm_min.setToolTip(msg.ToolTips.t78)
        widget.le_norm_max = QLineEdit(
            placeholderText=msg.Widgets.i11, maximumWidth=50, objectName='le_norm_max')
        widget.le_norm_max.setToolTip(msg.ToolTips.t78)

        layout = QGridLayout()
        layout.addWidget(widget.check_normalize, 0, 0)
        layout.addWidget(widget.le_norm_min, 0, 1)
        layout.addWidget(widget.le_norm_max, 0, 2)

        if abs_values:
            widget.check_abs_value = QCheckBox(
                "Absolute Values", objectName='check_abs_value')
            widget.check_abs_value.setToolTip(msg.ToolTips.t79)
            layout.addWidget(widget.check_abs_value, 1, 0)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def select_fit(label, exclusive_check=False):
        def on_checkbox_clicked(current, check_list):
            """
            Custom slot to implement exclusive behavior.
            If the current checkbox is being checked, uncheck all others.
            If it's already checked and clicked (i.e. toggled off), leave all unchecked.
            """
            if current.isChecked():
                # Uncheck all other checkboxes in the list.
                for cb in check_list:
                    if cb is not current:
                        cb.setChecked(False)
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t80)

        # Create 9 checkboxes with object names 'fit1' to 'fit9', all initially hidden
        widget.check_fits = [
            QCheckBox(objectName=f'fit{i}', visible=False) for i in range(1, 10)]

        # Set up a grid layout and add a label spanning three columns
        layout = QGridLayout()
        layout.addWidget(QLabel(label), 0, 0, 1, 3)

        if exclusive_check:
            # Instead of using autoExclusive with QButtonGroup, we set autoExclusive to False
            # and connect a custom slot to handle exclusivity manually.
            for checkbox in widget.check_fits:
                checkbox.setAutoExclusive(False)
                # Use a lambda to capture the current checkbox.
                checkbox.clicked.connect(
                    lambda checked, cb=checkbox: on_checkbox_clicked(cb, widget.check_fits))

        # Add the checkboxes to the layout in a 3-column grid
        for idx, checkbox in enumerate(widget.check_fits):
            row, col = (idx // 3) + 1, idx % 3
            layout.addWidget(checkbox, row, col)

        widget.setLayout(layout)
        return widget

    @staticmethod
    def view_manipulation2D(ss=True):
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t81)
        widget.le_linlog = QLineEdit(
            placeholderText=msg.Widgets.i07, maximumWidth=50, objectName='le_linlog')
        widget.le_linlog.setToolTip(msg.ToolTips.t22)
        widget.sb_ss_ratio = QSpinBox(
            minimum=1, maximum=100, value=2, minimumWidth=60, maximumWidth=60, objectName='sb_ss_ratio')
        widget.sb_ss_ratio.setEnabled(False)
        widget.sb_lin_ratio = QSpinBox(
            minimum=1, maximum=100, value=2, minimumWidth=60, maximumWidth=60, objectName='sb_lin_ratio')
        widget.sb_log_ratio = QSpinBox(
            minimum=1, maximum=100, value=4, minimumWidth=60, maximumWidth=60, objectName='sb_log_ratio')
        layout = QGridLayout()

        if ss:
            layout.addWidget(
                QLabel("Lin/Log Transition"), 0, 0, 1, 2,)
            layout.addWidget(widget.le_linlog, 0, 2, 1, 2)
            layout.addWidget(
                QLabel("Ratio (Lin|Log|ss)", ), 1, 0)
            layout.addWidget(widget.sb_ss_ratio, 1, 3)
        else:
            layout.addWidget(
                QLabel("Lin/Log Transition"), 0, 0, )
            layout.addWidget(widget.le_linlog, 0, 1)
            layout.addWidget(
                QLabel("Ratio (Lin|Log)", ), 1, 0)
        layout.addWidget(widget.sb_lin_ratio, 1, 1)
        layout.addWidget(widget.sb_log_ratio, 1, 2)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def colormap2D():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t82)
        widget.cb_select_cmap = QComboBox(objectName='cb_select_cmap')
        widget.cb_select_cmap.addItems(
            ['-', 'default_cmap', 'nipy_spectral', 'seismic', 'jet', 'gist_ncar', 'gnuplot2', 'viridis', 'gray'])
        widget.check_cmap = QCheckBox("Colormap", objectName='check_cmap')
        widget.cb_pos_cmap = QComboBox(objectName='cb_pos_cmap')
        widget.cb_pos_cmap.addItems(['right', 'bottom'])
        layout = QGridLayout()
        layout.addWidget(widget.check_cmap, 0, 0)
        layout.addWidget(widget.cb_select_cmap, 0, 1)
        layout.addWidget(widget.cb_pos_cmap, 0, 2)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def show_delA_2D():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t83)
        widget.check_show_delA_cuts = QCheckBox(
            "Display ΔA Cuts", objectName='check_show_delA_cuts')
        layout = QGridLayout()
        layout.addWidget(widget.check_show_delA_cuts, 0, 0)
        widget.setLayout(layout)
        return widget

    @staticmethod
    def show_kin_2d():
        widget = QWidget()
        widget.setToolTip(msg.ToolTips.t83)
        widget.check_show_kin_cuts = QCheckBox(
            "Display Kinetic Cuts", objectName='check_show_kin_cuts')
        layout = QGridLayout()
        layout.addWidget(widget.check_show_kin_cuts, 0, 0)
        widget.setLayout(layout)
        return widget


class RadioMenu(QRadioButton):
    ''' custom RTadioButton with dropdown menu, eg for global fit selection + plot type
    the radiobutton.toggled is emitted every time an item from the menu is selected (even wihtin the same rb)
    BUG: can lead to multiple execution due to stacked signal connections when RB clicked, but no menu selection

    '''

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._menu = None
        self._menu_selected = None

    def setMenu(self, menu: QMenu):
        self._menu = menu

    def menuSelected(self):
        return self._menu_selected

    def mousePressEvent(self, event):
        # Instead of immediately updating the state,
        # if we have a menu, we use popup() (non-blocking) so the UI keeps animating.
        if self._menu:
            pos = self.mapToGlobal(self.rect().bottomLeft())
            # Show the menu non-blocking so the event loop keeps running.
            self._menu.popup(pos)
            # Connect the triggered signal so that when the user selects an item,
            # we update the radio button’s state.
            self._menu.triggered.connect(self.handleMenuTriggered)
        else:
            super().mouseReleaseEvent(event)

    def handleMenuTriggered(self, action):
        self._menu_selected = action.objectName()
        if self.isChecked():
            # Save the current autoExclusive state.
            auto_exclusive = self.autoExclusive()
            # Temporarily disable auto-exclusive to allow unchecking.
            self.setAutoExclusive(False)
            self.setChecked(False)
            self.setChecked(True)
            # Restore the auto-exclusive setting.
            self.setAutoExclusive(auto_exclusive)
        # Only update the state when an action is selected.
        else:
            self.setChecked(True)

        # Disconnect the signal to avoid duplicate connections
        self._menu.triggered.disconnect(self.handleMenuTriggered)


class SelectPlotWidget(QWidget):
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

        self.gb_plots = QGroupBox("Plot")
        self.rb_2d_plot = QRadioButton("2D Plot", self, objectName='2D_plot',)
        self.rb_2d_plot.setToolTip(msg.ToolTips.t84)
        self.rb_2d_plot.setChecked(True)
        self.rb_delA_plot = QRadioButton(
            "ΔA Plot", self, objectName='delA_plot')
        self.rb_delA_plot.setToolTip(msg.ToolTips.t85)
        self.rb_kin_trace = QRadioButton(
            "Kin Trace", self, objectName='kin_trace')
        self.rb_kin_trace.setToolTip(msg.ToolTips.t86)
        self.rb_local_fit = RadioMenu("Local Fit", self,)
        self.rb_local_fit.setToolTip(msg.ToolTips.t87)
        self.rb_local_fit.setObjectName('local_fit')

        MENU_ACTIONS = {
            'kin_trace': 'Kin Trace',
            'emcee': 'Posterior Dist',
        }

        menu = QMenu()
        for obj_name, text in MENU_ACTIONS.items():
            action = menu.addAction(text)
            action.setObjectName(obj_name)

        self.rb_local_fit.setMenu(menu)

        self.rb_global_fit = RadioMenu("Global Fit", self,)
        self.rb_global_fit.setToolTip(msg.ToolTips.t88)
        self.rb_global_fit.setObjectName('global_fit')

        MENU_ACTIONS = {
            'plot_2d': '2D Plot',
            'eas_das': 'EAS/DAS/SAS',
            'conc': 'Concentration',
            'dela_plot': 'ΔA Plot',
            'kinetic_trace': 'Kinetic Trace',
            'emcee': 'Posterior Dist'}

        menu = QMenu()
        for obj_name, text in MENU_ACTIONS.items():
            action = menu.addAction(text)
            action.setObjectName(obj_name)

        self.rb_global_fit.setMenu(menu)

        buttonGroup = QButtonGroup()
        buttonGroup.addButton(self.rb_2d_plot)
        buttonGroup.addButton(self.rb_delA_plot)
        buttonGroup.addButton(self.rb_kin_trace)
        buttonGroup.addButton(self.rb_local_fit)
        buttonGroup.addButton(self.rb_global_fit)

        self.w_view_layout = QGridLayout()
        # self.w_view_layout.addWidget(QLabel("View"))
        self.w_view_layout.addWidget(self.rb_2d_plot, 0, 0)
        self.w_view_layout.addWidget(self.rb_delA_plot, 1, 0)
        self.w_view_layout.addWidget(self.rb_kin_trace, 2, 0)
        self.w_view_layout.addWidget(self.rb_local_fit, 0, 1)
        self.w_view_layout.addWidget(self.rb_global_fit, 1, 1)

        self.gb_plots.setLayout(self.w_view_layout)

        self.w_style = QGroupBox("Figure Style")
        self.w_style.setToolTip(msg.ToolTips.t89)
        self.le_style = QLabel("Default", )
        self.pb_add_rcParam = QPushButton("Add rcParam")
        self.pb_add_rcParam.setToolTip(msg.ToolTips.t90)
        self.le_fig_size_w = QLineEdit(objectName='fig_size_w')
        self.le_fig_size_w.setToolTip(msg.ToolTips.t91)
        self.le_fig_size_h = QLineEdit(objectName='fig_size_h')
        self.le_fig_size_h.setToolTip(msg.ToolTips.t92)
        self.check_display_size = QCheckBox('Show Real Size?')
        self.check_display_size.setToolTip(msg.ToolTips.t93)
        self.l_DPI = QLabel("Figure DPI:")
        self.le_fig_dpi = QLineEdit(objectName='fig_dpi')
        self.le_fig_dpi.setToolTip(msg.ToolTips.t94)
        int_validator = QIntValidator()
        int_validator.setRange(0, 2000)
        self.le_fig_dpi.setValidator(int_validator)
        self.l_DPI.setVisible(True)
        self.le_fig_dpi.setVisible(True)

        self.check_svg_text2path = QCheckBox('Render Text as Path?')
        self.check_svg_text2path.setToolTip(msg.ToolTips.t06)

        self.check_svg_text2path.setVisible(False)
        
        def restrict_qfontcombobox_to_matplotlib(font_cb: QFontComboBox) -> None:
            """
            Replace `font_cb`'s model so it lists only the font families that
            Matplotlib can actually render.  PyQt-6 safe (static QFontDatabase API).
            """
            # 1) Families Matplotlib knows (case-folded for robust matching)
            mpl_families = {fe.name.lower() for fe in fm.fontManager.ttflist}
         
            # 2) Intersect with what Qt sees
            allowed = [fam for fam in QFontDatabase.families()
                       if fam.lower() in mpl_families]
         
            # 3) Build a new model
            model = QStandardItemModel()
            for fam in sorted(allowed, key=str.casefold):
                item = QStandardItem(fam)                    # visible text
                item.setData(QFont(fam), Qt.ItemDataRole.FontRole)  # preview glyphs
                model.appendRow(item)
         
            font_cb.setModel(model)
            font_cb.setCurrentIndex(-1)          # no pre-selection; keeps completer ok
        
        
        self.font_cb = QFontComboBox(objectName='fig_font')
        restrict_qfontcombobox_to_matplotlib(self.font_cb) 
        self.cb_font_style = QComboBox(objectName='fig_font_style')
        self.font_cb.setToolTip(msg.ToolTips.t99)

        def populate_styles(qfont):
            self.cb_font_style.clear()
            fam = qfont.family()
            for style in QFontDatabase.styles(fam):
                self.cb_font_style.addItem(style)

        self.font_cb.currentFontChanged.connect(populate_styles)
        populate_styles(self.font_cb.currentFont())

        self.sb_label_size = QSpinBox(
            minimum=1, maximum=40, suffix=' pt', objectName='label_size')
        self.sb_label_size.setToolTip(msg.ToolTips.t96)
        self.sb_tick_size = QSpinBox(
            minimum=1, maximum=40, suffix=' pt', objectName='tick_size')
        self.sb_tick_size.setToolTip(msg.ToolTips.t97)
        self.cb_fig_format = QComboBox(objectName='fig_format')
        self.cb_fig_format.setToolTip(msg.ToolTips.t100)
        self.cb_fig_format.addItems(['jpg', 'pdf', 'png', 'svg', 'tiff'])
        self.le_results_dir = QLineEdit('', objectName='save_dir')
        self.le_results_dir.setToolTip(msg.ToolTips.t95)
        self.pb_save_fig = QPushButton("Save", objectName='save_fig')
        self.pb_save_fig.setToolTip(msg.ToolTips.t98)
        self.pb_save_as_fig = QPushButton(
            "Save as ...", objectName='save_as_fig')
        self.pb_save_as_fig.setToolTip(msg.ToolTips.t98)

        self.cb_fig_format.currentIndexChanged.connect(self.change_dpi_vs_path)

        self.w_style_layout = QGridLayout()
        self.w_style_layout.addWidget(QLabel("Current Style:"), 0, 0)
        self.w_style_layout.addWidget(self.le_style, 0, 1, )
        self.w_style_layout.addWidget(self.pb_add_rcParam, 0, 2, )
        self.w_style_layout.addWidget(QLabel("Figure Size:"), 1, 0,)
        self.w_style_layout.addWidget(self.le_fig_size_w, 1, 1)
        self.w_style_layout.addWidget(self.le_fig_size_h, 1, 2)
        self.w_style_layout.addWidget(self.check_display_size, 2, 0, 1, 2)

        self.w_style_layout.addWidget(QLabel("Figure Font:"), 3, 0,)
        self.w_style_layout.addWidget(self.font_cb, 3, 1, 1, 2)
        self.w_style_layout.addWidget(QLabel("Font Style:"), 4, 0,)
        self.w_style_layout.addWidget(self.cb_font_style, 4, 1, 1, 2)
        self.w_style_layout.addWidget(QLabel("Label Size:"), 5, 0,)
        self.w_style_layout.addWidget(self.sb_label_size, 5, 1, )
        self.w_style_layout.addWidget(QLabel("Ticks Size:"), 6, 0,)
        self.w_style_layout.addWidget(self.sb_tick_size, 6, 1,)
        self.w_style_layout.addWidget(QLabel("Format:"), 7, 0,)
        self.w_style_layout.addWidget(self.cb_fig_format, 7, 1)
        self.w_style_layout.addWidget(self.l_DPI, 8, 0,)
        self.w_style_layout.addWidget(self.le_fig_dpi, 8, 1)
        self.w_style_layout.addWidget(self.check_svg_text2path, 8, 0, 1, 2)
        self.w_style_layout.addWidget(QLabel("Result Dir:"), 9, 0,)
        self.w_style_layout.addWidget(self.le_results_dir, 9, 1, 1, 3)
        self.w_style_layout.addWidget(self.pb_save_fig, 10, 0)
        self.w_style_layout.addWidget(self.pb_save_as_fig, 10, 2)

        self.w_style.setLayout(self.w_style_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_ds, )
        self.layout.addWidget(self.gb_plots, )
        self.layout.addWidget(self.w_style, )
        self.layout.addStretch()
        self.setMaximumWidth(310)

    def change_dpi_vs_path(self):
        if self.sender().currentText() == "svg":
            self.l_DPI.setVisible(False)
            self.le_fig_dpi.setVisible(False)
            self.check_svg_text2path.setVisible(True)
        else:
            self.check_svg_text2path.setVisible(False)
            self.l_DPI.setVisible(True)
            self.le_fig_dpi.setVisible(True)


class CanvasWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.gb_canvas = QGroupBox('Figure Preview')
        self.w_canvas = QLabel(msg.Widgets.i13)
        self.view_layout = QVBoxLayout(self)
        # self.layout.addWidget(self.group_box)
        self.view_layout.addWidget(self.w_canvas)
        self.gb_canvas.setLayout(self.view_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.layout.addWidget(self.gb_canvas)


class PropertiesWidget(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.gb_properties = QGroupBox('Plot Properties')
        self.w_current_properties = QWidget()
        self.view_layout = QVBoxLayout(self)

        self.view_layout.addWidget(self.w_current_properties)
        self.view_layout.setContentsMargins(0, 0, 0, 0)
        self.gb_properties.setLayout(self.view_layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.gb_properties)
        self.layout.addStretch()
        self.setMinimumWidth(370)
        self.setMaximumWidth(370)


class SurfPlotProperties(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.w_trimm = Widgets.trimm_widget(surf_plot=True)

        self.w_hide_area = Widgets.hide_area()

        self.w_view_manipulations = Widgets.view_manipulation2D()

        self.w_colormap = Widgets.colormap2D()

        self.w_show_ss = QWidget()
        self.w_show_ss.check_show_ss = QCheckBox(
            "Display Steady-State Data", objectName='check_show_ss')
        layout = QGridLayout()
        layout.addWidget(self.w_show_ss.check_show_ss, 0, 0)
        self.w_show_ss.setLayout(layout)
        self.w_show_ss.setEnabled(False)

        self.w_show_delA_cuts = Widgets.show_delA_2D()

        self.w_show_kin_cuts = Widgets.show_kin_2d()

        self.w_show_second_ax = Widgets.show_2nd_ax_widget()

        self.w_show_info = Widgets.show_info_widget()

        self.w_show_pump = Widgets.show_pump_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)

        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_view_manipulations)
        self.layout.addWidget(self.w_hide_area)
        self.layout.addWidget(self.w_colormap)
        self.layout.addWidget(self.w_show_ss)
        self.layout.addWidget(self.w_show_delA_cuts)
        self.layout.addWidget(self.w_show_kin_cuts)
        self.layout.addWidget(self.w_show_second_ax)
        self.layout.addWidget(self.w_show_info)
        self.layout.addWidget(self.w_show_pump)
        self.layout.addStretch()


class DelAPlotProperties(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.w_trimm = Widgets.trimm_widget(exclude='Delay')

        self.w_select_data_color = Widgets.select_data_color_widget('Delay')
        self.w_show_legend = Widgets.show_legend()

        self.w_hide_area = Widgets.hide_area()

        self.w_show_second_ax = Widgets.show_2nd_ax_widget()

        self.w_show_info = Widgets.show_info_widget()

        self.w_show_pump = Widgets.show_pump_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_show_legend)
        self.layout.addWidget(self.w_select_data_color)
        self.layout.addWidget(self.w_hide_area)
        self.layout.addWidget(self.w_show_second_ax)
        self.layout.addWidget(self.w_show_info)
        self.layout.addWidget(self.w_show_pump)
        self.layout.addStretch()


class KinTraceProperties(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.w_trimm = Widgets.trimm_widget(exclude='Wavelength')
        self.w_show_legend = Widgets.show_legend()

        self.w_select_data_color = Widgets.select_data_color_widget(
            'Wavelength')

        self.w_normalize = Widgets.normalize_widget()

        self.w_show_info = Widgets.show_info_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_show_legend)
        self.layout.addWidget(self.w_select_data_color)
        self.layout.addWidget(self.w_normalize)
        self.layout.addWidget(self.w_show_info)
        self.layout.addStretch()


class LocalFitProperties(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_trimm = Widgets.trimm_widget(exclude='Wavelength')

        self.w_select_fit = Widgets.select_fit(
            label='Select Kinetic Fits:', exclusive_check=False)

        self.w_select_data_color = Widgets.select_data_color_widget('Local')

        self.w_normalize = Widgets.normalize_widget()

        self.w_show_data = Widgets.show_experimental()

        self.w_show_conc = QWidget()
        self.w_show_conc.check_show_conc = QCheckBox(
            "Show Components", objectName='check_show_conc')
        self.w_show_conc_layout = QGridLayout()
        self.w_show_conc_layout.addWidget(
            self.w_show_conc.check_show_conc, 0, 0)
        self.w_show_conc.setLayout(self.w_show_conc_layout)

        self.w_show_tau = QWidget()
        self.w_show_tau.check_show_tau = QCheckBox(
            "Show Lifetimes", objectName='check_show_tau')
        self.w_show_tau_layout = QGridLayout()
        self.w_show_tau_layout.addWidget(self.w_show_tau.check_show_tau, 0, 0)
        self.w_show_tau.setLayout(self.w_show_tau_layout)

        self.w_show_residuals = Widgets.show_residuals()
        self.w_show_legend = Widgets.show_legend()

        self.w_show_info = Widgets.show_info_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_select_fit)
        self.layout.addWidget(Widgets.separator())
        self.layout.addWidget(self.w_select_data_color)
        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_show_legend)
        self.layout.addWidget(self.w_normalize)
        self.layout.addWidget(self.w_show_data)
        self.layout.addWidget(self.w_show_conc)
        self.layout.addWidget(self.w_show_tau)
        self.layout.addWidget(self.w_show_residuals)
        self.layout.addWidget(self.w_show_info)
        self.layout.addStretch()


class GlobalFitProperties_2D(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):
        def change_merge_visibility(sender):

            if sender.cb_2D_plot.currentText() == 'merge':
                sender.le_merge_thresh.setVisible(True)
            else:
                sender.le_merge_thresh.setVisible(False)

        self.w_select_fit = Widgets.select_fit(
            label='Select Global Fit:', exclusive_check=True)

        self.w_select_2D_plot = QWidget()
        self.w_select_2D_plot.cb_2D_plot = QComboBox(objectName='cb_2D_plot')
        self.w_select_2D_plot.cb_2D_plot.addItems(
            ['simulated', 'residuals', 'merge'])
        self.w_select_2D_plot.cb_2D_plot.setToolTip(msg.ToolTips.t127)
        self.w_select_2D_plot.le_merge_thresh = QLineEdit(
            placeholderText=msg.Widgets.i12, maximumWidth=50, objectName='le_merge_thresh')
        self.w_select_2D_plot.le_merge_thresh.setToolTip(msg.ToolTips.t128)
        self.w_select_2D_plot.le_merge_thresh.setVisible(False)
        layout = QGridLayout()
        layout.addWidget(QLabel('Type'), 0, 0)
        layout.addWidget(self.w_select_2D_plot.cb_2D_plot, 0, 1)
        layout.addWidget(self.w_select_2D_plot.le_merge_thresh, 0, 2)
        self.w_select_2D_plot.setLayout(layout)

        self.w_select_2D_plot.cb_2D_plot.currentIndexChanged.connect(
            lambda index: change_merge_visibility(self.w_select_2D_plot))

        self.w_trimm = Widgets.trimm_widget(surf_plot=True)
        self.w_hide_area = Widgets.hide_area()
        self.w_view_manipulations = Widgets.view_manipulation2D(ss=False)
        self.w_colormap = Widgets.colormap2D()
        self.w_show_second_ax = Widgets.show_2nd_ax_widget()
        self.w_show_info = Widgets.show_info_widget()
        self.w_show_pump = Widgets.show_pump_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_select_fit)
        self.layout.addWidget(Widgets.separator())
        self.layout.addWidget(self.w_select_2D_plot)
        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_view_manipulations)
        self.layout.addWidget(self.w_colormap)
        self.layout.addWidget(self.w_hide_area)
        self.layout.addWidget(self.w_show_second_ax)
        self.layout.addWidget(self.w_show_info)
        self.layout.addWidget(self.w_show_pump)

        self.layout.addStretch()


class GlobalFitProperties_EASDAS(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_select_fit = Widgets.select_fit(
            label='Select Global Fit:', exclusive_check=True)

        self.w_trimm = Widgets.trimm_widget(exclude='Delay')
        self.w_show_legend = Widgets.show_legend()
        self.w_normalize = Widgets.normalize_widget(abs_values=False)
        self.w_select_data_color = Widgets.select_data_color_widget('Global')

        self.w_hide_area = Widgets.hide_area()

        self.w_show_second_ax = Widgets.show_2nd_ax_widget()

        self.w_show_info = Widgets.show_info_widget()

        self.w_show_pump = Widgets.show_pump_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_select_fit)
        self.layout.addWidget(Widgets.separator())

        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_show_legend)
        self.layout.addWidget(self.w_normalize)
        self.layout.addWidget(self.w_select_data_color)
        self.layout.addWidget(self.w_hide_area)
        self.layout.addWidget(self.w_show_second_ax)
        self.layout.addWidget(self.w_show_info)
        self.layout.addWidget(self.w_show_pump)
        self.layout.addStretch()

        # self.layout.addWidget(self.show_info)

        self.layout.addStretch()


class GlobalFitProperties_conc(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_select_fit = Widgets.select_fit(
            label='Select Global Fit:', exclusive_check=True)

        self.w_trimm = Widgets.trimm_widget(exclude='Wavelength')
        self.w_trimm.z_label.setText("Conc")
        self.w_show_legend = Widgets.show_legend()
        self.w_select_data_color = Widgets.select_data_color_widget('Local')

        self.w_show_info = Widgets.show_info_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_select_fit)
        self.layout.addWidget(Widgets.separator())
        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_show_legend)
        self.layout.addWidget(self.w_select_data_color)
        self.layout.addWidget(self.w_show_info)
        self.layout.addStretch()


class GlobalFitProperties_DelA(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_select_fit = Widgets.select_fit(
            label='Select Global Fit:', exclusive_check=True)
        self.w_trimm = Widgets.trimm_widget(exclude='Delay')
        self.w_show_legend = Widgets.show_legend()

        self.w_select_data_color = Widgets.select_data_color_widget('Delay')

        self.w_hide_area = Widgets.hide_area()

        self.w_show_data = Widgets.show_experimental()
        self.w_show_residuals = Widgets.show_residuals()
        self.w_show_second_ax = Widgets.show_2nd_ax_widget()

        self.w_show_info = Widgets.show_info_widget()

        self.w_show_pump = Widgets.show_pump_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_select_fit)
        self.layout.addWidget(Widgets.separator())
        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_show_legend)
        self.layout.addWidget(self.w_select_data_color)
        self.layout.addWidget(self.w_hide_area)
        self.layout.addWidget(self.w_show_data)
        self.layout.addWidget(self.w_show_residuals)
        self.layout.addWidget(self.w_show_second_ax)
        self.layout.addWidget(self.w_show_info)
        self.layout.addWidget(self.w_show_pump)
        self.layout.addStretch()

        # self.layout.addWidget(self.show_info)

        self.layout.addStretch()


class GlobalFitProperties_KinTrace(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_select_fit = Widgets.select_fit(
            label='Select Global Fit:', exclusive_check=True)
        self.w_trimm = Widgets.trimm_widget(exclude='Wavelength')
        self.w_show_legend = Widgets.show_legend()

        self.w_select_data_color = Widgets.select_data_color_widget(
            'Wavelength')

        self.w_normalize = Widgets.normalize_widget()
        self.w_show_data = Widgets.show_experimental()

        self.w_show_residuals = Widgets.show_residuals()

        self.w_show_info = Widgets.show_info_widget()

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_select_fit)
        self.layout.addWidget(Widgets.separator())
        self.layout.addWidget(self.w_trimm)
        self.layout.addWidget(self.w_show_legend)
        self.layout.addWidget(self.w_select_data_color)
        self.layout.addWidget(self.w_normalize)
        self.layout.addWidget(self.w_show_data)
        self.layout.addWidget(self.w_show_residuals)
        self.layout.addWidget(self.w_show_info)
        self.layout.addStretch()

        # self.layout.addWidget(self.show_info)

        self.layout.addStretch()


class GlobalFitProperties_emcee(QWidget):
    def __init__(self, ):
        super().__init__()
        self.initUI()

    def initUI(self):

        self.w_select_fit = Widgets.select_fit(
            label='Select Fit:', exclusive_check=True)
        self.w_corner_manipulation = QWidget()
        self.w_corner_manipulation.sb_bins = QSpinBox(
            minimum=2, maximum=100, value=5, objectName='sb_bins', maximumWidth=100)
        self.w_corner_manipulation.check_truth = QCheckBox(
            'Show Values', objectName='check_truth')
        self.w_corner_manipulation.sb_label_pad = QSpinBox(
            minimum=0, maximum=100, value=2, objectName='sb_label_pad', maximumWidth=100)
        self.w_corner_manipulation.sb_subplot_pad = QSpinBox(
            minimum=0, maximum=100, value=5, objectName='sb_subplot_pad', maximumWidth=100)
        self.w_corner_manipulation.sb_tick_num = QSpinBox(
            minimum=0, maximum=100, value=5, objectName='sb_tick_num', maximumWidth=100)
        self.w_corner_manipulation.check_plot_datapoints = QCheckBox(
            'Show Datapoints', objectName='check_plot_datapoints')
        self.w_corner_manipulation.check_show_titles = QCheckBox(
            'Show Errors', objectName='check_show_titles')
        self.w_corner_manipulation.check_show_quantiles = QCheckBox(
            'Show Median ± σ', objectName='check_show_quantiles')
        self.w_corner_manipulation.check_plot_contours = QCheckBox(
            'Show Contours', objectName='check_plot_contours')
        self.w_corner_manipulation.check_plot_density = QCheckBox(
            'Plot Density', objectName='check_plot_density')

        layout = QFormLayout()
        layout.addRow('Bins', self.w_corner_manipulation.sb_bins)
        layout.addRow('Label Pad', self.w_corner_manipulation.sb_label_pad)
        layout.addRow('Subplot Pad', self.w_corner_manipulation.sb_subplot_pad)
        layout.addRow('Max # of Ticks', self.w_corner_manipulation.sb_tick_num)
        layout.addRow(self.w_corner_manipulation.check_truth)
        layout.addRow(self.w_corner_manipulation.check_show_titles)
        layout.addRow(self.w_corner_manipulation.check_show_quantiles)
        layout.addRow(self.w_corner_manipulation.check_plot_datapoints)
        layout.addRow(self.w_corner_manipulation.check_plot_contours)
        layout.addRow(self.w_corner_manipulation.check_plot_density)

        self.w_corner_manipulation.sb_bins.setToolTip(msg.ToolTips.t129)
        self.w_corner_manipulation.check_truth.setToolTip(msg.ToolTips.t133)
        self.w_corner_manipulation.sb_label_pad.setToolTip(msg.ToolTips.t130)
        self.w_corner_manipulation.sb_subplot_pad.setToolTip(msg.ToolTips.t131)
        self.w_corner_manipulation.sb_tick_num.setToolTip(msg.ToolTips.t132)
        self.w_corner_manipulation.check_plot_datapoints.setToolTip(msg.ToolTips.t136)
        self.w_corner_manipulation.check_show_titles.setToolTip(msg.ToolTips.t134)
        self.w_corner_manipulation.check_show_quantiles.setToolTip(msg.ToolTips.t135)
        self.w_corner_manipulation.check_plot_contours.setToolTip(msg.ToolTips.t137)
        self.w_corner_manipulation.check_plot_density.setToolTip(msg.ToolTips.t138)

        self.w_corner_manipulation.setLayout(layout)

        self.layout = QVBoxLayout(self)
        self.setLayout(self.layout)
        self.layout.addWidget(self.w_select_fit)
        self.layout.addWidget(Widgets.separator())

        self.layout.addWidget(self.w_corner_manipulation)
        self.layout.addStretch()

        # self.layout.addWidget(self.show_info)

        self.layout.addStretch()
