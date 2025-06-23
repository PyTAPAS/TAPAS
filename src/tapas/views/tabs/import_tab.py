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
from pathlib import Path

# Third-Party Imports
import numpy as np
from PyQt6.QtWidgets import (QWidget, QLabel, QVBoxLayout, QGridLayout,
                             QFileDialog, QGroupBox, QMessageBox)
from PyQt6.QtCore import Qt
from matplotlib import colors
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Local Application Imports
from ...utils import utils
from ...configurations import messages as msg
from ...views.tabwidgets.import_tabwidgets import (
    ImportWidget, MetadataWidget, PreviewContainerWidget)


logger = logging.getLogger(__name__)


class ImportTab(QWidget):
    def __init__(self, tab, abs_model, em_model, ta_model, controller, config):
        super().__init__()

        self.abs_model = abs_model
        self.em_model = em_model
        self.ta_model = ta_model
        self.import_controller = controller
        self.config = config
        self.need_GUI_update = False
        self.tab = tab
        self.InitUI()

    def InitUI(self):
        # -------- create Widgets ------------------------------------------------------------------
        self.absorbance_input = ImportWidget(
            label="Absorbance Data", placeholder=msg.Widgets.i01, energy_unit=True)
        self.emission_input = ImportWidget(
            label="Emission Data", placeholder=msg.Widgets.i02, energy_unit=True)
        self.ta_input = ImportWidget(
            label="TA Data", placeholder=msg.Widgets.i03, time_unit=True, energy_unit=True,
            delA_unit=True, matrix_orientation=True)
        self.solvent_input = ImportWidget(
            label="Solvent", placeholder=msg.Widgets.i03, time_unit=True, energy_unit=True,
            delA_unit=True, matrix_orientation=True)
        self.tw_metadata_container = MetadataWidget()
        self.tw_preview_container = PreviewContainerWidget()
        self.update_config()

        # -------- add Widgets to layout -----------------------------------------------------------
        self.gb_import = QGroupBox('Imports')
        # self.gb_import.setFixedHeight(380)
        self.gb_import_layout = QVBoxLayout()
        self.gb_import_layout.addWidget(self.ta_input)
        self.gb_import_layout.addWidget(self.solvent_input)
        self.gb_import_layout.addWidget(self.absorbance_input)
        self.gb_import_layout.addWidget(self.emission_input)
        self.gb_import.setLayout(self.gb_import_layout)
        self.tw_import_container = QWidget()
        self.tw_import_container_layout = QGridLayout()
        self.tw_import_container_layout.addWidget(self.gb_import)
        self.tw_import_container.setLayout(self.tw_import_container_layout)
        self.import_layout = QGridLayout()
        self.setLayout(self.import_layout)

        import_scroll = utils.Converter.create_scrollable_widget(
            self.tw_import_container, min_height=150, max_height=600)
        meta_scroll = utils.Converter.create_scrollable_widget(
            self.tw_metadata_container, min_width=400, max_width=400, min_height=150,
            max_height=600, horizontal_scroll=False)

        self.import_layout.setContentsMargins(0, 0, 0, 0)
        self.import_layout.addWidget(import_scroll, 0, 0, alignment=Qt.AlignmentFlag.AlignTop)
        self.import_layout.addWidget(meta_scroll, 0, 1,
                                     alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop)
        self.import_layout.addWidget(self.tw_preview_container, 1, 0, 1, 2,)

        # -------- connect Widgets to view / controller --------------------------------------------
        self.absorbance_input.pb_browse.clicked.connect(
            lambda: self.open_file_dialog(raw_ds='abs'))
        self.absorbance_input.pb_load.clicked.connect(
            lambda: self.import_controller.change_path(
                raw_ds='abs',
                paths=self.absorbance_input.le_path.text()))
        self.absorbance_input.pb_load.clicked.connect(
            lambda: self.import_controller.load_data(
                raw_ds='abs',
                delimiter_idx=self.absorbance_input.cb_delimiter.currentIndex(),
                header=self.absorbance_input.sb_header.value(),
                energy_unit=self.absorbance_input.cb_energy_unit.currentText()))
        self.absorbance_input.pb_clear.clicked.connect(
            lambda: self.import_controller.clear_all(raw_ds='abs'))

        self.emission_input.pb_browse.clicked.connect(
            lambda: self.open_file_dialog(raw_ds='em'))
        self.emission_input.pb_load.clicked.connect(
            lambda: self.import_controller.change_path(
                raw_ds='em',
                paths=self.emission_input.le_path.text()))
        self.emission_input.pb_load.clicked.connect(
            lambda: self.import_controller.load_data(
                raw_ds='em',
                delimiter_idx=self.emission_input.cb_delimiter.currentIndex(),
                header=self.emission_input.sb_header.value(),
                energy_unit=self.emission_input.cb_energy_unit.currentText()))
        self.emission_input.pb_clear.clicked.connect(
            lambda: self.import_controller.clear_all(raw_ds='em'))

        self.ta_input.pb_browse.clicked.connect(
            lambda: self.open_file_dialog('ta'))
        self.ta_input.pb_load.clicked.connect(
            lambda: self.import_controller.change_path(
                raw_ds='ta',
                paths=self.ta_input.le_path.text()))
        self.ta_input.pb_load.clicked.connect(self.ask_average)
        self.ta_input.pb_clear.clicked.connect(
            lambda: self.import_controller.clear_all('ta'))

        self.solvent_input.pb_browse.clicked.connect(
            lambda: self.open_file_dialog(raw_ds='solvent'))
        self.solvent_input.pb_load.clicked.connect(
            lambda: self.import_controller.change_path(raw_ds='solvent', paths=self.solvent_input.le_path.text()))
        self.solvent_input.pb_load.clicked.connect(
            lambda: self.import_controller.load_data(
                raw_ds='solvent',
                delimiter_idx=self.solvent_input.cb_delimiter.currentIndex(),
                header=self.solvent_input.sb_header.value(),
                time_unit=self.solvent_input.cb_time_unit.currentIndex(),
                energy_unit=self.solvent_input.cb_energy_unit.currentText(),
                delA_unit=self.solvent_input.cb_delA_unit.currentText(),
                matrix_orientation =self.solvent_input.cb_matrix_orientation.currentIndex()))
        self.solvent_input.pb_clear.clicked.connect(
            lambda: self.import_controller.clear_all('solvent'))

        self.tw_metadata_container.le_exp_name.editingFinished.connect(
            lambda: self.import_controller.save_metadata(
                key='experiment',
                value=self.tw_metadata_container.le_exp_name.text()))
        self.tw_metadata_container.le_sample_name.editingFinished.connect(
            lambda: self.import_controller.save_metadata(
                key='sample',
                value=self.tw_metadata_container.le_sample_name.text()))
        self.tw_metadata_container.le_exc_wavelen.editingFinished.connect(
            lambda: self.import_controller.save_metadata(
                key='excitation wavelength',
                value=self.tw_metadata_container.le_exc_wavelen.text()))
        self.tw_metadata_container.le_exc_int.editingFinished.connect(
            lambda: self.import_controller.save_metadata(
                key='excitation intensity',
                value=self.tw_metadata_container.le_exc_int.text()))
        self.tw_metadata_container.le_solovent.editingFinished.connect(
            lambda: self.import_controller.save_metadata(
                key='solvent',
                value=self.tw_metadata_container.le_solovent.text()))
        self.tw_metadata_container.te_notes.textChanged.connect(
            lambda: self.import_controller.save_metadata(
                key='notes',
                value=self.tw_metadata_container.te_notes.toPlainText()))

        # -------- listen to model event signals ---------------------------------------------------
        models = (self.abs_model, self.em_model, self.ta_model)
        for i in models:
            i.rawdata_changed.connect(
                lambda raw_ds, data='data': self.queue_update_GUI(raw_ds, data='data'))
            i.path_changed.connect(
                lambda raw_ds, data='path': self.queue_update_GUI(raw_ds, data='path'))
            i.metadata_changed.connect(self.update_metadata_GUI)
        self.ta_model.rawdata_changed.connect(
            lambda sender: self.import_controller.delete_ds() if sender == 'ta' else None)

    def update_metadata_GUI(self) -> None:
        ''' called by main window if new project is loaded '''
        meta_dict = self.import_controller.get_metadata()

        if meta_dict:
            self.tw_metadata_container.le_exp_name.setText(
                meta_dict['experiment'])
            self.tw_metadata_container.le_sample_name.setText(
                meta_dict['sample'])
            self.tw_metadata_container.le_exc_wavelen.setText(
                meta_dict['excitation wavelength'])
            self.tw_metadata_container.le_exc_int.setText(
                meta_dict['excitation intensity'])
            self.tw_metadata_container.le_solovent.setText(
                meta_dict['solvent'])
            self.tw_metadata_container.te_notes.setText(meta_dict['notes'])

    def queue_update_GUI(self, raw_ds: str, data: str) -> None:
        ''' called, if the model path or data is changed. GUI update waits till tab is selected

        Parameters
        ----------
        raw_ds : TYPE str
            either 'abs', 'em', 'ta' or 'solvent'. specifies which model has been changed
        data : TYPE str
            either 'data' or 'path'. specifies what has been changed

        Returns
        -------
        None.

        '''
        self.need_GUI_update = True
        if self.tab.currentIndex() == 0:
            self.update_GUI(raw_ds, data)

    def update_GUI(self, raw_ds="all", data='data') -> None:
        '''
        function called directly by the main window everytime the Tab is clicked
        or if the Tab is active and data was changed (handled by queue_update_GUI).
        Tab is updated if needed (handled by the need_GUI_update boolean).

        Parameters
        ----------
        raw_ds : TYPE str
            either 'abs', 'em', 'ta', 'solvent' or 'all'. specifies which model has been changed.
            The default is "all" which updates everything
        data : TYPE str
            either 'data' or 'path'. specifies what has been changed

        Returns
        -------
        None.

        '''
        if self.need_GUI_update:
            if raw_ds == 'all':
                self.set_path(raw_ds='abs')
                self.set_path(raw_ds='em')
                self.set_path(raw_ds='ta')
                self.preplot_data('abs')
                self.preplot_data(raw_ds='em')
                self.preplot_data(raw_ds='ta')
                self.preplot_data(raw_ds='solvent')

            else:
                if data == 'path':
                    self.set_path(raw_ds)
                else:
                    self.preplot_data(raw_ds=raw_ds)
            self.need_GUI_update = False

    def update_config(self) -> None:
        '''updates configuration and standard values of QWidgets'''

        self.config.add_handler('import_cb_abs_delimiter',
                                self.absorbance_input.cb_delimiter)
        self.config.add_handler('import_sb_abs_header',
                                self.absorbance_input.sb_header)
        self.config.add_handler('import_cb_em_delimiter',
                                self.emission_input.cb_delimiter)
        self.config.add_handler('import_sb_em_header',
                                self.emission_input.sb_header)
        self.config.add_handler('import_cb_solvent_delimiter',
                                self.solvent_input.cb_delimiter)
        self.config.add_handler('import_sb_solvent_header', self.solvent_input.sb_header)
        self.config.add_handler('import_cb_solvent_time_unit',
                                self.solvent_input.cb_time_unit)
        self.config.add_handler('import_cb_ta_delimiter',
                                self.ta_input.cb_delimiter)
        self.config.add_handler('import_sb_ta_header', self.ta_input.sb_header)
        self.config.add_handler('import_cb_ta_time_unit',
                                self.ta_input.cb_time_unit)

    def open_file_dialog(self, raw_ds: str) -> None:
        ''' opens a file dialog and forwards the path to the controller '''
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select one or multiple Files",)

        if not filenames:
            return
        paths = [Path(f) for f in filenames]
        self.import_controller.change_path(raw_ds, paths)

    def ask_average(self) -> None:
        ''' checks for multiple data import and whether user wants to average before sending it to controller '''
        path = self.import_controller.get_model_path('ta')
        if not path:
            self.import_controller.call_statusbar("error", msg.Error.e03)
            return
        if len(path) != 1:
            answer = QMessageBox.question(self,
                                          'Multiple Data detected',
                                          'Do you want to average over multiple scans?',
                                          QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)

            if answer == QMessageBox.StandardButton.Yes:
                self.import_controller.load_data(raw_ds='ta',
                                                 delimiter_idx=self.ta_input.cb_delimiter.currentIndex(),
                                                 header=self.ta_input.sb_header.value(),
                                                 time_unit=self.ta_input.cb_time_unit.currentIndex(),
                                                 energy_unit=self.ta_input.cb_energy_unit.currentText(),
                                                 delA_unit=self.ta_input.cb_delA_unit.currentText(),
                                                 matrix_orientation =self.ta_input.cb_matrix_orientation.currentIndex())
            else:
                return
        else:
            self.import_controller.load_data(raw_ds='ta',
                                             delimiter_idx=self.ta_input.cb_delimiter.currentIndex(),
                                             header=self.ta_input.sb_header.value(),
                                             time_unit=self.ta_input.cb_time_unit.currentIndex(),
                                             energy_unit=self.ta_input.cb_energy_unit.currentText(),
                                             delA_unit=self.ta_input.cb_delA_unit.currentText(),
                                             matrix_orientation =self.ta_input.cb_matrix_orientation.currentIndex())

    def set_path(self, raw_ds: str,) -> None:
        '''
        selectively sets modelpath (list object) to easy to read line edit string in GUI

        Parameters
        ----------
        raw_ds : TYPE str
            either 'abs', 'em', 'ta' or 'solvent'

        Returns
        -------
        None.

        '''
        model = self.import_controller.get_rawdata_model(raw_ds)
        paths = model.path or [] if raw_ds != 'solvent' else model.solvent_path or []
        text = ' , '.join(p.as_posix() for p in paths)
        if raw_ds == "abs":
            self.absorbance_input.le_path.setText(text)
        elif raw_ds == "em":
            self.emission_input.le_path.setText(text)
        elif raw_ds == "ta":
            self.ta_input.le_path.setText(text)
        else:
            self.solvent_input.le_path.setText(text)

    def preplot_data(self, raw_ds: str) -> None:
        ''' tries to plot the updated data model, checks data for inconsistency '''

        # -------- create Canvas -------------------------------------------------------------------
        self.sc = utils.PlotCanvas(self, width=5, height=5, dpi=100)
        fig = self.sc.fig
        ax = self.sc.fig.add_subplot(111)
        sc = self.sc
        ax.cla()
        setattr(self, f"sc_{raw_ds}", sc)
        sc.axes_mapping = {}
        sc.mpl_connect(
            "scroll_event",
            lambda event, _sc=sc: _sc._zoom_TA(event, _sc.axes_mapping))

        toolbar = NavigationToolbar2QT(sc)
        layout = QVBoxLayout()
        layout.addWidget(toolbar)
        layout.addWidget(sc)

        # -------- plot TA data --------------------------------------------------------------------
        if raw_ds == "ta":
            dataX, dataY, dataZ = self.import_controller.get_ta_data()
            if dataX is None:
                self.tw_preview_container.ta_window.setParent(
                    None)  # remove widget from layout
                self.tw_preview_container.ta_window = QLabel(msg.Widgets.i06)
                self.tw_preview_container.inner_layout.addWidget(
                    self.tw_preview_container.ta_window, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)
                return
            normalization = colors.TwoSlopeNorm(vmin=-5, vmax=5, vcenter=0)
            try:
                X, Y = np.meshgrid(dataX,
                                   dataY)
                self.pcolormesh_plot = ax.pcolormesh(
                    X, Y, dataZ, shading='auto', norm=normalization)
            except (IndexError, ValueError):
                self.import_controller.clear_data(raw_ds)
                self.import_controller.call_statusbar("error", msg.Error.e02)

                return
            except Exception:
                self.import_controller.clear_data(raw_ds)
                logger.exception("unknown exception occurred")
                self.import_controller.call_statusbar("error", msg.Error.e01)
                return
            else:
                ax.set_ylabel(msg.Labels.delay)
                ax.set_xlabel(msg.Labels.wavelength)
                ax.yaxis.set_major_formatter(sc.delay_formatter0)
                ax.xaxis.set_major_formatter(sc.nm_formatter_ax)
                ax.set_title("TA Data")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)
                self.cb = fig.colorbar(mappable=self.pcolormesh_plot, cax=cax, location="right",
                                       shrink=0.6, label=msg.Labels.delA)
                self.cb.minorticks_on()
                self.tw_preview_container.ta_window.setParent(None)  # remove widget from layout
                self.tw_preview_container.ta_window = QWidget()
                self.tw_preview_container.ta_window.setLayout(layout)
                self.tw_preview_container.inner_layout.addWidget(
                    self.tw_preview_container.ta_window, 0, 0, alignment=Qt.AlignmentFlag.AlignCenter)

            self.sc.axes_mapping[ax] = (self.pcolormesh_plot, self.cb)

        # -------- plot solvent data ---------------------------------------------------------------
        elif raw_ds == 'solvent':
            dataX, dataY, dataZ = self.import_controller.get_solvent_data()
            if dataX is None:
                self.tw_preview_container.solvent_window.setParent(
                    None)  # remove widget from layout
                self.tw_preview_container.solvent_window = QLabel(
                    msg.Widgets.i06)
                self.tw_preview_container.inner_layout.addWidget(
                    self.tw_preview_container.solvent_window, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
                return
            normalization = colors.TwoSlopeNorm(vmin=-5, vmax=5, vcenter=0)
            try:
                X, Y = np.meshgrid(dataX,
                                   dataY)
                self.pcolormesh_plot = ax.pcolormesh(
                    X, Y, dataZ, shading='auto', norm=normalization)
            except (IndexError, ValueError):
                self.import_controller.clear_data(raw_ds)
                self.import_controller.call_statusbar("error", msg.Error.e02)

                return
            except Exception:
                logger.exception("unknown exception occurred")
                self.import_controller.clear_data(raw_ds)
                self.import_controller.call_statusbar("error", msg.Error.e01)
                return
            else:

                ax.set_ylabel(msg.Labels.delay)
                ax.set_xlabel(msg.Labels.wavelength)
                ax.yaxis.set_major_formatter(sc.delay_formatter0)
                ax.xaxis.set_major_formatter(sc.nm_formatter_ax)

                ax.set_title("TA Data")
                divider = make_axes_locatable(ax)
                cax = divider.append_axes('right', size='5%', pad=0.05)

                self.cb = fig.colorbar(mappable=self.pcolormesh_plot, cax=cax, location="right",
                                       shrink=0.6, label=msg.Labels.delA, )
                self.cb.minorticks_on()
                self.tw_preview_container.solvent_window.setParent(
                    None)  # remove widget from layout
                self.tw_preview_container.solvent_window = QWidget()
                self.tw_preview_container.solvent_window.setLayout(layout)
                self.tw_preview_container.inner_layout.addWidget(
                    self.tw_preview_container.solvent_window, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
            self.sc.axes_mapping[ax] = (self.pcolormesh_plot, self.cb)

        # -------- plot steady-state abs or em data ------------------------------------------------
        else:

            data_before, data_after = self.import_controller.get_ss_data(
                raw_ds)

            if data_before is None:
                if raw_ds == 'abs':
                    self.tw_preview_container.abs_window.setParent(
                        None)  # remove widget from layout
                    self.tw_preview_container.abs_window = QLabel(msg.Widgets.i04)
                    self.tw_preview_container.inner_layout.addWidget(
                        self.tw_preview_container.abs_window, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
                    return
                else:  # raw_ds = em
                    self.tw_preview_container.em_window.setParent(None)  # remove widget from layout
                    self.tw_preview_container.em_window = QLabel(msg.Widgets.i05)
                    self.tw_preview_container.inner_layout.addWidget(
                        self.tw_preview_container.em_window, 0, 3, alignment=Qt.AlignmentFlag.AlignCenter)
                    return

            try:
                ax.plot(
                    data_before[:, 0], data_before[:, 1], label="before")
                if data_after is not None:
                    ax.plot(
                        data_after[:, 0], data_after[:, 1], label="after")
                    ax.legend()

            except (IndexError, ValueError):
                self.import_controller.clear_data(raw_ds)
                self.import_controller.call_statusbar("error", msg.Error.e02)

                return
            except Exception:
                logger.exception("unknown exception occurred")
                self.import_controller.clear_data(raw_ds)
                self.import_controller.call_statusbar("error", msg.Error.e01)
                return
            else:
                if raw_ds == 'abs':
                    ax.set_title("Absorbance Data")
                    ax.set_xlabel(msg.Labels.wavelength)
                    ax.set_ylabel(msg.Labels.absorbance)
                    ax.xaxis.set_major_formatter(sc.nm_formatter_ax)

                    self.tw_preview_container.abs_window.setParent(
                        None)  # remove widget from layout
                    self.tw_preview_container.abs_window = QWidget()
                    self.tw_preview_container.abs_window.setLayout(layout)
                    self.tw_preview_container.inner_layout.addWidget(
                        self.tw_preview_container.abs_window, 0, 2, alignment=Qt.AlignmentFlag.AlignCenter)
                else:
                    ax.set_title("Emission Data")
                    ax.set_xlabel(msg.Labels.wavelength)
                    ax.set_ylabel(msg.Labels.intensity)
                    ax.xaxis.set_major_formatter(sc.nm_formatter_ax)

                    self.tw_preview_container.em_window.setParent(
                        None)  # remove widget from layout
                    self.tw_preview_container.em_window = QWidget()
                    self.tw_preview_container.em_window.setLayout(layout)
                    self.tw_preview_container.inner_layout.addWidget(
                        self.tw_preview_container.em_window, 0, 3, alignment=Qt.AlignmentFlag.AlignCenter)

        # -------- inform controller if plotting succeeded -----------------------------------------
        self.import_controller.call_statusbar("info", msg.Status.s02)
