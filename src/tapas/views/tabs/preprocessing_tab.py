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
from pathlib import Path
import logging
import logging.config

# PyQt6 Imports
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QGridLayout,
    QFileDialog,
)

# Matplotlib Imports
from matplotlib import colors
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backend_bases import MouseEvent

# Other Third‑Party Libraries
import h5py
import numpy as np

# Local Application Imports
from ...utils import utils
from ...configurations import exceptions as exc, messages as msg
from ...views.tabwidgets.preprocessing_tabwidgets import SelectViewWidget, CanvasWidget, ProcessWidget

logger = logging.getLogger(__name__)


class PreprocTab(QWidget):
    def __init__(self, tab, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3, controller, config):
        super().__init__()

        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.preproc_controller = controller
        self.tab = tab
        self.config = config
        self.view = 'full_view'
        self.trimm_xmin, self.trimm_xmax, self.trimm_ymin, self.trimm_ymax = None, None, None, None
        self.ds = '1'
        self.chirp_coeff = None
        self.buffer_dataZ = None
        self.buff_backgrnd_corr = {}
        self.buff_resampled_data = {}
        self.need_GUI_update = False
        self.InitUI()
        

    def InitUI(self):
        # -------- create Widgets ------------------------------------------------------------------
        self.tw_select_view = SelectViewWidget()
        self.tw_canvas = CanvasWidget()
        self.tw_process_view = ProcessWidget()
        self.update_config()

        # -------- add Widgets to layout -----------------------------------------------------------
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(utils.Converter.create_scrollable_widget(
            self.tw_select_view, max_width=310), 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.tw_canvas, 0, 1)
        layout.addWidget(utils.Converter.create_scrollable_widget(
            self.tw_process_view, min_width=370, max_width=370), 0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

        # -------- connect Widgets to view / controller --------------------------------------------
        self.tw_select_view.rb_ds_1.toggled.connect(self.update_dataset)
        self.tw_select_view.rb_ds_2.toggled.connect(self.update_dataset)
        self.tw_select_view.rb_ds_3.toggled.connect(self.update_dataset)

        self.tw_select_view.rb_full_view.toggled.connect(self.get_view)
        self.tw_select_view.rb_t0_view.toggled.connect(self.get_view)
        self.tw_select_view.le_linlog.editingFinished.connect(self.update_view)

        self.tw_process_view.le_xmin.editingFinished.connect(self.get_trimm)
        self.tw_process_view.le_xmax.editingFinished.connect(self.get_trimm)
        self.tw_process_view.le_ymin.editingFinished.connect(self.get_trimm)
        self.tw_process_view.le_ymax.editingFinished.connect(self.get_trimm)
        self.tw_process_view.pb_apply_trimm.clicked.connect(lambda _, : self.preproc_controller.apply_trimm(
            self.ds, self.trimm_xmin, self.trimm_xmax, self.trimm_ymin, self.trimm_ymax))

        self.tw_process_view.pb_resample.clicked.connect(self.get_resampled_data)
        self.tw_process_view.pb_regularize.clicked.connect(self.get_regularized_data)
        self.tw_process_view.pb_apply_resample.clicked.connect(self.apply_resampling)
        self.tw_process_view.pb_delete_resample.clicked.connect(
            lambda: self.plot_resampled(remove_correction=True))

        self.tw_process_view.pb_autocorr_chirp.clicked.connect(
            self.auto_set_chirp)
        self.tw_process_view.pb_manually_chirp.toggled.connect(
            self.set_t0_view)
        self.tw_process_view.pb_fit_chirp.clicked.connect(self.fit_chirp)
        self.tw_process_view.pb_del_chirp_fit.clicked.connect(
            self.remove_chirp)
        self.tw_process_view.pb_fromfile_chirp.clicked.connect(
            self.get_chirp_from_file)
        self.tw_process_view.pb_show_chirp_correction.clicked.connect(
            self.get_corrected_chirp)
        self.tw_process_view.pb_delete_chirp.clicked.connect(
            lambda: self.plot_corr_delA(remove_correction=True))
        self.tw_process_view.pb_apply_chirp.clicked.connect(
            lambda _, ds=self.ds, chirp_coeff=self.chirp_coeff: self.preproc_controller.apply_chirp(ds=self.ds, chirp_coeff=self.chirp_coeff))

        self.tw_process_view.pb_corr_background.clicked.connect(
            self.get_area_background_correction)
        self.tw_process_view.pb_fromfile_background.clicked.connect(
            self.get_background_from_file)
        self.tw_process_view.pb_corr_from_solvent.clicked.connect(
            self.get_background_from_solvent)
        self.tw_process_view.pb_apply_background.clicked.connect(
            self.apply_background_correction)
        self.tw_process_view.pb_delete_background.clicked.connect(
            lambda: self.plot_corr_delA(remove_correction=True))

        self.tw_process_view.le_t0.editingFinished.connect(self.get_t0)
        self.tw_process_view.pb_delete_t0.clicked.connect(
            lambda: self.plot_corr_delay(remove_correction=True))
        self.tw_process_view.pb_apply_t0.clicked.connect(
            lambda: self.get_t0(apply2model=True))

        self.tw_process_view.pb_show_filter.clicked.connect(
            lambda: self.get_filter(apply2model=False))
        self.tw_process_view.pb_apply_savgol.clicked.connect(
            lambda: self.get_filter(apply2model=True))
        self.tw_process_view.pb_delete_filter.clicked.connect(
            lambda: self.plot_corr_delA(remove_correction=True))

        self.tw_select_view.pb_clear_ds.clicked.connect(
            lambda _, ds=self.ds: self.preproc_controller.clear_ds(ds=self.ds))

        # -------- listen to model event signals ---------------------------------------------------
        models = (self.ta_model, self.ta_model_ds1,
                  self.ta_model_ds2, self.ta_model_ds3,)
        for i in models:
            i.data_changed.connect(self.queue_update_GUI)

    def queue_update_GUI(self) -> None:
        ''' called, if the model path or data is changed. GUI update waits till tab is selected '''
        self.need_GUI_update = True
        if self.tab.currentIndex() == 1:
            self.update_GUI()

    def update_GUI(self) -> None:
        ''' function called directly by the main window everytime the Tab is clicked
        or if the Tab is active and data was changed (handled by queue_update_GUI).
        Tab is updated if needed (handled by the need_GUI_update boolean). '''
        if self.need_GUI_update:
            self.plot_data()
            self.need_GUI_update = False

    def update_config(self) -> None:
        '''updates configuration and standard values of QWidgets'''
        self.config.add_handler('preproc_le_linlog',
                                self.tw_select_view.le_linlog)
        self.config.add_handler('preproc_le_trim_x_min',
                                self.tw_process_view.le_xmin)
        self.config.add_handler('preproc_le_trim_x_max',
                                self.tw_process_view.le_xmax)
        self.config.add_handler('preproc_le_trim_y_min',
                                self.tw_process_view.le_ymin)
        self.config.add_handler('preproc_le_trim_y_max',
                                self.tw_process_view.le_ymax)
        self.config.add_handler(
            'preproc_le_min_resample', self.tw_process_view.le_min_resample)
        self.config.add_handler(
            'preproc_le_max_resample', self.tw_process_view.le_max_resample)
        self.config.add_handler(
            'preproc_le_resample_factor', self.tw_process_view.le_resample_factor)
        self.config.add_handler(
            'preproc_cb_resample_axis', self.tw_process_view.cb_resample_axis)
        self.config.add_handler(
            'preproc_cb_resample_method', self.tw_process_view.cb_resample_method)
        self.config.add_handler(
            'preproc_sb_regularize_points', self.tw_process_view.sb_regularize_points)
        self.config.add_handler(
            'preproc_cb_regularize_axis', self.tw_process_view.cb_regularize_axis)
        self.config.add_handler(
            'preproc_cb_regularize_method', self.tw_process_view.cb_regularize_method)
        self.config.add_handler(
            'preproc_le_chirp_threshold', self.tw_process_view.le_threshold_chirp)
        self.config.add_handler(
            'preproc_cb_chirp_thr_unit', self.tw_process_view.cb_threshold_unit)
        self.config.add_handler(
            'preproc_le_ymin_area_background', self.tw_process_view.le_ymin_background)
        self.config.add_handler(
            'preproc_le_ymax_area_background', self.tw_process_view.le_ymax_background)
        self.config.add_handler(
            'preproc_le_corr_max_background', self.tw_process_view.le_ymax_background)
        self.config.add_handler(
            'preproc_cb_background_method', self.tw_process_view.cb_background_method)
        self.config.add_handler(
            'preproc_le_t0', self.tw_process_view.le_t0)
        self.config.add_handler(
            'preproc_cb_filter', self.tw_process_view.cb_filter)
        self.config.add_handler(
            'preproc_sb_filter_window', self.tw_process_view.sb_filter_window)
        self.config.add_handler('preproc_sb_savgol_order',
                                self.tw_process_view.sb_savgol_order)
        self.config.add_handler(
            'preproc_cb_filter_axis', self.tw_process_view.cb_filter_axis)

    def get_filter(self, apply2model: bool = False) -> None:
        '''  interprets user input, caches updated data from controller and triggers plotting.
        requests controller to save data to model if apply2model '''
        filter_idx = self.tw_process_view.cb_filter.currentIndex()
        filter_axis = self.tw_process_view.cb_filter_axis.currentIndex()
        window = self.tw_process_view.sb_filter_window.value()
        order = self.tw_process_view.sb_savgol_order.value()
        if filter_idx == 0:
            if order >= window:
                self.preproc_controller.call_statusbar("error", msg.Error.e10)
                return

            try:
                self.buffer_dataZ = self.preproc_controller.savgol_filter_data(
                    ds=self.ds, window=window, order=order, filter_axis=filter_axis)
            except exc.NoDataError:  # no raw-data given
                self.preproc_controller.call_statusbar("error", msg.Error.e05)
                return

        elif filter_idx == 1:  # median
            try:
                self.buffer_dataZ = self.preproc_controller.moving_filter_data(
                    ds=self.ds, method='median', size=window, filter_axis=filter_axis)
            except exc.NoDataError:  # no raw-data given
                self.preproc_controller.call_statusbar("error", msg.Error.e05)
                return

        elif filter_idx == 2:  # average
            try:
                self.buffer_dataZ = self.preproc_controller.moving_filter_data(
                    ds=self.ds, method='average', size=window, filter_axis=filter_axis)
            except exc.NoDataError:
                self.preproc_controller.call_statusbar("error", msg.Error.e05)
                return

        if apply2model:
            self.preproc_controller.apply_filter(
                ds=self.ds, corrected_dsZ=self.buffer_dataZ, filter_idx=filter_idx,
                filter_axis=filter_axis, window=window, order=order)
        else:
            self.plot_corr_delA()
            self.preproc_controller.call_statusbar("info", msg.Status.s14)

    def get_t0(self, apply2model: bool = False) -> None:
        '''  interprets user input, caches updated data from controller and triggers plotting.
        requests controller to save data to model if apply2model '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return
        try:
            self.t0 = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_t0.text())

        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e02)
            self.t0 = None
        if self.t0 is None:
            self.preproc_controller.call_statusbar("error", msg.Error.e02)
            return
        try:
            self.new_dsY = self.preproc_controller.correct_t0(
                ds=self.ds, t0=self.t0)
        except exc.NoDataError:

            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return
        if apply2model:
            self.preproc_controller.apply_t0(
                ds=self.ds, new_dsY=self.new_dsY, t0=self.t0)
        else:
            self.plot_corr_delay()

    def set_t0_view(self) -> None:
        ''' helper that triggers time-zero view '''
        if self.sender():
            # function is not executed if manual chirp button is deactivated
            if self.sender().objectName() == "manual_chirp":
                if not self.tw_process_view.pb_manually_chirp.isChecked():
                    return
        self.tw_select_view.rb_manual_view.setChecked(True)  # resets view
        self.tw_select_view.rb_t0_view.setChecked(True)  # triggers get_view -> update_view

    def get_regularized_data(self) -> None:
        '''  interprets user input, caches updated data from controller and triggers plotting. '''
        self.buff_resampled_data = {}
        if self.preproc_controller.verify_rawdata() is False:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            self.clear_gui()
            return

        datapoints = self.tw_process_view.sb_regularize_points.value()
        axis = self.tw_process_view.cb_regularize_axis.currentText()
        method = self.tw_process_view.cb_regularize_method.currentText()

        self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ = self.preproc_controller.regularize_grid(
            ds=self.ds, datapoints=datapoints, axis_str=axis, method=method)

        self.buff_resampled_data['meta'] = f'regularized {axis} axis to evenly spaced {datapoints} datapoints by using {method} interpolation.'
        self.buff_resampled_data['wavelength'] = self.buffer_dataX
        self.buff_resampled_data['delay'] = self.buffer_dataY
        self.buff_resampled_data['delA'] = self.buffer_dataZ

        self.plot_resampled()
        self.preproc_controller.call_statusbar("info", msg.Status.s25)

    def get_resampled_data(self) -> None:
        '''  interprets user input, caches updated data from controller and triggers plotting. '''
        self.buff_resampled_data = {}

        if self.preproc_controller.verify_rawdata() is False:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            self.clear_gui()
            return

        try:
            factor = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_resample_factor.text())
        except ValueError:  # falls back to default if wrong input
            self.preproc_controller.call_statusbar("error", msg.Error.e45)
            return
        if not factor:
            self.preproc_controller.call_statusbar("error", msg.Error.e45)
            return

        try:
            min_val = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_min_resample.text())
        except ValueError:  # falls back to default if wrong input
            min_val = None
        try:
            max_val = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_max_resample.text())
        except ValueError:  # falls back to default if wrong input
            max_val = None

        interval = (min_val, max_val)
        resample_delay = True if self.tw_process_view.cb_resample_axis.currentIndex() == 0 else False
        method = self.tw_process_view.cb_resample_method.currentText()
        try:
            self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ = self.preproc_controller.resample_grid(
                ds=self.ds, factor=abs(factor), interval=interval,
                resample_delay=resample_delay, method=method)
        except exc.InputLogicError:
            self.preproc_controller.call_statusbar("error", msg.Error.e06)
            return

        axis = 'delay time' if resample_delay else 'wavelength'
        min_val = 'lowest value' if not min_val else f'{min_val:.0}'
        max_val = 'highest value' if not max_val else f'{max_val:.0}'
        meta = f'resampled {axis} from {min_val} to {max_val} by multiplying by {factor} and using {method} interpolation.'
        self.buff_resampled_data['meta'] = meta
        self.buff_resampled_data['wavelength'] = self.buffer_dataX
        self.buff_resampled_data['delay'] = self.buffer_dataY
        self.buff_resampled_data['delA'] = self.buffer_dataZ

        self.plot_resampled()
        self.preproc_controller.call_statusbar("info", msg.Status.s25)

    def apply_resampling(self) -> None:
        ''' reads the cached self.buff_backgrnd_corr dict and calls the controller
        to change model delA data and metadata '''
        if not self.buff_resampled_data:
            self.preproc_controller.call_statusbar("error", msg.Error.e47)
            return
        self.preproc_controller.apply_resampling(
            ds=self.ds, x=self.buff_resampled_data['wavelength'],
            y=self.buff_resampled_data['delay'],
            z=self.buff_resampled_data['delA'], meta=self.buff_resampled_data['meta'])

    def apply_background_correction(self) -> None:
        ''' reads the cached self.buff_backgrnd_corr dict and calls the controller
        to change model delA data and metadata '''
        if not self.buff_backgrnd_corr:
            self.preproc_controller.call_statusbar("error", msg.Error.e41)
            return
        if self.buff_backgrnd_corr['type'] == 'area':
            self.preproc_controller.apply_background(corr_type=self.buff_backgrnd_corr['type'],
                                                     ds=self.ds,
                                                     corrected_dsZ=self.buff_backgrnd_corr['data'],
                                                     method=self.buff_backgrnd_corr['method'],
                                                     area_lim=self.buff_backgrnd_corr['area_lim'],
                                                     ymax=self.buff_backgrnd_corr['y_max'])
        elif self.buff_backgrnd_corr['type'] == 'file':
            self.preproc_controller.apply_background(corr_type=self.buff_backgrnd_corr['type'],
                                                     ds=self.ds, corrected_dsZ=self.buff_backgrnd_corr['data'],
                                                     meta_file=self.buff_backgrnd_corr['file'],
                                                     meta_ds=self.buff_backgrnd_corr['ds'],
                                                     meta_background=self.buff_backgrnd_corr['background'])
        elif self.buff_backgrnd_corr['type'] == 'solvent':
            self.preproc_controller.apply_background(corr_type=self.buff_backgrnd_corr['type'],
                                                     ds=self.ds, corrected_dsZ=self.buff_backgrnd_corr['data'])

    def get_area_background_correction(self) -> None:
        '''
        triggered by pb_corr_background: reads area limits and method, caches 
        the correcded data and triggers plotting of corrected background.

        Returns
        -------
        None.

        '''
        self.buff_backgrnd_corr = {}
        if self.preproc_controller.verify_rawdata() is False:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            self.clear_gui()
            return

        method = self.tw_process_view.cb_background_method.currentText()

        # limits of background area
        ymin_str = self.tw_process_view.le_ymin_background.text()
        ymax_str = self.tw_process_view.le_ymax_background.text()

        # background will be substracted up to this limit
        corr_max_str = self.tw_process_view.le_corr_max_background.text()
        try:
            y_min = utils.Converter.convert_str_input2float(ymin_str)
        except ValueError:  # falls back to default if wrong input
            y_min = None
        try:
            y_max = utils.Converter.convert_str_input2float(ymax_str)
        except ValueError:  # falls back to default if wrong input
            y_max = None

        if y_max is None:
            y_max = -0.5e-12
        try:
            corr_max = utils.Converter.convert_str_input2float(corr_max_str)
        except ValueError:  # falls back to default if wrong input
            corr_max = None

        try:
            self.buffer_dataZ = self.preproc_controller.substract_area(
                self.ds, method, y_min, y_max, corr_max)
        except exc.InputLogicError:
            self.preproc_controller.call_statusbar("error", msg.Error.e06)
            return
        self.buff_backgrnd_corr['type'] = 'area'
        self.buff_backgrnd_corr['data'] = self.buffer_dataZ
        self.buff_backgrnd_corr['method'] = method
        self.buff_backgrnd_corr['area_lim'] = [y_min, y_max]
        self.buff_backgrnd_corr['y_max'] = corr_max
        self.plot_corr_delA()
        self.preproc_controller.call_statusbar("info", msg.Status.s08)

    def get_background_from_solvent(self) -> None:
        '''
        triggered by pb_corr_from_solvent: reads solvent data, caches
        the correcded data and triggers plotting of corrected background.

        Returns
        -------
        None.

        '''
        self.buff_backgrnd_corr = {}
        try:
            _, _, background = self.preproc_controller.get_solvent()
        except exc.NoDataError:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return
        except KeyError:
            self.preproc_controller.call_statusbar("error", msg.Error.e43)
            return
        try:
            self.buffer_dataZ = self.preproc_controller.substract_surface(
                self.ds, background)
        except exc.NoDataError:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return
        except exc.InputLogicError:
            self.preproc_controller.call_statusbar("error", msg.Error.e42)
            return
        self.buff_backgrnd_corr['type'] = 'solvent'
        self.buff_backgrnd_corr['data'] = self.buffer_dataZ

        self.plot_corr_delA()
        self.preproc_controller.call_statusbar("info", msg.Status.s08)

    def get_background_from_file(self) -> None:
        '''
        triggered by pb_fromfile_background: reads project, caches
        the correcded data and triggers plotting of corrected background.

        Returns
        -------
        None.

        '''
        self.buff_backgrnd_corr = {}
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', filter="*.hdf5")
        if not filename:
            return
        project_path = Path(filename)
        ds = self.tw_process_view.cb_fromfile_background.currentText()
        try:
            with h5py.File(project_path, "r") as p:

                ta_data_group = p.get("TA Data")
                if ta_data_group is None:
                    self.preproc_controller.call_statusbar(
                        'error', msg.Error.e37)
                    return
                if ds == 'raw':
                    background = p['TA Data/raw Data/delA'][()]
                else:

                    background = p[f'TA Data/{ds}/Intensity'][()]
        except (UnboundLocalError, KeyError):
            self.preproc_controller.call_statusbar("error", msg.Error.e37)
            return
        except Exception:
            logger.exception("unknown exception occurred")
            self.preproc_controller.call_statusbar("error", msg.Error.e01)
            return
        try:
            self.buffer_dataZ = self.preproc_controller.substract_surface(
                self.ds, background)
        except exc.NoDataError:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return
        except exc.InputLogicError:
            self.preproc_controller.call_statusbar("error", msg.Error.e42)
            return
        self.buff_backgrnd_corr['type'] = 'file'
        self.buff_backgrnd_corr['data'] = self.buffer_dataZ
        self.buff_backgrnd_corr['background'] = background
        self.buff_backgrnd_corr['file'] = project_path.stem
        self.buff_backgrnd_corr['ds'] = ds

        self.plot_corr_delA()
        self.preproc_controller.call_statusbar("info", msg.Status.s08)

    def get_chirp_from_file(self) -> None:
        ''' reads in HDF file, looks for chirp_coeff and plots the correction '''
        filename, _ = QFileDialog.getOpenFileName(
            self, 'Open Project', filter="*.hdf5")
        if not filename:
            return
        project_path = Path(filename)
        ds = self.tw_process_view.cb_fromfile_ds_chirp.currentText()

        with h5py.File(project_path, "r") as p:

            ta_data_group = p.get("TA Data")
            if ta_data_group is None:
                self.preproc_controller.call_statusbar('error', msg.Error.e37)
                return

            ds_group = ta_data_group.get(ds)
            if ds_group is None:
                self.preproc_controller.call_statusbar('error', msg.Error.e37)
                return

            if 'chirp_coeffs' in ds_group.attrs:
                self.del_chirp_data()
                self.chirp_coeff = ds_group.attrs['chirp_coeffs']
                self.preproc_controller.call_statusbar('info', msg.Status.s22)
            else:
                self.preproc_controller.call_statusbar('error', msg.Error.e07)
                return
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return

        self.set_t0_view()
        self.chirp_plot_fit = self.ax.plot(self.buffer_dataX, np.polynomial.polynomial.polyval(
            self.buffer_dataX, self.chirp_coeff), color="C{}".format(0),)

    def get_corrected_chirp(self) -> None:
        ''' hands the cached chirp_coeff to the controller,
        gets the corrected spectrum and triggers plotting '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return
        try:
            self.buffer_dataZ = self.preproc_controller.calculate_chirp_corrected_dsZ(
                ds=self.ds, chirp_coeff=self.chirp_coeff)
        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e07)
            return
        self.plot_corr_delA()

    def plot_resampled(self, remove_correction: bool = False) -> None:
        ''' plots the cached resamped data or removes it from the Canvas if remove_correction is True '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return
        if remove_correction:
            self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ = self.preproc_controller.get_ds_data(
                self.ds)
            self.buff_resampled_data = {}

        try:
            self.pcolormesh_plot.remove()
        except (ValueError, AttributeError):
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return

        # Create the new 2D pcolormesh plot.
        self.X, self.Y = np.meshgrid(self.buffer_dataX, self.buffer_dataY)
        norm = self.sc._last_norms.get(self.ax, self.normalization)
        self.pcolormesh_plot = self.ax.pcolormesh(
            self.X, self.Y, self.buffer_dataZ, shading='auto', norm=norm)
        if hasattr(self, 'cb'):
            self.cb.update_normal(self.pcolormesh_plot)
        else:
            self.cb = self.fig.colorbar(
                mappable=self.pcolormesh_plot,
                cax=self.cax,
                label=msg.Labels.delA,
                shrink=0.6,
                location='right'
            )
            self.cb.minorticks_on()

        self.sc.axes_mapping = {self.ax: (self.pcolormesh_plot, self.cb)}
        self.cursor.update_data(
            self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ)
        self.sc.draw()

    def plot_corr_delA(self, remove_correction: bool = False) -> None:
        ''' plots the cached corrected data or removes it from the Canvas if remove_correction is True '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return
        if remove_correction:
            _, _, self.buffer_dataZ = self.preproc_controller.get_ds_data(self.ds)
            self.chirp_coeff = None
            self.buff_backgrnd_corr = {}
            self.del_chirp_data()
            self.preproc_controller.call_statusbar("info", msg.Status.s21)

        # Create the new 2D pcolormesh plot.
        try:
            self.pcolormesh_plot.remove()
        except (ValueError, AttributeError):
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return
        self.buffer_dataX, self.buffer_dataY, _ = self.preproc_controller.get_ds_data(self.ds)
        self.X, self.Y = np.meshgrid(self.buffer_dataX, self.buffer_dataY)
        norm = self.sc._last_norms.get(self.ax, self.normalization)
        self.pcolormesh_plot = self.ax.pcolormesh(
            self.X, self.Y, self.buffer_dataZ, shading='auto', norm=norm)
        if hasattr(self, 'cb'):
            self.cb.update_normal(self.pcolormesh_plot)
        else:
            self.cb = self.fig.colorbar(
                mappable=self.pcolormesh_plot,
                cax=self.cax,
                label=msg.Labels.delA,
                shrink=0.6,
                location='right'
            )
            self.cb.minorticks_on()

        self.sc.axes_mapping = {self.ax: (self.pcolormesh_plot, self.cb)}
        self.cursor.update_data(
            self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ)
        self.sc.draw()

    def plot_corr_delay(self, remove_correction: bool = False) -> None:
        ''' plots the cached corrected delay or removes it from the Canvas if remove_correction is True '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return
        if remove_correction:
            _, self.new_dsY, _ = self.preproc_controller.get_ds_data(self.ds)

        try:
            self.pcolormesh_plot.remove()
        except (ValueError, AttributeError):
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return

        # Create the new 2D pcolormesh plot.
        self.buffer_dataX, _, self.buffer_dataZ = self.preproc_controller.get_ds_data(self.ds)
        self.X, self.Y = np.meshgrid(self.buffer_dataX, self.new_dsY)
        norm = self.sc._last_norms.get(self.ax, self.normalization)
        self.pcolormesh_plot = self.ax.pcolormesh(
            self.X, self.Y, self.buffer_dataZ, shading='auto', norm=norm)
        if hasattr(self, 'cb'):
            self.cb.update_normal(self.pcolormesh_plot)
        else:
            self.cb = self.fig.colorbar(
                mappable=self.pcolormesh_plot,
                cax=self.cax,
                label=msg.Labels.delA,
                shrink=0.6,
                location='right'
            )
            self.cb.minorticks_on()

        self.sc.axes_mapping = {self.ax: (self.pcolormesh_plot, self.cb)}
        self.cursor.update_data(
            self.buffer_dataX, self.new_dsY, self.buffer_dataZ)
        self.set_t0_view()
        self.sc.draw()

    def _manual_set_chirp(self, event: MouseEvent) -> None:
        ''' triggered by the Canvas if user manually sets chirp points.
        left mouse button click will add the coordinates to self.chirp_points, right click deletes it '''

        # initiate scatter plot to be populated in manual selection
        if not hasattr(self, 'chirp_plot_points'):
            self._plot_chirp()

        buffer_chirp_points = self.chirp_plot_points.get_offsets()

        # find new chirp point in data closest to input click
        idx_x_new_chirp_point = (
            np.abs(self.buffer_dataX - event.xdata)).argmin()
        idx_y_new_chirp_point = (
            np.abs(self.buffer_dataY - event.ydata)).argmin()
        new_chirp_point = np.array(
            [self.buffer_dataX[idx_x_new_chirp_point], self.buffer_dataY[idx_y_new_chirp_point]])

        if event.button == 1:
            # insert new scatter point into chirp points array at leftclick position
            if new_chirp_point[0] not in buffer_chirp_points[:, 0]:

                new_chirp_points = np.insert(
                    buffer_chirp_points, 0, new_chirp_point, axis=0)
                new_chirp_points = new_chirp_points[np.argsort(
                    new_chirp_points[:, 0])]
                self.chirp_plot_points.set_offsets(new_chirp_points)
                self.sc.draw_idle()

        elif event.button == 3:
            # remove chirp point from chirp points array, if found within 2% of rightclick x-coordinates
            low_lim = new_chirp_point[0] - new_chirp_point[0] * 0.01
            high_lim = new_chirp_point[0] + new_chirp_point[0] * 0.01

            for x in buffer_chirp_points[:, 0]:
                if x >= low_lim and x <= high_lim:

                    new_chirp_points = np.delete(buffer_chirp_points, np.where(
                        buffer_chirp_points[:, 0] == x), axis=0)
                    self.chirp_plot_points.set_offsets(new_chirp_points)
            self.sc.draw_idle()

        self.chirp_points = new_chirp_points

    def fit_chirp(self) -> None:
        ''' forwards the chirp_points to the controller to get the fitted chirp_coeff,
        triggers plotting of chirp line '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return
        if not hasattr(self, 'chirp_points'):
            self.preproc_controller.call_statusbar("error", msg.Error.e08)
            return
        if len(self.chirp_points) < 5:
            self.preproc_controller.call_statusbar("error", msg.Error.e09)
            return

        self.chirp_coeff = self.preproc_controller.fit_chirp(self.chirp_points)
        self._plot_chirp()
        self.preproc_controller.call_statusbar("info", msg.Status.s10)

    def remove_chirp(self) -> None:
        ''' removes the complete chirp correction '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return
        self.del_chirp_data()
        self._plot_chirp()
        self.preproc_controller.call_statusbar("info", msg.Status.s11)

    def del_chirp_data(self) -> None:
        ''' deletes the complete chirp fit points and coefficients '''
        if hasattr(self, 'chirp_plot_points') and self.chirp_plot_points is not None:
            self.chirp_plot_points.remove()
        if hasattr(self, 'chirp_plot_fit') and self.chirp_plot_fit is not None:
            for line in self.chirp_plot_fit:
                if line in self.ax.lines:
                    line.remove()
        if hasattr(self, 'chirp_points'):
            del self.chirp_points

        if hasattr(self, 'chirp_plot_points'):
            del self.chirp_plot_points
        self.chirp_coeff = None

    def auto_set_chirp(self) -> None:
        ''' interpretes user input and calls the controller for the auto chirp_correction  '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return

        try:
            self.threshold_chirp = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_threshold_chirp.text())
        except ValueError:  # falls back to default if wrong input
            self.preproc_controller.call_statusbar("error", msg.Error.e02)
            self.threshold_chirp = None
            return

        try:

            self.chirp_points = self.preproc_controller.autofind_chirp(
                self.ds, self.threshold_chirp,
                self.tw_process_view.cb_threshold_unit.currentIndex())

        except exc.NoDataError:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return
        except ValueError:  # thrown if chirp_coeff not found?
            self.preproc_controller.call_statusbar("error", msg.Error.e08)

            return
        self.fit_chirp()

    def _plot_chirp(self) -> None:
        ''' plots the chirp line using the cached chirp_points '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return

        # remove already drawn chirp fit lines and points
        if hasattr(self, 'chirp_plot_points') and self.chirp_plot_points is not None:
            self.chirp_plot_points.remove()
        if hasattr(self, 'chirp_plot_fit') and self.chirp_plot_fit is not None:
            for line in self.chirp_plot_fit:
                if line in self.ax.lines:
                    line.remove()

        # initiate scatter plot to be populated in manual selection
        if not hasattr(self, 'chirp_points'):

            self.chirp_points = np.zeros([1, 2])
            self.chirp_plot_points = self.ax.scatter(
                self.chirp_points[:, 0], self.chirp_points[:, 1], color="C{}".format(0),
                marker="x", zorder=3)
            self.chirp_points = np.delete(self.chirp_points, 0, axis=0)
            self.chirp_plot_points.set_offsets(self.chirp_points)
        # or just plot the fit
        else:
            self.chirp_plot_points = self.ax.scatter(
                self.chirp_points[:, 0], self.chirp_points[:, 1],
                color="C{}".format(0), marker="x", zorder=3)

            self.chirp_plot_fit = self.ax.plot(self.buffer_dataX, np.polynomial.polynomial.polyval(
                self.buffer_dataX, self.chirp_coeff), color="C{}".format(0),)

        self.set_t0_view()

    def update_dataset(self, state: bool) -> None:
        '''
        detects if different dataset is set with radiobuttons and request replotting.

        Parameters
        ----------
        state : bool
            state of rb_ds_1, rb_ds_2, rb_ds_3.

        Returns
        -------
        None.

        '''

        if state:

            if self.sender().objectName() == "ds1":
                self.ds = '1'
            elif self.sender().objectName() == "ds2":
                self.ds = '2'

            elif self.sender().objectName() == "ds3":
                self.ds = '3'

            self.chirp_coeff = None
            self.buff_backgrnd_corr = {}
            self.buff_resampled_data = {}
            self.plot_data()

    def get_view(self, state: bool) -> None:
        '''
        helper fcn that updates cached view depending on selected view radiobutton
        and requests update_view()

        Parameters
        ----------
        state : bool
            state of rb_full_view, rb_t0_view.

        Returns
        -------
        None.

        '''

        if state:
            self.view = self.sender().objectName()

            self.update_view()

    def get_trimm(self) -> None:
        '''
        converts trimm GUI inputs to floats and calls update_view with the new setting to update the Canvas.

        empty, invalid or min value is larger than max value sets value to None
        which results in no change and raises an error message to the status bar.

        Called by self.tw_process_view.LINEEDIT.editingFinished and calls update_view()
        Returns
        -------
        None.

        '''

        try:
            self.trimm_xmin = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_xmin.text())

        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e02)
            self.trimm_xmin = None
        try:
            self.trimm_xmax = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_xmax.text())
        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e02)
            self.trimm_xmax = None

        if self.trimm_xmin is not None and self.trimm_xmax is not None:
            if (self.trimm_xmin >= self.trimm_xmax):
                self.trimm_xmin, self.trimm_xmax = None, None
                self.preproc_controller.call_statusbar("error", msg.Error.e06)

        try:
            self.trimm_ymin = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_ymin.text())
        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e02)
            self.trimm_ymin = None
        try:
            self.trimm_ymax = utils.Converter.convert_str_input2float(
                self.tw_process_view.le_ymax.text())
        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e02)
            self.trimm_ymax = None

        if self.trimm_ymin is not None and self.trimm_ymax is not None:
            if (self.trimm_ymin >= self.trimm_ymax):
                self.trimm_ymin, self.trimm_ymax = None, None
                self.preproc_controller.call_statusbar("error", msg.Error.e06)

        self.view = 'manual_view'
        self.tw_select_view.rb_manual_view.setChecked(
            True)
        self.update_view()

    def update_view(self) -> None:
        '''
        updates the cached self.sc.fig axes with the cached limits. No slow redraw.
        Called by   - plot_data()
                    - get_trimm()
                    - get_view()
                    - set_t0_view()
                    - self.tw_select_view.le_linlog.editingFinished


        Returns
        -------
        None
            DESCRIPTION.

        '''
        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return

        try:

            self.linthresh = utils.Converter.convert_str_input2float(
                self.tw_select_view.le_linlog.text())
            if self.linthresh is None:
                self.ax.set_yscale('linear')
                self.ax_kin.set_xscale('linear')
            else:
                self.ax.set_yscale(
                    'symlog', linthresh=self.linthresh, linscale=1)

                self.ax_kin.set_xscale(
                    'symlog', linthresh=self.linthresh, linscale=1)

            if self.view == 'full_view':
                self.ax.set_xlim(np.min(self.buffer_dataX),
                                 np.max(self.buffer_dataX))
                self.ax.set_ylim(np.min(self.buffer_dataY),
                                 np.max(self.buffer_dataY))
                self.ax_kin.set_xlim(self.ax.get_ylim())
                self.ax_delA.set_xlim(self.ax.get_xlim())

            elif self.view == 't0_view':
                self.ax.set_xlim(np.min(self.buffer_dataX),
                                 np.max(self.buffer_dataX))
                self.ax.set_ylim(np.min(self.buffer_dataY), -
                                 np.min(self.buffer_dataY))
                self.ax_delA.set_xlim(self.ax.get_xlim())
                self.ax_kin.set_xlim(
                    np.min(self.buffer_dataY), -np.min(self.buffer_dataY))
            else:  # manual from get trimm
                self.ax.set_xlim(self.trimm_xmin, self.trimm_xmax)
                self.ax.set_ylim(self.trimm_ymin, self.trimm_ymax)
                self.ax_kin.set_xlim(self.ax.get_ylim())
                self.ax_delA.set_xlim(self.ax.get_xlim())

            self.ax.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
            self.ax.yaxis.set_major_formatter(self.sc.delay_formatter0)
            self.ax_delA.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
            self.ax_kin.xaxis.set_major_formatter(self.sc.delay_formatter0)
            self.sc.draw_idle()
        except AttributeError:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e02)

    def plot_data(self) -> None:
        '''
        plots the dataset using the cached self.ds. All the other cached parameters are ignored.
        Called by   - update_dataset()
                    - update_gui()

        Calls       - update_view()

        Returns
        -------
        None.

        '''
        def onclick(event):

            if event.button == 2:  # MMB
                self.cursor.on_mouse_press(event)

            elif (event.button == 1 or event.button == 3) and self.tw_process_view.pb_manually_chirp.isChecked():
                if not event.inaxes == self.ax:
                    return
                self._manual_set_chirp(event)

            else:
                return

        if self.preproc_controller.verify_rawdata() is False:
            self.clear_gui()
            return

        if hasattr(self, 'chirp_points'):
            del self.chirp_points

        if hasattr(self, 'chirp_plot_points'):
            del self.chirp_plot_points

        self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ = self.preproc_controller.get_ds_data(
            ds=self.ds)

        # Create or reuse the canvas.
        if not hasattr(self, 'sc'):
            self.sc = utils.PlotCanvas(self, width=5, height=5, dpi=100)
            self.toolbar = NavigationToolbar2QT(self.sc)
            layout = QVBoxLayout()
            layout.addWidget(self.toolbar)
            layout.addWidget(self.sc)
            self.tw_canvas.w_canvas.hide()
            self.tw_canvas.w_canvas.setParent(None)
            self.tw_canvas.w_canvas = QWidget(self.tw_canvas)
            self.tw_canvas.w_canvas.setLayout(layout)
            self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas)

            self.fig = self.sc.fig
            self.ax = self.fig.add_subplot(2, 2, 3)
            self.ax_delA = self.fig.add_subplot(2, 2, 1)
            self.ax_kin = self.fig.add_subplot(2, 2, 4)

            divider = make_axes_locatable(self.ax)
            self.cax = divider.append_axes('right', size='5%', pad=0.05)
            self.normalization = colors.TwoSlopeNorm(vmin=-5, vmax=5, vcenter=0)

            self.sc._event_connections = []

        else:
            # Clear the figure to update the plot.
            for ax in (self.ax, self.ax_delA, self.ax_kin):
                ax.cla()

        try:
            self.X, self.Y = np.meshgrid(self.buffer_dataX, self.buffer_dataY)
            norm = self.sc._last_norms.get(self.ax, self.normalization)
            self.pcolormesh_plot = self.ax.pcolormesh(
                self.X, self.Y, self.buffer_dataZ, shading='auto', norm=norm)

        except IndexError:
            return
        except ValueError:
            self.preproc_controller.call_statusbar("error", msg.Error.e05)
            return
        except Exception:
            logger.exception("unknown exception occurred")
            self.preproc_controller.call_statusbar("error", msg.Error.e01)
            return
        self.ax.set_title('move the crosshair by dragging the middle mouse button')
        self.ax.set_ylabel(msg.Labels.delay)
        self.ax.set_xlabel(msg.Labels.wavelength)

        if hasattr(self, 'cb'):
            self.cb.update_normal(self.pcolormesh_plot)
        else:
            self.cb = self.fig.colorbar(
                mappable=self.pcolormesh_plot,
                cax=self.cax,
                label=msg.Labels.delA,
                shrink=0.6,
                location='right'
            )
            self.cb.minorticks_on()

        self.sc.axes_mapping = {
            self.ax: (self.pcolormesh_plot, self.cb)}  # mapping needed for applying scroll zooming

        self.tw_select_view.rb_manual_view.setChecked(True)
        self.tw_select_view.rb_full_view.setChecked(
            True)  # triggers update view

        self.ax_delA.set_xlabel(msg.Labels.wavelength)
        self.ax_delA.set_ylabel(msg.Labels.delA)
        self.ax_delA.axhline(y=0, color=self.ax.xaxis.label.get_color(
        ), linestyle="--", linewidth=1, zorder=0)
        self.ax_kin.axhline(y=0, color=self.ax.xaxis.label.get_color(
        ), linestyle="--", linewidth=1, zorder=0)
        self.ax_kin.set_xlabel(msg.Labels.delay)
        self.ax_kin.set_ylabel(msg.Labels.delA)

        # Disconnect previous event connections.
        if hasattr(self.sc, '_event_connections'):
            for cid in self.sc._event_connections:
                self.sc.mpl_disconnect(cid)
            self.sc._event_connections = []

        if hasattr(self, 'cursor'):
            try:

                del self.cursor
            except AttributeError:  # if clear button is pressed multiple times
                pass

        self.cursor = utils.ClickCursor(
            self.ax, self.ax_delA, self.ax_kin,
            self.buffer_dataX, self.buffer_dataY, self.buffer_dataZ,
            self.sc.nm_formatter_ax, self.sc.delay_formatter0)

        # Now, connect the event handlers and store their connection IDs.
        cid = self.sc.mpl_connect('button_press_event', onclick)
        self.sc._event_connections.append(cid)
        cid = self.sc.mpl_connect(
            'motion_notify_event', lambda event: self.cursor.on_mouse_move(event))
        self.sc._event_connections.append(cid)
        cid = self.sc.mpl_connect(
            'button_release_event', lambda event: self.cursor.on_mouse_release(event))
        self.sc._event_connections.append(cid)
        cid = self.sc.mpl_connect(
            'scroll_event', lambda event: self.sc._zoom_TA(event, self.sc.axes_mapping))
        self.sc._event_connections.append(cid)

        self.sc.draw()

    def clear_gui(self) -> None:
        '''
        called, when raw data in import tab is cleared. Refreshes GUI


        Returns
        -------
        None.

        '''

        if hasattr(self, "fig"):
            del self.fig
            del self.sc
        try:

            del self.cursor
        except AttributeError:  # if clear button is pressed multiple times
            return

        self.tw_canvas.w_canvas.hide()
        self.tw_canvas.w_canvas.setParent(None)  # remove widget from layout
        self.tw_canvas.w_canvas.deleteLater()
        self.tw_canvas.w_canvas = QLabel(msg.Widgets.i06, self.tw_canvas)
        self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)
