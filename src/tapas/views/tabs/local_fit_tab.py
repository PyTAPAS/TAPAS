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

# Third‑Party Imports

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHeaderView,
    QVBoxLayout,
    QTableWidgetItem,
    QGridLayout,
    QFileDialog
)
from PyQt6.QtCore import Qt

# Matplotlib Imports
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import matplotlib.ticker as tk

# Other Third‑Party Libraries
import numpy as np
import jax.numpy as jnp
import corner

# Local Application Imports
from ...utils import utils
from ...configurations import exceptions as exc, messages as msg
from ...views.tabwidgets.local_fit_tabwidgets import InputWidget, CanvasWidget, ResultsWidget

logger = logging.getLogger(__name__)


class LocalFitTab(QWidget):
    def __init__(self, tabwidget, ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3, controller, config):
        super().__init__()
        self.tab = tabwidget
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.local_fit_controller = controller
        self.config = config
        self.ds = '1'
        self.project_path = None
        self.results_dir = None
        self.y_min, self.y_linlog, self.y_max = None, None, None
        self.z_min, self.z_max, self.z_center = None, None, 0
        self.final = None
        self.fit_results = None
        self.emcee_final_result = {}
        self.need_GUI_update = False
        self.InitUI()

    def InitUI(self):
        # -------- create Widgets ------------------------------------------------------------------
        self.tw_input = InputWidget()
        self.tw_canvas = CanvasWidget()
        self.tw_results = ResultsWidget()
        self.update_config()

        # -------- add Widgets to layout -----------------------------------------------------------
        layout = QGridLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(utils.Converter.create_scrollable_widget(
            self.tw_input, min_width=310, max_width=310, horizontal_scroll=False), 0, 0,
            alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.tw_canvas, 0, 1,)
        layout.addWidget(utils.Converter.create_scrollable_widget(
            self.tw_results, min_width=310, max_width=310, horizontal_scroll=False, ), 0, 2,
            alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

        # -------- connect Widgets to view / controller --------------------------------------------
        self.tw_input.rb_ds_1.toggled.connect(self.update_dataset)
        self.tw_input.rb_ds_2.toggled.connect(self.update_dataset)
        self.tw_input.rb_ds_3.toggled.connect(self.update_dataset)
        self.tw_input.le_wavelength.editingFinished.connect(self.plot_data)
        self.tw_input.sp_wavelength_area.editingFinished.connect(self.plot_data)

        self.tw_input.le_ymin.editingFinished.connect(self.check_delay_input)
        self.tw_input.le_ymin.editingFinished.connect(
            lambda: self.plot_data(show_fit=True))
        self.tw_input.le_ymax.editingFinished.connect(self.check_delay_input)
        self.tw_input.le_ymax.editingFinished.connect(
            lambda: self.plot_data(show_fit=True))
        self.tw_input.le_linlog_y.editingFinished.connect(
            lambda: self.plot_data(show_fit=True))
        self.tw_input.le_linlog_z.editingFinished.connect(
            lambda: self.plot_data(show_fit=True))

        self.tw_input.le_zmin.editingFinished.connect(self.check_z_input)
        self.tw_input.le_zmin.editingFinished.connect(
            lambda: self.plot_data(show_fit=True))
        self.tw_input.le_zmax.editingFinished.connect(self.check_z_input)
        self.tw_input.le_zmax.editingFinished.connect(
            lambda: self.plot_data(show_fit=True))
        self.tw_input.cb_yscale.currentIndexChanged.connect(
            lambda: self.plot_data(show_fit=True))
        self.tw_input.cb_zscale.currentIndexChanged.connect(
            lambda: self.plot_data(show_fit=True))

        self.tw_input.pb_fit.pressed.connect(self.fit_data)
        self.tw_input.pb_show_guess.pressed.connect(self.show_initial_fit)
        self.tw_results.pb_save_fit.pressed.connect(
            lambda ds=self.ds: self.local_fit_controller.save_current_fit(ds=self.ds))
        self.tw_results.pb_delete_fit.pressed.connect(self.delete_selected_fit)
        self.tw_results.tab_ds_fits.horizontalHeader(
        ).sectionClicked.connect(self.show_clicked_fit)

        self.tw_results.pb_run_emcee.pressed.connect(self.run_emcee)
        self.tw_results.pb_cancel_emcee.pressed.connect(
            self.local_fit_controller.abort_emcee)
        self.tw_results.pb_save_emcee.pressed.connect(
            lambda: self.local_fit_controller.save_emcee_result_2model(results=self.emcee_final_result))

        # -------- connect emcee worker ------------------------------------------------------------
        self.local_fit_controller.worker_progress.connect(self.emcee_update_progress)
        self.local_fit_controller.worker_results.connect(self.emcee_update_results)
        self.local_fit_controller.emcee_finished.connect(self.emcee_show_final)

        # -------- listen to model event signals ---------------------------------------------------
        models = (self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        for i in models:
            i.data_changed.connect(self.queue_update_GUI)
            i.local_fit_changed.connect(
                lambda: self.queue_update_GUI(only_fitting_list=True))

    def emcee_update_progress(self, info: str) -> None:
        ''' helper that prints emcee worker info to gui '''
        self.emcee_window.te_progress.append(info)

    def emcee_show_final(self, results: object | None, abort: bool) -> None:
        ''' triggered after final emcee run, hides progress bar, prints and caches results '''
        self.emcee_window.progress_bar.hide()
        for i in [self.tw_input, self.tw_results.w_results_output, self.tw_results.w_fitting_list,
                  self.tw_results.sb_burn, self.tw_results.sb_init, self.tw_results.sb_thin,
                  self.tw_results.sb_target_ratio, self.tw_results.pb_run_emcee, self.tw_results.pb_save_emcee]:
            i.setEnabled(True)
        if results is None:
            self.emcee_window.te_progress.append('finished without a result')
            self.emcee_final_result = {}
            return
        elif abort:
            self.emcee_window.te_progress.append(
                'finished earlier due to user input or runtime')
        else:
            self.emcee_window.te_progress.append('finished succesfully')
        report_str = self.local_fit_controller.get_emcee_print(results)
        self.emcee_window.te_results.setText(report_str)
        self.emcee_window.label_results.setText(f'results after final run:')
        self.emcee_final_result['meta']['finished_early'] = abort
        self.emcee_final_result['params'] = results.params
        self.emcee_final_result['output'] = report_str
        self.emcee_final_result['flatchain'] = results.flatchain.to_numpy().astype(
            np.float32)
        self.emcee_final_result['meta']['#samples'] = results.chain.shape[0]

    def emcee_update_results(self, results: object, run: int) -> None:
        ''' triggered after each emcee run, prints and plots results to gui '''
        report_str = self.local_fit_controller.get_emcee_print(results)
        self.emcee_window.te_results.setText(report_str)
        self.emcee_window.label_results.setText(f'results after {run} run(s):')
        self.sc = utils.PlotCanvas()
        self.fig = self.sc.fig
        self.fig.clear()
        num_params = len(results.var_names)

        square_container = utils.SquareContainer(self.sc, size=(200*num_params, 200*num_params))

        scrollable_container = utils.Converter.create_scrollable_widget(
            square_container, use_container=False)
        layout = QVBoxLayout()
        layout.addWidget(scrollable_container)

        varying_names, labels, truths = [], [], []
        for name, par in results.params.items():
            if not par.vary:
                continue
            # keep the name for slicing flatchain
            varying_names.append(name)
            truths.append(par.value)

            if name == "t0":
                labels.append(r"$t_0$")
            elif name == "IRF":
                labels.append("IRF")
            elif name.startswith("t") and name[1:].isdigit():
                labels.append(fr"$t_{name[1:]}$")
            elif name.startswith("__ln"):
                labels.append(r"ln(σ/mOD)")
            else:
                labels.append(name)        # fallback

        f = corner.corner(results.flatchain.to_numpy(), labels=labels, truths=truths, fig=self.fig)

        axes = np.array(f.axes).reshape((num_params, num_params))
        bottom_row = axes[-1, :]
        for ax in bottom_row:
            ax.xaxis.set_major_formatter(self.sc.emcee_formatter)

        left_column = axes[:, 0]
        for ax in left_column:
            ax.yaxis.set_major_formatter(self.sc.emcee_formatter)

        self.sc.draw()
        self.emcee_window.corner_canvas.hide()
        self.emcee_window.corner_canvas.setParent(None)
        self.emcee_window.corner_canvas.deleteLater()
        corner_canvas = QWidget()
        corner_canvas.setLayout(layout)
        self.emcee_window.corner_canvas = corner_canvas
        self.emcee_window.layout.addWidget(corner_canvas, 4, 0, 1, 2)
        self.emcee_window.layout.setRowStretch(4, 1)

    def run_emcee(self) -> None:
        ''' interprets user inputs, freezes the gui and requests the controler to perform emcee analysis '''
        self.emcee_final_result = {}
        self.emcee_final_result['meta'] = {}
        current_col = self.tw_results.tab_ds_fits.currentColumn()
        if current_col == -1:  # no selection
            self.local_fit_controller.call_statusbar("error", msg.Error.e36)
            return
        else:
            fit_results, ukey = self.local_fit_controller.get_fit(
                ds=self.ds, selected_fit=current_col)
        self.emcee_final_result['ukey'] = ukey
        self.emcee_final_result['ds'] = self.ds
        self.emcee_final_result['meta']['burn'] = self.tw_results.sb_burn.value()
        self.emcee_final_result['meta']['thin'] = self.tw_results.sb_thin.value()
        self.emcee_final_result['meta']['target_ratio'] = self.tw_results.sb_target_ratio.value()

        if self.fit_results is None:
            self.local_fit_controller.call_statusbar("error", msg.Error.e24)
            return
        for i in [self.tw_input, self.tw_results.w_results_output, self.tw_results.w_fitting_list,
                  self.tw_results.sb_burn, self.tw_results.sb_init, self.tw_results.sb_thin,
                  self.tw_results.sb_target_ratio, self.tw_results.pb_run_emcee, self.tw_results.pb_save_emcee]:
            i.setEnabled(False)

        self.emcee_window = CanvasWidget.create_emcee_canvas()
        self.tw_canvas.w_canvas.hide()
        self.tw_canvas.w_canvas.setParent(None)
        self.tw_canvas.w_canvas = self.emcee_window
        self.tw_canvas.view_layout.addWidget(self.emcee_window)
        self.emcee_window.title.setText(
            "+++ Do not change any internal data state during calculations +++")
        self.local_fit_controller.run_emcee(
            ds=self.ds, results=self.fit_results,  burn=self.tw_results.sb_burn.value(),
            init=self.tw_results.sb_init.value(), thin=self.tw_results.sb_thin.value(),
            target_ratio=self.tw_results.sb_target_ratio.value())

    def queue_update_GUI(self, only_fitting_list: bool = False) -> None:
        ''' called, if the raw or ds data is changed. GUI update waits till tab is selected
        if local_fit data is changed, only the fitting list will be updated (only_fitting_list = True)
        '''
        self.need_GUI_update = True
        if self.tab.currentIndex() == 5:
            self.update_GUI(only_fitting_list=only_fitting_list)

    def update_GUI(self, only_fitting_list: bool = False) -> None:
        ''' function called directly by the main window everytime the Tab is clicked
        or if the Tab is active and data was changed (handled by queue_update_GUI).
        Tab is updated if needed (handled by the need_GUI_update boolean). '''
        if self.need_GUI_update:
            if not only_fitting_list:
                self._reset_fitting_cache()
            else:
                self._update_fitting_list()
            self.need_GUI_update = False

    def export_data(self) -> None:
        ''' gets Path from file dialog, writes fit and meta data to txt '''
        if self.project_path is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Choose a file name", str(self.project_path.parent), filter='*.txt')
        else:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Choose a file name", filter='*.txt')
        if not file_name:
            self.local_fit_controller.call_statusbar("error", msg.Error.e17)
            return
        results_path = Path(file_name)

        if results_path is None:
            self.local_fit_controller.call_statusbar("error", msg.Error.e17)
            return
        meta_data = self.local_fit_controller.get_fitting_meta(
            self.fit_results)
        basic_results = self.local_fit_controller.get_fitting_print(
            self.fit_results)
        wavelength = utils.Converter.convert_str_input2float(str(self.fit_results['wavelength']))
        emcee_results = ''
        if self.emcee_final_result:
            if 'output' in self.emcee_final_result:
                emcee_results += self.emcee_final_result['output']
                emcee_results += f"\nburn: {self.emcee_final_result['meta']['burn']}, thin: {self.emcee_final_result['meta']['thin']}"

        with open(results_path, "w") as file:
            file.write(f'local fit results: {wavelength}m\n')
            file.write(meta_data)
            file.write('\n-----------------------------------------------------\n')
            file.write('\n--- Results from Jacobian Analysis: ---\n\n')
            file.write(basic_results)
            file.write('\n-----------------------------------------------------\n\n')
            file.write(emcee_results)

    def show_clicked_fit(self, column_index: int) -> None:
        '''
        prints and plots fit results of selected table column. Caches the data

        Parameters
        ----------
        column_index : int
            column index of clicked fit results table.

        Returns
        -------
        None

        '''
        self._reset_fitting_cache()

        fit_results, _ = self.local_fit_controller.get_fit(
            ds=self.ds, selected_fit=column_index)

        self.tw_results.te_results.setText(fit_results['output'])
        self.fit_results = fit_results
        if 'emcee' in fit_results:
            self.emcee_final_result = fit_results['emcee']
        self.plot_data(show_fit=True, use_wavelength=fit_results['wavelength'])
        self.local_fit_controller.call_statusbar("info", msg.Status.s32)

    def delete_selected_fit(self) -> None:
        ''' removes selevted fit form gui and requests controller to delete its data '''
        self._reset_fitting_cache()
        selected_fit = self.tw_results.tab_ds_fits.currentColumn()
        if selected_fit < 0:
            self.local_fit_controller.call_statusbar("error", msg.Error.e22)
            return
        self.local_fit_controller.delete_fit(
            ds=self.ds, selected_fit=selected_fit)
        self.local_fit_controller.call_statusbar("info", msg.Status.s33)

    def _clear_fitting_canvas(self) -> None:
        ''' cleanely deletes canvas '''
        self.tw_canvas.w_canvas.hide()
        self.tw_canvas.w_canvas.setParent(None)  # remove widget from layout
        self.tw_canvas.w_canvas.deleteLater()
        self.tw_canvas.w_canvas = QLabel(msg.Widgets.i13, self.tw_canvas)
        self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)
        self.tw_results.te_results.setText('')

    def check_delay_input(self) -> None:
        ''' retrieves, validates and caches ylims for plotting. Plotting will be triggered if changed '''
        try:
            self.y_min = utils.Converter.convert_str_input2float(
                self.tw_input.le_ymin.text())
        except ValueError:
            self.local_fit_controller.call_statusbar("error", msg.Error.e02)
            self.y_min = None
        try:
            self.y_max = utils.Converter.convert_str_input2float(
                self.tw_input.le_ymax.text())
        except ValueError:
            self.visualize_controller.call_statusbar("error", msg.Error.e02)
            self.y_max = None

        if self.y_min is not None and self.y_max is not None:
            if (self.y_min >= self.y_max):
                self.y_min, self.y_max = None, None
                self.local_fit_controller.call_statusbar(
                    "error", msg.Error.e06)

    def check_z_input(self) -> None:
        ''' retrieves, validates and caches delA lims for plotting. Plotting will be triggered if changed '''
        try:
            self.z_min = utils.Converter.convert_str_input2float(
                self.tw_input.le_zmin.text())
        except ValueError:
            self.local_fit_controller.call_statusbar("error", msg.Error.e02)
            return
        try:
            self.z_max = utils.Converter.convert_str_input2float(
                self.tw_input.le_zmax.text())
        except ValueError:
            self.local_fit_controller.call_statusbar("error", msg.Error.e02)
            return

        if self.z_min is not None and self.z_max is not None:
            if (self.z_min >= self.z_max):
                self.z_min, self.z_max = None, None
                self.local_fit_controller.call_statusbar(
                    "error", msg.Error.e06)

    def update_config(self) -> None:
        '''updates configuration and standard values of QWidgets'''
        self.config.add_handler('local_w_le_ymin', self.tw_input.le_ymin)
        self.config.add_handler('local_w_le_ymax', self.tw_input.le_ymax)
        self.config.add_handler('local_w_le_zmin', self.tw_input.le_zmin)
        self.config.add_handler('local_w_le_zmax', self.tw_input.le_zmax)
        self.config.add_handler('local_w_cb_yscale', self.tw_input.cb_yscale)
        self.config.add_handler('local_w_cb_zscale', self.tw_input.cb_zscale)
        self.config.add_handler('local_w_le_linlog_y', self.tw_input.le_linlog_y)
        self.config.add_handler('local_w_le_linlog_z', self.tw_input.le_linlog_z)
        self.config.add_handler('local_w_le_wavelength', self.tw_input.le_wavelength)
        self.config.add_handler('local_w_sb_components', self.tw_input.sb_components)
        self.config.add_handler('local_w_check_infinte', self.tw_input.check_infinte)
        self.config.add_handler('local_w_cb_model', self.tw_input.cb_model)
        self.config.add_handler('local_w_cb_method', self.tw_input.cb_method)

    def _sort_fit_params_input(self, components: int) -> None:
        ''' sorts parameter input table to increasing lifetimes. Important for sequential analysis
        if no value or invalid value given, parameter will be sorted to the end
        Parameters
        ----------
        components : int
            number of components, exclude infinite componets.

        Returns
        -------
        None.

        '''
        if components == 1:
            return  # one components does not need any sorting

        value_array = []

        # get value input for all components (row+2), handle invalid or empty as 0:
        for i in range(0, components):
            try:
                value_array.append(utils.Converter.convert_str_input2float(
                    self.tw_input.tab_fit_params.item(i + 2, 0).text()))
            except (AttributeError, ValueError):
                value_array.append(np.inf)
            if value_array[i] is None:
                value_array[i] = np.inf

        sorted_index_array = np.argsort(value_array)

        # get the input table as object array:
        table = np.zeros((components, 4), dtype=object)
        for i in range(0, components):
            for j in range(4):
                try:
                    table[i, j] = self.tw_input.tab_fit_params.item(
                        i + 2, j).text()
                except (AttributeError, ValueError):
                    table[i, j] = ''

        # set the input table according to the sorted values:
        for i in range(0, components):
            for j in range(4):
                item = QTableWidgetItem(table[sorted_index_array[i], j])
                self.tw_input.tab_fit_params.setItem(i + 2, j, item)

    def _update_fitting_list(self) -> None:
        ''' gets whole fit dict from controller and updates GUI table accordingly '''
        fit_dict = self.local_fit_controller.get_local_fit_list(ds=self.ds)

        # adjust columns:
        self.tw_results.tab_ds_fits.setColumnCount(len(fit_dict))
        for column, fit in enumerate(list(fit_dict)):
            self.tw_results.tab_ds_fits.setHorizontalHeaderItem(
                column, QTableWidgetItem(fit))
            self.tw_results.header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents)

            self.tw_results.tab_ds_fits.setCellWidget(
                0, column, QLabel(fit_dict[fit]['meta']['model']))
            self.tw_results.tab_ds_fits.setCellWidget(
                1, column, QLabel(f"{fit_dict[fit]['meta']['r2']:.3f}"))
            self.tw_results.tab_ds_fits.setCellWidget(
                2, column, QLabel(str(fit_dict[fit]['meta']['Ainf'])))
            if self.tw_results.tab_ds_fits.rowCount() < len(list(fit_dict[fit]['opt_params'])) + 3:
                self.tw_results.tab_ds_fits.setRowCount(
                    len(list(fit_dict[fit]['opt_params'])) + 3)

            for row, parameter in enumerate(list(fit_dict[fit]['opt_params'])):

                lifetime = (fit_dict[fit]['opt_params'][parameter])
                formatted_lifetime = str(tk.EngFormatter(
                    places=1, sep="\N{THIN SPACE}")(lifetime) + 's ')
                self.tw_results.tab_ds_fits.setCellWidget(
                    row + 3, column, QLabel(formatted_lifetime))
        # adjust rows:
        self.tw_results.tab_ds_fits.setVerticalHeaderLabels(
            self.tw_results.component_labels)

    def _reset_fitting_cache(self) -> None:
        ''' triggered, when data changed. resets all the cached data and Canvas '''
        self._update_fitting_list()
        self._clear_fitting_canvas()
        self.emcee_final_result = {}
        self.local_fit_controller.initialize_fitting_cache()

    def update_dataset(self, state: bool) -> None:
        ''' triggred if dataset radiobutton is clicked. updates the cached fitting values and plots the data '''
        if state:
            if self.sender().objectName() == "ds1":
                self.ds = '1'
            elif self.sender().objectName() == "ds2":
                self.ds = '2'
            elif self.sender().objectName() == "ds3":
                self.ds = '3'
            self._reset_fitting_cache()
            self.plot_data(show_fit=False)

    def show_initial_fit(self) -> None:
        ''' retrives and plots the initial guess parameters. returns diagnotsic error when failed '''

        # -------- reads input and converts it to usable parameters --------------------------------
        if not self.local_fit_controller.verify_rawdata():
            self.local_fit_controller.call_statusbar("error", msg.Error.e05)
            return
        self._sort_fit_params_input(components=self.tw_input.sb_components.value())
        try:
            params = self.local_fit_controller.get_params(QtTable=self.tw_input.tab_fit_params)
        except ValueError:
            self.tw_results.te_results.setText("Fitting did not succeed")
            return

        missing_params = [name for name, par in params.items() if np.isinf(par.value)]
        if missing_params:
            self.local_fit_controller.call_statusbar(
                'error', f"Missing initial value for: {missing_params}")
            return

        try:
            wavelength, delay, delA = self.local_fit_controller.get_data(
                ds=self.ds, wavelength_input=self.tw_input.le_wavelength.text(),
                wavelength_area=self.tw_input.sp_wavelength_area.value())
        except ValueError:
            return

        use_threshold_t0 = True if self.tw_input.cb_t0_def.currentText() == '5% Threshold' else False
        Ainf = self.tw_input.check_infinte.isChecked()
        model = self.tw_input.cb_model.currentText()
        ca_order = self.tw_input.cb_ca_order.currentIndex()

        # -------- requests controller to model the parameters -------------------------------------
        try:
            fit_results = {}
            fit_results['delA_calc'], fit_results['conc'], fit_results['Amp'],  = self.local_fit_controller.model_theta_wrapper(
                params=params, delay=delay, delA=delA, Ainf=Ainf,  model=model,
                weights=jnp.array([1]), use_threshold_t0=use_threshold_t0, substeps=10, ca_order=ca_order, output=True)
            fit_results['opt_params'] = params
            self.tw_results.te_results.setText("Initial fit succeed")
            self.local_fit_controller.call_statusbar("info", msg.Status.s30)
        except np.linalg.LinAlgError:
            self.tw_results.te_results.setText("Fitting did not succeed")
            self.local_fit_controller.call_statusbar("error", msg.Error.e23)
            return

        # -------- caches the results and plots the guess ------------------------------------------
        fit_results['meta'] = {}
        fit_results['meta']['components'] = self.local_fit_controller.get_component_labels(
            model=self.tw_input.cb_model.currentText(), Ainf=self.tw_input.check_infinte.isChecked(),
            num=self.tw_input.sb_components.value(), ca_order=ca_order, local = True)
        self.fit_results = fit_results

        try:
            self.plot_data(show_fit=True)
        except OverflowError:
            self.tw_results.te_results.setText("Fitting did not succeed.")
            self.local_fit_controller.call_statusbar("error", msg.Error.e19)

    def fit_data(self) -> None:
        ''' retrives the input parameters and informs controller to perform and evaluate the fit.
        Fit results are cached, printed and plotted to the GUI.
        '''
        # -------- reads input and converts it to usable parameters --------------------------------
        if not self.local_fit_controller.verify_rawdata():
            self.local_fit_controller.call_statusbar("error", msg.Error.e05)
            return
        self.local_fit_controller.create_ds(self.ds)  # check if ds is empty and fill with raw_data
        self._sort_fit_params_input(components=self.tw_input.sb_components.value())
        try:
            params = self.local_fit_controller.get_params(QtTable=self.tw_input.tab_fit_params)
        except ValueError:
            self.tw_results.te_results.setText("Fitting did not succeed")
            return

        use_threshold_t0 = True if self.tw_input.cb_t0_def.currentText() == '5% Threshold' else False
        input_wavelength = self.tw_input.le_wavelength.text()
        wavelength_area = self.tw_input.sp_wavelength_area.value()
        Ainf = self.tw_input.check_infinte.isChecked()
        model = self.tw_input.cb_model.currentText()
        method = self.tw_input.cb_method.currentText()
        ca_order = self.tw_input.cb_ca_order.currentIndex()

        # -------- requests controller to model the parameters -------------------------------------
        try:
            optimized_params = self.local_fit_controller.optimize_params(params=params, ds=self.ds,
                                                                         input_wavelength=input_wavelength, wavelength_area=wavelength_area,
                                                                         Ainf=Ainf, model=model, method=method, use_threshold_t0=use_threshold_t0, ca_order=ca_order)
        except exc.FittingError:
            self.tw_results.te_results.setText("Fitting did not succeed")
            return

        # -------- caches the results, estimates statistics and plots the fit ----------------------
        fit_results = self.local_fit_controller.calculate_fitting_output(
            fit_results=optimized_params)
        self.tw_results.te_results.setText(fit_results['output'])
        self.local_fit_controller.call_statusbar("info", msg.Status.s31)

        self.fit_results = fit_results
        self.plot_data(show_fit=True)

    def plot_data(self, show_fit: bool = False, use_wavelength: bool | float = False) -> None:
        '''
        plots the experimental and fitted data (if fit succeeded)

        Parameters
        ----------
        show_fit : bool, optional
            DESCRIPTION. if true, the fit/guess will be plotted as well. The default is False.
        use_wavelength : bool | float, optional
            DESCRIPTION. normally, the GUI wavelength inpult will be used for plotting the linecut.
            use_wavelength surpasses this. Used when fit is loaded. The default is False.

        Returns
        -------
        None
            DESCRIPTION.

        '''
        # -------- load data and ranges ------------------------------------------------------------
        if not self.local_fit_controller.verify_rawdata():
            self._clear_fitting_canvas()
            return

        if use_wavelength is False:
            wavelength_input = self.tw_input.le_wavelength.text()
        else:
            wavelength_input = str(use_wavelength)

        try:
            wavelength, delay, delA = self.local_fit_controller.get_data(
                ds=self.ds, wavelength_input=wavelength_input,
                wavelength_area=self.tw_input.sp_wavelength_area.value())
        except ValueError:
            self._clear_fitting_canvas()
            return

        self.check_delay_input()
        self.check_z_input()
        if self.y_min is None:
            self.y_min = np.nanmin(delay)
        if self.y_linlog is None:
            self.y_linlog = 2e-12
        if self.y_max is None:
            self.y_max = np.nanmax(delay)
        if self.z_min is None:
            self.z_min = np.nanmin(delA)
        if self.z_max is None:
            self.z_max = np.nanmax(delA)

        # -------- init Canvas ---------------------------------------------------------------------
        self.sc = utils.PlotCanvas()
        self.toolbar = NavigationToolbar2QT(self.sc, )
        self.toolbar.addWidget(QLabel('   '))
        self.pb_export_csv = QPushButton(' export fitting results ')
        self.toolbar.addWidget(self.pb_export_csv)
        self.pb_export_csv.pressed.connect(self.export_data)

        self.fig = self.sc.fig
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.sc)

        gs = self.fig.add_gridspec(2, 1, height_ratios=[5, 1])
        axs = gs.subplots(sharex=True)

        # -------- plot data -----------------------------------------------------------------------
        ax1 = axs[0]
        ax1.axhline(y=0, linestyle='dashed', color='black', alpha=0.5)
        ax1.set_xlabel(msg.Labels.delay)
        ax1.set_ylabel(msg.Labels.delA)
        ax1.tick_params(labelleft=True, right=True,)

        ax1.set(xlim=(self.y_min, self.y_max))
        ax1.set(ylim=(self.z_min, self.z_max))

        if self.tw_input.cb_yscale.currentText() == 'lin':
            ax1.set_xscale('linear')
            ax1.xaxis.set_minor_locator(tk.AutoMinorLocator())
        if self.tw_input.cb_yscale.currentText() == 'log':
            ax1.set_xscale('log')
            ax1.xaxis.set_minor_locator(tk.LogLocator(base=10.0, subs="all", numticks=10))
        if self.tw_input.cb_yscale.currentText() == 'linlog':
            try:
                linthreshy = utils.Converter.convert_str_input2float(
                    self.tw_input.le_linlog_y.text())
            except ValueError:
                linthreshy = None
            if linthreshy is None:
                linthreshy = 1e-12

            ax1.set_xscale('symlog', linthresh=linthreshy)
            ax1.xaxis.set_minor_locator(tk.SymmetricalLogLocator(
                base=10.0, linthresh=linthreshy, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))

        if self.tw_input.cb_zscale.currentText() == 'lin':
            ax1.set_yscale('linear')
            ax1.yaxis.set_minor_locator(tk.AutoMinorLocator())
        if self.tw_input.cb_zscale.currentText() == 'log':
            ax1.set_yscale('log')
            ax1.yaxis.set_minor_locator(tk.LogLocator(base=10.0, subs="all", numticks=10))
        if self.tw_input.cb_zscale.currentText() == 'linlog':
            try:
                linthreshz = utils.Converter.convert_str_input2float(
                    self.tw_input.le_linlog_z.text())
            except ValueError:
                linthreshz = None
            if linthreshz is None:
                linthreshz = 1e-12

            ax1.set_yscale('symlog', linthresh=linthreshz)
            ax1.yaxis.set_minor_locator(tk.SymmetricalLogLocator(
                base=10.0, linthresh=linthreshz, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)))

        ax1.xaxis.set_major_formatter(self.sc.delay_formatter0)
        fit_colorlist = utils.Converter.fitting_colorlist(
            plt.rcParams["axes.prop_cycle"].by_key()["color"])
        label = f"{self.sc.nm_formatter_ax(wavelength)} nm"
        ax1.plot(delay, delA, 'x', alpha=0.5, markersize=5, label=label, zorder=0)

        # -------- plot fit ------------------------------------------------------------------------
        if show_fit:
            if self.fit_results:
                if 'delA_calc' in self.fit_results:
                    labels = self.fit_results['meta']['components']
                    labels_tex = [f'${label}$' for label in labels]

                    ax1.plot(delay, self.fit_results['delA_calc'],
                             color=fit_colorlist[0], alpha=1, label='fit', zorder=2)

                    # plotting conc only makes sense if more than 1 comp
                    if self.fit_results['conc'].shape[1] >= 2:
                        for i in range((self.fit_results['conc'].shape[1])):
                            ax1.plot(delay, self.fit_results['conc'][:, i] *
                                     self.fit_results['Amp'][i], '--', label=labels_tex[i])

                    ax2 = axs[1]
                    ax2.plot(
                        delay,  delA - self.fit_results['delA_calc'], 'b', label='residuals')

        ax1.legend(bbox_to_anchor=(1.02, 1), borderpad=0.0, loc='upper left',
                   borderaxespad=0, edgecolor='1', title='wavelength')

        self.tw_canvas.w_canvas.hide()
        self.tw_canvas.w_canvas.setParent(None)  # remove widget from layout
        self.tw_canvas.w_canvas.deleteLater()      # schedule it for deletion
        self.tw_canvas.w_canvas = QWidget(self.tw_canvas)
        self.tw_canvas.w_canvas.setLayout(layout)
        self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)
