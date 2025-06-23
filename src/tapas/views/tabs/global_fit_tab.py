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
import re

# Third-Party Imports

# PyQt6 Imports
from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QPushButton,
    QHeaderView,
    QVBoxLayout,
    QTableWidgetItem,
    QGridLayout,
    QFileDialog,
)
from PyQt6.QtCore import Qt

# Matplotlib and Related Imports
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
import matplotlib.ticker as tk
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Other Third-Party Libraries
import numpy as np
import jax.numpy as jnp
import corner

# Local Application Imports
from ...utils import utils
from ...configurations import exceptions as exc, messages as msg
from ...views.tabwidgets.global_fit_tabwidgets import InputWidget, CanvasWidget, ResultsWidget

logger = logging.getLogger(__name__)


class GlobalFitTab(QWidget):
    logger = logging.getLogger("views.tabs.global_fit_tab.GlobalFitTab")

    def __init__(self, tabwidget,  ta_model, ta_model_ds1, ta_model_ds2, ta_model_ds3, controller, config):
        super().__init__()
        self.tab = tabwidget
        self.ta_model = ta_model
        self.ta_model_ds1 = ta_model_ds1
        self.ta_model_ds2 = ta_model_ds2
        self.ta_model_ds3 = ta_model_ds3
        self.global_fit_controller = controller
        self.config = config
        self.ds = '1'
        self.project_path = None
        self.results_dir = None
        self.y_min, self.y_linlog, self.y_max = None, None, None
        self.z_min, self.z_max, self.z_center = None, None, 0
        self.fit_results = None
        self.need_GUI_update = False
        self.emcee_final_result = {}
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
            self.tw_input, min_width=310, max_width=310, horizontal_scroll=False,
            use_container=False), 0, 0, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.tw_canvas, 0, 1,)
        layout.addWidget(utils.Converter.create_scrollable_widget(
            self.tw_results, min_width=310, max_width=310, horizontal_scroll=False, ),
            0, 2, alignment=Qt.AlignmentFlag.AlignRight)
        self.setLayout(layout)

        # -------- connect Widgets to view / controller --------------------------------------------
        self.tw_input.rb_ds_1.toggled.connect(self.update_dataset)
        self.tw_input.rb_ds_2.toggled.connect(self.update_dataset)
        self.tw_input.rb_ds_3.toggled.connect(self.update_dataset)
        self.tw_input.le_linlog.editingFinished.connect(self.plot_fit)
        self.tw_input.check_normalize.stateChanged.connect(self.plot_fit)
        self.tw_input.check_normalized_resid.stateChanged.connect(self.plot_fit)
        self.tw_input.pb_show_guess.pressed.connect(self.show_initial_fit)
        self.tw_input.pb_fit.pressed.connect(self.fit_data)
        self.tw_input.check_gs.stateChanged.connect(self.show_gs_use_abs)

        self.fitting_models = ['parallel', 'sequential' 'model1']
        self.tw_input.cb_model.currentIndexChanged.connect(
            self.update_fitting_input_gui)

        self.tw_input.sb_weights1_max.valueChanged.connect(self.update_ranges)
        self.tw_input.sb_weights2_max.valueChanged.connect(self.update_ranges)
        self.tw_input.sb_weights3_max.valueChanged.connect(self.update_ranges)
        self.tw_input.sb_weights4_max.valueChanged.connect(self.update_ranges)
        self.tw_input.sb_weights5_max.valueChanged.connect(self.update_ranges)
        self.tw_results.tab_ds_fits.horizontalHeader(
        ).sectionClicked.connect(self.show_clicked_fit)
        self.tw_results.pb_run_emcee.pressed.connect(self.run_emcee)
        self.tw_results.pb_cancel_emcee.pressed.connect(
            self.global_fit_controller.abort_emcee)
        self.tw_results.pb_save_emcee.pressed.connect(
            lambda: self.global_fit_controller.save_emcee_result_2model(results=self.emcee_final_result))
        self.tw_results.pb_save_fit.pressed.connect(
            lambda ds=self.ds: self.global_fit_controller.save_current_fit(ds=self.ds))
        self.tw_results.pb_delete_fit.pressed.connect(self.delete_selected_fit)

        # -------- connect emcee worker ------------------------------------------------------------
        self.global_fit_controller.worker_progress.connect(self.emcee_update_progress)
        self.global_fit_controller.worker_results.connect(self.emcee_update_results)
        self.global_fit_controller.emcee_finished.connect(self.emcee_show_final)

        # -------- listen to model event signals ---------------------------------------------------
        models = (self.ta_model, self.ta_model_ds1, self.ta_model_ds2, self.ta_model_ds3)
        for i in models:
            i.data_changed.connect(self.queue_update_GUI)
            i.global_fit_changed.connect(lambda: self.queue_update_GUI(only_fitting_list=True))

    def show_gs_use_abs(self) -> None:
        ''' enables the "use steady-state absorbance button, if data is present and GS should be modeled '''
        if self.tw_input.check_gs.isChecked():
            if self.global_fit_controller.verify_abs_data():
                self.tw_input.gs_use_ss_abs.setEnabled(True)
        else:
            self.tw_input.gs_use_ss_abs.setChecked(False)
            self.tw_input.gs_use_ss_abs.setEnabled(False)

    def update_fitting_input_gui(self) -> None:
        ''' updates the possible lifetime/rate inputs in the gui.
        screens custom model for number of lifetimes k (nC_mk_1) and sets the number accordingly'''
        selected_model = self.tw_input.cb_model.currentText()

        if selected_model in ['sequential', 'parallel']:
            self.tw_input.sb_components.setEnabled(True)
            return

        m = re.search(r'_(\d+)k_', selected_model)
        if m:
            comp = int(m.group(1))
        else:
            comp = 1
            self.global_fit_controller.call_statusbar("error", msg.Error.e49)

        self.tw_input.sb_components.setValue(comp)
        self.tw_input.sb_components.setEnabled(False)

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
        report_str = self.global_fit_controller.get_emcee_print(results)
        self.emcee_window.te_results.setText(report_str)
        self.emcee_window.label_results.setText('results after final run:')
        self.emcee_final_result['meta']['finished_early'] = abort
        self.emcee_final_result['params'] = results.params
        self.emcee_final_result['output'] = report_str
        self.emcee_final_result['flatchain'] = results.flatchain.to_numpy().astype(np.float32)
        self.emcee_final_result['meta']['#samples'] = results.chain.shape[0]

    def emcee_update_results(self, results: object, run: int) -> None:
        ''' triggered after each emcee run, prints and plots results to gui '''
        report_str = self.global_fit_controller.get_emcee_print(results)
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
            self.global_fit_controller.call_statusbar("error", msg.Error.e36)
            return
        else:
            fit_results, ukey = self.global_fit_controller.get_fit(
                ds=self.ds, selected_fit=current_col)
        self.emcee_final_result['ukey'] = ukey
        self.emcee_final_result['ds'] = self.ds
        self.emcee_final_result['meta']['burn'] = self.tw_results.sb_burn.value()
        self.emcee_final_result['meta']['thin'] = self.tw_results.sb_thin.value()
        self.emcee_final_result['meta']['target_ratio'] = self.tw_results.sb_target_ratio.value()

        if self.fit_results is None:
            self.global_fit_controller.call_statusbar("error", msg.Error.e24)
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

        self.global_fit_controller.run_emcee(
            ds=self.ds, results=self.fit_results, burn=self.tw_results.sb_burn.value(),
            init=self.tw_results.sb_init.value(), thin=self.tw_results.sb_thin.value(),
            target_ratio=self.tw_results.sb_target_ratio.value())

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
        fit_results, _ = self.global_fit_controller.get_fit(
            ds=self.ds, selected_fit=column_index)

        self.tw_results.te_results.setText(fit_results['output'])
        self.fit_results = fit_results
        if 'emcee' in fit_results:
            self.emcee_final_result = fit_results['emcee']
        self.plot_fit()
        self.global_fit_controller.call_statusbar("info", msg.Status.s32)

    def export_data(self) -> None:
        ''' gets Path from file dialog, writes fit and meta data to txt '''
        if self.project_path is not None:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Choose a file name", str(self.project_path.parent), filter='*.txt')
        else:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Choose a file name", filter='*.txt')
        if not file_name:
            self.global_fit_controller.call_statusbar("error", msg.Error.e17)
            return
        results_path = Path(file_name)

        if results_path is None:
            self.global_fit_controller.call_statusbar("error", msg.Error.e17)
            return
        meta_data = self.global_fit_controller.get_fitting_meta(
            self.fit_results)
        basic_results = self.global_fit_controller.get_fitting_print(
            self.fit_results)

        emcee_results = ''
        if self.emcee_final_result:
            if 'output' in self.emcee_final_result:
                emcee_results += self.emcee_final_result['output']
                emcee_results += f"\nburn: {self.emcee_final_result['meta']['burn']}, thin: {self.emcee_final_result['meta']['thin']}"

        with open(results_path, "w") as file:
            file.write('global fit results:\n')
            file.write(meta_data)
            file.write('\n-----------------------------------------------------\n')
            file.write('\n--- Results from Jacobian Analysis: ---\n\n')
            file.write(basic_results)
            file.write('\n-----------------------------------------------------\n\n')
            file.write(emcee_results)

    def _reset_fitting_cache(self) -> None:
        ''' triggered, when data changed. resets all the cached data '''
        self.fit_results = None
        self.emcee_final_result = {}
        self._init_weights()
        self._clear_fitting_canvas()
        self._update_fitting_list()
        self.global_fit_controller.initialize_fitting_cache()

    def _clear_fitting_canvas(self) -> None:
        ''' cleanely deletes canvas '''
        self.tw_canvas.w_canvas.hide()
        self.tw_canvas.w_canvas.setParent(None)  # remove widget from layout
        self.tw_canvas.w_canvas.deleteLater()
        self.tw_canvas.w_canvas = QLabel(msg.Widgets.i13, self.tw_canvas)
        self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas,)
        self.tw_results.te_results.setText('')

    def delete_selected_fit(self) -> None:
        ''' removes selevted fit form gui and requests controller to delete its data '''
        self._reset_fitting_cache()
        selected_fit = self.tw_results.tab_ds_fits.currentColumn()
        if selected_fit < 0:
            self.global_fit_controller.call_statusbar("error", msg.Error.e22)
            return
        self.global_fit_controller.delete_fit(
            ds=self.ds, selected_fit=selected_fit)
        self.global_fit_controller.call_statusbar("info", msg.Status.s33)

    def _update_fitting_list(self) -> None:
        ''' gets whole fit dict from controller and updates GUI table accordingly '''
        fit_dict = self.global_fit_controller.get_global_fit_list(ds=self.ds)

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
                1, column, QLabel(str(fit_dict[fit]['meta']['r2'])))
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

    def _init_weights(self) -> None:
        ''' clusters and sets bounds to weights widget '''
        self.spinboxes_min = [
            self.tw_input.sb_weights1_min,
            self.tw_input.sb_weights2_min,
            self.tw_input.sb_weights3_min,
            self.tw_input.sb_weights4_min,
            self.tw_input.sb_weights5_min,
        ]
        self.spinboxes_max = [
            self.tw_input.sb_weights1_max,
            self.tw_input.sb_weights2_max,
            self.tw_input.sb_weights3_max,
            self.tw_input.sb_weights4_max,
            self.tw_input.sb_weights5_max,
        ]
        # Note: container1 is assumed always visible; additional containers for intervals 2-5:
        self.containers = [
            self.tw_input.container2,
            self.tw_input.container3,
            self.tw_input.container4,
            self.tw_input.container5,
        ]

        self.weights = [
            self.tw_input.sb_weights1_value,
            self.tw_input.sb_weights2_value,
            self.tw_input.sb_weights3_value,
            self.tw_input.sb_weights4_value,
            self.tw_input.sb_weights5_value,
        ]
        try:
            self.wavelength, _, _ = self.global_fit_controller.get_data(self.ds)
        except ValueError:
            self.tw_input.container1.setEnabled(False)
            for container in self.containers:
                container.hide()
            return
        self.tw_input.container1.setEnabled(True)
        self.min_value = (int(round(np.nanmin(self.wavelength), 9) * 1e9))
        self.max_value = (int(round(np.nanmax(self.wavelength), 9) * 1e9))
        for sb in self.spinboxes_min:
            sb.setRange(self.min_value, self.max_value)
            sb.setValue(self.max_value)
        for sb in self.spinboxes_max:
            sb.setRange(self.min_value, self.max_value)
            sb.setValue(self.max_value)
        for container in self.containers:
            container.hide()
        self.tw_input.sb_weights1_min.setValue(self.min_value)

    def update_ranges(self) -> None:
        ''' adds, removes and adjust limits for the weights GUI widget  '''
        if self.tw_input.sb_weights1_max.value() < self.max_value:
            self.tw_input.container2.show()
            self.tw_input.sb_weights2_min.setValue(
                self.tw_input.sb_weights1_max.value())
            self.tw_input.sb_weights2_max.setMinimum(
                self.tw_input.sb_weights1_max.value())

            if self.tw_input.sb_weights2_max.value() < self.max_value:
                self.tw_input.container3.show()
                self.tw_input.sb_weights3_min.setValue(
                    self.tw_input.sb_weights2_max.value())

                self.tw_input.sb_weights3_max.setMinimum(
                    self.tw_input.sb_weights2_max.value())
                if self.tw_input.sb_weights3_max.value() < self.max_value:

                    self.tw_input.container4.show()
                    self.tw_input.sb_weights4_min.setValue(
                        self.tw_input.sb_weights3_max.value())

                    self.tw_input.sb_weights4_max.setMinimum(
                        self.tw_input.sb_weights3_max.value())

                    if self.tw_input.sb_weights4_max.value() < self.max_value:
                        self.tw_input.container5.show()
                        self.tw_input.sb_weights5_min.setValue(
                            self.tw_input.sb_weights4_max.value())
                        self.tw_input.sb_weights5_max.setValue(self.max_value)

                    else:
                        self.tw_input.sb_weights4_min.setValue(
                            self.tw_input.sb_weights3_max.value())
                        self.tw_input.sb_weights5_max.setValue(self.max_value)
                        self.tw_input.sb_weights5_min.setValue(self.max_value)
                        self.tw_input.container5.hide()
                else:
                    self.tw_input.sb_weights3_min.setValue(
                        self.tw_input.sb_weights2_max.value())
                    self.tw_input.sb_weights4_max.setValue(self.max_value)
                    self.tw_input.sb_weights4_min.setValue(self.max_value)
                    self.tw_input.container4.hide()
                    self.tw_input.sb_weights5_max.setValue(self.max_value)
                    self.tw_input.sb_weights5_min.setValue(self.max_value)
                    self.tw_input.container5.hide()
            else:
                self.tw_input.sb_weights2_min.setValue(
                    self.tw_input.sb_weights1_max.value())
                self.tw_input.sb_weights3_max.setValue(self.max_value)
                self.tw_input.sb_weights3_min.setValue(self.max_value)
                self.tw_input.container3.hide()
                self.tw_input.sb_weights4_max.setValue(self.max_value)
                self.tw_input.sb_weights4_min.setValue(self.max_value)
                self.tw_input.container4.hide()
                self.tw_input.sb_weights5_max.setValue(self.max_value)
                self.tw_input.sb_weights5_min.setValue(self.max_value)
                self.tw_input.container5.hide()

        else:
            self.tw_input.sb_weights2_max.setValue(self.max_value)
            self.tw_input.sb_weights2_min.setValue(self.max_value)
            self.tw_input.container2.hide()
            self.tw_input.sb_weights3_max.setValue(self.max_value)
            self.tw_input.sb_weights3_min.setValue(self.max_value)
            self.tw_input.container3.hide()
            self.tw_input.sb_weights4_max.setValue(self.max_value)
            self.tw_input.sb_weights4_min.setValue(self.max_value)
            self.tw_input.container4.hide()
            self.tw_input.sb_weights5_max.setValue(self.max_value)
            self.tw_input.sb_weights5_min.setValue(self.max_value)
            self.tw_input.container5.hide()
        self.tw_input.adjustSize()

    def queue_update_GUI(self, only_fitting_list: bool = False) -> None:
        ''' called, if the raw or ds data is changed. GUI update waits till tab is selected
        if local_fit data is changed, only the fitting list will be updated (only_fitting_list = True)
        '''
        self.need_GUI_update = True
        if self.tab.currentIndex() == 6:
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

    def update_config(self) -> None:
        '''updates configuration and standard values of QWidgets'''
        self.config.add_handler('global_w_le_linlog', self.tw_input.le_linlog)
        self.config.add_handler('global_w_sb_components', self.tw_input.sb_components)
        self.config.add_handler('global_w_check_infinte', self.tw_input.check_infinte)
        self.config.add_handler('global_w_cb_model', self.tw_input.cb_model)
        self.config.add_handler('global_w_results_posterior_sb_burn', self.tw_results.sb_burn)
        self.config.add_handler('global_w_results_posterior_sb_init', self.tw_results.sb_init)
        self.config.add_handler('global_w_results_posterior_sb_thin', self.tw_results.sb_thin)
        self.config.add_handler(
            'global_w_results_posterior_sb_target_ratio', self.tw_results.sb_target_ratio)

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

    def _sort_fit_params_input(self, components: None) -> None:
        '''
        sorts parameter input table to increasing lifetimes. Important for sequential analysis
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

    def show_initial_fit(self) -> None:
        ''' retrives and plots the initial guess parameters. returns diagnotsic error when failed '''
        # -------- reads input and converts it to usable parameters --------------------------------
        if not self.global_fit_controller.verify_rawdata():
            self.global_fit_controller.call_statusbar("error", msg.Error.e05)
            return
        if self.tw_input.cb_model.currentText() in ['parallel', 'sequential']:
            self._sort_fit_params_input(components=self.tw_input.sb_components.value())
        try:
            params = self.global_fit_controller.get_params(QtTable=self.tw_input.tab_fit_params)
        except ValueError:
            self.tw_results.te_results.setText("Fitting did not succeed")
            return

        missing_params = [name for name, par in params.items() if np.isinf(par.value)]
        if missing_params:
            self.global_fit_controller.call_statusbar(
                'error', f"Missing initial value for: {missing_params}")
            return
        try:
            wavelength, delay, delA = self.global_fit_controller.get_data(ds=self.ds)
        except ValueError:
            return

        weight_dummy = jnp.array(np.full(wavelength.shape, 1, dtype=float))
        substeps = self.tw_input.sp_substeps.value()
        use_threshold_t0 = True if self.tw_input.cb_t0_def.currentText() == '5% Threshold' else False
        gs = self.tw_input.check_gs.isChecked()
        gs_spec = False
        if self.tw_input.gs_use_ss_abs.isChecked():
            gs_spec = self.global_fit_controller.get_gs_spec(wavelength)
            regular_grid = self.global_fit_controller.verify_regular_grid(wavelength)

            if gs_spec is not False and regular_grid:
                params.add('gs_shift', value=1e-9, min=-5e-9, max=5e-9)
                params.add('gs_sigma', value=1e-9, min=0, max=5e-9)
            else:
                self.global_fit_controller.call_statusbar("error", msg.Error.e46)
                return
        Ainf = self.tw_input.check_infinte.isChecked()
        ca_order = self.tw_input.cb_ca_order.currentIndex()

        # -------- requests controller to model the parameters -------------------------------------
        try:
            fit_results = {}

            if self.tw_input.cb_model.currentText() == 'sequential':
                fit_results['delA_calc'], fit_results['conc'], fit_results['EAS'],  = self.global_fit_controller.model_theta_wrapper(
                    params=params, delay=delay, delA=delA, Ainf=Ainf,  model='sequential',
                    weights=weight_dummy, use_threshold_t0=use_threshold_t0, substeps=substeps, gs=gs, gs_spec=gs_spec,ca_order=ca_order, output=True)
                _, _, fit_results['DAS'] = self.global_fit_controller.model_theta_wrapper(
                    params=params, delay=delay, delA=delA, Ainf=Ainf,  model='parallel',
                    weights=weight_dummy, use_threshold_t0=use_threshold_t0, substeps=substeps, gs=gs, gs_spec=gs_spec,ca_order=ca_order, output=True)

            elif self.tw_input.cb_model.currentText() == 'parallel':
                fit_results['delA_calc'], fit_results['conc'], fit_results['DAS'],  = self.global_fit_controller.model_theta_wrapper(
                    params=params, delay=delay, delA=delA, Ainf=Ainf,  model='parallel',
                    weights=weight_dummy, use_threshold_t0=use_threshold_t0, substeps=substeps, gs=gs, gs_spec=gs_spec,ca_order=ca_order, output=True)
                _, _, fit_results['EAS'] = self.global_fit_controller.model_theta_wrapper(
                    params=params, delay=delay, delA=delA, Ainf=Ainf,  model='sequential',
                    weights=weight_dummy, use_threshold_t0=use_threshold_t0, substeps=substeps, gs=gs, gs_spec=gs_spec,ca_order=ca_order, output=True)

            else:
                fit_results['delA_calc'], fit_results['conc'], fit_results['SAS'],  = self.global_fit_controller.model_theta_wrapper(
                    params=params, delay=delay, delA=delA, Ainf=Ainf,  model=self.tw_input.cb_model.currentText(),
                    weights=weight_dummy, use_threshold_t0=use_threshold_t0, substeps=substeps, gs=gs, gs_spec=gs_spec,ca_order=ca_order, output=True)

            fit_results['opt_params'] = params
            self.tw_results.te_results.setText("Initial fit succeed")
            self.global_fit_controller.call_statusbar("info", msg.Status.s30)

        except np.linalg.LinAlgError:
            self.global_fit_controller.call_statusbar("error", msg.Error.e23)
            self.tw_results.te_results.setText("Fitting did not succeed")
            return

        # -------- caches the results and plots the guess ------------------------------------------
        fit_results['residuals'] = delA - fit_results['delA_calc']
        fit_results['meta'] = {}
        fit_results['meta']['model'] = self.tw_input.cb_model.currentText()
        fit_results['meta']['components'] = self.global_fit_controller.get_component_labels(
            model=self.tw_input.cb_model.currentText(), Ainf=self.tw_input.check_infinte.isChecked(),
            num=self.tw_input.sb_components.value(), gs=self.tw_input.check_gs.isChecked(),ca_order= ca_order)
        self.fit_results = fit_results
        try:
            self.plot_fit()
        except OverflowError:
            self.tw_results.te_results.setText("Fitting did not succeed.")
            self.global_fit_controller.call_statusbar("error", msg.Error.e19)

    def fit_data(self) -> None:
        ''' retrives the input parameters and informs controller to perform and evaluate the fit.
        Fit results are cached, printed and plotted to the GUI.
        '''
        # -------- reads input and converts it to usable parameters --------------------------------
        if not self.global_fit_controller.verify_rawdata():
            self.global_fit_controller.call_statusbar("error", msg.Error.e05)
            return

        self.global_fit_controller.create_ds(self.ds)  # check if ds is empty and fill with raw_data
        if self.tw_input.cb_model.currentText() in ['parallel', 'sequential']:
            self._sort_fit_params_input(components=self.tw_input.sb_components.value())
        try:
            params = self.global_fit_controller.get_params(
                QtTable=self.tw_input.tab_fit_params)
        except ValueError:
            self.tw_results.te_results.setText("Fitting did not succeed")
            return

        weight_intervals = (list(zip([x.value() for x in self.spinboxes_min], [
                            x.value() for x in self.spinboxes_max], [x.value()/100 for x in self.weights])))
        weights_vector = self.global_fit_controller.create_weight_vector(
            wavelength=self.wavelength, intervals=weight_intervals)

        substeps = self.tw_input.sp_substeps.value()
        use_threshold_t0 = True if self.tw_input.cb_t0_def.currentText() == '5% Threshold' else False
        gs = self.tw_input.check_gs.isChecked()
        gs_spec = False
        if self.tw_input.gs_use_ss_abs.isChecked():
            gs_spec = self.global_fit_controller.get_gs_spec(self.wavelength)
            regular_grid = self.global_fit_controller.verify_regular_grid(self.wavelength)
            if gs_spec is not False and regular_grid:
                params.add('gs_shift', value=1e-9, min=-5e-9, max=5e-9)
                params.add('gs_sigma', value=1e-9, min=0, max=5e-9)
            else:
                self.global_fit_controller.call_statusbar("error", msg.Error.e46)
                return

        Ainf = self.tw_input.check_infinte.isChecked()
        model = self.tw_input.cb_model.currentText()
        method = self.tw_input.cb_method.currentText()
        ca_order = self.tw_input.cb_ca_order.currentIndex()

        # -------- requests controller to model the parameters -------------------------------------
        try:
            optimized_params = self.global_fit_controller.optimize_params(params=params, ds=self.ds,
                                                                          Ainf=Ainf, model=model,
                                                                          method=method, weights=weights_vector,
                                                                          use_threshold_t0=use_threshold_t0,
                                                                          substeps=substeps, gs=gs, gs_spec=gs_spec,ca_order=ca_order)
        except exc.FittingError:
            self.tw_results.te_results.setText("Fitting did not succeed")
            self.fit_results = None
            self.plot_fit()
            return

        # -------- caches the results, estimates statistics and plots the fit ----------------------
        fit_results = self.global_fit_controller.calculate_fitting_output(
            fit_results=optimized_params)

        self.tw_results.te_results.setText(fit_results['output'])
        self.global_fit_controller.call_statusbar("info", msg.Status.s31)
        self.fit_results = fit_results
        self.plot_fit()

    def plot_fit(self) -> None:
        ''' plots EAS/DAS/SAS, conc profile, simmulated delA, residuals and SVD of residuals  '''
        # -------- load data, results and SVD ------------------------------------------------------
        if not self.fit_results:
            self._clear_fitting_canvas()
            return
        if not self.global_fit_controller.verify_rawdata():
            self._clear_fitting_canvas()
            return

        try:
            wavelength, delay, delA = self.global_fit_controller.get_data(ds=self.ds)
        except ValueError:
            return

        try:
            U, s_norm, Vh = self.global_fit_controller.get_SVD(
                residuals=self.fit_results['residuals'])
        except ValueError:
            return

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

        with plt.rc_context({
            'axes.titlesize': 10,
            'axes.labelsize': 8,
            'legend.fontsize': 8,
            'xtick.labelsize': 6,
            'ytick.labelsize': 6
        }):

            gs = self.sc.fig.add_gridspec(2, 3)
            self.ax_conc = self.sc.fig.add_subplot(gs[0, 2])
            self.ax_delA_calc = self.sc.fig.add_subplot(gs[1, 0])
            self.ax_res = self.sc.fig.add_subplot(gs[1, 1])
            nested_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1, 2])
            self.ax_Vh = self.sc.fig.add_subplot(nested_gs[0])
            self.ax_U = self.sc.fig.add_subplot(nested_gs[1])

            # -------- init scales -----------------------------------------------------------------
            try:
                self.linthresh = utils.Converter.convert_str_input2float(
                    self.tw_input.le_linlog.text())
                if self.linthresh is None:
                    self.ax_delA_calc.set_yscale('linear')
                    self.ax_res.set_yscale('linear')
                    self.ax_conc.set_xscale('linear')
                    self.ax_U.set_xscale('linear')
                else:
                    self.ax_delA_calc.set_yscale('symlog', linthresh=self.linthresh, linscale=1)
                    self.ax_res.set_yscale('symlog', linthresh=self.linthresh, linscale=1)

                    self.ax_conc.set_xscale('symlog', linthresh=self.linthresh, linscale=1)
                    self.ax_U.set_xscale('symlog', linthresh=self.linthresh, linscale=1)
            except AttributeError:
                self.global_fit_controller.call_statusbar("error", msg.Error.e05)
            except ValueError:
                self.global_fit_controller.call_statusbar("error", msg.Error.e02)

            # -------- init labels and titles ------------------------------------------------------
            self.ax_delA_calc.set_xlabel(msg.Labels.wavelength)
            self.ax_delA_calc.set_ylabel(msg.Labels.delay)
            self.ax_res.set_xlabel(msg.Labels.wavelength)
            self.ax_res.set_ylabel(msg.Labels.delay)
            self.ax_Vh.set_xlabel(msg.Labels.wavelength)
            self.ax_U.set_xlabel(msg.Labels.delay)

            self.ax_Vh.set_ylabel("spectral Vectors")
            self.ax_U.set_ylabel("temporal Vectors")
            self.ax_Vh.tick_params(axis='y', which='both', labelleft=False, labelright=False)
            self.ax_U.tick_params(axis='y', which='both', labelleft=False, labelright=False)

            self.ax_conc.set_xlabel(msg.Labels.delay)
            self.ax_conc.set_ylabel('Norm. Intensity')

            self.ax_Vh.set_title("SVD of Residuals")
            self.ax_conc.set_title("Concentrations")
            self.ax_delA_calc.set_title("Calculated Matrix")
            self.ax_res.set_title("Residuals")
            self.ax_delA_calc.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
            self.ax_delA_calc.yaxis.set_major_formatter(self.sc.delay_formatter0)
            self.ax_res.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
            self.ax_res.yaxis.set_major_formatter(self.sc.delay_formatter0)
            self.ax_Vh.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
            self.ax_conc.xaxis.set_major_formatter(self.sc.delay_formatter0)
            self.ax_U.xaxis.set_major_formatter(self.sc.delay_formatter0)

            self.ax_conc.axhline(y=0, color=self.ax_conc.xaxis.label.get_color(
            ), linestyle="--", linewidth=1, zorder=0)
            self.ax_Vh.axhline(y=0, color=self.ax_conc.xaxis.label.get_color(
            ), linestyle="--", linewidth=1, zorder=0)
            self.ax_U.axhline(y=0, color=self.ax_conc.xaxis.label.get_color(
            ), linestyle="--", linewidth=1, zorder=0)

            labels = self.fit_results['meta']['components']
            labels_tex = [f'${label}$' for label in labels]

            # -------- plot EAS/DAS & conc ---------------------------------------------------------
            if self.fit_results['meta']['model'] in ['parallel', 'sequential']:

                self.ax_EAS = self.sc.fig.add_subplot(gs[0, 0])
                self.ax_DAS = self.sc.fig.add_subplot(gs[0, 1])
                self.ax_EAS.set_xlabel(msg.Labels.wavelength)
                self.ax_DAS.set_xlabel(msg.Labels.wavelength)
                if self.tw_input.check_normalize.isChecked():
                    self.ax_EAS.set_ylabel(msg.Labels.delA_norm)
                    self.ax_DAS.set_ylabel(msg.Labels.delA_norm)
                else:
                    self.ax_EAS.set_ylabel(msg.Labels.delA)
                    self.ax_DAS.set_ylabel(msg.Labels.delA)

                self.ax_EAS.set_title("EAS")
                self.ax_DAS.set_title("DAS")
                self.ax_EAS.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
                self.ax_DAS.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
                self.ax_DAS.axhline(y=0, color=self.ax_DAS.xaxis.label.get_color(
                ), linestyle="--", linewidth=1, zorder=0)
                self.ax_EAS.axhline(y=0, color=self.ax_DAS.xaxis.label.get_color(
                ), linestyle="--", linewidth=1, zorder=0)

                if self.tw_input.check_normalize.isChecked():
                    for i in range((self.fit_results['conc'].shape[1])):
                        norm_factor_DAS = np.amax(
                            abs(self.fit_results['DAS'][:, i]))

                        norm_factor_EAS = np.amax(
                            abs(self.fit_results['EAS'][:, i]))
                        self.ax_DAS.plot(
                            wavelength, self.fit_results['DAS'][:, i]/norm_factor_DAS,)
                        self.ax_EAS.plot(
                            wavelength, self.fit_results['EAS'][:, i]/norm_factor_EAS,)
                        self.ax_conc.plot(
                            delay, self.fit_results['conc'][:, i], label=labels_tex[i])

                else:
                    for i in range((self.fit_results['conc'].shape[1])):
                        self.ax_DAS.plot(
                            wavelength, self.fit_results['DAS'][:, i],)
                        self.ax_EAS.plot(
                            wavelength, self.fit_results['EAS'][:, i],)
                        self.ax_conc.plot(
                            delay, self.fit_results['conc'][:, i], label=labels_tex[i])

            # -------- plot SAS & conc -------------------------------------------------------------
            else:
                self.ax_SAS = self.sc.fig.add_subplot(gs[0, 0])
                self.ax_SAS.set_xlabel(msg.Labels.wavelength)
                if self.tw_input.check_normalize.isChecked():
                    self.ax_SAS.set_ylabel(msg.Labels.delA_norm)

                else:
                    self.ax_SAS.set_ylabel(msg.Labels.delA)

                self.ax_SAS.set_title("SAS")
                self.ax_SAS.xaxis.set_major_formatter(self.sc.nm_formatter_ax)
                self.ax_SAS.axhline(y=0, color=self.ax_SAS.xaxis.label.get_color(
                ), linestyle="--", linewidth=1, zorder=0)

                if self.tw_input.check_normalize.isChecked():
                    for i in range((self.fit_results['conc'].shape[1])):

                        norm_factor_SAS = np.amax(
                            abs(self.fit_results['SAS'][:, i]))

                        self.ax_SAS.plot(
                            wavelength, self.fit_results['SAS'][:, i]/norm_factor_SAS,)
                        self.ax_conc.plot(
                            delay, self.fit_results['conc'][:, i], label=labels_tex[i])

                else:
                    for i in range((self.fit_results['conc'].shape[1])):

                        self.ax_SAS.plot(
                            wavelength, self.fit_results['SAS'][:, i],)
                        self.ax_conc.plot(
                            delay, self.fit_results['conc'][:, i], label=labels_tex[i])

            self.ax_conc.legend(bbox_to_anchor=(1.02, 1), borderpad=0.0, loc='upper left',
                                borderaxespad=0, edgecolor='1', )

            # -------- plot residuals SVD  ---------------------------------------------------------
            for idx, s1 in enumerate(s_norm):
                self.ax_Vh.plot(wavelength, Vh[idx, :] * (-s1),)
                self.ax_U.plot(delay, U[:, idx] * (-s1),)

            # -------- plot residuals  -------------------------------------------------------------
            normalization = colors.TwoSlopeNorm(vmin=-2, vmax=2, vcenter=0)
            X, Y = np.meshgrid(wavelength, delay)
            divider = make_axes_locatable(self.ax_res)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            if self.tw_input.check_normalized_resid.isChecked():
                denom = np.where(delA == 0, 1e-8, delA)
                pct_dev = (self.fit_results['residuals'] / denom) * 100
                self.pcolormesh_plot_error = self.ax_res.pcolormesh(
                    X, Y, pct_dev, shading='auto', norm=normalization)
                cb_error = self.fig.colorbar(mappable=self.pcolormesh_plot_error, cax=cax,
                                             location="right", shrink=0.6, label=msg.Labels.norm_residuals)
                normalization = colors.TwoSlopeNorm(vmin=-20, vmax=20, vcenter=0)
            else:
                self.pcolormesh_plot_error = self.ax_res.pcolormesh(
                    X, Y, self.fit_results['residuals'], shading='auto', norm=normalization)
                cb_error = self.fig.colorbar(mappable=self.pcolormesh_plot_error, cax=cax,
                                             location="right", shrink=0.6, label=msg.Labels.residuals)
                normalization = colors.TwoSlopeNorm(vmin=-10, vmax=10, vcenter=0)

            cb_error.minorticks_on()

            # -------- plot simmulated delA  -------------------------------------------------------
            self.pcolormesh_plot_delA = self.ax_delA_calc.pcolormesh(
                X, Y, self.fit_results['delA_calc'], shading='auto', norm=normalization)
            divider = make_axes_locatable(self.ax_delA_calc)
            cax = divider.append_axes('right', size='5%', pad=0.05)

            cb_delA = self.fig.colorbar(mappable=self.pcolormesh_plot_delA, cax=cax,
                                        location="right", shrink=0.6, label=msg.Labels.delA)
            cb_delA.minorticks_on()
            self.sc.axes_mapping = {
                self.ax_res: (self.pcolormesh_plot_error, cb_error),
                self.ax_delA_calc: (self.pcolormesh_plot_delA, cb_delA)}

            self.sc.mpl_connect('scroll_event', lambda event: self.sc._zoom_TA(
                event, self.sc.axes_mapping))
            self.tw_canvas.w_canvas.hide()
            self.tw_canvas.w_canvas.setParent(None)  # remove widget from layout
            self.tw_canvas.w_canvas.deleteLater()      # schedule it for deletion
            self.tw_canvas.w_canvas = QWidget(self.tw_canvas)
            self.tw_canvas.w_canvas.setLayout(layout)
            self.tw_canvas.view_layout.addWidget(self.tw_canvas.w_canvas)
