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

class License:
    notes = '''This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

The font assets included in this distribution are licensed under the SIL Open
Font License, Version 1.1. They remain under the terms of that license regardless
of the licensing of the overall software'''

from .. import __version__

class Widgets:
    i01 = 'Enter ground-state absorbance data path...'
    i02 = 'Enter steady-state emission data path...'
    i03 = 'Enter transient absorbance data path...'
    i04 = 'No Absorbance Data imported'
    i05 = 'No Emission Data imported'
    i06 = 'No TA Data imported'
    i07 = 'delay'
    i08 = 'x-pos'
    i09 = 'y-pos'
    i10 = 'min'
    i11 = 'max'
    i12 = 'threshold'
    i13 = 'No Data'
    i14 = '*.hdf project path'
    i15 = 'no Project imported'
    i16 = 'no combined project'
    i17 = 'Delay1, Delay2, ..'
    i18 = 'Wavelength1, Wavelength2, ..'
    i19 = 'Color1, Color2, ..'
    i20 = 'This creates a new project. Unsaved changes in the old project won\'t be saved. Continue anyway?'
    i21 = '>0'



class ToolTips:
    t01 = 'Create a new project'
    t02 = 'Open a project'
    t03 = 'Save the project'
    t04 = 'Exit'
    t05 = 'About'
    t06 = ('leaving this unchecked lets you edit the text in programs like\n'
           'Inkscape or Illustrator but may lead to improper rendering if the\n'
           'font is not installed on the local machine')
    t07 = 'Save the project as'
    t08 = 'use copy-paste, drag-drop, or Browse Button to insert datapath'
    t09 = 'opens file manager to select one or multiple files'
    t10 = 'delimiter used in the loaded file'
    t11 = 'how many starting rows are ignored when importing the data'
    t12 = 'original unit of the delay time'
    t13 = 'original unit of the spectral wavelength'
    t14 = 'original unit of the differential absorbance'
    t15 = 'loads the data file'
    t16 = 'clears the loaded data and subsequent processing'
    t17 = 'predefined plotting styles'
    t18 = 'working copy of the raw data'
    t19 = 'resets dataset and subsequent fits to raw data'
    t20 = 'show the full data range'
    t21 = 'show the data close to time zero'
    t22 = 'point where delay time axis changes from linear to logarithmic scale'
    t23 = 'Crop the dataset to the given interval'
    t24 = 'save the previewed change to the dataset'
    t25 = 'up/down sample a given grid'
    t26 = 'intervall over which resampling is applied'
    t27 = '0-1: downsample | 1-99: upsample'
    t28 = 'interpolation method'
    t29 = 'change affects only current axis'
    t30 = 'evenly spread data over datapoints'
    t31 = 'number of samples on current axis'
    t32 = 'use heuristic on the right to calculate chirp'
    t33 = 'threshold used for autocorrection'
    t34 = 'use chirpfunction from hdf project'
    t35 = 'dataset from which function is used'
    t36 = 'if active, left-click in Canvas: add chirp point | right-click: remove chirp point'
    t37 = 'fit current chirp points'
    t38 = 'delete fit & chirp points'
    t39 = 'correct for chirp'
    t40 = 'delete chirp correction preview'
    t41 = 'delete resampling preview'
    t42 = 'subtract averaged area of intervall'
    t43 = 'delay intervall used to calculate background'
    t44 = 'subtract background up to delay'
    t45 = 'average background data using'
    t46 = 'subtract surface from hdf project'
    t47 = 'subtract solvent surface imported to this project'
    t48 = 'delete previewed correction'
    t49 = 'delete previewed filter'
    t50 = 'preview filter'
    t51 = "new 'time zero' delay time"
    t52 = 'filter algorithm'
    t53 = 'filter subset length/size'
    t54 = 'order of the polynomial used to fit the data'
    t55 = 'dateset from which surface is used'
    t56 = 'dimensions of the imported dataset'
    t57 = 'deletes the current imported project'
    t58 = 'select what happens when both imports share wavelengths or delay times'
    t59 = 'method used to project data over new unified grid'
    t60 = 'any missing grid values will be set to 0 or estimated by linearly interpolating from the nearest data points'
    t61 = 'preview the combined project'
    t62 = 'create a new project with the combined data as new root raw data'
    t63 = 'display the meta information from the import tab as the title'
    t64 = 'set x-axis intervall which will not be plotted'
    t65 = 'set the axis scale. linlog uses a lin scale up to a thershold and the a log scale'
    t66 = 'center point used for the ΔA colorbar'
    t67 = 'display the pump as vertical line if set as meta data in import tab'
    t68 = 'display eV axis at the top'
    t69 = "show experimental data as 'x'"
    t70 = 'displays ΔA<sub>exp</sub> – ΔA<sub>calc</sub> below the plot'
    t71 = 'display legend at selected position'
    t72 = 'colors of the lines. Set explicitly, use a continous colormap, or the style defaults'
    t73 = 'line colors are evenly spread over continous colormap'
    t74 = 'set colors explicitly via name or hex code'
    t75 = 'plots comma-separated list of delay times'
    t76 = 'plots comma-separated list of wavelengths'
    t77 = 'normalize y-axis so that the maximum value witin the interval is 1'
    t78 = 'x-axis normalization interval'
    t79 = 'display the absolute values of ΔA'
    t80 = 'select fit to display or create and save a fit first'
    t81 = 'set relative ratio of linear logarithmic and optionally steady-state area'
    t82 = 'display and position colorbar or change ΔA colorscheme'
    t83 = 'if checked, the selected line profiles from the DelA or Kin Trace plot are overlayed'
    t84 = 'plot the 2D ΔA contour map'
    t85 = 'plot spectral ΔA slices'
    t86 = 'plot kinetic traces'
    t87 = 'plot the results of the local kinetic fits'
    t88 = 'plot the results of the global fits'
    t89 = 'style used for saving the figure'
    t90 = 'overwrite or add specific Matplotlib rc parameters'
    t91 = 'set figure width'
    t92 = 'set figure height'
    t93 = 'approximate and display real figure size'
    t94 = 'printed DPI for raster graafics'
    t95 = ('directory where the figures or exported data are saved to.\n'
           'Subfolder structure is created automatically based on meta information')
    t96 = 'font size of all the main labels and titles'
    t97 = 'font size of the tick labes and handles'
    t98 = 'save current figure'
    t99 = 'select font from list of installed fonts on your os'
    t100 = 'choose image file format'
    t101 = 'number of leading singular components to calculate and plot'
    t102 = 'change plotting limits to the given intervals'
    t103 = 'select wavelength which will be fitted'
    t104 = 'define an area around target wavelength which will be averaged'
    t105 = 'number of individual components used for fitting'
    t106 = 'if checked, an additional non decaying component is added to the fit model'
    t107 = 'fitting model used'
    t108 = "define 'time zero' as the 5% rise of the IRF or the maximum of the IRF"
    t109 = ('Nelder-Mead: simplex method which does not require derivatives. robust but slow\n'
            'leastsq: Gradient-based non-linear least-squares solver. fast but can fail if ill-conditioned problem\n'
            'Differential Evolution: Population-based global optimizer. slow but only relies on bounds')
    t110 = 'define initial guess, bounds and if parameter is fixed to starting value (vary=False)'
    t111 = 'model the data only with the initial starting values. Good to check if guess is justified'
    t112 = 'try to fit the data and estimate fit metrics'
    t113 = 'add a component to model the ground-state bleaching'
    t114 = ('use the steady-state absorbance data to model the ground state.\n'
            'a spectral shift and broadening parameter will be added')
    t115 = 'Per-sample importance factors. Residuals are scaled by √weight to account for varying uncertainties'
    t116 = 'calculated amplitude spectra will be normalized to 1'
    t117 = 'calculate each residual as a percentage of the measured ΔA'
    t118 = 'use Markov chain Monte Carlo (MCMC) to obtain the posterior probability distribution of parameters'
    t119 = 'number of samples to discard at the beginning of the sampling regime'
    t120 = 'number of initial samples drawn from the distribution'
    t121 = 'thin the samples'
    t122 = 'the chain should be longer than x times the integrated autocorr time'
    t123 = ('performs multiple runs which can take time:\n'
            '1) warmup\n'
            '2) initial run\n'
            '3) owards additional runs to satisfy target ratio')
    t124 = 'cancel evaluation after the next run'
    t125 = 'Number of RK4 sub-steps to take within each interval'
    t126 = 'define the structure of the imported 2D matrix'
    t127 = 'plot either the fitted 2D map, the residuals or a combination of both'
    t128 = 'below the noise threshold the residual structure is not displayed'
    t129 = 'Histogram resolution: number of bins used for each 1-D distribution'
    t130 = 'Extra space between axis label and ticks (in points)'
    t131 = 'Gap between subplots, given as a fraction of the panel size'
    t132 = 'Upper limit on tick marks per axis to keep labels readable'
    t133 = 'Reference values to mark with lines in every subplot'
    t134 = 'Show median ± σ statistics as titles on the diagonal plots'
    t135 = 'Percentiles to draw as vertical lines (16 % & 84 %)'
    t136 = 'Overlay individual sample points on the 2-D panels'
    t137 = 'Draw confidence-level contours on the 2-D histograms'
    t138 = 'Fill 2-D histograms with shaded density'
    t139 = 'model coherent artefacts'
    t140 = ('zero order: use the Gaussian IRF to model the CA\n'
            'zero + first order: use the Gaussian IRF and the 1st derivative to model the CA')


class Status:
    s_splash = [
        "Loading core modules…",
        "Initializing analysis engine…",
        "Warming up computational backend…",
        "Preparing user interface…",
        "Almost there…",
        "First launch cached — next start will be faster!"]
    s00 = f'TAPAS: Transient Absorption Processing & Analysis Software\n\nTAPAS Version {__version__} stable\n\nPython 3.11.11\nPyQt 6.9\n\nCopyright © 2025 Philipp Frech\nUniversity of Tübingen\n\n'+ ('_' * 50)+'\n\nLicensed under GPL-3.0-or-later\n'
    s01 = 'path changed'
    s02 = 'data fetched'
    s03 = 'data cleared'
    s04 = 'maximum zoom reached'
    s05 = 'minimum zoom reached'
    s06 = 'project saved'
    s07 = 'project loaded'
    s08 = 'background subtracted in plot'
    s09 = 'background applied to dataset'
    s10 = 'chirp function calculated'
    s11 = 'chirp function deleted'
    s12 = 'chirp function applied'
    s13 = 'trim applied to dataset'
    s14 = 'filter applied in plot'
    s15 = 'filter applied to dataset'
    s16 = 'configuration file loaded'
    s17 = 'configuration file saved'
    s18 = 'populate delay times first'
    s19 = 'plot saved'
    s20 = 'data exported'
    s21 = 'correction removed'
    s22 = 'chirpfile loaded'
    s23 = 'SVD calculated'
    s24 = 'current fit saved to project'
    s25 = 'resampling in canvas'
    s26 = 'resampling applied to dataset'
    s27 = 'time zero correction applied to dataset'
    s28 = 'two projects merged succesfully in preview'
    s29 = 'two projects merged succesfully to new project'
    s30 = 'modeling initial guess succeeded'
    s31 = 'fitting algorithm finished successfully'
    s32 = 'fit loaded successfully'
    s33 = 'fit deleted from project'
    s34 = 'emcee analysis will start now'
    s35 = 'emcee analysis will be aborted after the next run. Please wait'
    s36 = 'emcee results saved to project'
    

class Error:
    e01 = 'unknown exception occurred. See log for more detail'
    e02 = 'wrong Input Format. '
    e03 = 'File not found'
    e04 = 'Project already opened elsewhere - unable to save the project. '
    e05 = 'data needs to be fetched first'
    e06 = 'min value must be smaller than max value, intervall must be within data'
    e07 = 'no chirp correction found to apply'
    e08 = 'no valid points found'
    e09 = 'at least 5 points needed for chirp fit'
    e10 = 'The order of the polynomial msut be less than the window length'
    e13 = 'no change to apply'
    e14 = 'min value, center and max value must be in ascending order'
    e15 = 'Font not found. Fallback used'
    e16 = 'tight layout not applicable. Figsize is slightly adjusted'
    e17 = 'directory not found'
    e18 = 'wavelength needs to be entered first'
    e19 = 'fitting algorithm failed. Try different starting parameters'
    e20 = 'fitting algorithm failed. Nothing to optimize.'
    e21 = 'no fit to save'
    e22 = 'no fit selected'
    e23 = 'computation does not converge. Try a different model'
    e24 = 'no fitted data found'
    e25 = 'currently only the fist 9 fits can be displayed'
    e26 = 'add pump wavelength to the metadata first'
    e27 = 'this parameter does not exist'
    e28 = 'pump outside of plotting window'
    e29 = 'bound interval needs to exceed 100fs or value should be fixed'
    e30 = 'Probably wrong delimiter input. '
    e31 = 'Probably wrong file format. '
    e32 = 'ambiguous input: if you provide two files, every file can only contain one dataset'
    e33 = 'ambiguous input: if you provide one file, only up to two datasets (before - after) are supported.'
    e34 = 'ambiguous input: only up to two files comtaining data before and after the TA experiment are supported'
    e35 = 'layout engine cannot prevent clipping of some artists. An unclipped figure is also saved.'
    e36 = 'select a fit in the table above first'
    e37 = 'dataset not found'
    e38 = 'selected dataset is empty'
    e39 = 'project does not contain TA data'
    e40 = 'unexpected project structure - could not load data'
    e41 = 'a background correction method needs to be performed first'
    e42 = 'data and correction surface must have the same shape'
    e43 = 'add solvent data in Import Tab first'
    e44 = 'The model produces non-finite numbers. Try different starting values, test them with the initial guess and add realistic bounds.'
    e45 = 'wrong factor value input. Should be a positive float or int.'
    e46 = 'the TA wavelength grid is too irregualar - resample to a regular grid for explicit abs data modeling.'
    e47 = 'resample the data first'
    e48 = 'file needs to have the correct hdf format'
    e49 = "cannot extract number of components n. Custom model needs '_nk' signature"
    

class Labels:
    wavelength = 'Wavelength (nm)'
    delay = 'Delay Time (s)'
    delA = r'$\Delta A$ (mOD)'
    absorbance = "Absorbance (a.u.)"
    intensity = "Intensity (a.u.)"
    delA_norm = r'$\Delta A$ $(norm.)$'
    delA_error = r'$\Delta A- \Delta A_{calc}$ (mO)'
    residuals = r'(weighted) residuals (mOD)'
    norm_residuals = r'Percent residuals (%)'