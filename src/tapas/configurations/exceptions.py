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


class InputLogicError(Exception):
    """ exception used when input is ambigous or intervals are invalid """


class FittingError(Exception):
    """ exception used when fitting did not succeed """


class NoSelectionError(Exception):
    """ exception used when user input is missing """


class NoDataError(Exception):
    """ exception used when no raw or processed data is available """
