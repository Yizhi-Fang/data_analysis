"""I/O module.

This module creates a File class that can fetch data and/or plot data.

Available classes are one of following:
File:
    MCPDataFile;
    TESConfigFile;
    SPECParamFile;
    ScanFileAPS29;
    SCanFileSSRL13.
"""

__author__ = "Yizhi Fang"
__version__ = "2017.04.17"

import re
from os.path import join
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns


class File:
    """File class.

    Attributes:
        path: File path.
        file_name: File name.
    """

    # Make it an Abstract Base Class which is only meant to be inherited from.
    __metaclass__ = ABCMeta

    # Regular expression signals start and end of data.
    _signs = ("", "")
    # Extra lines between signal lines and data, default is (1, 1).
    _extra_lines = (1, 1)
    # Separation in data file, default is space.
    _sep = "\s+"

    def __init__(self, path, file_name):
        self._path = path
        self._file_name = file_name

    def search_line(self):
        """Search signal lines for data.

        Returns:
            skiprows:  Number of rows to skip from start.
            nrows: Number of rows to read.
        """
        file = join(self._path, self._file_name)
        with open(file, "r") as f:
            for i, line in enumerate(f):
                if re.search(self._signs[0], line):
                    start = i + self._extra_lines[0]
                if re.search(self._signs[1], line):
                    end = i - self._extra_lines[1]

        # Check if end exists. If not, returns NameError.
        # If it doesn't exist then return the last line number.
        try:
            end
        except NameError:
            end = i

        skiprows = start
        nrows = end - start + 1

        return skiprows, nrows

    def fetch_data(self, as_df=False):
        """Fetch all data from file.

        Args:
            as_df: Boolean indicating whether return data as data frame,
                   default is False.

        Returns:
            2D array (data) if as_df is False;
            data frame (data) if as_df is True.
        """
        file = join(self._path, self._file_name)
        data = pd.read_table(file,
                             skiprows=self.search_line()[0],
                             nrows=self.search_line()[1],
                             sep=self._sep,
                             header=None)
        if not as_df:
            data = np.asarray(data)
        return data

    @abstractmethod
    def plot_data(self, *kwargs):
        pass

    @abstractmethod
    def file_type(self):
        pass


class MCPDataFile(File):
    """MCP data file (*.mcp) class.

    Attributes:
        path: File path.
        file_name: File name.
    """

    # Regular expression signals start and end of data.
    _signs = ("^\[CDAT0", "^\[CDAT1")
    # ExtraLines between signal lines and data.
    _extra_lines = (14, 1)
    # Separation in data file.
    _sep = "\t"
    # x, y dimensions.
    _dim = (1024, 1024)

    def fetch_data(self, as_df=False):
        """Fetch data from MCP data file.

        Args:
            as_df: Boolean indicating whether return data as data frame,
                   default is False.

        Returns:
            data: 2D array.
        """
        df = super().fetch_data()
        data = np.zeros((self._dim[0], self._dim[1]))
        # Recall x position is column index while y is row
        data[df[1], df[0]] = df[2]
        return data

    def plot_data(self, saturate_factor=1.0):
        """Plot data from MCP data.

        Data must be 2D array.
        All 0 values are replaced by nan for imshow() convenience.

        Args:
            saturate_factor: Factor to be divided by max, default is 1.0.
        """
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 6))
        data = self.fetch_data()
        max_val = np.max(data) # Couldn't find max if array contains nan.
        data[data == 0] = np.nan
        im = ax.imshow(data,
                       origin="lower left",
                       interpolation="nearest",
                       extent=[0, self._dim[0], 0, self._dim[1]],
                       vmax=max_val / saturate_factor,
                       cmap=cm.viridis)
        plt.colorbar(im)
        plt.setp(ax,
                 xlabel="x", ylabel="y",
                 title=re.sub("\.[a-z0-9]+$", "", self._file_name))
        fig.set_tight_layout(True)
        plt.show()

    def file_type(self):
        return "MCP data file (*.mcp)"


class TESConfigFile(File):
    """TES configuration file (*.cfg) class.

    Attributes:
        path: File path.
        file_name: File name.
    """

    # Regular expression signals start and end of data.
    _signs = ("^spacing", "None")

    def plot_data(self, ncols, nrows):
        """Plot TES configuration.

        Args:
            ncols: Number of columns active in TES.
            nrows: Number of rows active in TES.
        """
        class Rectangle(object):
            """Rectangular pixel to draw.

            Attributes:
                data: Information for rectangle (row in configuration data).
                fc: A color palette (seaborn) indicating the face color
                    of rectangle object.
            """

            def __init__(self, data, fc):
                self.data = data
                self.fc = fc

            def draw(self):
                """Draw rectangle.

                Returns:
                    Rectangle patch.
                """

                x = self.data[1] - self.data[4]/2
                y = self.data[2] - self.data[5]/2
                w = self.data[4]
                h = self.data[5]
                return plt.Rectangle((x, y), w, h,
                                     fc=self.fc,
                                     ec="none")

            def add_text(self):
                """Add channel number.

                Returns:
                    x: Channel number position x.
                    y: Channel number position y.
                    text: Channel number.
                """
                x = self.data[1]
                y = self.data[2]
                text = str(self.data[0])
                return x, y, text

        config_data = self.fetch_data()

        sns.set_style("white")
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 9))

        # Color for each column.
        color_col = sns.color_palette("husl", ncols)
        for idx_col, c in zip(range(0, len(config_data), nrows), color_col):
            for data in config_data[idx_col:idx_col+nrows, :]:
                rectangle = Rectangle(data, c)
                ax.add_patch(rectangle.draw())
                x = rectangle.add_text()[0]
                y = rectangle.add_text()[1]
                text = rectangle.add_text()[2]
                ax.text(x, y, text, size=10, ha="center", va="center")

        ax.axhline(y=0, ls="--", c="k")
        ax.axvline(x=0, ls="--", c="k")

        # This is important in order to see the patch.
        plt.axis("scaled")
        plt.setp(ax,
                 xlim=[-5000, 5000],
                 ylim=[-5000, 5000],
                 title="{:d} columns x {:d} rows".format(ncols, nrows))
        fig.set_tight_layout(True)
        plt.show()

    def file_type(self):
        return "TES configuration file (*.cfg)"


class SPECParamFile(File):
    """SPEC parameter file (*.txt) class.

    Attributes:
        path: File path.
        file_name: File name.
    """

    # Regular expression signals start and end of data.
    _signs = ("^Orientation", "^Primary")
    # ExtraLines between signal lines and data.
    _extra_lines = (1, 2)

    def file_type(self):
        return "SPEC parameters file (*.txt)"


class ScanFileAPS29(File):
    """1D Scan file at APS Sector 29 (*.asc) class.

    Attributes:
        path: File path.
        file_name: File name.
    """

    # Regular expression signals start and end of data.
    _signs = ("^\# 1-D", "None")

    def fetch_data(self, as_df=False):
        """Fetch scan data at APS Sector 29.

        The first column is just index so ignore it.

        Args:
            as_df: Boolean indicating whether return data as data frame,
                   default is False.

        Returns:
            2D array, each column represents a 1D data.
        """
        data = super().fetch_data()
        return data[:, 1:]

    def col_num(self, chans=[]):
        """Find out column number for certain variables.
        
        Args:
            chans: A list of channels recorded.

        Returns:
            Dictionary with variable names as key and column numbers as value.
        """
        col_table = {}
        with open(join(self._path, self._file_name), "r") as f:
            for i, line in enumerate(f):
                if re.search("29idb:ca14:read", line):
                    col_table["I0"] = int(line.split()[1]) - 2
                if re.search("29idd:ca2:read", line):
                    col_table["TEY"] = int(line.split()[1]) - 2
                if re.search("29idd:ca3:read", line):
                    col_table["PhDiode"] = int(line.split()[1]) - 2
                if re.search("29ID:TES:trigger_rate", line):
                    col_table["TES_total"] = int(line.split()[1]) - 2
                for c in chans:
                    if re.search("29ID:TES:chan{:d}_trigger_rate".format(c),
                                 line):
                        col_table["TES_chan{:d}".format(c)] = (
                            int(line.split()[1]) - 2)
        return col_table

    def plot_data(self):
        """Plot scan record at APS Sector 29.

        All data points will be normalized with I0 except I0.
        """
        col_table = self.col_num()
        data = self.fetch_data()

        # To add frame and remove the background color and grids.
        sns.set_style("white")
        sns.set_palette("husl", len(col_table))
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 6))
        for key in col_table.keys():
            if key != "I0":
                y = data[:, col_table[key]] / data[:, col_table["I0"]]
            else:
                y = data[:, col_table[key]]
            if y.all() == 0:
                y_norm = y  # Otherwise y_norm is nan.
            else:
                y_norm = (y-np.min(y)) / (np.max(y)-np.min(y))
            ax.plot(data[:, 0], y_norm, "-o", ms=6, lw=2, label=key)
        # seaborn sets frame off by default.
        ax.legend()
        plt.setp(ax,
                 title=re.sub("\.[a-z0-9]+$", "", self._file_name),
                 ylim=[-0.1, 1.1])
        fig.set_tight_layout(True)
        plt.show()

    def file_type(self):
        return "1D Scan file at APS Sector 29 (*.asc)"


class ScanFileSSRL13(File):
    """1D scan file from SPEC at SSRL 13-3.

    All scans are stored in one single file in SPEC so it needs an extra
    attribute - scan number.

    Attributes:
        path: File path.
        file_name: File name.
        scan_num: Scan number.
    """

    # ExtraLines between signal lines and data.
    _extra_lines = (16, 2)

    def __init__(self, path, file_name, scan_num):
        self._path = path
        self._file_name = file_name
        self._scan_num = scan_num

    @property
    def scan_num(self):
        return self._scan_num

    @scan_num.setter
    def scan_num(self, val):
        if val <= 0:
            raise ValueError("Scan number starts from 1!")
        else:
            self._scan_num = val

    def search_line(self):
        """Search signal lines for data.

        Returns:
            skiprows:  Number of rows to skip from start.
            nrows: Number of rows to read.
        """
        self._signs = ("^\#S\s+{:d}\s+".format(self._scan_num),
                       "^\#S\s+{:d}\s+".format(self._scan_num + 1))
        return super().search_line()

    def col_num(self):
        """Find out column number for certain variables.

        Returns:
            Dictionary with variable names as key and column numbers as value.
        """
        file = join(self._path, self._file_name)
        skiprows, _ = self.search_line()
        with open(file, "r") as f:
            header = f.readlines()[skiprows - 1]
        header = header.split()[1:]
        col_table = {}
        for key in ["H", "K", "L", "ChnTron", "TEY", "PdNorm", "Monitor"]:
            try:
                col_table[key] = header.index(key)
            except ValueError:
                pass
        return col_table

    def plot_data(self):
        """Plot scan record at SSRL 13-3.

        All data points will be normalized with I0 except I0.
        """
        col_table = self.col_num()
        data = self.fetch_data()

        # To add frame and remove the background color and grids.
        sns.set_style("white")
        sns.set_palette("husl", len(col_table))
        plt.ion()
        fig, ax = plt.subplots(figsize=(9, 6))
        for key in col_table.keys():
            if key != "Monitor":
                y = data[:, col_table[key]] / data[:, col_table["Monitor"]]
            else:
                y = data[:, col_table["Monitor"]]
            if y.all() == 0:
                y_norm = y  # Otherwise y_norm is nan.
            else:
                y_norm = (y-np.min(y)) / (np.max(y)-np.min(y))
            ax.plot(data[:, 0], y_norm, "-o", ms=6, lw=2, label=key)
        # seaborn sets frame off by default.
        ax.legend()
        plt.setp(ax,
                 title=re.sub("\.[a-z0-9]+$", "", self._file_name),
                 ylim=[-0.1, 1.1])
        fig.set_tight_layout(True)
        plt.show()

    def file_type(self):
        return "1D scan file at SSRL 13-3"
