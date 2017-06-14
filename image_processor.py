"""Image processor module.

This module creates 2D Image class that contains data plotting, peak searching
and plotting, line cutting (which returns 1D Line class).

This module creates 1D Line class that contains plotting, fitting including
scipy.curve_fit, lmfit and kernel density estimation.

Available classes are one of following:
Image2D
Line1D:
    LineCut;
    Line1DAPS29;
    Line1DSSRL13;
    Hist1DTES.
"""

__author__ = "Yizhi Fang"
__version__ = "2017.05.31"

import re
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import binary_erosion
from scipy.special import wofz
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from lmfit.models import *
from lmfit import Model
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import seaborn as sns


class Image2D:
    """2D Image class.

    Attributes:
        file_name: File name of original data.
        data: 2D array.
    """

    def __init__(self, file_name, data):
        self._file_name = file_name
        self.data = data

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, arr):
        if len(arr.shape) != 2:
            raise ValueError("Image must be built on 2D array!")
        else:
            self._data = arr

    def plot_data(self, saturate_factor=1.0):
        """Plot original data.

        Args:
            saturate_factor: Factor to be divided by max, default is 1.0.
        """
        m, n = self.data.shape
        max_val = np.max(self.data)

        # Somehow simply do original_data = self.data links them so change
        # original_data will also change data.
        original_data = np.zeros(self.data.shape)
        original_data = original_data + self.data
        original_data[original_data == 0] = np.nan

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(original_data,
                       origin="lower left",
                       interpolation="nearest",
                       extent=[0, n, 0, m],
                       vmax=max_val / saturate_factor,
                       cmap=cm.viridis)
        plt.colorbar(im)
        plt.setp(ax,
                 xlabel="x", ylabel="y",
                 title=re.sub("\.[a-z0-9]*$", "", self._file_name))
        fig.set_tight_layout(True)
        plt.show()

    def search_peaks(self, thres_rel):
        """Find local maximum in an image.

        Use dilation methods.

        Args:
            thres_rel: Threshold relative to max of peak value.

        Returns:
            coords: Coordinates of peaks (x, y) i.e. (j, i).
            intensities: Intensities of all peaks.
        """
        # Remove all values smaller than thres_rel * max.
        threshold = thres_rel * np.max(self.data)

        # Somehow simply do reduced_data = self.data links them so change
        # reduced_data will also change data.
        reduced_data = np.zeros(self.data.shape)
        reduced_data = reduced_data + self.data
        reduced_data[reduced_data < threshold] = 0.0

        # Smooth noisy background before passing to maximum filter.
        smoothed_image = gaussian_filter(reduced_data,
                                         sigma=8,
                                         mode="constant")

        # Create 3 x 3 all-connected structure.
        struct = np.ones((3, 3), dtype=bool)

        # Apply the local maximum filter.
        # All pixel is replaced by max within footprint.
        local_max = (maximum_filter(smoothed_image, footprint=struct)
                     == smoothed_image)

        # local_max is a mask that contains the peaks we are looking for but
        # also the background. In order to isolate the peaks we must remove
        # the background from the mask.

        # Create a mask of background.
        background = smoothed_image == 0

        # A little technicality: we must erode the background in order to
        # successfully subtract it form local_max, otherwise a line will appear
        # along the background border (artifact of the local maximum filter).
        eroded_background = binary_erosion(background,
                                           structure=struct,
                                           border_value=1)

        # We obtain the final mask containing only peaks by removing the
        # background from the local_max mask.
        peaks = local_max - eroded_background

        # Here is a tuple with i and j in two 1D array.
        wh = np.where(peaks == True)
        coords = np.zeros((len(wh[0]), 2), dtype=int)
        # Recall x position is column index while y is row.
        coords[:, 0] = wh[1]
        coords[:, 1] = wh[0]

        print("{:d} peaks found in ".format(len(coords))
              + self._file_name + " at pixel (x, y) with pixel intensity:")
        intensities = []
        for i in range(len(coords)):
            # Recall x position is column index while y is row.
            intensity = self.data[coords[i, 1], coords[i, 0]]
            print("Peak at ({:d}, {:d}) "
                  "has intensity {:d}".format(coords[i, 0], coords[i, 1],
                                                int(intensity)))
            intensities.append(intensity)

        return coords, intensities

    def plot_peaks(self, thres_rel, saturate_factor=1.0):
        """Plot original data and peaks.

        Args:
            thres_rel: Threshold relative to max of peak value.
            saturate_factor: Factor to be divided by max, default is 1.0.
        """
        m, n = self.data.shape
        max_val = np.max(self.data)
        coords, intensities = self.search_peaks(thres_rel)

        # Somehow simply do original_data = self.data links them so change
        # original_data will also change data.
        original_data = np.zeros(self.data.shape)
        original_data = original_data + self.data
        original_data[original_data == 0] = np.nan

        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(original_data,
                       origin="lower left",
                       interpolation="nearest",
                       extent=[0, n, 0, m],
                       vmax=max_val / saturate_factor,
                       cmap=cm.viridis)
        plt.colorbar(im)
        # Recall coords are actually [x, y].
        ax.scatter(coords[:, 0], coords[:, 1],
                   marker="x",
                   s=40,
                   c="k",
                   label="Peaks")
        ax.legend(frameon=False)
        plt.setp(ax,
                 xlim=[0, n], ylim=[0, m],
                 xlabel="x", ylabel="y",
                 title=("{:d} peaks found in ".format(len(coords))
                        + re.sub("\.[a-z0-9]*$", "", self._file_name)))
        fig.set_tight_layout(True)
        plt.show()

    def cut_line(self, start_point, end_point):
        """Perform line cuts.

        Args:
            start_point: Start point.
            end_point: End point.

        Returns:
            One CutLine object if cut is along either x or y axis or
            two CutLine objects (one projected along x and the other along y).
        """
        # Line with selected cut range as function of x
        def line_cut_x(x, *args):
            x0, y0, x1, y1 = args
            return (y1-y0) * (x-x0) / (x1-x0) + y0

        # Line with selected cut range as function of y
        def line_cut_y(y, *args):
            x0, y0, x1, y1 = args
            return (x1-x0) * (y-y0) / (y1-y0) + x0

        x0, y0 = start_point
        x1, y1 = end_point
        args = (x0, y0, x1, y1)

        # Recall x position is column index while y is row.
        m = y1 - y0 + 1
        n = x1 - x0 + 1

        cut_range_x = np.linspace(x0, x1, n)
        cut_range_y = np.linspace(y0, y1, m)

        # LineCut object projected in x axis.
        cuts = []
        for i, x in zip(range(n), cut_range_x):
            y = line_cut_x(x, *args)
            data = interp1d(range(self.data.shape[0]),
                            self.data[:, int(x)]).__call__(y)
            cuts.append(data)

        line_x = LineCut(cut_range_x, np.asarray(cuts, dtype=float))

        # LineCut object projected in y axis.
        cuts = []
        for i, y in zip(range(m), cut_range_y):
            x = line_cut_y(y, *args)
            data = interp1d(range(self.data.shape[1]),
                            self.data[int(y), :]).__call__(x)
            cuts.append(data)

        line_y = LineCut(cut_range_y, np.asarray(cuts, dtype=float))

        if m == 1:
            return line_x
        if n == 1:
            return line_y
        else:
            return line_x, line_y


class Line1D:
    """1D line class.

    Attributes:
        x: x data.
        y: y data.
        fig, ax: Figure and axis to be plotted.
        lines: List to store all plotted lines.
    """

    # Make it an Abstract Base Class which is only meant to be inherited from.
    __metaclass__ = ABCMeta

    def __init__(self, x, y, fig=None, ax=None, lines=None):
        self.x = x
        self.y = y
        self.fig = fig
        self.ax = ax
        self.lines = lines

    def plot_line(self, new_plot=False, norm=False):
        """Plot 1D line.

        Args:
            new_plot: Boolean indicating whether crates a new figure,
                      default is False.
            norm: Boolean indicating whether line needs to be normalized,
                  default is False.
        """
        if norm:
            y = (self.y-np.min(self.y)) / (np.max(self.y)-np.min(self.y))
        else:
            y = self.y

        if new_plot:
            lines = []
            # To add frame and remove the background color and grids.
            sns.set_style("ticks", {"xtick.direction": "in",
                                    "ytick.direction": "in"})
            fig, ax = plt.subplots(figsize=(8, 6))
            self.fig = fig
            self.ax = ax
            self.lines = lines

        try:
            plt.ion()
            line, = self.ax.plot(self.x, y, "-o", ms=6, lw=2)
            self.lines.append(line)

            # Change line colors later based on number of lines.
            colors = sns.color_palette("husl", len(self.lines))
            for line, c in zip(self.lines, colors):
                line.set_color(c)

            self.fig.set_tight_layout(True)
            plt.show()

        except AttributeError:
            print("Change new_plot argument to True (default is False)!")

    def fit_by_lmfit(self, p0, method, gamma_default=True, norm=False):
        """Fit a line with defined functions/builtin models by using lmfit,

        lmfit is not a builtin module in Anaconda3 so you need to install
        it by the following command:

        pip install lmfit

        lmfit uses non-linear least-square curve fitting which yields similar
        results with scipy.curve_fit but comes with some standard models and
        more complexity and flexibility.

        The website is https://lmfit.github.io/lmfit-py/intro.html.

        Args:
            p0: List of initial guesses of amplitudes, peak centers and sigmas.
            method: String of fitting methods. Each character indicates a
                    basic function. Choose one of
                    {"g" for Gaussian, "l" for Lorentz and "v" for Voigt}.
                    E.g. "ggg" means fitting with 3 Gaussian functions.
            gamma_default: In VoigtModel only. Boolean indicating whether to
                           use default gamma definition which is constrained
                           to have equal values with sigma.
            norm: Boolean indicating whether line needs to be normalized,
                  default is False.

        Returns:
            result: lmfit ModelResult class,
                    full report can be viewed by print(result.fit_report()),
                    fitted parameters can be accessed by result.params,
                    new values can be calculated by result.eval(x=new_x).
        """
        try:
            if len(p0) % 3 != 0:
                raise ValueError("Initial guess must have 3*num_peak length!")
            elif len(p0) // len(method) != 3:
                raise ValueError("Model length must equal number of peaks!")
            else:
                num_peak = len(p0) // 3  # Number of peaks to be fitted.
                a0, x0, sigma0 = [p0[i:i + num_peak]
                                  for i in range(0, len(p0), num_peak)]

            if norm:
                y = (self.y-np.min(self.y)) / (np.max(self.y)-np.min(self.y))
            else:
                y = self.y

            model = ConstantModel()
            init_params = model.make_params()
            init_params["c"].set(0)

            for i in range(num_peak):
                # Add Gaussian fit.
                if method[i] == "g":
                    temp = GaussianModel(prefix="m{:d}_".format(i))
                # Add Lorentz fit.
                if method[i] == "l":
                    temp = LorentzianModel(prefix="m{:d}_".format(i))
                # Add Voigt fit.
                if method[i] == "v":
                    temp = VoigtModel(prefix="m{:d}_".format(i))

                init_params.update(temp.make_params())
                init_params["m{:d}_amplitude".format(i)].set(a0[i])
                init_params["m{:d}_center".format(i)].set(x0[i])
                init_params["m{:d}_sigma".format(i)].set(sigma0[i])

                # In VoigtModel case, there's an option to set gamma as
                # constrained (default) or independent.
                if not gamma_default:
                    init_params["m{:d}_gamma".format(i)].set(sigma0[i],
                                                             min=0,
                                                             vary=True,
                                                             expr=None)
                    expr = ("1.0692*m{:d}_gamma+sqrt(0.8664*m{:d}_gamma**2"
                            "+5.5451774*m{:d}_sigma**2)".format(i, i, i))
                    init_params["m{:d}_fwhm".format(i)].set(3.60131,
                                                            expr=expr)

                model += temp

            result = model.fit(y, init_params, x=self.x)

            line_fit, = self.ax.plot(self.x, result.best_fit,
                                     lw=2,
                                     ls="--",
                                     label="{:s} fitting".format(method))
            self.lines.append(line_fit)

            # Change line colors later based on number of lines.
            colors = sns.color_palette("husl", len(self.lines))
            for line, c in zip(self.lines, colors):
                line.set_color(c)

            # seaborn sets frame off by default.
            self.ax.legend()
            self.fig.set_tight_layout(True)
            plt.show()

            names = model.param_names[1:]
            # Number of parameters per peak.
            num_params = len(names) // num_peak

            print("\nParameters found for {:s} fitting are:".format(method))
            print("(Peak order does not necessarily match the graphic order.)")
            for i, j in zip(range(num_peak), range(0, len(names), num_params)):
                print("\nPeak #{:d} parameters are:".format(i + 1))
                # Fitted parameters are stored in result.params while
                # init_params ONLY stores initial parameters.
                for n in names[j:j + num_params]:
                    print("{} = {:.5g}".format(n, result.params[n].value))
            print("\nTotal offset is "
                  "{:.5g}".format(result.params["c"].value))

            return result

        except AttributeError:
            print("Plot original data first!")

    def fit_by_kde(self):
        """kde fitting.

        kde fitting creates a new figure, plot original data as step function
        and plot the normalized kde function (kde max matches counts max).

        Returns:
            fig, ax: Current new figure and axis of kde fitting.
        """
        # Create input data for kde fitting that each pixel point in cut_range
        # repeats itself for cuts (counts) times.
        counts = np.asarray(self.y, dtype=int)
        input_data = np.repeat(self.x, counts)

        # To add frame and remove the background color and grids.
        sns.set_style("ticks", {"xtick.direction": "in",
                                "ytick.direction": "in"})
        plt.ion()
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(input_data, bins=self.x, histtype="step", where="mid",
                color="g", lw=2)

        # Add kde fitting.
        kde = sm.nonparametric.KDEUnivariate(input_data)
        kde.fit()

        # To normalize the kde to counts assuming it's linear.
        kde_norm = np.max(self.y) * kde.density / np.max(kde.density)

        # Add kde fitting line.
        ax.plot(kde.support, kde_norm, "k--", lw=2, label="kde fitting")

        # seaborn sets frame off by default.
        ax.legend()
        fig.set_tight_layout(True)
        plt.show()

        return fig, ax

    def gaussian(self, x, a, mu, sigma, offset):
        return (a*np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))
                + offset)

    def lorentz(self, x, a, mu, gamma, offset):
        return a*gamma/(np.pi*((x-mu)**2 + gamma**2)) + offset

    def voigt(self, x, a, mu, sigma, gamma, offset):
        return (a*np.real(wofz((x - mu + 1j*gamma)/(sigma*np.sqrt(2))))
                /(sigma*np.sqrt(2*np.pi)) + offset)

    def remove_line(self, line_index):
        """Remove lines.

        This only remove line from plot but the object is still in self.lines.

        Args:
            line_index: Line index to remove, starting from original data (0).
        """
        self.lines[line_index].remove()

    def add_line(self, line_index):
        """Add removed lines back.

        Args:
            line_index: Line index to remove, starting from original data (0).
        """
        self.ax.add_line(self.lines[line_index])

    @abstractmethod
    def line_type(self):
        pass


class LineCut(Line1D):
    """Line cut class from 2D image.

    Attributes:
        x: Cut range projected along x or y.
        y: Cut data.
        fig, ax: Figure and axis to be plotted.
        lines: List to store all plotted lines.
    """

    def line_type(self):
        return "Line cut from 2D image"


class Line1DAPS29(Line1D):
    """1D scan class at APS Sector 29.

    Attributes:
        x: x data.
        y: y data.
        fig, ax: Figure and axis to be plotted.
        lines: List to store all plotted lines.
    """

    def line_type(self):
        return "1D scan at APS Sector 29"


class Line1DSSRL13(Line1D):
    """1D scan class at SSRL 13-3.

    Attributes:
        x: x data.
        y: y data.
        fig, ax: Figure and axis to be plotted.
        lines: List to store all plotted lines.
    """

    def line_type(self):
        return "1D scan at SSRL 13-3"


class Hist1DTES(Line1D):
    """1D histogram class for TES spectrometer.

    Attributes:
            x: x data.
            y: y data.
            fig, ax: Figure and axis to be plotted.
            lines: List to store all plotted lines.
    """

    def plot_line(self, new_plot=False, norm=False):
        """Plot 1D histogram.

        TES is counting pulses (pulse height approximately equal to photon
        energy) over a period of time so histogram plot is more appropriate to
        plot counts vs photon energies.

        Args:
            new_plot: Boolean indicating whether crates a new figure,
                      default is False.
            norm: Boolean indicating whether line needs to be normalized,
                  default is False.
        """
        if norm:
            y = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))
        else:
            y = self.y

        if new_plot:
            lines = []
            # To add frame and remove the background color and grids.
            sns.set_style("ticks", {"xtick.direction": "in",
                                    "ytick.direction": "in"})
            fig, ax = plt.subplots(figsize=(8, 6))
            self.fig = fig
            self.ax = ax
            self.lines = lines

        try:
            plt.ion()
            line, = self.ax.step(self.x, y, lw=2)
            self.lines.append(line)

            # Change line colors later based on number of lines.
            colors = sns.color_palette("husl", len(self.lines))
            for line, c in zip(self.lines, colors):
                line.set_color(c)

            self.fig.set_tight_layout(True)
            plt.show()

        except AttributeError:
            print("Change new_plot argument to True (default is False)!")

    def resp_func(self):
        """TES response function.

        It actually doesn't have fixed form in general but in a small range of
        energies we could tune parameters by fitting calibration and assume
        it's independent of incident energy.

        In this model, the resp_func function is consisted of one rectangular
        lower energy tail (30 eV wide and 25% weight, a triangular low energy
        tail (4 eV wide and 30% weight) then convolved with Gaussian (1.5 eV
        FWHM).

        Credit by Young IL Joe.
        """
        det_resolution = 1.5
        det_resolution_sigma = det_resolution / (2*np.sqrt(2*np.log(2)))
        tail_fraction = .12
        tail_width = 30
        near_tail_fraction = .24
        near_tail_width = 8

        # Delta functin part.
        det_x = np.linspace(-100, 100, 2001)
        det_y = np.zeros_like(det_x)
        det_y[1000] = 1.0 - (tail_fraction + near_tail_fraction)

        # Rectangular tail part.
        tail_y = np.zeros_like(det_x)
        tail_y[(det_x > -tail_width) & (det_x < 0)] += 1
        tail_y /= np.sum(tail_y) / tail_fraction
        det_y += tail_y

        # Triangular tail part.
        near_tail_y = np.zeros_like(det_x)
        near_tail_y[(det_x > -near_tail_width) & (det_x < 0)] = (det_x
                + near_tail_width)[(det_x > -near_tail_width) & (det_x < 0)]
        near_tail_y /= np.sum(near_tail_y) / near_tail_fraction
        det_y += near_tail_y

        gaussian_x = np.linspace(-50, 50, 1001)
        gaussian_y = np.exp(-gaussian_x**2 / (2*det_resolution_sigma**2))
        gaussian_y /= np.sum(gaussian_y)
        det_y = np.convolve(gaussian_y, det_y, mode="full")[500:-500]

        return det_y

    def basic_fit_func(self, method, num_fit=None):
        """Define fitting function for various methods.

        Args:
            method: Fitting method. Choose one of
                    {"Gaussian", "Lorentz", "Voigt", "Gaussian+Lorentz"}.
            num_fit: Number of first fitting functions in combined methods.

        Return:
            Fitting function and parameter names.
        """
        # Add Gaussian fitting.
        if method == "Gaussian":
            def func(x, *args):
                num_peak = len(args) // 3
                a, mu, sigma = [args[i:i+num_peak]
                                for i in range(0, len(args), num_peak)]
                sum = 0
                for i in range(num_peak):
                    sum += self.gaussian(x, a[i], mu[i], sigma[i], 0)
                return sum

            param_names = ["amplitude", "mu", "sigma"]

        # Add Lorentz fitting.
        if method == "Lorentz":
            def func(x, *args):
                num_peak = len(args) // 3
                a, mu, gamma = [args[i:i+num_peak]
                                for i in range(0, len(args), num_peak)]
                sum = 0
                for i in range(num_peak):
                    sum += self.lorentz(x, a[i], mu[i], gamma[i], 0)
                return sum

            param_names = ["amplitude", "mu", "gamma"]

        # Add Voigt fitting.
        if method == "Voigt":
            def func(x, *args):
                num_peak = len(args) // 3
                a, mu, sigma = [args[i:i+num_peak] for i in
                                range(0, len(args), num_peak)]
                # Default gamma are constrained to have values equal to sigma.
                gamma = sigma
                sum = 0
                for i in range(num_peak):
                    sum += self.voigt(x, a[i], mu[i], sigma[i], gamma[i], 0)
                return sum

            param_names = ["amplitude", "mu", "sigma"]

        # Add Gaussian and Lorentz combined method (Gaussian first)
        if method == "Gaussian+Lorentz":
            def func(x, *args):
                num_peak = len(args) // 3
                a, mu, sigma = [args[i:i+num_peak] for i in
                                range(0, len(args), num_peak)]
                sum = 0
                if num_fit == None:
                    raise TypeError("Must assign number of first fittings!")
                else:
                    for i in range(num_fit):
                        sum += self.gaussian(x, a[i], mu[i], sigma[i], 0)
                    for j in range(num_fit, num_peak):
                        sum += self.lorentz(x, a[j], mu[j], sigma[j], 0)
                return sum

            param_names = ["amplitude", "mu", "sigma/gamma"]

        return func, param_names

    def TESmodel(self, a0, x0, method, num_fit=None, norm=False):
        """Fitting model for TES emission spectra at Sector 29.

        This model is consisted of three parts:
            Elastic peak determined by beam and detector resolution (delta
        function);
            Emission peak coming from harmonic background which is considered
        as constant because they're far away from resonance (Gaussian
        function);
            Emission peak coming from fundamental which is "Raman-like"
        (Gaussian function).

        Args:
            a0: Sequence of initial guesses of peak amplitude.
            x0: Sequence of initial guesses of peak center.
            method: Fitting method. Choose one of
                    {"Gaussian", "Lorentz", "Voigt", "Gaussian+Lorentz"}.
            num_fit: Number of first fitting functions in combined methods.
            norm: Boolean indicating whether line needs to be normalized,
                  default is False.

        Returns:
            params: Parameters found in fitting.
            fwhm: Full width at half maximum.
            fitted_y: Fitted y values.
        """
        try:
            if (len(np.shape(a0)) != 1) or (len(np.shape(x0)) != 1):
                raise TypeError("a0 and x0 need to be sequence!")
            elif len(a0) != len(x0):
                raise ValueError("a0 and x0 must have same length!")
            else:
                num_peak = len(x0)  # Number of peaks to be fitted.

            if norm:
                y = (self.y-np.min(self.y)) / (np.max(self.y)-np.min(self.y))
            else:
                y = self.y

            peaks = self.basic_fit_func(method, num_fit)[0]
            param_names = self.basic_fit_func(method, num_fit)[1]

            def final_fit(x, *args):
                return np.convolve(peaks(x, *args), self.resp_func(),
                                   mode="full")[1000:-1000]

            init_params = [a0,
                           x0,
                           np.ones(num_peak)]
            p0 = [num for param in init_params for num in param]

            params, _ = curve_fit(final_fit, self.x, y, p0)

            # Compute starting index for sigma/gamma and force sigma/gamma to
            # be positive.
            index = range(0, len(params), num_peak)[2]
            params[index:index + num_peak] = [abs(num) for num in
                                              params[index:index + num_peak]]

            fitted_y = final_fit(self.x, *params)

            # # Calculate FWHM for each fitting.
            # fwhm = []
            # if method == "Gaussian":
            #     for sigma in params[index:index+num_peak]:
            #         temp = 2 * sigma * np.sqrt(2*np.log(2))
            #         fwhm.append(temp)
            # if method == "Lorentz":
            #     for gamma in params[index:index+num_peak]:
            #         temp = 2 * gamma
            #         fwhm.append(temp)
            # if method == "Voigt":
            #     sigmas = params[index:index+num_peak]
            #     gammas = sigmas
            #     for sigma, gamma in zip(sigmas, gammas):
            #         fwhm_g = 2 * sigma * np.sqrt(2*np.log(2))
            #         fwhm_l = 2 * gamma
            #         temp = (0.5346*fwhm_l
            #                 + np.sqrt(0.2166*fwhm_l**2 + fwhm_g**2))
            #         fwhm.append(temp)

            line_fit, = self.ax.plot(self.x, fitted_y,
                                     lw=2,
                                     ls="--",
                                     label="{:s} fitting".format(method))
            self.lines.append(line_fit)

            # Change line colors later based on number of lines.
            colors = ["green", "black"]
            for c, line in zip(colors, self.lines):
                line.set_color(c)

            # seaborn sets frame off by default.
            self.ax.legend()
            self.fig.set_tight_layout(True)
            plt.show()

            print("\nParameters found for {:s} fitting are:".format(method))
            print("(Peak order does not necessarily match the graphic order.)")
            for i in range(num_peak):
                print("\nPeak #{:d} parameters are:".format(i + 1))
                if method == "Voigt":
                    print("gamma is constrained to be equal to sigma")
                for j, n in zip(range(i, len(params), num_peak), param_names):
                    print("{} = {:.5g}".format(n, params[j]))
                # print("FWHM estimated to be {:.5g}"
                #       " (only pure Gaussian/Lorentz/Voigt)".format(fwhm[i]))

            return params

        except AttributeError:
            print("Plot original data first!")

    def line_type(self):
        return "1D histogram class for TES spectrometer"