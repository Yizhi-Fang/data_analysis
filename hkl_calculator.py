"""hkl mapping module.

This module creates AreaDetector class that can calculate hkl for each pixel
given all angles in 4-circle diffractometer.

This module is accurate with given orientation matrix UB calculated from spec;
however, with respect to building orientation matrix UB, there's algorithmic
error between my calculation and spec ;(.

Available classes are one of following:
AreaDetector:
    MCP;
    TES.
"""

__author__ = "Yizhi Fang"
__version__ = "2016.11.21"

from abc import ABCMeta, abstractmethod

import numpy as np
import numpy.linalg as la
import pandas as pd
from tabulate import tabulate


class AreaDetector:
    """Area detector class.

    Attributes:
        distance: Distance from sample to detector center.
    """

    # Make it an Abstract Base Class which is only meant to be inherited from.
    __metaclass__ = ABCMeta

    # Center pixel, default is (0, 0).
    _i0, _j0 = (0, 0)
    # Pixel size, default is (1.0, 1.0).
    _size = (1.0, 1.0)
    # Default primary and secondary reflections.
    _refs = pd.DataFrame(data=np.array([[1.54, 60, 30, 90, 0, 0, 0, 1],
                                        [1.54, 60, 120, 90, 0, 1, 0, 0]]),
                         index=["primary", "secondary"],
                         columns=["lambda", "tth", "th",
                                 "chi", "phi", "H", "K", "L"],
                         dtype=float)
    # Orientation matrix.
    _ub = np.identity(3)
    # Lattice parameters.
    _lat = pd.DataFrame(data=np.array([[1, 1, 1, 90, 90, 90],
                                       [6.2832, 6.28312, 6.2832, 90, 90, 90]]),
                        index=["real", "reciprocal"],
                        columns=["a", "b", "c", "alpha", "beta", "gamma"],
                        dtype=float)

    def __init__(self, distance):
        self._distance = distance

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, val):
        if val <= 0:
            raise ValueError("Distance has to be positive!")
        else:
            self._distance = val

    def rot_x(self, angle):
        """Rotation matrix around x.

        Left-handed rotation.
        Right-handed angle positive.

        Args:
            angle: Angle in degree unit.

        Returns:
            Rotation matrix around x.
        """
        radian = np.deg2rad(angle)
        return np.array([[1, 0, 0],
                         [0, np.cos(radian), np.sin(radian)],
                         [0, -np.sin(radian), np.cos(radian)]], dtype=float)

    def rot_y(self, angle):
        """Rotation matrix around y.

        Left-handed rotation.
        Right-handed angle positive.

        Args:
            angle: Angle in degree unit.

        Returns:
            Rotation matrix around y.
        """
        radian = np.deg2rad(angle)
        return np.array([[np.cos(radian), 0, -np.sin(radian)],
                         [0, 1, 0],
                         [np.sin(radian), 0, np.cos(radian)]], dtype=float)

    def rot_z(self, angle):
        """Rotation matrix around z.

        Left-handed rotation.
        Right-handed angle positive.

        Args:
            angle: Angle in degree unit.

        Returns:
            Rotation matrix around z.
        """
        radian = np.deg2rad(angle)
        return np.array([[np.cos(radian), np.sin(radian), 0],
                         [-np.sin(radian), np.cos(radian), 0, ],
                         [0, 0, 1]], dtype=float)

    def get_triple(self, v1, v2):
        """Construct orthogonal 3x3 matrix from two vectors.

        Args:
            v1, v2: Two non-parallel vectors.

        Returns:
            Unit orthogonal 3x3 matrix.
        """

        # Triple vector1 - parallel to v1.
        t1 = v1 / la.norm(v1)

        # Triple vector2 - normal to the plane of (v1, v2)
        t2 = np.cross(v1, v2, axis=0)  # axis = 0 means column vector.
        t2 = t2 / la.norm(t2)

        # Triple vector3 - in plane of (v1, v2) and normal to v1.
        t3 = np.cross(v1, t2, axis=0)
        t3 = t3 / la.norm(t3)

        return np.concatenate((t1, t2, t3), axis=1)

    def get_pixel_tth(self, i, j):
        """Calculate pixel_tth for pixel (i, j).

        Args:
            i, j: Pixel (i, j).

        Returns:
            pixel_tth: pixel position in detector frame.
        """
        x = (i - self._i0) * self._size[0]
        y = self._distance
        z = (j - self._j0) * self._size[1]
        return np.array([[x],
                         [y],
                         [z]], dtype=float)

    def get_pixel(self, pixel_tth):
        """Calculate pixel (i, j) for pixel_tth.

        Args:
            pixel_tth: pixel position in detector frame.

        Returns:
            Pixel i, j.
        """
        i = pixel_tth[0, 0]/self._size[0] + self._i0
        j = pixel_tth[2, 0]/self._size[1] + self._j0
        return i, j

    def calc_b(self, lat_params):
        """Calculate matrix B.

        Args:
            lat_params: Lattice parameters, a, b, c, alpha, beta and gamma.

        Returns:
            Transformation matrix B from reciprocal to crystal.
        """
        a, b, c, alpha, beta, gamma = lat_params

        # Real space unit vectors.
        a1 = np.array([[a],
                       [0],
                       [0]])
        a2 = np.array([[0],
                       [b],
                       [0]])
        a3 = np.array([[0],
                       [0],
                       [c]])

        # Volume of unit cell.
        vol = a1.T @ np.cross(a2, a3, axis=0)

        # Reciprocal space unit vectors.
        b1 = 2 * np.pi * np.cross(a2, a3, axis=0) / vol
        b2 = 2 * np.pi * np.cross(a3, a1, axis=0) / vol
        b3 = 2 * np.pi * np.cross(a1, a2, axis=0) / vol

        # Angles in reciprocal unit cell.
        alpha1 = float(np.arccos(b1.T@b2/(la.norm(b1)*la.norm(b2))))
        beta1 = float(np.arccos(b2.T @ b3/(la.norm(b2)*la.norm(b3))))
        gamma1 = float(np.arccos(b1.T @ b3/(la.norm(b1)*la.norm(b3))))

        recip = [la.norm(b1), la.norm(b2), la.norm(b3),
                 np.rad2deg(alpha1), np.rad2deg(beta1), np.rad2deg(gamma1)]
        self._lat.iloc[1] = recip

        # Calculate matrix B.
        col1 = np.array([[la.norm(b1)],
                         [0],
                         [0]])
        col2 = np.array([[la.norm(b2)*np.cos(gamma1)],
                         [la.norm(b2)*np.sin(gamma1)],
                         [0]])
        col3 = np.array([[la.norm(b3)*np.cos(beta1)],
                         [-la.norm(b3)*np.sin(beta1)*np.cos(alpha)],
                         [2*np.pi/la.norm(a3)]])

        return np.concatenate((col1, col2, col3), axis=1)

    def calc_q_phi(self, i, j, *args):
        """Calculate q_phi for pixel (i, j).

        Args:
            i, j: Pixel (i, j).
            *args: lamb, tth, th, chi, phi.

        Returns:
            q_phi: Momentum transfer in phi frame.
        """
        lamb, tth, th, chi, phi = args

        # Transformation matrices for all frames.
        tth_trans = self.rot_z(tth)
        th_trans = self.rot_z(th)
        chi_trans = self.rot_y(-chi)
        phi_trans = self.rot_z(phi)

        pixel_lab = tth_trans @ self.get_pixel_tth(i, j)
        wave_vec = 2.0 * np.pi / lamb
        kf_lab = wave_vec * pixel_lab / la.norm(pixel_lab)
        ki_lab = wave_vec * np.array([[0],
                                      [1],
                                      [0]], dtype=float)
        q_lab = kf_lab - ki_lab

        return (la.inv(phi_trans) @ la.inv(chi_trans)
                @ la.inv(th_trans) @ q_lab)

    def set_or(self, pixel1=(0, 0), pixel2=(0, 0)):
        """Construct orientation matrix UB.

        Reflections and lattice parameters will be automatically updated.
        There's algorithmic error between my calculation and spec ;(.

        Args:
            pixel1: (i1, j1), first pixel for primary reflection.
            pixel2: (i2, j2), second pixel for secondary reflection.

        Returns:
            ub: Orientation matrix.
        """
        # Initialize lattice parameters.
        print("\nEnter lattice parameters:")
        for name in self._lat.columns:
            temp = input(name + " {:g}: ".format(self._lat.ix[0, name]))
            if temp != "":
                self._lat.ix[0, name] = float(temp)

        lat_params = self._lat.iloc[0]
        b_matrix = self.calc_b(lat_params)

        # Initialize primary and secondary reflections.
        for ref in self._refs.index:
            print("\nEnter {:s} reflection:".format(ref))
            for arg in self._refs.columns:
                temp = input(arg
                             + " ({:g}): ".format(self._refs.ix[ref, arg]))
                if temp != "":
                    self._refs.ix[ref, arg] = float(temp)

        q_phi_all = []
        q_c_all = []
        for ref, pixel in zip(self._refs.index, [pixel1, pixel2]):
            args = (self._refs.ix[ref, "lambda"],
                    self._refs.ix[ref, "tth"],
                    self._refs.ix[ref, "th"],
                    self._refs.ix[ref, "chi"],
                    self._refs.ix[ref, "phi"])
            q_phi = self.calc_q_phi(pixel[0], pixel[1], *args)
            q_phi_all.append(q_phi)

            q = np.array([[self._refs.ix[ref, "H"]],
                          [self._refs.ix[ref, "K"]],
                          [self._refs.ix[ref, "L"]]])
            q_c = b_matrix @ q
            q_c_all.append(q_c)

        q_phi_tri = self.get_triple(q_phi_all[0], q_phi_all[1])
        q_c_tri = self.get_triple(q_c_all[0], q_c_all[1])

        ub = q_phi_tri @ la.inv(q_c_tri) @ b_matrix

        # Correct UB with secondary reflection.
        q_sec = np.array([[self._refs.ix[1, "H"]],
                          [self._refs.ix[1, "K"]],
                          [self._refs.ix[1, "L"]]])
        corr = q_phi_all[1] / (ub@q_sec)
        self._ub = corr * ub

        return self._ub

    def print_or(self):
        """Print out all current parameters."""
        print("Orientation matrix:")
        ub = tabulate(self._ub,
                      numalign="left",
                      tablefmt="plain",
                      showindex=False)
        print(ub)

        print("\nPrimary and secondary reflections:")
        refs = tabulate(self._refs,
                        headers=self._refs.columns,
                        numalign="left",
                        tablefmt="plain")
        print(refs)

        print("\nLattice parameters:")
        lat = tabulate(self._lat,
                       headers=self._lat.columns,
                       numalign="left",
                       tablefmt="plain")
        print(lat)

    def pixel2hkl(self, i, j, *args):
        """Calculate h, k, l for pixel (i, j).

        Args:
            i, j: Pixel (i, j).
            *args: lamb, tth, th, chi, phi, ub.
        """
        lamb, tth, th, chi, phi, ub = args

        # Transformation matrices for all frames.
        tth_trans = self.rot_z(tth)
        th_trans = self.rot_z(th)
        chi_trans = self.rot_y(-chi)
        phi_trans = self.rot_z(phi)

        pixel_lab = tth_trans @ self.get_pixel_tth(i, j)
        wave_vec = 2.0 * np.pi / lamb
        kf_lab = wave_vec * pixel_lab / la.norm(pixel_lab)
        ki_lab = wave_vec * np.array([[0],
                                      [1],
                                      [0]], dtype=float)
        q_lab = kf_lab - ki_lab

        q = (la.inv(ub) @ la.inv(phi_trans) @ la.inv(chi_trans)
             @ la.inv(th_trans) @ q_lab)

        h = q[0, 0]
        k = q[1, 0]
        l = q[2, 0]

        print("For pixel = ({:d}, {:d}): "
              "(h, k, l) = ({:.2f}, {:.2f}, {:.2f})".format(i, j, h, k, l))

        return h, k, l

    def hkl2pixel(self, h, k, l, *args):
        """Calculate pixel (i, j) for (h, k, l).

        Args:
            h, k, l: a set of (h, k, l).
            *args: lamb, tth, th, chi, phi, ub.
        """
        lamb, tth, th, chi, phi, ub = args

        # Transformation matrices for all frames.
        tth_trans = self.rot_z(tth)
        th_trans = self.rot_z(th)
        chi_trans = self.rot_y(-chi)
        phi_trans = self.rot_z(phi)

        q = np.array([[h],
                      [k],
                      [l]], dtype=float)
        q_lab = th_trans @ chi_trans @ phi_trans @ ub @ q

        wave_vec = 2.0 * np.pi / lamb
        ki_lab = wave_vec * np.array([[0],
                                      [1],
                                      [0]], dtype=float)
        kf_lab = q_lab + ki_lab

        # Detector center in detector frame.
        c_tth = self._distance * np.array([[0],
                                           [1],
                                           [0]], dtype=float)
        c_lab = tth_trans @ c_tth
        # Alpha is angle between center of detector and kf.
        cos_alpha = kf_lab.T @ c_lab / (la.norm(kf_lab)*la.norm(c_lab))
        # Distance from sample to pixel (i, j)
        r = self._distance / cos_alpha

        pixel_lab = r * kf_lab / wave_vec
        pixel_tth = la.inv(tth_trans) @ pixel_lab
        pixel = self.get_pixel(pixel_tth)

        # round() returns float while int() doesn't round.
        i = int(round(pixel[0]))
        j = int(round(pixel[1]))

        print("For (h, k, l) = ({:.2f}, {:.2f}, {:.2f}): "
              "pixel = ({:d}, {:d})".format(h, k, l, i, j))

        return pixel[0], pixel[1]

    @abstractmethod
    def detector_type(self):
        pass


class MCP(AreaDetector):
    """MCP class.

    Attributes:
        distance: Distance from sample to detector center in mm.
    """

    # Center pixel.
    _i0, _j0 = (512, 512)
    # Pixel size in mm.
    _size = (0.0625, 0.0625)

    def detector_type(self):
        return "MCP"


class TES(AreaDetector):
    """TES class.

    TES has uneven channel distribution depending on certain config file.

    TES pixel actually indicates the real distance with respect to center.

    Attributes:
        config_data: Configuration of channels for TES.
        distance: Distance from sample to detector center in um.
    """

    # Center pixel.
    _i0, _j0 = (0, 0)

    def __init__(self, config_data, distance):
        self._config_data = config_data
        self._distance = distance

    def search_chan(self, i, j):
        """Search channel for pixel (i, j).

        Args:
            i, j: Pixel (i, j).

        Returns:
            Channel number or None if no such channel.
        """
        x = (i - self._i0) * self._size[0]
        y = (j - self._j0) * self._size[1]

        check = 0    # Check if channel is found.
        for chan in range(len(self._config_data)):
            x1 = self._config_data[chan, 1] - self._config_data[chan, 4]/2
            x2 = self._config_data[chan, 1] + self._config_data[chan, 4]/2
            y1 = self._config_data[chan, 2] - self._config_data[chan, 5]/2
            y2 = self._config_data[chan, 2] + self._config_data[chan, 5]/2
            if x1 <= x <= x2 and y1 <= y <= y2:
                print("Pixel ({:d}, {:d}) is on channel"
                      " = {:d}".format(i, j, self._config_data[chan, 0]))
                check += 1
                return self._config_data[chan, 0]

        if check == 0:
            print("Pixel ({:d}, {:d}) is NOT on a valid channel!".format(i, j))
            return None

    def detector_type(self):
        return "TES"