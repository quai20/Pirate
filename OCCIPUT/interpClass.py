import numpy as np
from scipy import interpolate

#######################################

class InterpProfile:
    """ Internal machinery for the interpolation of vertical profiles

        This class is called once at PCM instance initialisation and
        whenever data to be classified are not on the PCM feature axis.

        Here we consume numpy arrays
        @gmaze
    """

    def __init__(self, axis=None, method='linear'):
        self.axis = axis
        self.doINTERPz = False
        self.method = method

    def lagrange_interp_at_zref(self, CT, z, zref):
        """Interpolate V, dV/dz from their native depths z to zref

        Method: we use piecewise Lagrange polynomial interpolation

        For each zref[k], we select a list of z[j] that are close to
        zref[k], imposing to have z[j] that are above and below zref[k]
        (except near the boundaries)

        If only two z[j] are found then the result is a linear interpolation

        If n z[j] are found then the result is a n-th order interpolation.

        For interior points we may go up to 6-th order

        For the surface level (zref==0), we do extrapolation

        For the bottom level (zref=2000), we do either extrapolation or
        interpolation if data deeper than 2000 are available.

        G. Roullet, 2018/04/26

        """
        # print CT.shape
        # print z.shape
        # print zref.shape

        nref = len(zref)
        CTi = np.zeros((nref,), dtype=float)
        dCTdzi = np.zeros((nref,), dtype=float)

        nbpi, ks = self.lagrange_select_depth(zref, z)
        nupper = np.zeros((nref,), dtype=int)
        nlower = np.zeros((nref,), dtype=int)

        # count the number of data that are lower and upper than zref[k]
        for k in range(nref):
            if k > 0:
                nlower[k] += nbpi[k - 1]
            if k > 1:
                nlower[k] += nbpi[k - 2]
            if k < nref:
                nupper[k] += nbpi[k]
            if k < nref - 1:
                nupper[k] += nbpi[k + 1]

        # for each zref, form the list of z[j] used for the interpolation
        # if the list has at least two elements (a linear interpolation is possible)
        # then do it, otherwise, skip that depth
        for k in range(nref):
            idx = []
            if k == 0:
                if nupper[k] >= 2:
                    idx = (ks[0] + ks[1] + ks[2])[:3]
            elif k == 1:
                if (nlower[k] >= 1) and (nupper[k] >= 1):
                    idx = ks[0][:-3] + (ks[1] + ks[2])[:3]
            elif k == (nref - 1):
                if (nlower[k] + nupper[k]) >= 2:
                    idx = (ks[k - 2] + ks[k - 1])[-3:] + ks[k][:3]
            elif k == (nref - 2):
                if (nlower[k] >= 1) and (nupper[k] >= 1):
                    idx = (ks[k - 2] + ks[k - 1])[-3:] + (ks[k] + ks[k + 1])[:3]
            else:
                if (nlower[k] >= 1) and (nupper[k] >= 1):
                    idx = (ks[k - 2] + ks[k - 1])[-3:] + (ks[k] + ks[k + 1])[:3]
            # print idx

            if len(idx) >= 2:
                # print('****', k, idx, z[idx].data)
                cs, ds = self.lagrange_poly(zref[k], z[idx])
                # the meaning of the weights computed by lagrangepoly should
                # be clear in the code below
                #
                # cs[i] (resp. ds[i]) is the weight to apply on CT[idx[i]]
                # sitting at z[idx[i]] to compute CT (resp. dCT/dz) at zref[k]
                #
                CTi[k] = np.sum(cs * CT[idx])
                dCTdzi[k] = np.sum(ds * CT[idx])
            else:
                CTi[k] = np.nan
                dCTdzi[k] = np.nan

        # return CTi, dCTdzi
        return CTi

    def lagrange_select_depth(self, zref, z):
        """Return the number of data points we have between successive zref.

        for each interval k, we select the z_j such that

        zref[k] <= z_j < zref[k+1], for k=0 .. nref-2

        zref[nref-1] <= z_j < zextra, for k=nref-1

        and return

        nbperintervale[k] = number of z_j

        kperint[k] = list of j's

        with zextra = 2*zref[-1] - zref[-2]

        """
        zref, z = np.abs(zref), np.abs(z)  # Make sure we're working with positive values

        nz = len(z)
        nref = len(zref)
        zextra = 2 * zref[-1] - zref[-2]
        zrefextended = list(zref) + [zextra]
        nbperintervale = np.zeros((nref,), dtype=int)
        kperint = []
        zprev = -1.
        j = 0
        for k, z0 in enumerate(zrefextended[1:]):
            n = 0
            ks = []
            while (j < nz) and (z[j] < z0):
                # for a few profiles it may happens that two consecutive
                # data sit at the same depth this causes a division by
                # zero in the interpolation routine.  Here we fix this by
                # simply skipping depths that are already used.
                if z[j] > zprev:
                    n += 1
                    ks.append(j)
                zprev = z[j]
                j += 1
            nbperintervale[k] = n
            kperint.append(ks)
        return nbperintervale, kperint

    def lagrange_poly(self, x0, xi):
        """Weights for polynomial interpolation at x0 given a list of xi
        return both the weights for function (cs) and its first derivative
        (ds)

        Example:
        lagrangepoly(0.25, [0, 1])
        >>> [0.75, 0.25,], [1, -1]

        """
        xi = np.asarray(xi)
        ncoef = len(xi)
        cs = np.ones((ncoef,))
        ds = np.zeros((ncoef,))

        denom = np.zeros((ncoef, ncoef))
        for i in range(ncoef):
            for j in range(ncoef):
                if i != j:
                    dx = xi[i] - xi[j]
                    if dx == 0:
                        # should not happen because select_depth removes
                        # duplicate depths
                        raise ValueError('division by zero in lagrangepoly')
                    else:
                        denom[i, j] = 1. / dx

        for i in range(ncoef):
            for j in range(ncoef):
                if i != j:
                    cff = 1.
                    cs[i] *= (x0 - xi[j]) * denom[i, j]
                    for k in range(ncoef):
                        if (k != i) and (k != j):
                            cff *= (x0 - xi[k]) * denom[i, k]
                    ds[i] += cff * denom[i, j]
        return cs, ds

    def isnecessary(self, C, Caxis):
        """Check wether or not the input data vertical axis is different
            from the PCM one, if not, avoid interpolation
        """
        # todo We should be smarter and recognize occurences of z in DPTmodel
        # or viceversa in order to limit interpolation as much as possible !
        z = np.float32(Caxis)
        self.doINTERPz = not np.array_equiv(self.axis, Caxis)
        return self.doINTERPz

    def mix(self, x):
        """
            Homogeneize the upper water column:
            Set 1st nan value to the first non-NaN value
        """
        # izmixed = np.argwhere(np.isnan(x))
        izok = np.where(~np.isnan(x))[0][0]
        # x[izmixed] = x[izok]
        x[0] = x[izok]
        return x

    def fit_transform(self, C, Caxis):
        """
            Interpolate data on the PCM vertical axis
        """
        if (self.isnecessary(C, Caxis)):
            if len(C.shape) == 1:
                C = C[np.newaxis]
            [Np, Nz] = C.shape
            # print [Np, Nz]
            # Possibly Create a mixed layer for the interpolation to work
            # smoothly at the surface
            if ((Caxis[0] < 0.) & (self.axis[0] == 0.)):
                #print "Create a Mixed Layer because SDL starts at the surface and not the original axis"
                Caxis = np.concatenate((np.zeros(1), Caxis))
                x = np.empty((Np, 1))
                x.fill(np.nan)
                C = np.concatenate((x, C), axis=1)
                np.apply_along_axis(self.mix, 1, C)
            # Linear interpolation of profiles onto the model grid:
            if Np > 1:
                if self.method == 'linear':
                    f = interpolate.interp2d(Caxis, np.arange(Np), C, kind='linear')
                    C = f(self.axis, np.arange(Np))
                elif self.method == 'lagrange':
                    #print "Multiple lagrange profile interp"
                    # We should be using ufunc instead of a loop here
                    # Ci = np.empty((Np, self.axis.shape[0]))
                    # for ip in np.arange(0, Np, 1):
                    #     Ci[ip,:] = self.lagrange_interp_at_zref(C[ip,:].T, Caxis, self.axis)
                    # C = Ci
                    C = np.apply_along_axis(self.lagrange_interp_at_zref, 1, C.T, Caxis, self.axis)
            else:
                if self.method == 'linear':
                    #print "Single linear profile interp"
                    f = interpolate.interp1d(Caxis, C, kind='linear')
                    C = f(self.axis)[0]
                elif self.method == 'lagrange':
                    #print "Single lagrange profile interp"
                    C = self.lagrange_interp_at_zref(C[0,:].T, np.abs(Caxis), np.abs(self.axis))

        return C
