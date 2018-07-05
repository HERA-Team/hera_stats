import numpy as np
import hera_pspec as hp
import copy

class JKSet():
    def __init__(self, pc_uvp, jktype="None"):
        """
        JKSet is a class to handle sets of single spectra outputted by jackknives
        and usable for other purposes. At the core is a list of UVPSpecs with single
        spectra, which JKSet uses to create an array of spectra. One can load a
        PSpecContainer or a 2D list of UVPSpecs, and easily access the spectra for
        plotting and statistical purposes.

        Parameters
        ----------
        pc_uvp: PSpecContainer of 2D list
            The input for the JKSet. Each UVPSpec in either the container or the list
            must contain only one spectrum, and a stats_array entry named
            "bootstrap_errors."

        jktype: string, optional
            If this comes from a jackknife, especially when loading from a container,
            jktype must be specified. It is used to extract data from the container,
            whose UVPSpecs are named by the jackknife type. Default: "None".
        """
        if isinstance(pc_uvp, hp.container.PSpecContainer):
            self._load_pc(pc_uvp, jktype)

        elif isinstance(pc_uvp, (list, tuple, np.ndarray)):
            self._load_uvp(pc_uvp, jktype)

    def _load_pc(self, pc, jktype):
        """
        Loads a PSpecContainer.

        Parameters
        ----------
        pc: PSpecContainer
            Loaded container that has jackknife data in it.

        jktype: string
            Jackknife type to look for inside the container.
        """
        # Get jackknife name data
        jkf_strs = pc.spectra("jackknives")
        jkf_groups = [a.split(".") for a in pc.spectra("jackknives")]
        all_jktypes = np.unique([a[0] for a in jkf_groups])

        # Get specific jackknive type
        assert jktype in all_jktypes, "Specified jackknife type not found in container."
        all_jkfs = [j for j in jkf_groups if j[0] == jktype]

        # Create dictionary so that the indices can be sorted while still maintaining
        # the correct spectra
        refdic = {}
        jkf_groups = [[int(a[1]), int(a[2][3:])] for a in all_jkfs]
        for n, i in jkf_groups:
            if n not in refdic.keys(): refdic[n] = {}
            refdic[n][i] = jktype + "." + str(n) + ".grp" + str(i)

        # Load UVPSpecs to 2d list
        uvp_list = [[pc.get_pspec("jackknives", refdic[n][i])
                     for i in sorted(refdic[n].keys())]
                    for n in sorted(refdic.keys())]

        self._load_uvp(uvp_list, jktype)

    def _load_uvp(self, uvp_list, jktype, proj=None):
        """
        Loads a 2D list of UVPSpecs.

        Parameters
        ----------
        uvp_list: ndarray (ndim = 2)
            List or array of 

        jktype: string
            String that indicates the jackknife type.
        """
        if proj == None:
            proj = lambda x: x.real

        setattr(self, "jktype", jktype)

        uvp_list = np.array(uvp_list)
        ndim = uvp_list.ndim
        assert ndim == 2, "Expected uvp_list to have 2 dimensions, got {}.".format(ndim)
        assert isinstance(uvp_list[0][0], hp.UVPSpec), "uvp_list must consist of UVPSpecs"

        # Indicate which attributes to save
        to_save = ["spectra", "errs", "grps", "times", "vis_units", "nsamples",
                   "integrations", "units", "_uvp_list"]

        # Create dictionary of empty array for each attr
        data = dict([(ts, []) for ts in to_save])

        # iterate over uvp_list
        for uvp_l in uvp_list:
            # Create another dictionary of empty arrays named by first 3 chars of attr
            dat_l = dict([(ts[:3], []) for ts in to_save])

            for uvp in uvp_l:
                # get delays, spectra, errors.
                dlys = uvp.get_dlys(0) * 10**9
                key = uvp.get_all_keys()[0]
                avspec = proj(uvp.get_data(key)[0])
                errspec = proj(uvp.get_stats("bootstrap_errs", key)[0])

                # Set saved values with list comprehension, should be same order as to_save names.
                to_save_vals = [avspec, errspec, list(uvp.labels),
                                uvp.time_avg_array[0], uvp.vis_units,
                                uvp.nsample_array[0][0], uvp.integration_array[0][0],
                                uvp.units, uvp]

                # Save first to small dictionary
                [dat_l[to_save[i][:3]].append(to_save_vals[i]) for i in range(len(to_save))]

            # Save to larger dictionary
            [data[ts].append(dat_l[ts[:3]]) for ts in to_save]

        # Set class metadata
        for meta in ["units", "vis_units"]:
            val = np.unique(data[meta])
            assert len(val) == 1, "Got {} different values for {}, expected 1.".format(len(val), meta)
            setattr(self, meta, val[0])

        # Set class arrays
        for dset in ["spectra", "errs", "grps", "times", "nsamples",
                    "integrations", "_uvp_list"]:
            setattr(self, dset, np.array(data[dset]))

        self.dlys = dlys
        self.shape = self._uvp_list.shape
        self._validate()

    def __getitem__(self, key):
        """
        Handles indexing.
        """
        # Let numpy handle the indexing
        new_uvp = self._uvp_list[key]

        # Make 2D if not already 2d.
        if isinstance(new_uvp, hp.UVPSpec):
            new_uvp = np.array([[new_uvp]])
        if new_uvp.ndim == 1:
            new_uvp = new_uvp[None]

        # Create new JKSet with sliced uvp_list
        newjk = JKSet(new_uvp, self.jktype)
        return newjk

    def __repr__(self):
        """
        Handles printing.
        """
        # Recreate normal class representation
        name = "<hera_stats.jkset.JKSet instance at {}>".format(hex(id(self)))

        # Add some informative information
        info = ("Jackknife Data\n"
                "--------------\n"
                "jktype: {}\n"
                "data shape: {}\n").format(self.jktype, self._uvp_list.shape)
        return name + "\n\n" + info
    
    def __eq__(self, jk2, just_meta=False):
        """
        Handles ==
        """
        # Validate both data sets
        self._validate(), jk2._validate()

        # Attributes to check if equal (add actual data to list if not just_meta)
        attrs = ["nsamples", "times", "units", "vis_units", "integrations", "grps", "jktype"]
        if not just_meta:
            attrs.extend(["spectra", "errs"])

        # Check if equal
        for attr in attrs:
            if not np.all(getattr(self, attr) == getattr(jk2, attr)):
                return False

        return True

    def add(self, jkset2, axis=1, inplace=False):
        """
        Stacks together the current jkset and another along a specific axis.

        Parameters
        ----------
        jkset2: JKSet
            The other JKSet to stack with this one.

        axis: int, 1 or 0, optinoal
            Axis along which to add. Default: 1.

        inplace: boolean, optional
            If true, keeps stacked data in this object. If false, returns new object. Default: False.

        Returns
        -------
        new_jkset: JKSet
            If inplace == False, returns a jkset with spectra of both this class and jkset2
        """
        # Test shape to see if stackable
        assert axis in [0, 1], "axis must either be 1 or 0."
        assert self.shape[not axis] == jkset2.shape[not axis], "jksets don't match shape in dimension {}.".format(not axis)

        # Vstack or hstack uvp lists
        if axis == 1:
            new_uvp = np.hstack([self._uvp_list, jkset2._uvp_list])
        elif axis == 0:
            new_uvp = np.vstack([self._uvp_list, jkset2._uvp_list])

        # Return or set data
        if inplace:
            self._load_uvp(new_uvp)
        else:
            return JKSet(new_uvp, self.jktype)

    def set_data(self, spectra, errs):
        """
        Sets the spectra data and errors of this class.

        Parameters
        ----------
        spectra: 3d ndarray
            Spectra to set. First two dimensions must match the shape of this object, and
            third dimension must match the number of delays for this class.

        errs: 3d ndarray
            Errors to set. Same restrictions as above.
        """
        # Check is spectra and errors have same first 2 dimensions as class
        msg = "First two axes of {} {} and {} {} must match."
        assert spectra.shape[:2] == self.shape, msg.format("spectra", spectra.shape, "this JKSet", self.shape)
        assert errs.shape[:2] == self.shape, msg.format("errs", errs.shape,"this JKSet", self.shape)

        # Check is spectra shape matches errors shape
        msg = "Shape of {} {} and {} {} must match exactly."
        assert spectra.shape == errs.shape, msg.format("spectra", spectra.shape, "errs", errs.shape)

        # Check if number of delays is consistent
        msg = "Number of class delay modes {} must the number of delay modes for {} ({})"
        ndlys = len(self.dlys)
        assert spectra.shape[2] == ndlys, msg.format(ndlys, "spectra", spectra.shape[2])
        assert errs.shape[2] == ndlys, msg.format(ndlys, "spectra", errs.shape[2])

        # Copy uvp_list and set spectra and errors
        uvp_list = copy.deepcopy(self._uvp_list)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                uvp = uvp_list[i][j]
                uvp.data_array[0] = np.expand_dims(spectra[i][j][None], 2)
                uvp.stats_array["bootstrap_errs"][0] = np.expand_dims(errs[i][j][None], 2)

        self._load_uvp(uvp_list, self.jktype)

    def flatten(self):
        """
        Flattens the jkset into  a shape (1, n) JKSet.

        Returns
        -------
        jkset: JKSet
            New jkset that has been flattened.
        """
        return JKSet(self._uvp_list.flatten()[None], self.jktype)

    def reshape(self, *args):
        """
        Rehspaes the jkset.
        
        Returns
        -------
        jkset: JKSet
            New jkset in the shape specified.
        """
        return JKSet(self._uvp_list.reshape(*args), self.jktype)

    def T(self):
        """
        Transposes the data of JKSet.

        Returns
        -------
        jkset: JKSet
            The transposed JKSet.
        """
        return JKSet(self._uvp_list.T, self.jktype)

    def _validate(self):
        """
        Validates the data. Checks if all arrays have the same shape, and if data is consistent.
        """
        shape = self.shape

        # See if all arrays have same first two dimensions.
        for attr in ["spectra", "errs", "nsamples", "times",
                     "integrations", "grps"]:

            array = getattr(self, attr)
            assert isinstance(array, (list, tuple, np.ndarray)), "{} must be an array".format(attr)
            msg =  "Shape of {} {} does not match class shape {}.".format(attr, array.shape, shape)
            assert array.shape[:2] == shape, msg

        # Check is delay modes are consistent
        for spec_like in ["spectra", "errs"]:
            array = getattr(self, spec_like)
            assert array.shape[2] == len(self.dlys), "{} ({}) and dlys ({}) have a different number of delay modes.".format(spec_like, array.shape[2], len(self.dlys))

        # Check jackknife type
        assert isinstance(self.jktype, str), "Expected jktype to be string, got {}".format(type(self.jktype.__name__))

        # Check that units match
        for attr in ["units", "vis_units", "jktype"]:
            val = getattr(self, attr)
            assert isinstance(val, (str, np.str)), "Expected meta attribute {} to be string.".format(attr)