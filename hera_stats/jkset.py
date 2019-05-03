import numpy as np
import hera_pspec as hp
import copy


class JKSet(object):

    def __init__(self, pc_uvp, jktype, error_field="bs_std"):
        """
        JKSet is a class to handle sets of single spectra outputted by jackknives
        and usable for other purposes. At the core is a list of UVPSpecs with single
        spectra, which JKSet uses to create an array of spectra. One can load a
        PSpecContainer or a 2D list of UVPSpecs, and easily access the spectra for
        plotting and statistical purposes.

        Parameters
        ----------
        pc_uvp: PSpecContainer or 2D list
            The input for the JKSet. Each UVPSpec in either the container or the list
            must contain only one spectrum, and a stats_array entry "error_field"

        jktype: string, optional
            If this comes from a jackknife, especially when loading from a container,
            jktype must be specified. It is used to extract data from the container,
            whose UVPSpecs are named by the jackknife type. Default: "None".
        """
        # Load PSpecContainer
        if isinstance(pc_uvp, hp.container.PSpecContainer):
            self._load_pc(pc_uvp, jktype, error_field=error_field)

        # Or, load UVPSpec array or list
        elif isinstance(pc_uvp, (list, tuple, np.ndarray)):
            self._load_uvp(pc_uvp, jktype, error_field=error_field)

        else:
            raise AssertionError("Expected pc_uvp to be either a PSpecContainer or a list, got %s." % type(pc_uvp).__name__)

    def _load_pc(self, pc, jktype, error_field="bs_std"):
        """
        Loads a PSpecContainer.

        Parameters
        ----------
        pc: PSpecContainer
            Loaded container that has jackknife data in it.

        jktype: string
            Jackknife type to look for inside the container.

        error_field: string
            The error field of the UVPSpec stats_array from which to
            load errors. Default: "bs_std" (which is set by
            hera_pspec.grouping.bootsrap_resampled_error).
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
        jkf_groups = [[int(a[1]), int(a[2])] for a in all_jkfs]
        for n, i in jkf_groups:
            if n not in list(refdic.keys()): refdic[n] = {}
            refdic[n][i] = jktype + "." + str(n) + "." + str(i)

        # Load UVPSpecs to 2d list
        uvp_list = [[pc.get_pspec("jackknives", refdic[n][i])
                     for i in sorted(refdic[n].keys())]
                    for n in sorted(refdic.keys())]

        self._load_uvp(uvp_list, jktype, error_field)

    def _load_uvp(self, uvp_list, jktype, error_field="bs_std", proj=None):
        """
        Loads a 2D list of UVPSpecs.

        Parameters
        ----------
        uvp_list: ndarray (ndim = 2)
            List or array of 

        jktype: string
            String that indicates the jackknife type.

        error_field: string
            The error field of the UVPSpec stats_array from which to
            load errors. Default: "bs_std" (which is set by
            hera_pspec.grouping.bootsrap_resampled_error).

        proj: function, optional
            In development, can specify how to project the data when loading it.
            If None, sets to lambda x: x.real. Default: None.
        """
        if proj == None:
            proj = lambda x: x.real

        setattr(self, "jktype", jktype)

        uvp_list = np.array(uvp_list)
        assert isinstance(uvp_list.flatten()[0], hp.UVPSpec), "uvp_list must consist of UVPSpecs"

        # Indicate which attributes to save
        attrs  = ["spectra", "errs", "grps", "times", "vis_units", "nsamples",
                   "integrations", "units", "_uvp_list", "dlys"]

        # Functions used to get attrs and arrays.
        load_attr = [lambda uvp: proj(uvp.get_data((0, uvp.get_blpairs()[0], "xx"))[0]),
                     lambda uvp: proj(uvp.get_stats(error_field, (0, uvp.get_blpairs()[0], "xx"))[0]),
                     lambda uvp: uvp.labels,
                     lambda uvp: uvp.time_avg_array[0],
                     lambda uvp: uvp.vis_units,
                     lambda uvp: uvp.nsample_array[0][0],
                     lambda uvp: uvp.integration_array[0][0],
                     lambda uvp: uvp.units,
                     lambda uvp: uvp,
                     lambda uvp: uvp.get_dlys(0) * 10**9]

        # Create recursion function
        def map_uvp(func, obj):
            if isinstance(obj, hp.UVPSpec):
                return func(obj)
            elif isinstance(obj, (list, np.ndarray)):
                return np.array([map_uvp(func, ob) for ob in obj])

        # Recursively applies a function of load_attr to uvp_list.
        # Sets resulting array as element of dictionary with attrs as name.
        dic = dict([(attrs[i], map_uvp(load_attr[i], uvp_list)) for i in range(len(attrs))])    

        # Set class metadata
        for meta in ["units", "vis_units"]:
            # Make sure all values are the same and set attribute
            val = np.unique(dic[meta])
            assert len(val) == 1, "Got {} different values for {}, expected 1.".format(len(val), meta)
            setattr(self, meta, val[0])

        # Set class arrays
        for dset in ["spectra", "errs", "grps", "times", "nsamples",
                    "integrations", "_uvp_list", "dlys"]:
            setattr(self, dset, dic[dset])

        # Set more metadata
        key = tuple([0] * (self.dlys.ndim - 1) + [slice(None, None, 1)])
        self.dlys = self.dlys[key]
        self.shape = self._uvp_list.shape
        self.ndim = self._uvp_list.ndim
        self._error_field = error_field
        self._validate()

    def __getitem__(self, key):
        """
        Handles indexing.
        """
        # Let numpy handle the indexing
        new_uvp = self._uvp_list[key]

        # Make array if index happens to shrink list down to single UVPSpec
        if isinstance(new_uvp, hp.UVPSpec):
            new_uvp = np.array([new_uvp])

        # Create new JKSet with sliced uvp_list
        newjk = JKSet(new_uvp, self.jktype, self._error_field)
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
        Handles "=="
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
        if self.ndim != 1:
            # Test shape to see if stackable
            assert axis < self.ndim,"axis outside of range (axis <= %i)." % (self.ndim - 1)
            otheraxes = np.arange(self.ndim) != axis
            assert all([self.shape[a] == jkset2.shape[a] for a in otheraxes]), "Axes other than the one specified must match."

        # Vstack or hstack uvp lists
        if axis == 1:
            new_uvp = np.hstack([self._uvp_list, jkset2._uvp_list])
        elif axis == 0:
            new_uvp = np.vstack([self._uvp_list, jkset2._uvp_list])

        # Return or set data
        if inplace:
            self._load_uvp(new_uvp, self.jktype, self._error_field)
        else:
            return JKSet(new_uvp, self.jktype, self._error_field)

    def set_data(self, spectra, errs, error_field='bs_std'):
        """
        Sets the spectra data and errors of this class.

        Parameters
        ----------
        spectra: 3d ndarray
            Spectra to set. First two dimensions must match the shape of this object, and
            third dimension must match the number of delays for this class.

        errs: 3d ndarray
            Errors to set. Same restrictions as above.

        error_field: string
            The error field of the UVPSpec stats_array from which to
            load errors. Default: "bs_std" (which is set by
            hera_pspec.grouping.bootsrap_resampled_error).
        """
        # Check is spectra and errors have same first 2 dimensions as class
        msg = "First two axes of {} {} and {} {} must match."
        assert spectra.shape[:self.ndim] == self.shape, msg.format("spectra", spectra.shape, "this JKSet", self.shape)
        assert errs.shape[:self.ndim] == self.shape, msg.format("errs", errs.shape,"this JKSet", self.shape)

        # Check is spectra shape matches errors shape
        msg = "Shape of {} {} and {} {} must match exactly."
        assert spectra.shape == errs.shape, msg.format("spectra", spectra.shape, "errs", errs.shape)

        # Check if number of delays is consistent
        msg = "Number of class delay modes {} must the number of delay modes for {} ({})"
        ndlys = len(self.dlys)
        assert spectra.shape[self.ndim] == ndlys, msg.format(ndlys, "spectra", spectra.shape[self.ndim])
        assert errs.shape[self.ndim] == ndlys, msg.format(ndlys, "spectra", errs.shape[self.ndim])

        # Recursive function for setting spectra and errors.
        def recursive_set(obj, spectra, errs):
            if isinstance(obj, hp.UVPSpec):
                obj.data_array[0] = np.asarray(np.expand_dims(spectra[None], 2), dtype=np.complex128)
                obj.stats_array[error_field][0] = np.asarray(np.expand_dims(errs[None], 2), dtype=np.complex128)
                obj.check()
                return obj
            elif isinstance(obj, (list, np.ndarray)):
                return np.array([recursive_set(obj[i], spectra[i], errs[i]) for i in range(len(obj))])

        # Recursively set data and load the new uvp_list
        uvpl = copy.deepcopy(self._uvp_list)
        uvp_list = recursive_set(uvpl, spectra, errs)
        self._load_uvp(uvp_list, self.jktype, self._error_field)

    def flatten(self):
        """
        Flattens the jkset into  a shape (1, n) JKSet.

        Returns
        -------
        jkset: JKSet
            New jkset that has been flattened.
        """
        return JKSet(self._uvp_list.flatten(), self.jktype, self._error_field)

    def reshape(self, *args):
        """
        Rehspaes the jkset.
        
        Returns
        -------
        jkset: JKSet
            New jkset in the shape specified.
        """
        return JKSet(self._uvp_list.reshape(*args), self.jktype, self._error_field)

    def T(self):
        """
        Transposes the data of JKSet.

        Returns
        -------
        jkset: JKSet
            The transposed JKSet.
        """
        return JKSet(self._uvp_list.T, self.jktype, self._error_field)

    def _validate(self):
        """
        Validates the data. Checks if all arrays have the same shape, and if data is consistent.
        """
        shape = self.shape

        assert self.ndim <= 2, "Cannot take more than 2 dimensions."

        # See if all arrays have same first two dimensions.
        for attr in ["spectra", "errs", "nsamples", "times",
                     "integrations", "grps"]:

            array = getattr(self, attr)
            assert isinstance(array, (list, tuple, np.ndarray)), "{} must be an array".format(attr)
            msg =  "Shape of {} {} does not match class shape {}.".format(attr, array.shape, shape)
            assert array.shape[:self.ndim] == shape, msg

        # Check is delay modes are consistent
        for spec_like in ["spectra", "errs"]:
            array = getattr(self, spec_like)
            assert array.shape[self.ndim] == len(self.dlys), "{} ({}) and dlys ({}) have a different number of delay modes.".format(spec_like, array.shape[self.ndim], len(self.dlys))

        # Check jackknife type
        assert isinstance(self.jktype, str), "Expected jktype to be string, got {}".format(type(self.jktype.__name__))

        # Check that units are valid
        for attr in ["units", "vis_units", "jktype"]:
            val = getattr(self, attr)
            assert isinstance(val, (str, np.str)), "Expected meta attribute {} to be string.".format(attr)

def peek(pc):
    """
    Lists the jackknife types and shapes in a PSpecContainer
    
    Parameters
    ----------
    pc: hera_pspec.container.PSpecContainer
        A PSpecContainer to peek into.
    """
    assert isinstance(pc, hp.container.PSpecContainer)

    # Load spectra labels
    try:
        sp = pc.spectra("jackknives")
    except:
        print("No jackknives found in PSpecContainer")
    sp = np.array([s.split(".") for s in sp])

    # Extract jackknife ypes and array shapes
    jktypes = np.unique(sp[:, 0])
    nj = [sum(sp[:, 0] == jkt) for jkt in jktypes]
    n = [sum((sp[:, 0] == jkt) * (sp[:, 1] == "0")) for jkt in jktypes]
    shapes = [(nj[i]/n[i], n[i]) for i in range(len(n))]

    for shp, jkt in zip(shapes, jktypes):
        print("%s:" % jkt)
        print("   shape: %s" % str(shp))
