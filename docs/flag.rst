Flagging algorithms and helper functions
========================================

The ``hera_stats.flag`` module contains flagging algorithms and utilities, including a way to randomly flag frequency channels (``apply_random_flags``), a convenience function to flag whole ranges of channels at once (``flag_channels``), and an implementation of a 'greedy' flagging algorithm (``construct_factorizable_mask``) that can construct factorizable (in time and frequency) masks that flag as small a total fraction of the data as possible.

.. automodule:: hera_stats.flag
    :members:

