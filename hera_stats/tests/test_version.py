"""Tests for version.py."""
import nose.tools as nt
import sys
import os
from StringIO import StringIO
import subprocess
import hera_pspec


def test_main():
    version_info = hera_pspec.version.construct_version_info()

    saved_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        hera_pspec.version.main()
        output = out.getvalue()
        nt.assert_equal(output, 'Version = {v}\ngit origin = {o}\n'
                        'git branch = {b}\ngit description = {d}\n'
                        .format(v=version_info['version'],
                                o=version_info['git_origin'],
                                b=version_info['git_branch'],
                                d=version_info['git_description']))
    finally:
        sys.stdout = saved_stdout