"""
Tests for version.py.
"""
import nose.tools as nt
import sys
import os
import json
try:
    # Python 2
    from cStringIO import StringIO
except:
    # Python 3
    from io import StringIO
import subprocess
import hera_stats


def test_main():
    version_info = hera_stats.version.construct_version_info()
    
    saved_stdout = sys.stdout
    try:
        out = StringIO()
        sys.stdout = out
        hera_stats.version.main()
        output = out.getvalue()
        nt.assert_equal(output, 'Version = {v}\ngit origin = {o}\n'
                        'git branch = {b}\ngit description = {d}\n'
                        .format(v=version_info['version'],
                                o=version_info['git_origin'],
                                b=version_info['git_branch'],
                                d=version_info['git_description']))
    finally:
        sys.stdout = saved_stdout
    
    # Test history string function
    hera_stats.version.history_string()


def test_get_gitinfo_file():
    dir = hera_stats.version.hera_stats_dir

    git_file = os.path.join(dir, 'GIT_INFO')
    if not os.path.exists(git_file):
        # write a file to read in
        temp_git_file = os.path.join(dir, 'GIT_INFO')
        version_info = hera_stats.version.construct_version_info()
        data = [version_info['git_origin'], version_info['git_origin'],
                version_info['git_origin'], version_info['git_origin']]
        with open(temp_git_file, 'w') as outfile:
            json.dump(data, outfile)
        git_file = temp_git_file

    with open(git_file) as data_file:
        data = [hera_stats.version._unicode_to_str(x) for x in json.loads(data_file.read().strip())]
        git_origin = data[0]
        git_hash = data[1]
        git_description = data[2]
        git_branch = data[3]

    test_file_info = {
        'git_origin': git_origin, 'git_hash': git_hash,
        'git_description': git_description, 'git_branch': git_branch
    }

    if 'temp_git_file' in locals():
        file_info = hera_stats.version._get_gitinfo_file(git_file=temp_git_file)
        os.remove(temp_git_file)
    else:
        file_info = hera_stats.version._get_gitinfo_file()

    assert file_info == test_file_info
