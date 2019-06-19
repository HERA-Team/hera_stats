from setuptools import setup
import sys
import os
from hera_stats import version
import json

data = [version.git_origin, version.git_hash, version.git_description, version.git_branch]
with open(os.path.join('hera_stats', 'GIT_INFO'), 'w') as outfile:
    json.dump(data, outfile)

def package_files(package_dir, subdirectory):
    # walk the input package_dir/subdirectory
    # return a package_data list
    paths = []
    directory = os.path.join(package_dir, subdirectory)
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            path = path.replace(package_dir + '/', '')
            paths.append(os.path.join(path, filename))
    return paths
data_files = package_files('hera_stats', 'data')

setup_args = {
    'name':         'hera_stats',
    'author':       'HERA Team',
    'url':          'https://github.com/HERA-Team/hera_stats',
    'license':      'BSD',
    'version':      version.version,
    'description':  'HERA Jackknives and Statistical Analyses.',
    'packages':     ['hera_stats'],
    'package_dir':  {'hera_stats': 'hera_stats'},
    'package_data': {'hera_stats': data_files},
    'install_requires': ['numpy>=1.10', 'scipy>=0.19', 'matplotlib',],
    'extras_require': {'automate': ['jupyter',]},
    'include_package_data': True,
    'zip_safe':     False,
}

if __name__ == '__main__':
    setup(**setup_args)
