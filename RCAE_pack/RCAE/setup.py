import os
import numpy
from Cython.Build import cythonize
from RCAE_pack.Public import cg


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('RCAE', parent_package, top_path)

    config.add_extension('RCAE_',
                         sources=['RCAE_.pyx'],
                         extra_compile_args=cg.ext_comp_args,
                         extra_link_args=cg.ext_link_args,
                         include_dirs=[numpy.get_include()],
                         language="c++",
                         libraries=cg.libraries)

    config.ext_modules = cythonize(config.ext_modules, compiler_directives={'language_level': 3})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
