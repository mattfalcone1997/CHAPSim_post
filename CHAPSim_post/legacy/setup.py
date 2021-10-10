from numpy.distutils.core import setup
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import Extension
import os
from Cython.Build import cythonize


current_loc = os.path.dirname(__file__)
sources = [os.path.join(current_loc,"_libs/autocorr_parallel32.pyx"),
            os.path.join(current_loc,"_libs/autocorr_parallel64.pyx")]

cy_ext32 = Extension(name = "CHAPSim_post.legacy.post._cy_ext32_base",
                sources = [sources[0]],
                extra_compile_args = ["-fopenmp","-O3"],
                extra_link_args = ["-fopenmp","-O3"])

cy_ext64 = Extension(name = "CHAPSim_post.legacy.post._cy_ext64_base",
            sources = [sources[1]],
            extra_compile_args = ["-fopenmp","-O3"],
            extra_link_args = ["-fopenmp","-O3"])

ext_list = cythonize([cy_ext32,cy_ext64])

config = Configuration(package_name='CHAPSim_post.legacy',
                        description="Package containing post-processing routines for the CHAPSim DNS Solver",
                        package_path="CHAPSim_post/legacy",
                        ext_modules = ext_list)

config.add_subpackage(subpackage_name='post',
                        subpackage_path="CHAPSim_post/legacy/post")

config.add_subpackage(subpackage_name="utils",
                        subpackage_path="CHAPSim_post/legacy/utils")

config.add_subpackage(subpackage_name="POD",
                        subpackage_path="CHAPSim_post/legacy/POD")

setup(**config.todict())
