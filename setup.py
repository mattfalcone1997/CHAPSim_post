
if __name__ == "__main__":
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import Extension
    from numpy.distutils import fcompiler
    from Cython.Build import cythonize


    cy_ext = Extension(name = "CHAPSim_post.post._cy_ext_base",
                sources = ["src/autocorr_parallel.pyx"],
                extra_compile_args = ["-fopenmp","-O3"],
                extra_link_args = ["-fopenmp","-O3"])

    if fcompiler.get_default_fcompiler():
        f90_ext = Extension(name = 'CHAPSim_post.post._f90_ext_base',
                            sources = ["src/autocorr_parallel.f90"],
                            extra_link_args=["-lgomp"],
                            extra_f90_compile_args=['-O3','-fopenmp'])
        ext_list = [cy_ext,f90_ext]
    else:
        ext_list = [cy_ext]
            
    config = Configuration(package_name='CHAPSim_post',
                            description="Package containing post-processing routines for the CHAPSim DNS Solver",
                            package_path="core/CHAPSim_post",
                            ext_modules = cythonize(ext_list))


    config.add_subpackage(subpackage_name='post',
                        subpackage_path="core/CHAPSim_post/post")
    
    config.add_subpackage(subpackage_name="utils",
                        subpackage_path="core/CHAPSim_post/utils")

    config.add_subpackage(subpackage_name="POD",
                        subpackage_path="core/CHAPSim_post/POD")

    # config.add_extension(name = 'CHAPSim_post._f90_ext_base',
    #                     sources = ["src/autocorr_parallel.f90"],
    #                     extra_link_args=["-lgomp"],
    #                     extra_f90_compile_args=['-O3','-fopenmp'])

    # config.add_extension(name = "CHAPSim_post._cy_ext_base",
    #                 sources = ["src/autocorr_parallel.pyx"],
    #                 extra_compile_args = ["-fopenmp","-03"],
    #                 extra_link_args = ["-fopenmp","-03"])
    # config.add_extension(cythonize(ext))
    setup(**config.todict())
