
if __name__ == "__main__":
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import Extension
    import os
    from Cython.Build import cythonize

    sources = ["CHAPSim_post/_libs/autocorr_parallel.pyx",
                "CHAPSim_post/_libs/gradient_parallel.pyx"]
    cy_parallel = []
    cy_parallel_legacy = []

    sources_legacy = ["CHAPSim_post/legacy/_libs/autocorr_parallel%d.pyx"%x for x in [32,64]]
    names_legacy = [source.replace('/','.') for source in sources_legacy]




    for source in sources:
        module = os.path.splitext(os.path.basename(source))[0]
        name = "CHAPSim_post._libs" + "." + module
        cy_parallel.append( Extension(name = name,
                                     sources = [source],
                                     extra_compile_args = ["-fopenmp","-O3"],
                                     extra_link_args = ["-fopenmp","-O3"]))

    for source, name in zip(sources_legacy,names_legacy):
        cy_parallel_legacy.append( Extension(name = name,
                                     sources = [source],
                                     extra_compile_args = ["-fopenmp","-O3"],
                                     extra_link_args = ["-fopenmp","-O3"]))


    ext_list = cythonize(cy_parallel+cy_parallel_legacy)

    config = Configuration(package_name='CHAPSim_post',
                            description="Package containing post-processing routines for the CHAPSim DNS Solver",
                            package_path="CHAPSim_post",
                            ext_modules = ext_list)


    config.add_subpackage(subpackage_name='post',
                        subpackage_path="CHAPSim_post/post")
    
    config.add_subpackage(subpackage_name="utils",
                        subpackage_path="CHAPSim_post/utils")

    config.add_subpackage(subpackage_name="dtypes",
                        subpackage_path="CHAPSim_post/dtypes")

    config.add_subpackage(subpackage_name="POD",
                        subpackage_path="CHAPSim_post/POD")

    config.add_subpackage(subpackage_name='CHAPSim_post.legacy')

    config.add_subpackage(subpackage_name='CHAPSim_post.legacy.post')

    config.add_subpackage(subpackage_name="CHAPSim_post.legacy.utils")

    config.add_subpackage(subpackage_name="CHAPSim_post.legacy.POD")                        





    setup(**config.todict())
