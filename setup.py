
if __name__ == "__main__":
    from numpy.distutils.core import setup
    from numpy.distutils.misc_util import Configuration
    from numpy.distutils.core import Extension
    import numpy
    import os
    from Cython.Build import cythonize
    
    def create_cython_ext(folder,**other_args):
        sources = [os.path.join(folder,file) for file in os.listdir(folder) \
                        if os.path.splitext(file)[-1] == '.pyx']
        names = [os.path.splitext(source)[0].replace('/','.')\
                    for source in sources]
        include_dirs = [numpy.get_include()],
        if 'include_dirs' in other_args:
            other_args['include_dirs'] += include_dirs
        else:
            other_args['include_dirs'] = include_dirs
        ext_list = []
        for name, source in zip(names,sources):
            ext_list.append(Extension(name=name,
                                      sources=[source],
                                      **other_args))
            
        return ext_list
        
    path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(path)

    cy_parallel = create_cython_ext("CHAPSim_post/_libs",
                                    extra_compile_args = ["-fopenmp","-O3"],
                                    extra_link_args = ["-fopenmp","-O3"])
    
    cy_parallel_legacy = create_cython_ext("CHAPSim_post/legacy/_libs",
                                    extra_compile_args = ["-fopenmp","-O3"],
                                    extra_link_args = ["-fopenmp","-O3"])

    ext_list = cythonize(cy_parallel+cy_parallel_legacy,
                         compiler_directives={'language_level':3})

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

    config.add_subpackage(subpackage_name='legacy',
                        subpackage_path="CHAPSim_post/legacy")
    
    # config.add_subpackage(subpackage_name='legacy.post',
    #                          subpackage_path="CHAPSim_post/legacy/post")

    # config.add_subpackage(subpackage_name="legacy.utils",
    #                     subpackage_path="CHAPSim_post/legacy/utils")

    # config.add_subpackage(subpackage_name="legacy.POD",
    #                     subpackage_path="CHAPSim_post/legacy/POD")                     




    setup(**config.todict())
