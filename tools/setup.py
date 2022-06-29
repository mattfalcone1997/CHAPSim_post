
if __name__ == "__main__":
    from numpy.distutils.core import setup
    
    #---------------------------------------------------------------
    
    """
    Building independent tools to be used with but not dependent on CHAPSim_post
    These only contain modules packaged with base python so do not have external dependencies
    """

    # pgfto
    setup(name='pgfto',
         description="Tool to convert .pgf files to .eps",
         py_modules=['pgfto'],
         scripts=['pgftoeps'])