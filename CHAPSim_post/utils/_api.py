import sys

def module_name_change(original_name,new_name):
    if sys.modules.get(original_name,False):
        warnings.warn(("This module has been renamed `{new_name}',"
                        " soon this will raise an error"),
                        category=DeprecationWarning)