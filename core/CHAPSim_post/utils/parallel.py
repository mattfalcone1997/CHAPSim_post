
from contextlib import contextmanager
import multiprocessing
import os
import concurrent
import itertools
import warnings
import sys
from CHAPSim_post import rcParams
import copy 

class ParallelConcurrent():
    def __init__(self,*args,**kwargs):
        self._executor = concurrent.futures.ThreadPoolExecutor(*args,**kwargs)

    def set_func(self,func):
        self._func = func

    def set_args_lists(self,args_list,kwargs_list):
        if len(args_list) != len(kwargs_list):
            msg = ("The length of the argument an keyword"
                    " argument lists must be the same")
            raise ValueError(msg)

        new_args_list=[]
        for a in args_list:
            if not hasattr(a,'__len__'):
                new_args_list.append(tuple([a]))
            else:
                new_args_list.append(tuple(a))

        self._args_list = new_args_list
        self._kwargs_list = kwargs_list

    def __call__(self):
        if rcParams['use_parallel']:
            try:
                return self._run_parallel()
                
            except Exception as exec:
                msg = ("Error was raised in parallel execution running sequentially:\n"
                        "{0}".format(exec))
                warnings.warn(msg)
                return self._run_sequential()
        else:
            return self._run_sequential()

    def _run_parallel(self):
        data_order=[]
        args_kwargs = list(zip(self._args_list,self._kwargs_list))
        parallel_list = {self._executor.submit(self._func,*args,**kwargs): i\
                             for i,(args, kwargs) in enumerate(args_kwargs)}

        for future in concurrent.futures.as_completed(parallel_list):
            data_order.append((future.result(),parallel_list[future]))


        data = [None]*len(args_kwargs)
        for d,i in data_order:
            data[i] = d 

        return data

    def _run_sequential(self):
        args_kwargs = zip(self._args_list,self._kwargs_list)
        data = []
        for args, kwargs in args_kwargs:
            data.append(self._func(*args,**kwargs))

        return data

class processCallable:
    def __init__(self,func):
        self._func = copy.deepcopy(func)
        
    def __call__(self, *args, **kwargs):
        return self._func(*args,**kwargs)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self,state):
        self.__dict__ = state

class ParallelOverlap:
    def __init__(self,*args,thread=True,**kwargs):
        if thread:
            self._executor = concurrent.futures.ThreadPoolExecutor(*args,**kwargs)
        else:
            self._executor = concurrent.futures.ProcessPoolExecutor(*args,**kwargs)
        self._funcs = []
        self._args = []
        self._kwargs = []
        self._thread_mode = thread
        
    def register_func(self,func,*args,**kwargs):
        if self._thread_mode:
            self._funcs.append(func)
        else:
            module = sys.modules[func.__module__]
            self._funcs.append(getattr(module,func.__name__))
        
        self._args.append(args)
        self._kwargs.append(kwargs)

    def run_parallel(self):
        if len(self._funcs) == 1:
            return self.run_sequential()
        data_order=[]

        func_args_kwargs = zip(self._funcs,self._args,self._kwargs)
        parallel_list = {self._executor.submit(func,*args,**kwargs): i\
                     for i,(func,args, kwargs) in enumerate(func_args_kwargs)}

        for future in concurrent.futures.as_completed(parallel_list):
            data_order.append((future.result(),parallel_list[future]))


        data = [None]*len(self._funcs)
        for d,i in data_order:
            data[i] = d 

        return data

    def run_sequential(self):
        data = []
        func_args_kwargs = list(zip(self._funcs,self._args,self._kwargs))
        for func,args,kwargs in func_args_kwargs:
            data.append(func(*args,**kwargs))

        return data

    def __call__(self):
        if rcParams['use_parallel']:
            try:
                return self.run_parallel()
                
            except Exception as exec:
                msg = ("Error was raised in parallel execution running sequentially:\n"
                        "{0}".format(exec))
                warnings.warn(msg)
                return self.run_sequential()
        else:
            return self.run_sequential()
