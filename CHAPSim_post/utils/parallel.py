
from contextlib import contextmanager
import multiprocessing
import os
import concurrent
import itertools
import warnings
import sys
from CHAPSim_post import rcParams
import copy 
from functools import wraps 

class ParallelConcurrent():
    def __init__(self,*args,**kwargs):
        
        if self.check_parallel('thread'):   
            self._executor = concurrent.futures.ThreadPoolExecutor(*args,**kwargs)
        elif self.check_parallel('process'):
            self._executor = concurrent.futures.ProcessPoolExecutor(*args,**kwargs)
        else:
            self._executor = None
    
    @staticmethod
    def check_parallel(mode):
        options = ['off','thread','process']
        if rcParams['use_parallel'] not in options:
            msg = (f"Invalid parallel setting {rcParams['use_parallel']}.\n"
                   f"Must be one of {options}")
            raise ValueError(msg)
        
        return rcParams['use_parallel'] == mode
    
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
        if self.check_parallel('off'):
            return self._run_sequential()
        else:
            return self._run_parallel()
    
    def map_async(self,func,iterable,*args,**kwargs):
        if self.check_parallel('off'):
            return [func(it,*args,**kwargs) for it in iterable]

        data_order=[]
        parallel_list = {self._executor.submit(func,it,*args,**kwargs): i\
                             for i,it in enumerate(iterable)}

        for future in concurrent.futures.as_completed(parallel_list):
            data_order.append((future.result(),parallel_list[future]))
        

        data = [None]*len(iterable)
        for d,i in data_order:
            data[i] = d 

        return data


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
    def stop(self):
        if self._executor is not None:
            self._executor.shutdown()
    def __del__(self):
        if self._executor is not None:
            self._executor.shutdown()

class processCallable:
    def __init__(self,func):
        self._func = copy.deepcopy(func)
        
    def __call__(self, *args, **kwargs):
        return self._func(*args,**kwargs)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self,state):
        self.__dict__ = state


def processWrap(func):
    module = sys.modules[__name__]
    setattr(module,func.__name__,func)

    @wraps(func)
    def wrapper(*args,**kwargs):
        return func(*args,**kwargs)

    return wrapper

class ParallelOverlap:
    def __init__(self,processes):
            
        self._pool = multiprocessing.Pool(processes)
        
        self._funcs = []
        self._args = {}
        self._kwargs = {}
    
    @property
    def Pool(self):
        return self._pool


    def register_func(self,func):
        self._funcs.append(func)

        def _func_setargs(*args,**kwargs):
            self._args[func.__name__] = args
            self._kwargs[func.__name__] = kwargs

        setattr(self,func.__name__, _func_setargs)



    def run_parallel(self):
        if len(self._funcs) == 1:
            return self.run_sequential()
        elif len(self._funcs) > self._pool._processes:
            self._pool = multiprocessing.Pool(len(self._funcs))
        
        async_list = []
        for func in self._funcs:
            async_list.append(self._pool.apply_async(func,
                                    self._args[func.__name__],
                                    self._kwargs[func.__name__]))
        data=[]
        for a_sync in async_list:
            data.append(a_sync.get())

        return data

    def run_sequential(self):
        data = []
        func_args_kwargs = list(zip(self._funcs,self._args,self._kwargs))
        for func,args,kwargs in func_args_kwargs:
            data.append(func(*args,**kwargs))

        return data
    def map_async(self,func,iterable, *args,**kwargs):
        if len(iterable) > self._pool._processes:
            self._pool = multiprocessing.Pool(len(iterable))

        result = self._pool.map_async(func,iterable,*args,**kwargs)
        
        return result.get()

    def __call__(self):
        if rcParams['use_parallel'] and sys.version >= '3.8' :
            return self.run_parallel()

        else:
            return self.run_sequential()
