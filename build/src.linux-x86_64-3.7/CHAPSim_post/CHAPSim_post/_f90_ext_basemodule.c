/* File: _f90_ext_basemodule.c
 * This file is auto-generated with f2py (version:2).
 * f2py is a Fortran to Python Interface Generator (FPIG), Second Edition,
 * written by Pearu Peterson <pearu@cens.ioc.ee>.
 * Generation date: Tue Dec 22 17:19:16 2020
 * Do not edit this file directly unless you know what you are doing!!!
 */

#ifdef __cplusplus
extern "C" {
#endif

/*********************** See f2py2e/cfuncs.py: includes ***********************/
#include "Python.h"
#include <stdarg.h>
#include "fortranobject.h"
#include <math.h>

/**************** See f2py2e/rules.py: mod_rules['modulebody'] ****************/
static PyObject *_f90_ext_base_error;
static PyObject *_f90_ext_base_module;

/*********************** See f2py2e/cfuncs.py: typedefs ***********************/
/*need_typedefs*/

/****************** See f2py2e/cfuncs.py: typedefs_generated ******************/
/*need_typedefs_generated*/

/********************** See f2py2e/cfuncs.py: cppmacros **********************/
#if defined(PREPEND_FORTRAN)
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F
#else
#define F_FUNC(f,F) _##f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) _##F##_
#else
#define F_FUNC(f,F) _##f##_
#endif
#endif
#else
#if defined(NO_APPEND_FORTRAN)
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F
#else
#define F_FUNC(f,F) f
#endif
#else
#if defined(UPPERCASE_FORTRAN)
#define F_FUNC(f,F) F##_
#else
#define F_FUNC(f,F) f##_
#endif
#endif
#endif
#if defined(UNDERSCORE_G77)
#define F_FUNC_US(f,F) F_FUNC(f##_,F##_)
#else
#define F_FUNC_US(f,F) F_FUNC(f,F)
#endif

#define rank(var) var ## _Rank
#define shape(var,dim) var ## _Dims[dim]
#define old_rank(var) (PyArray_NDIM((PyArrayObject *)(capi_ ## var ## _tmp)))
#define old_shape(var,dim) PyArray_DIM(((PyArrayObject *)(capi_ ## var ## _tmp)),dim)
#define fshape(var,dim) shape(var,rank(var)-dim-1)
#define len(var) shape(var,0)
#define flen(var) fshape(var,0)
#define old_size(var) PyArray_SIZE((PyArrayObject *)(capi_ ## var ## _tmp))
/* #define index(i) capi_i ## i */
#define slen(var) capi_ ## var ## _len
#define size(var, ...) f2py_size((PyArrayObject *)(capi_ ## var ## _tmp), ## __VA_ARGS__, -1)

#define CHECKSCALAR(check,tcheck,name,show,var)\
    if (!(check)) {\
        char errstring[256];\
        sprintf(errstring, "%s: "show, "("tcheck") failed for "name, var);\
        PyErr_SetString(_f90_ext_base_error,errstring);\
        /*goto capi_fail;*/\
    } else 
#ifdef DEBUGCFUNCS
#define CFUNCSMESS(mess) fprintf(stderr,"debug-capi:"mess);
#define CFUNCSMESSPY(mess,obj) CFUNCSMESS(mess) \
    PyObject_Print((PyObject *)obj,stderr,Py_PRINT_RAW);\
    fprintf(stderr,"\n");
#else
#define CFUNCSMESS(mess)
#define CFUNCSMESSPY(mess,obj)
#endif

#ifndef max
#define max(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef min
#define min(a,b) ((a < b) ? (a) : (b))
#endif
#ifndef MAX
#define MAX(a,b) ((a > b) ? (a) : (b))
#endif
#ifndef MIN
#define MIN(a,b) ((a < b) ? (a) : (b))
#endif


/************************ See f2py2e/cfuncs.py: cfuncs ************************/
static int f2py_size(PyArrayObject* var, ...)
{
  npy_int sz = 0;
  npy_int dim;
  npy_int rank;
  va_list argp;
  va_start(argp, var);
  dim = va_arg(argp, npy_int);
  if (dim==-1)
    {
      sz = PyArray_SIZE(var);
    }
  else
    {
      rank = PyArray_NDIM(var);
      if (dim>=1 && dim<=rank)
        sz = PyArray_DIM(var, dim-1);
      else
        fprintf(stderr, "f2py_size: 2nd argument value=%d fails to satisfy 1<=value<=%d. Result will be 0.\n", dim, rank);
    }
  va_end(argp);
  return sz;
}

static int int_from_pyobj(int* v,PyObject *obj,const char *errmess) {
    PyObject* tmp = NULL;
    if (PyInt_Check(obj)) {
        *v = (int)PyInt_AS_LONG(obj);
        return 1;
    }
    tmp = PyNumber_Int(obj);
    if (tmp) {
        *v = PyInt_AS_LONG(tmp);
        Py_DECREF(tmp);
        return 1;
    }
    if (PyComplex_Check(obj))
        tmp = PyObject_GetAttrString(obj,"real");
    else if (PyString_Check(obj) || PyUnicode_Check(obj))
        /*pass*/;
    else if (PySequence_Check(obj))
        tmp = PySequence_GetItem(obj,0);
    if (tmp) {
        PyErr_Clear();
        if (int_from_pyobj(v,tmp,errmess)) {Py_DECREF(tmp); return 1;}
        Py_DECREF(tmp);
    }
    {
        PyObject* err = PyErr_Occurred();
        if (err==NULL) err = _f90_ext_base_error;
        PyErr_SetString(err,errmess);
    }
    return 0;
}


/********************* See f2py2e/cfuncs.py: userincludes *********************/
/*need_userincludes*/

/********************* See f2py2e/capi_rules.py: usercode *********************/


/* See f2py2e/rules.py */
extern void F_FUNC_US(autocov_calc_z,AUTOCOV_CALC_Z)(double*,double*,double*,int*,int*,int*,int*);
extern void F_FUNC_US(autocov_calc_x,AUTOCOV_CALC_X)(double*,double*,double*,int*,int*,int*,int*);
/*eof externroutines*/

/******************** See f2py2e/capi_rules.py: usercode1 ********************/


/******************* See f2py2e/cb_rules.py: buildcallback *******************/
/*need_callbacks*/

/*********************** See f2py2e/rules.py: buildapi ***********************/

/******************************* autocov_calc_z *******************************/
static char doc_f2py_rout__f90_ext_base_autocov_calc_z[] = "\
r_z = autocov_calc_z(fluct1,fluct2,r_z,[max_z_step,ncl1,ncl2,ncl3])\n\nWrapper for ``autocov_calc_z``.\
\n\nParameters\n----------\n"
"fluct1 : input rank-3 array('d') with bounds (ncl3,ncl2,ncl1)\n"
"fluct2 : input rank-3 array('d') with bounds (ncl3,ncl2,ncl1)\n"
"r_z : input rank-3 array('d') with bounds (max_z_step,ncl2,ncl1)\n"
"\nOther Parameters\n----------------\n"
"max_z_step : input int, optional\n    Default: shape(r_z,0)\n"
"ncl1 : input int, optional\n    Default: shape(fluct1,2)\n"
"ncl2 : input int, optional\n    Default: shape(fluct1,1)\n"
"ncl3 : input int, optional\n    Default: shape(fluct1,0)\n"
"\nReturns\n-------\n"
"r_z : rank-3 array('d') with bounds (max_z_step,ncl2,ncl1)";
/* extern void F_FUNC_US(autocov_calc_z,AUTOCOV_CALC_Z)(double*,double*,double*,int*,int*,int*,int*); */
static PyObject *f2py_rout__f90_ext_base_autocov_calc_z(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,double*,double*,int*,int*,int*,int*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  double *fluct1 = NULL;
  npy_intp fluct1_Dims[3] = {-1, -1, -1};
  const int fluct1_Rank = 3;
  PyArrayObject *capi_fluct1_tmp = NULL;
  int capi_fluct1_intent = 0;
  PyObject *fluct1_capi = Py_None;
  double *fluct2 = NULL;
  npy_intp fluct2_Dims[3] = {-1, -1, -1};
  const int fluct2_Rank = 3;
  PyArrayObject *capi_fluct2_tmp = NULL;
  int capi_fluct2_intent = 0;
  PyObject *fluct2_capi = Py_None;
  double *r_z = NULL;
  npy_intp r_z_Dims[3] = {-1, -1, -1};
  const int r_z_Rank = 3;
  PyArrayObject *capi_r_z_tmp = NULL;
  int capi_r_z_intent = 0;
  PyObject *r_z_capi = Py_None;
  int max_z_step = 0;
  PyObject *max_z_step_capi = Py_None;
  int ncl1 = 0;
  PyObject *ncl1_capi = Py_None;
  int ncl2 = 0;
  PyObject *ncl2_capi = Py_None;
  int ncl3 = 0;
  PyObject *ncl3_capi = Py_None;
  static char *capi_kwlist[] = {"fluct1","fluct2","r_z","max_z_step","ncl1","ncl2","ncl3",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO|OOOO:_f90_ext_base.autocov_calc_z",\
    capi_kwlist,&fluct1_capi,&fluct2_capi,&r_z_capi,&max_z_step_capi,&ncl1_capi,&ncl2_capi,&ncl3_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable fluct1 */
  ;
  capi_fluct1_intent |= F2PY_INTENT_IN;
  capi_fluct1_tmp = array_from_pyobj(NPY_DOUBLE,fluct1_Dims,fluct1_Rank,capi_fluct1_intent,fluct1_capi);
  if (capi_fluct1_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _f90_ext_base_error,"failed in converting 1st argument `fluct1' of _f90_ext_base.autocov_calc_z to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    fluct1 = (double *)(PyArray_DATA(capi_fluct1_tmp));

  /* Processing variable ncl1 */
  if (ncl1_capi == Py_None) ncl1 = shape(fluct1,2); else
    f2py_success = int_from_pyobj(&ncl1,ncl1_capi,"_f90_ext_base.autocov_calc_z() 2nd keyword (ncl1) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(fluct1,2)==ncl1,"shape(fluct1,2)==ncl1","2nd keyword ncl1","autocov_calc_z:ncl1=%d",ncl1) {
  /* Processing variable ncl2 */
  if (ncl2_capi == Py_None) ncl2 = shape(fluct1,1); else
    f2py_success = int_from_pyobj(&ncl2,ncl2_capi,"_f90_ext_base.autocov_calc_z() 3rd keyword (ncl2) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(fluct1,1)==ncl2,"shape(fluct1,1)==ncl2","3rd keyword ncl2","autocov_calc_z:ncl2=%d",ncl2) {
  /* Processing variable ncl3 */
  if (ncl3_capi == Py_None) ncl3 = shape(fluct1,0); else
    f2py_success = int_from_pyobj(&ncl3,ncl3_capi,"_f90_ext_base.autocov_calc_z() 4th keyword (ncl3) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(fluct1,0)==ncl3,"shape(fluct1,0)==ncl3","4th keyword ncl3","autocov_calc_z:ncl3=%d",ncl3) {
  /* Processing variable r_z */
  r_z_Dims[1]=ncl2,r_z_Dims[2]=ncl1;
  capi_r_z_intent |= F2PY_INTENT_IN|F2PY_INTENT_OUT;
  capi_r_z_tmp = array_from_pyobj(NPY_DOUBLE,r_z_Dims,r_z_Rank,capi_r_z_intent,r_z_capi);
  if (capi_r_z_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _f90_ext_base_error,"failed in converting 3rd argument `r_z' of _f90_ext_base.autocov_calc_z to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    r_z = (double *)(PyArray_DATA(capi_r_z_tmp));

  /* Processing variable fluct2 */
  fluct2_Dims[0]=ncl3,fluct2_Dims[1]=ncl2,fluct2_Dims[2]=ncl1;
  capi_fluct2_intent |= F2PY_INTENT_IN;
  capi_fluct2_tmp = array_from_pyobj(NPY_DOUBLE,fluct2_Dims,fluct2_Rank,capi_fluct2_intent,fluct2_capi);
  if (capi_fluct2_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _f90_ext_base_error,"failed in converting 2nd argument `fluct2' of _f90_ext_base.autocov_calc_z to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    fluct2 = (double *)(PyArray_DATA(capi_fluct2_tmp));

  /* Processing variable max_z_step */
  if (max_z_step_capi == Py_None) max_z_step = shape(r_z,0); else
    f2py_success = int_from_pyobj(&max_z_step,max_z_step_capi,"_f90_ext_base.autocov_calc_z() 1st keyword (max_z_step) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(r_z,0)==max_z_step,"shape(r_z,0)==max_z_step","1st keyword max_z_step","autocov_calc_z:max_z_step=%d",max_z_step) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(fluct1,fluct2,r_z,&max_z_step,&ncl1,&ncl2,&ncl3);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_r_z_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*CHECKSCALAR(shape(r_z,0)==max_z_step)*/
  } /*if (f2py_success) of max_z_step*/
  /* End of cleaning variable max_z_step */
  if((PyObject *)capi_fluct2_tmp!=fluct2_capi) {
    Py_XDECREF(capi_fluct2_tmp); }
  }  /*if (capi_fluct2_tmp == NULL) ... else of fluct2*/
  /* End of cleaning variable fluct2 */
  }  /*if (capi_r_z_tmp == NULL) ... else of r_z*/
  /* End of cleaning variable r_z */
  } /*CHECKSCALAR(shape(fluct1,0)==ncl3)*/
  } /*if (f2py_success) of ncl3*/
  /* End of cleaning variable ncl3 */
  } /*CHECKSCALAR(shape(fluct1,1)==ncl2)*/
  } /*if (f2py_success) of ncl2*/
  /* End of cleaning variable ncl2 */
  } /*CHECKSCALAR(shape(fluct1,2)==ncl1)*/
  } /*if (f2py_success) of ncl1*/
  /* End of cleaning variable ncl1 */
  if((PyObject *)capi_fluct1_tmp!=fluct1_capi) {
    Py_XDECREF(capi_fluct1_tmp); }
  }  /*if (capi_fluct1_tmp == NULL) ... else of fluct1*/
  /* End of cleaning variable fluct1 */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/*************************** end of autocov_calc_z ***************************/

/******************************* autocov_calc_x *******************************/
static char doc_f2py_rout__f90_ext_base_autocov_calc_x[] = "\
r_x = autocov_calc_x(fluct1,fluct2,r_x,[max_x_step,ncl3,ncl2,ncl1])\n\nWrapper for ``autocov_calc_x``.\
\n\nParameters\n----------\n"
"fluct1 : input rank-3 array('d') with bounds (ncl3,ncl2,ncl1)\n"
"fluct2 : input rank-3 array('d') with bounds (ncl3,ncl2,ncl1)\n"
"r_x : input rank-3 array('d') with bounds (max_x_step,ncl2,ncl1-max_x_step)\n"
"\nOther Parameters\n----------------\n"
"max_x_step : input int, optional\n    Default: shape(r_x,0)\n"
"ncl3 : input int, optional\n    Default: shape(fluct1,0)\n"
"ncl2 : input int, optional\n    Default: shape(fluct1,1)\n"
"ncl1 : input int, optional\n    Default: shape(fluct1,2)\n"
"\nReturns\n-------\n"
"r_x : rank-3 array('d') with bounds (max_x_step,ncl2,ncl1-max_x_step)";
/* extern void F_FUNC_US(autocov_calc_x,AUTOCOV_CALC_X)(double*,double*,double*,int*,int*,int*,int*); */
static PyObject *f2py_rout__f90_ext_base_autocov_calc_x(const PyObject *capi_self,
                           PyObject *capi_args,
                           PyObject *capi_keywds,
                           void (*f2py_func)(double*,double*,double*,int*,int*,int*,int*)) {
  PyObject * volatile capi_buildvalue = NULL;
  volatile int f2py_success = 1;
/*decl*/

  double *fluct1 = NULL;
  npy_intp fluct1_Dims[3] = {-1, -1, -1};
  const int fluct1_Rank = 3;
  PyArrayObject *capi_fluct1_tmp = NULL;
  int capi_fluct1_intent = 0;
  PyObject *fluct1_capi = Py_None;
  double *fluct2 = NULL;
  npy_intp fluct2_Dims[3] = {-1, -1, -1};
  const int fluct2_Rank = 3;
  PyArrayObject *capi_fluct2_tmp = NULL;
  int capi_fluct2_intent = 0;
  PyObject *fluct2_capi = Py_None;
  double *r_x = NULL;
  npy_intp r_x_Dims[3] = {-1, -1, -1};
  const int r_x_Rank = 3;
  PyArrayObject *capi_r_x_tmp = NULL;
  int capi_r_x_intent = 0;
  PyObject *r_x_capi = Py_None;
  int max_x_step = 0;
  PyObject *max_x_step_capi = Py_None;
  int ncl3 = 0;
  PyObject *ncl3_capi = Py_None;
  int ncl2 = 0;
  PyObject *ncl2_capi = Py_None;
  int ncl1 = 0;
  PyObject *ncl1_capi = Py_None;
  static char *capi_kwlist[] = {"fluct1","fluct2","r_x","max_x_step","ncl3","ncl2","ncl1",NULL};

/*routdebugenter*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_clock();
#endif
  if (!PyArg_ParseTupleAndKeywords(capi_args,capi_keywds,\
    "OOO|OOOO:_f90_ext_base.autocov_calc_x",\
    capi_kwlist,&fluct1_capi,&fluct2_capi,&r_x_capi,&max_x_step_capi,&ncl3_capi,&ncl2_capi,&ncl1_capi))
    return NULL;
/*frompyobj*/
  /* Processing variable fluct1 */
  ;
  capi_fluct1_intent |= F2PY_INTENT_IN;
  capi_fluct1_tmp = array_from_pyobj(NPY_DOUBLE,fluct1_Dims,fluct1_Rank,capi_fluct1_intent,fluct1_capi);
  if (capi_fluct1_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _f90_ext_base_error,"failed in converting 1st argument `fluct1' of _f90_ext_base.autocov_calc_x to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    fluct1 = (double *)(PyArray_DATA(capi_fluct1_tmp));

  /* Processing variable ncl1 */
  if (ncl1_capi == Py_None) ncl1 = shape(fluct1,2); else
    f2py_success = int_from_pyobj(&ncl1,ncl1_capi,"_f90_ext_base.autocov_calc_x() 4th keyword (ncl1) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(fluct1,2)==ncl1,"shape(fluct1,2)==ncl1","4th keyword ncl1","autocov_calc_x:ncl1=%d",ncl1) {
  /* Processing variable ncl2 */
  if (ncl2_capi == Py_None) ncl2 = shape(fluct1,1); else
    f2py_success = int_from_pyobj(&ncl2,ncl2_capi,"_f90_ext_base.autocov_calc_x() 3rd keyword (ncl2) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(fluct1,1)==ncl2,"shape(fluct1,1)==ncl2","3rd keyword ncl2","autocov_calc_x:ncl2=%d",ncl2) {
  /* Processing variable ncl3 */
  if (ncl3_capi == Py_None) ncl3 = shape(fluct1,0); else
    f2py_success = int_from_pyobj(&ncl3,ncl3_capi,"_f90_ext_base.autocov_calc_x() 2nd keyword (ncl3) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(fluct1,0)==ncl3,"shape(fluct1,0)==ncl3","2nd keyword ncl3","autocov_calc_x:ncl3=%d",ncl3) {
  /* Processing variable fluct2 */
  fluct2_Dims[0]=ncl3,fluct2_Dims[1]=ncl2,fluct2_Dims[2]=ncl1;
  capi_fluct2_intent |= F2PY_INTENT_IN;
  capi_fluct2_tmp = array_from_pyobj(NPY_DOUBLE,fluct2_Dims,fluct2_Rank,capi_fluct2_intent,fluct2_capi);
  if (capi_fluct2_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _f90_ext_base_error,"failed in converting 2nd argument `fluct2' of _f90_ext_base.autocov_calc_x to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    fluct2 = (double *)(PyArray_DATA(capi_fluct2_tmp));

  /* Processing variable r_x */
  r_x_Dims[1]=ncl2;
  capi_r_x_intent |= F2PY_INTENT_IN|F2PY_INTENT_OUT;
  capi_r_x_tmp = array_from_pyobj(NPY_DOUBLE,r_x_Dims,r_x_Rank,capi_r_x_intent,r_x_capi);
  if (capi_r_x_tmp == NULL) {
    PyObject *exc, *val, *tb;
    PyErr_Fetch(&exc, &val, &tb);
    PyErr_SetString(exc ? exc : _f90_ext_base_error,"failed in converting 3rd argument `r_x' of _f90_ext_base.autocov_calc_x to C/Fortran array" );
    npy_PyErr_ChainExceptionsCause(exc, val, tb);
  } else {
    r_x = (double *)(PyArray_DATA(capi_r_x_tmp));

  /* Processing variable max_x_step */
  if (max_x_step_capi == Py_None) max_x_step = shape(r_x,0); else
    f2py_success = int_from_pyobj(&max_x_step,max_x_step_capi,"_f90_ext_base.autocov_calc_x() 1st keyword (max_x_step) can't be converted to int");
  if (f2py_success) {
  CHECKSCALAR(shape(r_x,0)==max_x_step,"shape(r_x,0)==max_x_step","1st keyword max_x_step","autocov_calc_x:max_x_step=%d",max_x_step) {
/*end of frompyobj*/
#ifdef F2PY_REPORT_ATEXIT
f2py_start_call_clock();
#endif
/*callfortranroutine*/
        (*f2py_func)(fluct1,fluct2,r_x,&max_x_step,&ncl3,&ncl2,&ncl1);
if (PyErr_Occurred())
  f2py_success = 0;
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_call_clock();
#endif
/*end of callfortranroutine*/
    if (f2py_success) {
/*pyobjfrom*/
/*end of pyobjfrom*/
    CFUNCSMESS("Building return value.\n");
    capi_buildvalue = Py_BuildValue("N",capi_r_x_tmp);
/*closepyobjfrom*/
/*end of closepyobjfrom*/
    } /*if (f2py_success) after callfortranroutine*/
/*cleanupfrompyobj*/
  } /*CHECKSCALAR(shape(r_x,0)==max_x_step)*/
  } /*if (f2py_success) of max_x_step*/
  /* End of cleaning variable max_x_step */
  }  /*if (capi_r_x_tmp == NULL) ... else of r_x*/
  /* End of cleaning variable r_x */
  if((PyObject *)capi_fluct2_tmp!=fluct2_capi) {
    Py_XDECREF(capi_fluct2_tmp); }
  }  /*if (capi_fluct2_tmp == NULL) ... else of fluct2*/
  /* End of cleaning variable fluct2 */
  } /*CHECKSCALAR(shape(fluct1,0)==ncl3)*/
  } /*if (f2py_success) of ncl3*/
  /* End of cleaning variable ncl3 */
  } /*CHECKSCALAR(shape(fluct1,1)==ncl2)*/
  } /*if (f2py_success) of ncl2*/
  /* End of cleaning variable ncl2 */
  } /*CHECKSCALAR(shape(fluct1,2)==ncl1)*/
  } /*if (f2py_success) of ncl1*/
  /* End of cleaning variable ncl1 */
  if((PyObject *)capi_fluct1_tmp!=fluct1_capi) {
    Py_XDECREF(capi_fluct1_tmp); }
  }  /*if (capi_fluct1_tmp == NULL) ... else of fluct1*/
  /* End of cleaning variable fluct1 */
/*end of cleanupfrompyobj*/
  if (capi_buildvalue == NULL) {
/*routdebugfailure*/
  } else {
/*routdebugleave*/
  }
  CFUNCSMESS("Freeing memory.\n");
/*freemem*/
#ifdef F2PY_REPORT_ATEXIT
f2py_stop_clock();
#endif
  return capi_buildvalue;
}
/*************************** end of autocov_calc_x ***************************/
/*eof body*/

/******************* See f2py2e/f90mod_rules.py: buildhooks *******************/
/*need_f90modhooks*/

/************** See f2py2e/rules.py: module_rules['modulebody'] **************/

/******************* See f2py2e/common_rules.py: buildhooks *******************/

/*need_commonhooks*/

/**************************** See f2py2e/rules.py ****************************/

static FortranDataDef f2py_routine_defs[] = {
  {"autocov_calc_z",-1,{{-1}},0,(char *)F_FUNC_US(autocov_calc_z,AUTOCOV_CALC_Z),(f2py_init_func)f2py_rout__f90_ext_base_autocov_calc_z,doc_f2py_rout__f90_ext_base_autocov_calc_z},
  {"autocov_calc_x",-1,{{-1}},0,(char *)F_FUNC_US(autocov_calc_x,AUTOCOV_CALC_X),(f2py_init_func)f2py_rout__f90_ext_base_autocov_calc_x,doc_f2py_rout__f90_ext_base_autocov_calc_x},

/*eof routine_defs*/
  {NULL}
};

static PyMethodDef f2py_module_methods[] = {

  {NULL,NULL}
};

static struct PyModuleDef moduledef = {
  PyModuleDef_HEAD_INIT,
  "_f90_ext_base",
  NULL,
  -1,
  f2py_module_methods,
  NULL,
  NULL,
  NULL,
  NULL
};

PyMODINIT_FUNC PyInit__f90_ext_base(void) {
  int i;
  PyObject *m,*d, *s, *tmp;
  m = _f90_ext_base_module = PyModule_Create(&moduledef);
  Py_SET_TYPE(&PyFortran_Type, &PyType_Type);
  import_array();
  if (PyErr_Occurred())
    {PyErr_SetString(PyExc_ImportError, "can't initialize module _f90_ext_base (failed to import numpy)"); return m;}
  d = PyModule_GetDict(m);
  s = PyString_FromString("$Revision: $");
  PyDict_SetItemString(d, "__version__", s);
  Py_DECREF(s);
  s = PyUnicode_FromString(
    "This module '_f90_ext_base' is auto-generated with f2py (version:2).\nFunctions:\n"
"  r_z = autocov_calc_z(fluct1,fluct2,r_z,max_z_step=shape(r_z,0),ncl1=shape(fluct1,2),ncl2=shape(fluct1,1),ncl3=shape(fluct1,0))\n"
"  r_x = autocov_calc_x(fluct1,fluct2,r_x,max_x_step=shape(r_x,0),ncl3=shape(fluct1,0),ncl2=shape(fluct1,1),ncl1=shape(fluct1,2))\n"
".");
  PyDict_SetItemString(d, "__doc__", s);
  Py_DECREF(s);
  _f90_ext_base_error = PyErr_NewException ("_f90_ext_base.error", NULL, NULL);
  /*
   * Store the error object inside the dict, so that it could get deallocated.
   * (in practice, this is a module, so it likely will not and cannot.)
   */
  PyDict_SetItemString(d, "__f90_ext_base_error", _f90_ext_base_error);
  Py_DECREF(_f90_ext_base_error);
  for(i=0;f2py_routine_defs[i].name!=NULL;i++) {
    tmp = PyFortranObject_NewAsAttr(&f2py_routine_defs[i]);
    PyDict_SetItemString(d, f2py_routine_defs[i].name, tmp);
    Py_DECREF(tmp);
  }


/*eof initf2pywraphooks*/
/*eof initf90modhooks*/

/*eof initcommonhooks*/


#ifdef F2PY_REPORT_ATEXIT
  if (! PyErr_Occurred())
    on_exit(f2py_report_on_exit,(void*)"_f90_ext_base");
#endif
  return m;
}
#ifdef __cplusplus
}
#endif