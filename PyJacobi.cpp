#include "PyJacobi.h"
#include "WorkArounds.h"
#include <cmath>
#include <iostream>

PyJacobi::PyJacobi(const int N) :
N(N), u_out(NULL), u_even(NULL), u_odd(NULL), rhs(NULL), nIter(0)
{

}

PyJacobi::~PyJacobi()
{
	fromdev(); //delete oacc data on the device
}

void PyJacobi::set_u_out(boost::python::object obj)
{
	PyObject* pobj = obj.ptr();
	Py_buffer pybuf;
	PyObject_GetBuffer(pobj,&pybuf,PyBUF_SIMPLE);
	void * buf = pybuf.buf;
	u_out = (double*) buf;
}

void PyJacobi::set_u_even(boost::python::object obj)
{
	PyObject* pobj = obj.ptr();
	Py_buffer pybuf;
	PyObject_GetBuffer(pobj,&pybuf,PyBUF_SIMPLE);
	void * buf = pybuf.buf;
	u_even = (double*) buf;
}

void PyJacobi::set_u_odd(boost::python::object obj)
{
	PyObject* pobj = obj.ptr();
	Py_buffer pybuf;
	PyObject_GetBuffer(pobj,&pybuf,PyBUF_SIMPLE);
	void * buf = pybuf.buf;
	u_odd = (double*) buf;
}

void PyJacobi::set_rhs(boost::python::object obj)
{
	PyObject* pobj = obj.ptr();
	Py_buffer pybuf;
	PyObject_GetBuffer(pobj,&pybuf,PyBUF_SIMPLE);
	void * buf = pybuf.buf;
	rhs = (double*) buf;
}

void PyJacobi::todev()
{
#pragma acc enter data copyin(this[0:1],u_out[0:N],u_even[0:N],u_odd[0:N],rhs[0:N])
}

void PyJacobi::fromdev()
{
#pragma acc exit data delete(u_out[0:N],u_even[0:N],u_odd[0:N],rhs[0:N], this[0:1])
}

void PyJacobi::updatehost()
{
#pragma acc update self(u_out[0:N],u_even[0:N],u_odd[0:N])
}

int PyJacobi::solve(double tol, int maxIter)
{

	todev(); // move data to the device
	nIter = 0;
	int exit_code = 0;
	bool KEEP_GOING = true;
	double normUpdate, normU, relUpdate;
//#pragma acc data present(u_even,u_odd,rhs,this) 
{
	while(KEEP_GOING)
	{
		nIter++; //increment iteration counter

		if(nIter%2==0){
//#pragma acc data present(u_even,u_odd,rhs,this)
#pragma acc parallel loop present(u_even,u_odd,rhs)
			for(int i=1;i<(N-1); i++){
				u_odd[i] = 0.5*(u_even[i-1] + u_even[i+1] - rhs[i]);
			}

            normU = 0.;

//#pragma acc data present(u_odd,this)
#pragma acc parallel loop reduction(+:normU) present(u_odd)
            for(int i=0;i<N;i++){
            	normU = normU + u_odd[i]*u_odd[i];
            }

		}else{
//#pragma acc data present(u_even,u_odd,rhs,this)
#pragma acc parallel loop present(u_even,u_odd,rhs,this)
			for(int i=1;i<(N-1);i++){
				u_even[i] = 0.5*(u_odd[i-1]+u_odd[i+1] - rhs[i]);
			}

//#pragma acc data present(u_even,this)
#pragma acc parallel loop reduction(+:normU)
            for(int i=0;i<N;i++){
            	normU = normU + u_even[i]*u_even[i];
            }

		}
		// check for convergence
		normUpdate = 0.;

//#pragma acc data present(u_even,u_odd,this)
#pragma acc parallel loop reduction(+:normUpdate) present(u_even,u_odd,this)
		for(int i=0;i<N;i++){
			normUpdate = normUpdate + (u_even[i] - u_odd[i])*(u_even[i] -u_odd[i]);
		}

		relUpdate = sqrt(normUpdate)/sqrt(normU);
		if(relUpdate < tol){
			KEEP_GOING = false;
			// copy most up-to-date data into u_out
			if(nIter%2==0){
//#pragma acc data present(u_out,u_odd,this)
#pragma acc parallel loop present(u_out,u_odd,this)
				for(int i=0;i<N;i++){
					u_out[i] = u_odd[i];
				}
			}else{
//#pragma acc data present(u_out,u_even,this)
#pragma acc parallel loop present(u_out,u_even,this)
				for(int i=0;i<N;i++){
					u_out[i] = u_even[i];
				}
			}
		}
		if(nIter == maxIter){
			KEEP_GOING = false;
			exit_code = -1;
		}

	}
}//end acc data region
    updatehost();
	return exit_code;

}

int PyJacobi::get_nIter()
{
  return nIter;
}

using namespace boost::python;
BOOST_PYTHON_MODULE(pyJacobi)
{
	class_<PyJacobi>("PyJacobi",init<int>())
				.def("set_u_out",&PyJacobi::set_u_out)
				.def("set_u_even",&PyJacobi::set_u_even)
				.def("set_u_odd",&PyJacobi::set_u_odd)
				.def("set_rhs",&PyJacobi::set_rhs)
				.def("solve",&PyJacobi::solve)
                                .def("get_nIter",&PyJacobi::get_nIter)
				;
}




