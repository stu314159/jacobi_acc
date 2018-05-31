#ifndef PYJACOBI_H_
#define PYJACOBI_H_

#include <boost/python.hpp>

class PyJacobi
{

public:
PyJacobi(const int N);
~PyJacobi();
void set_u_out(boost::python::object obj);
void set_u_even(boost::python::object obj);
void set_u_odd(boost::python::object obj);
void set_rhs(boost::python::object obj);
int solve(double tolerance, int maxIter);
int get_nIter();

void todev(); // for oacc - move data to device
void fromdev(); // for oacc- delete data from device
void updatehost(); // for oacc - update data on host


private:

const int N;
double * u_out;
double * u_even;
double * u_odd;
double * rhs;
int nIter;




};


#endif
