#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#include <memory.h>
#include <time.h>

#include <math.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_sf_psi.h>
#include <gsl/gsl_sf_gamma.h>

#include <vector>
using namespace std;

double log_sum(double log_a, double log_b);
double log_normalize(double * array, int nlen);
double log_normalize(vector<double> & vec, int nlen);
double log_subtract(double log_a, double log_b);
double log_factorial(int n, double a);
double similarity(const int* v1, const int* v2, int n);

bool   file_exists(const char * filename);
bool   dir_exists(const char * directory);

template <typename T> void free_vec_ptr(vector <T* > & v)
{
    int size = v.size();
    T* p = NULL;
    for (int i = 0; i < size; i ++)
    {
        p = v[i];
        delete [] p;
    }
    v.clear();
}

// find the max and argmax in an array
template <typename T> T max(const T * x, int n, int* argmax)
{
    *argmax = 0;
    T max_val = x[0];
    for (int i = 1; i < n; i++)
        if (x[i] > max_val)
        {
            max_val = x[i];
            *argmax = i;
        }
    return max_val;
}

// find the max and argmax in an vector
template <typename T> T max_vec(vector<T> & v, int n, int* argmax)
{
    *argmax = 0;
    T max_val = v[0];
    for (int i = 1; i < n; i++)
        if (v[i] > max_val)
        {
            max_val = v[i];
            *argmax = i;
        }
    return max_val;
}

// set a value to the entire array
template <typename T> void set_array(T * array, int size, T value)
{
    for (int i = 0; i < size; i++) array[i] = value;
}

// swap two elements in an array
template < typename T > void swap_array_element(T * array, int i, int j)
{
    if (i == j) return;
    T a = array[i];
    array[i] = array[j];
    array[j] = a;
}

// set a value to a entrie vector
template <typename T> void set_vector(vector<T> & v, T value)
{
    int size = v.size();
    for (int i = 0; i < size; i++) v[i] = value;
}

// swap two elements in vector
template < typename T > void swap_vec_element(vector<T> & v, int i, int j)
{
    if (i == j) return; // no need to swap
    T a = v[i];
    v[i] = v[j];
    v[j] = a;
}


/// gsl_wrappers
//double lgamma(double x); // linux has this
unsigned int rmultinomial(const double* p, int n, double tot_p=-1);
double rgamma(double a, double b);
double rbeta(double a, double b);
unsigned int rbernoulli(double p);
double runiform();
void rshuffle (void* base, size_t n, size_t size);
unsigned long int runiform_int(unsigned long int n);


#endif
