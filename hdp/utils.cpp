#include "utils.h"

extern gsl_rng * RANDOM_NUMBER;

const double half_ln_2pi = 0.91893853320467267;

/**
 * given log(a) and log(b), return log(a + b)
 *
 */

double log_sum(double log_a, double log_b)
{
    double v;

    if (log_a < log_b)
        v = log_b+log(1 + exp(log_a-log_b));
    else
        v = log_a+log(1 + exp(log_b-log_a));
    return v;
}


// give a_1, ..., a_n,
// return log(exp(a_1)+...+exp(a_n))
double log_normalize(double * array, int nlen)
{
   const double log_max = 100.0; // the log(maximum in double precision), make sure it is large enough.
   int argmax;
   double max_val = max(array, nlen, &argmax); //get the maximum value in the array to avoid overflow
   double log_shift = log_max - log(nlen + 1.0) - max_val;
   double sum = 0.0;
   for (int i = 0; i < nlen; i++)
       sum += exp(array[i] + log_shift); //shift it

   double log_norm = log(sum) - log_shift;
   for (int i = 0; i < nlen; i++)
       array[i] -= log_norm; //shift it back

   return log_norm;
}

// the vector version
double log_normalize(vector<double> & vec, int nlen)
{
   const double log_max = 100.0; // the log(maximum in double precision), make sure it is large enough.
   int argmax;
   double max_val = max_vec(vec, nlen, &argmax); //get the maximum value in the array to avoid overflow
   double log_shift = log_max - log(nlen + 1.0) - max_val;
   double sum = 0.0;
   for (int i = 0; i < nlen; i++)
       sum += exp(vec[i] + log_shift); //shift it

   double log_norm = log(sum) - log_shift;
   for (int i = 0; i < nlen; i++)
       vec[i] -= log_norm; //shift it back

   return log_norm;
}

/**
 * given log(a) and log(b), return log(a - b) a>b
 *
 */

double log_subtract(double log_a, double log_b)
{
    if (log_a < log_b) return -1000.0;

    double v;
    v = log_a + log(1 - exp(log_b-log_a));
    return v;
}

/**
*
* check if file exisits
*/
bool file_exists(const char * filename)
{
    if ( 0 == access(filename, R_OK))
        return true;
    return false;
}

/**
*
* check if directory exisits
*/
bool dir_exists(const char * directory)
{
    struct stat st;
    if(stat(directory, &st) == 0)
        return true;
    return false;
}

/**
 * return factorial log((n-1+a)...(a))
 *
 **/

double log_factorial(int n, double a)
{
    if (n == 0) return 0.0;
    double v = lgamma(n+a) - lgamma(a);
    return v;
}

/**
 * return the cosine similarity
 *
 **/

double similarity(const int* v1, const int* v2, int n)
{
    double sim = 0.0, norm1 = 0.0, norm2 = 0.0;
    for (int i = 0; i < n; i ++)
    {
        sim += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    return sim/sqrt(norm1*norm2);
}

/// gsl_wrappers
/* double lgamma(double x) 
{
    return gsl_sf_lngamma(x);
}
*/
unsigned int rmultinomial(const double* p, int n, double tot_p)
{
    int i;
    if (tot_p < 0)
    {
        tot_p = 0.0;
        for (i = 0; i < n; i ++) tot_p += p[i];
    }

    double u = runiform() * tot_p;
    double cum_p = 0.0;
    for (i = 0; i < n; i ++)
    {
        cum_p += p[i];
        if (u < cum_p) break;
    }
    return i;
}

double rgamma(double a, double b)
{
    return gsl_ran_gamma_mt(RANDOM_NUMBER, a, b);
}

double rbeta(double a, double b)
{
    return gsl_ran_beta(RANDOM_NUMBER, a, b);
}

unsigned int rbernoulli(double p)
{
    return gsl_ran_bernoulli(RANDOM_NUMBER, p);
}

double runiform()
{
    return gsl_rng_uniform_pos(RANDOM_NUMBER);
}

void rshuffle(void* base, size_t n, size_t size)
{
    gsl_ran_shuffle(RANDOM_NUMBER, base, n, size);
}

unsigned long int runiform_int(unsigned long int n)
{
    return gsl_rng_uniform_int(RANDOM_NUMBER, n);
}

// end of the file
