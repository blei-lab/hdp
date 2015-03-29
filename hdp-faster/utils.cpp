#include "utils.h"

extern gsl_rng * RANDOM_NUMBER;

const double half_ln_2pi = 0.91893853320467267;

/*
 * compare two ints
 * */

int compare (const void * a, const void * b) {
  return ( *(int*)a - *(int*)b );
}

/**
 * given log(a) and log(b), return log(a + b)
 *
 */

double log_sum(double log_a, double log_b) {
  double v;

  if (log_a < log_b)
    v = log_b+log(1 + exp(log_a-log_b));
  else
    v = log_a+log(1 + exp(log_b-log_a));
  return v;
}

// give a_1, ..., a_n,

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

/*
 * make a directory 
 */

void make_directory(const char* name) {
  mkdir(name, S_IRUSR|S_IWUSR|S_IXUSR);
}

// return log(exp(a_1)+...+exp(a_n))
double log_normalize(vct* x) {
  int nlen = x->size(); 
  const double log_max = 100.0; // the log(maximum in double precision), make sure it is large enough.
  int argmax;
  double max_val = vct_max(*x, &argmax); //get the maximum value in the array to avoid overflow
  double log_shift = log_max - log(nlen + 1.0) - max_val;
  double sum = 0.0;
  for (int i = 0; i < nlen; ++i)
    sum += exp(x->at(i) + log_shift); //shift it
 
  double log_norm = log(sum) - log_shift;
  for (int i = 0; i < nlen; ++i)
    x->at(i) -= log_norm; //shift it back
 
  return log_norm;
}


double vct_normalize(vct* x) {
  double sum = vct_sum(*x);
  if (sum == 0) return 0.0;
  size_t size = x->size();
  for (size_t i = 0; i < size; ++i) x->at(i) /= sum;
  return sum;
}

void vct_log(vct* x) {
  size_t size = x->size();
  for (size_t i = 0; i < size; ++i) x->at(i) = safe_log(x->at(i));
}

void vct_exp(vct* x) {
  size_t size = x->size();
  for (size_t i = 0; i < size; ++i) x->at(i) = exp(x->at(i));
}



/// gsl_wrappers
double digamma(double x) {
  return gsl_sf_psi(x);
}

unsigned int rmultinomial(const vct& v, double tot) {
  if (tot < 0) tot = vct_sum(v); 
  double u = runiform() * tot;
  double cum_sum = 0.0;
  size_t i = 0;
  for (; i < v.size(); ++i) {
    cum_sum += v.at(i);
    if (u < cum_sum) break;
  }
  return i;
}

double rgamma(double a, double b) {
  return gsl_ran_gamma_mt(RANDOM_NUMBER, a, b);
}

double rbeta(double a, double b) {
  return gsl_ran_beta(RANDOM_NUMBER, a, b);
}

unsigned int rbernoulli(double p) {
  return gsl_ran_bernoulli(RANDOM_NUMBER, p);
}

double runiform() {
  return gsl_rng_uniform_pos(RANDOM_NUMBER);
}

void rshuffle(void* base, size_t n, size_t size) {
  gsl_ran_shuffle(RANDOM_NUMBER, base, n, size);
}

unsigned long int runiform_int(unsigned long int n) {
  return gsl_rng_uniform_int(RANDOM_NUMBER, n);
}

void choose_k_from_n(int k, int n, int* result, int* src) {
  gsl_ran_choose(RANDOM_NUMBER, (void *) result,  k, (void *) src, n, sizeof(int));
}

void sample_k_from_n(int k, int n, vct_int* result) {
  if ((int)result->size() < k) result->resize(k);
  for (int i = 0; i < k; ++i) {
    result->at(i) = runiform_int(n);
  }
}

/*
 * new random number generator
 *
 */
gsl_rng * new_random_number_generator(long seed) {
  gsl_rng * random_number_generator = gsl_rng_alloc(gsl_rng_taus);
  gsl_rng_set(random_number_generator, (long) seed); // init the seed

  return random_number_generator;
}

/* 
 * free random number generator
 * */

void free_random_number_generator(gsl_rng * random_number_generator) {
  gsl_rng_free(random_number_generator);
}

// end of the file
