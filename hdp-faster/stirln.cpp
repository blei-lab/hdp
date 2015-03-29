#include "stirln.h"
#include "utils.h"

Stirling::Stirling() {
  log_stirling_num_.reserve(200);
}

Stirling::~Stirling() {
  vct_ptr_free(&log_stirling_num_);
}

/*
 * return the log of the stirling number log(s(n,m))
 * s(n, n) = 1
 * s(n, 0) = 0 if n > 0
 * s(n, m) = 0 if n < m
 * s(n+1, m) = s(n, m-1) + ns(n, m)
 */
double Stirling::get_log_stirling_num(size_t n, size_t m) {
  if (n < m)  return log_zero;
  size_t start = log_stirling_num_.size();
  for (size_t i = start; i < n+1; ++i) {
    double* v = new double[i+1];
    for (size_t j = 0; j < i + 1; ++j) { v[j] = log_zero; }
    log_stirling_num_.push_back(v);
    log_stirling_num_[i][i] = 0.0;
    for (size_t j = 1; j < i; ++j) {
      log_stirling_num_[i][j] = 
           log_sum(log_stirling_num_[i-1][j-1], 
             log(i-1) + log_stirling_num_[i-1][j]);
    }
  }
  return log_stirling_num_[n][m];
}
