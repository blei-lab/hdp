// Unsigned Stirling number of first kind in log space.

#ifndef _STIRLN_H
#define	_STIRLN_H

#include "utils.h"
#define log_zero -10000.0

class Stirling {
public:
  Stirling();
  ~Stirling();
  double get_log_stirling_num(size_t n, size_t m); 
  //return the log of stirling(n,m)
private:
  vector<double* > log_stirling_num_;
};

#endif	/* _STIRLN_H */

