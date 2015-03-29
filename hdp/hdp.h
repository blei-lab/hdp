#ifndef HDP_H
#define HDP_H

#include "state.h"


/// implement the Chinese restaurant franchies algorithm with split and merge
class hdp
{
public:
/// fixed parameters
    hdp_hyperparameter * m_hdp_param;

/// sampling state
    hdp_state * m_state;

public:
    hdp();
    virtual ~hdp();
public:
    void run(const char * directory);
    void run_test(const char * directory);

    void setup_state(const corpus * c,
                     double _eta, int init_topics,
                     hdp_hyperparameter * _hdp_param);
    void setup_state(const corpus * c, 
                     hdp_hyperparameter * _hdp_param);
    void load(char * model_path);

};

#endif // HDP_H
