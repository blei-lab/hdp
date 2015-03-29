#ifndef STATE_H
#define STATE_H

#include "corpus.h"
#include <map>

class hdp_hyperparameter
{
    /// hyperparameters
public:
    double m_gamma_a;
    double m_gamma_b;
    double m_alpha_a;
    double m_alpha_b;
    int    m_max_iter;
    int    m_save_lag;
    int    m_num_restricted_scans;

    bool m_sample_hyperparameter;
    bool m_split_merge_sampler;

public:
    void setup_parameters(double _gamma_a, double _gamma_b,
                        double _alpha_a, double _alpha_b,
                        int _max_iter, int _save_lag,
                        int _num_restricted_scans,
                        bool _sample_hyperparameter,
                        bool _split_merge_sampler)
    {
        m_gamma_a   = _gamma_a;
        m_gamma_b   = _gamma_b;
        m_alpha_a   = _alpha_a;
        m_alpha_b   = _alpha_b;
        m_max_iter  = _max_iter;
        m_save_lag  = _save_lag;
        m_num_restricted_scans = _num_restricted_scans;
        m_sample_hyperparameter = _sample_hyperparameter;
        m_split_merge_sampler = _split_merge_sampler;
    }
};

typedef vector<int> int_vec; // define the vector of int
typedef vector<double> double_vec; // define the vector of double
typedef map<int, int> word_stats;
enum ACTION {SPLIT, MERGE};

/// word info structure used in the main class
struct word_info
{
public:
    int m_word_index;
    int m_table_assignment;
    //int m_topic_assignment; // this is extra information
};

class doc_state
{
public:
    int m_doc_id; // document id
    int m_doc_length;  // document length
    int m_num_tables;  // number of tables in this document
    word_info * m_words;

    int_vec m_table_to_topic; // for a doc, translate its table index to topic index
    int_vec m_word_counts_by_t; // word counts for each table
    vector< word_stats > m_word_stats_by_t;

    //vector < vector<int> > m_words_by_zi; // stores the word idx indexed by z then i
public:
    doc_state();
    virtual ~doc_state();
public:
    void setup_state_from_doc(const document * doc);
    void free_doc_state();
};

class hdp_state
{
public:

/// doc information, fix value
    int m_size_vocab;
    int m_total_words;
    int m_num_docs;

/// document states
    doc_state** m_doc_states;

/// number of topics
    int m_num_topics;
/// total number of tables for all topics
    int m_total_num_tables;

/// by_z, by topic
/// by_d, by document, for each topic
/// by_w, by word, for each topic
/// by_t, by table for each document
    int_vec   m_num_tables_by_z; // how many tables each topic has
    int_vec   m_word_counts_by_z;   // word counts for each topic
    vector <int*> m_word_counts_by_zd; // word counts for [each topic, each doc]
    vector <int*> m_word_counts_by_zw; // word counts for [each topic, each word]

/// topic Dirichlet parameter
    double m_eta;

/// including concentration parameters
    double m_gamma;
    double m_alpha;
public:
    hdp_state();
    virtual ~hdp_state();
public:
    void   setup_state_from_corpus(const corpus* c);
    void   allocate_initial_space();
    void   free_state();
    void   init_gibbs_state_using_docs();
    void   init_gibbs_state_with_fixed_num_topics();
    void   iterate_gibbs_state(bool remove, bool permute,
                               hdp_hyperparameter* hdp_hyperparam,
                               bool table_sampling=false);

    void   sample_first_level_concentration(hdp_hyperparameter* hdp_hyperparam);
    void   sample_second_level_concentration(hdp_hyperparameter* hdp_hyperparam);
    void   sample_tables(doc_state* d_state, double_vec & q, double_vec & f);
    void   sample_table_assignment(doc_state* d_state, int t, int* words, double_vec & q, double_vec & f);
    void   sample_word_assignment(doc_state* d_state, int i, bool remove, double_vec & q, double_vec & f);
    void   doc_state_update(doc_state* d_state, int i, int update, int k=-1);
    void   compact_doc_state(doc_state* d_state, int* k_to_new_k);
    void   compact_hdp_state();
    double doc_partition_likelihood(doc_state* d_state);
    double table_partition_likelihood();
    double data_likelihood();
    double joint_likelihood(hdp_hyperparameter * hdp_hyperparam);
    void   save_state(char * name);
    void   save_state_ex(char * name);
    void   load_state_ex(char * name);

    /// the followings are the functions used in the split-merge algorithm
    void   copy_state(const hdp_state* state);
    ACTION select_mcmc_move(int& d0, int& d1, int& t0, int& t1);
    void   doc_table_state_update(doc_state* d_state, int t, int update, int k=-1);
    double sample_table_assignment_sm(doc_state* d_state, int t, bool remove,
                                      int k0, int k1, int target_k=-1);
    void   merge_two_topics(int k0, int k1);
    double split_sampling(int num_scans, int d0, int d1, int t0, int t1,
                                     hdp_state * target_state=NULL);
    friend double compute_split_ratio(const hdp_state* split_state, const hdp_state* merge_state, int k0, int k1);

    /// functions that should not be used when running the experiments
    bool   state_check_sum();
};

// double compute_split_ratio(const hdp_state* split_state, const hdp_state* merge_state, int k0, int k1);

#endif // STATE_H
