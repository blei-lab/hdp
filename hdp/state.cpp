#include "state.h"
#include "utils.h"
#include <assert.h>

#define SMALL_GIBBS_MAX_ITER 20
#define VERBOSE false
#define INIT_SIZE 50
#define INF -1e50

doc_state::doc_state()
{
    m_doc_id = -1; // document id
    m_doc_length = 0;  // document length
    m_num_tables = 0;  // number of tables in this document
    m_words = NULL;

    m_table_to_topic.clear(); // for a doc, translate its table index to topic index
    m_word_counts_by_t.clear(); // word counts for each table
    m_word_stats_by_t.clear();  // word stats infor each each table
}

doc_state::~doc_state()
{
    free_doc_state();
}

void doc_state::setup_state_from_doc(const document * doc)
{
    m_doc_id = doc->id;
    m_doc_length = doc->total;

    int word, count;
    m_words = new word_info[doc->total];
    int m = 0;
    for (int n = 0; n < doc->length; n++)
    {
        word  = doc->words[n];
        count = doc->counts[n];
        for (int j = 0; j < count; j++)
        {
            m_words[m].m_word_index = word;
            m_words[m].m_table_assignment = -1;
            //m_words[m].m_topic_assignment = -1;
            m++;
        }
    }
    //allocate some space
    m_table_to_topic.resize(INIT_SIZE, -1);
    m_word_counts_by_t.resize(INIT_SIZE, 0);
}

void doc_state::free_doc_state()
{
    m_table_to_topic.clear(); // for a doc, translate its table index to topic index
    m_word_counts_by_t.clear(); // word counts for each table
    m_word_stats_by_t.clear();

    delete [] m_words;
    m_words = NULL;
}

hdp_state::hdp_state()
{
    m_doc_states = NULL;
    m_size_vocab = 0;
    m_total_words = 0;;
    m_num_docs = 0;
    m_num_topics = 0;
    m_total_num_tables = 0;
    m_num_tables_by_z.clear();
    m_word_counts_by_z.clear();
    m_word_counts_by_zd.clear();
    m_word_counts_by_zw.clear();
}

hdp_state::~hdp_state()
{
    free_state();
}

void hdp_state::setup_state_from_corpus(const corpus * c)
{
    m_size_vocab = max(c->size_vocab, m_size_vocab);
    m_total_words += c->total_words;
    m_num_docs = c->num_docs;
    m_doc_states = new doc_state * [m_num_docs];

    for (unsigned int d = 0; d < c->docs.size(); d++)
    {
        document * doc      = c->docs[d];
        doc_state * d_state = new doc_state();
        m_doc_states[d]     = d_state;
        d_state->setup_state_from_doc(doc);
    }
}

void hdp_state::allocate_initial_space()
{
    // training
    if (m_num_tables_by_z.size() == 0)
    {
        m_num_tables_by_z.resize(INIT_SIZE, 0);
        m_word_counts_by_z.resize(INIT_SIZE, 0);
        m_word_counts_by_zd.resize(INIT_SIZE, NULL);
        m_word_counts_by_zw.resize(INIT_SIZE, NULL);
        int * p = NULL;
        for (int k = 0; k < INIT_SIZE; k++)
        {
            p = new int [m_num_docs];
            memset(p, 0, sizeof(int)*m_num_docs);
            m_word_counts_by_zd[k] = p;

            p = new int [m_size_vocab];
            memset(p, 0, sizeof(int)*m_size_vocab);
            m_word_counts_by_zw[k] = p;
        }
    }
    else // testing
    {
        while ((int)m_num_tables_by_z.size() < m_num_topics + 1)  
        {
            m_num_tables_by_z.push_back(0);
            m_word_counts_by_z.push_back(0);

            int* p = new int [m_size_vocab];
            memset(p, 0, sizeof(int)*m_size_vocab);
            m_word_counts_by_zw.push_back(p);
        }
        while ((int)m_word_counts_by_zd.size() < m_num_topics + 1)
        {
            int* p = new int [m_num_docs];
            memset(p, 0, sizeof(int)*m_num_docs);
            m_word_counts_by_zd.push_back(p);
        }
    }
}

void hdp_state::free_state()
{
    if (m_doc_states != NULL)
    {
        for (int d = 0; d < m_num_docs; d++)
        {
            doc_state * d_state = m_doc_states[d];
            delete d_state;
        }
        delete [] m_doc_states;
    }
    m_doc_states = NULL;

    m_size_vocab = 0;
    m_total_words = 0;;
    m_num_docs = 0;
    m_num_topics = 0;
    m_total_num_tables = 0;

    m_num_tables_by_z.clear();
    m_word_counts_by_z.clear();

    free_vec_ptr(m_word_counts_by_zd);
    free_vec_ptr(m_word_counts_by_zw);
}

void hdp_state::init_gibbs_state_using_docs()
{
    gsl_permutation* p = gsl_permutation_calloc(m_num_docs);
    rshuffle(p->data, m_num_docs, sizeof(size_t)); // shuffle the sequence

    /// allocate some space
    int k, d, i, j, doc_id, w;

    m_num_tables_by_z.resize(m_num_topics+1, 0);
    m_word_counts_by_z.resize(m_num_topics+1, 0);

    free_vec_ptr(m_word_counts_by_zd);
    free_vec_ptr(m_word_counts_by_zw);
    m_word_counts_by_zd.resize(m_num_topics+1, NULL);
    m_word_counts_by_zw.resize(m_num_topics+1, NULL);

    for (k = 0; k < m_num_topics+1; k ++)
    {
        m_word_counts_by_zd[k] = new int[m_num_docs];
        memset(m_word_counts_by_zd[k], 0, sizeof(int)*m_num_docs);
        m_word_counts_by_zw[k] = new int[m_size_vocab];
        memset(m_word_counts_by_zw[k], 0, sizeof(int)*m_size_vocab);
    }

    for (j = 0; j < m_num_topics; j ++) /// assign each doc a table and a topic
    {
        d = gsl_permutation_get(p, j);
        k = j;
        doc_state* d_state = m_doc_states[d];

        /// update the global book keepings
        m_total_num_tables ++; /// increase a topic
        m_num_tables_by_z[k] ++;
        m_word_counts_by_z[k] += d_state->m_doc_length;
        doc_id = d_state->m_doc_id;
        m_word_counts_by_zd[k][doc_id] = d_state->m_doc_length;

        /// update the local book keepings
        d_state->m_num_tables = 1;
        d_state->m_table_to_topic.resize(2, -1);
        d_state->m_word_counts_by_t.resize(2, 0);
        d_state->m_table_to_topic[0] = k;
        d_state->m_word_counts_by_t[0] = d_state->m_doc_length;
        for (i = 0; i < d_state->m_doc_length; i++)
        {
            w = d_state->m_words[i].m_word_index;
            m_word_counts_by_zw[k][w] ++;
            d_state->m_words[i].m_table_assignment = 0;
        }
    }

    double_vec q;
    double_vec f;
    for (j = m_num_topics; j < m_num_docs; j++)
    {
        d = gsl_permutation_get(p, j);
        k = d;
        doc_state* d_state = m_doc_states[d];
        for (i = 0; i < d_state->m_doc_length; i++)
        {
            sample_word_assignment(d_state, i, false, q, f);
        }
    }
    gsl_permutation_free(p);
}

void hdp_state::init_gibbs_state_with_fixed_num_topics()
{
    gsl_permutation* p = gsl_permutation_calloc(m_num_docs);
    rshuffle(p->data, m_num_docs, sizeof(size_t)); // shuffle the sequence

    /// allocate some space
    int k, d, i, j, doc_id, w;

    m_num_tables_by_z.resize(m_num_topics+1, 0);
    m_word_counts_by_z.resize(m_num_topics+1, 0);

    free_vec_ptr(m_word_counts_by_zd);
    free_vec_ptr(m_word_counts_by_zw);
    m_word_counts_by_zd.resize(m_num_topics+1, NULL);
    m_word_counts_by_zw.resize(m_num_topics+1, NULL);

    for (k = 0; k < m_num_topics+1; k ++)
    {
        m_word_counts_by_zd[k] = new int[m_num_docs];
        memset(m_word_counts_by_zd[k], 0, sizeof(int)*m_num_docs);
        m_word_counts_by_zw[k] = new int[m_size_vocab];
        memset(m_word_counts_by_zw[k], 0, sizeof(int)*m_size_vocab);
    }

    for (j = 0; j < m_num_topics; j ++) /// assign each doc a table and a topic
    {
        k = j;
        d = gsl_permutation_get(p, j);
        doc_state* d_state = m_doc_states[d];

        /// update the global book keepings
        m_total_num_tables ++; /// increase a topic
        m_num_tables_by_z[k] ++;
        m_word_counts_by_z[k] += d_state->m_doc_length;
        doc_id = d_state->m_doc_id;
        m_word_counts_by_zd[k][doc_id] += d_state->m_doc_length;

        /// update the local book keepings
        d_state->m_num_tables = 1;
        d_state->m_table_to_topic.resize(2, -1);
        d_state->m_word_counts_by_t.resize(2, 0);
        d_state->m_table_to_topic[0] = k;
        d_state->m_word_counts_by_t[0] = d_state->m_doc_length;
        for (i = 0; i < d_state->m_doc_length; i++)
        {
            w = d_state->m_words[i].m_word_index;
            m_word_counts_by_zw[k][w] ++;
            d_state->m_words[i].m_table_assignment = 0;
        }
    }

    int j0;
    double total_q;
    double prob;
    double* q  = new double[m_num_topics];
    int* v = new int[m_size_vocab];

    for (j = m_num_topics; j < m_num_docs; j++)
    {
        d = gsl_permutation_get(p, j);
        doc_state* d_state = m_doc_states[d];
        total_q = 0;

        memset(v, 0, sizeof(int) * m_size_vocab);
        for (i = 0; i < d_state->m_doc_length; i++)
        {
            w = d_state->m_words[i].m_word_index;
            v[w] ++;
        }
        for (j0 = 0; j0 < m_num_topics; j0 ++)
        {
            prob = similarity(v, m_word_counts_by_zw[k], m_size_vocab);
            total_q += prob;
            q[j0] = total_q;
        }

        double u = runiform() * total_q;
        for (j0 = 0; j0 < m_num_topics; j0++)
            if ( u < q[j0]) break;

        if (j0 == m_num_topics) j0 = runiform_int(m_num_topics);
        k = j0;
        m_total_num_tables ++; /// increase a topic
        m_num_tables_by_z[k] ++;
        m_word_counts_by_z[k] += d_state->m_doc_length;
        doc_id = d_state->m_doc_id;
        m_word_counts_by_zd[k][doc_id] += d_state->m_doc_length;

        d_state->m_num_tables = 1;
        d_state->m_table_to_topic.resize(2, -1);
        d_state->m_word_counts_by_t.resize(2, 0);
        d_state->m_table_to_topic[0] = k;
        d_state->m_word_counts_by_t[0] = d_state->m_doc_length;

        for (i = 0; i < d_state->m_doc_length; i++)
        {
            w = d_state->m_words[i].m_word_index;
            m_word_counts_by_zw[k][w] ++;
            d_state->m_words[i].m_table_assignment = 0;
        }
    }
    gsl_permutation_free(p);
    delete [] q; delete [] v;

}

void hdp_state::iterate_gibbs_state(bool remove, bool permute,
                                    hdp_hyperparameter * hdp_hyperparam,
                                    bool table_sampling)
{
    if (permute) // permuate data
    {
        rshuffle(m_doc_states, m_num_docs, sizeof(doc_state*));
        for (int j = 0; j < m_num_docs; j++)
            rshuffle(m_doc_states[j]->m_words, m_doc_states[j]->m_doc_length, sizeof(word_info));
    }

    double_vec q;
    double_vec f;
    doc_state* d_state = NULL;
    for (int j = 0; j < m_num_docs; j++)
    {
        d_state = m_doc_states[j];
        for (int i = 0; i < d_state->m_doc_length; i++)
        {
            sample_word_assignment(d_state, i, remove, q, f);
        }
        if (table_sampling) sample_tables(d_state, q, f);

    }
    compact_hdp_state();

    //if (!state_check_sum()) exit(0);

    /// sampling hyperparameters, including first and second levels
    if (hdp_hyperparam->m_sample_hyperparameter)
    {
        sample_first_level_concentration(hdp_hyperparam);
        sample_second_level_concentration(hdp_hyperparam);
    }
}

void hdp_state::compact_doc_state(doc_state* d_state, int* k_to_new_k)
{
    int num_tables_old = d_state->m_num_tables;
    int* t_to_new_t = new int[num_tables_old];

    int t, new_t, k, w;
    for (t = 0, new_t = 0; t < num_tables_old; t++)
    {
        if (d_state->m_word_counts_by_t[t] > 0)
        {
            t_to_new_t[t] = new_t;
            k = d_state->m_table_to_topic[t];
            d_state->m_table_to_topic[new_t] = k_to_new_k[k];
            swap_vec_element(d_state->m_word_counts_by_t, new_t, t);
            new_t ++;
        }
        else
            d_state->m_table_to_topic[t] = -1;
    }
    d_state->m_num_tables = new_t;

    d_state->m_word_stats_by_t.clear();
    d_state->m_word_stats_by_t.resize(d_state->m_num_tables, word_stats());

    for (int i = 0; i < d_state->m_doc_length; i++)
    {
        t = d_state->m_words[i].m_table_assignment;
        new_t =  t_to_new_t[t];
        d_state->m_words[i].m_table_assignment = new_t;
        w = d_state->m_words[i].m_word_index;
        d_state->m_word_stats_by_t[new_t][w] ++;
    }
    /*
         for (t = 0; t < d_state->m_num_tables; t ++)
         {
             int sum = 0;
             word_stats::iterator it;
             for (it = d_state->m_word_stats_by_t[t].begin(); it != d_state->m_word_stats_by_t[t].end(); it++)
             {
                 sum += (*it).second;
             }
             assert(sum == d_state->m_word_counts_by_t[t]);
         }
    */
    delete [] t_to_new_t;
}

//compress the unused tables and components
void hdp_state::compact_hdp_state()
{
    int num_topics_old = m_num_topics;
    int* k_to_new_k = new int[num_topics_old];
    int k, new_k;
    for (k = 0, new_k = 0; k < num_topics_old; k++)
    {
        if (m_word_counts_by_z[k] > 0)
        {
            k_to_new_k[k] = new_k;
            swap_vec_element(m_word_counts_by_z,  new_k, k);
            swap_vec_element(m_num_tables_by_z,   new_k, k);
            swap_vec_element(m_word_counts_by_zd, new_k, k);
            swap_vec_element(m_word_counts_by_zw, new_k, k);
            new_k ++;
        }
    }
    m_num_topics = new_k;

    doc_state* d_state = NULL;
    for (int j = 0; j < m_num_docs; j++)
    {
        d_state = m_doc_states[j];
        compact_doc_state(d_state, k_to_new_k);
    }

    delete [] k_to_new_k;
}

void hdp_state::sample_tables(doc_state* d_state, double_vec & q, double_vec & f)
{
    vector <int*> words_by_t;
    words_by_t.resize(d_state->m_num_tables, NULL);
    int* p = NULL;
    int t, word;
    for (t = 0; t < d_state->m_num_tables; t++)
    {
        if (d_state->m_word_counts_by_t[t] > 0)
        {
            p = new int[d_state->m_word_counts_by_t[t]];
            words_by_t[t] = p;
        }
    }

    int* i_by_t = new int[d_state->m_num_tables];
    memset(i_by_t, 0, sizeof(int) * d_state->m_num_tables);

    for (int i = 0; i < d_state->m_doc_length; i++)
    {
        word = d_state->m_words[i].m_word_index;
        t    = d_state->m_words[i].m_table_assignment;
        words_by_t[t][i_by_t[t]] =  i; // save the id for later use
        i_by_t[t]++;
    }

    for (t = 0; t < d_state->m_num_tables; t ++)
    {
        if (d_state->m_word_counts_by_t[t] > 0)
            sample_table_assignment(d_state, t, words_by_t[t], q, f);
        //eles no needs, since there is no data there
    }

    for (t = 0; t < d_state->m_num_tables; t++)
    {
        p = words_by_t[t];
        delete []p;
    }
    delete [] i_by_t;
}

void hdp_state::sample_table_assignment(doc_state* d_state, int t,
                                        int * words,
                                        double_vec & q, double_vec & f)
{
    //number of tables won't change at all
    int i, w, k, m, k_old, d;
    int* counts = new int[m_size_vocab];
    int* counts_copy = new int[m_size_vocab];
    memset(counts_copy, 0, sizeof(int)*m_size_vocab);
    int count_sum = d_state->m_word_counts_by_t[t];

    for (m = 0; m < d_state->m_word_counts_by_t[t]; m ++)
    {
        i = words[m];
        w = d_state->m_words[i].m_word_index;
        counts_copy[w] ++;
    }
    memcpy(counts, counts_copy, sizeof(int)*m_size_vocab);

    // compute the the log prob of being at a new cluster
    double f_new = lgamma(m_size_vocab*m_eta) - lgamma(count_sum + m_size_vocab*m_eta);

    for (m = 0; m < d_state->m_word_counts_by_t[t]; m ++)
    {
        i = words[m];
        w = d_state->m_words[i].m_word_index;
        if (counts[w] > 0)
        {
            f_new += lgamma(counts[w]+m_eta) - lgamma(m_eta);
            counts[w] = 0;
        }
    }

    if ((int)q.size() < m_num_topics + 1)
        q.resize(2 * m_num_topics+1, 0.0);

    if ((int)f.size() < m_num_topics)
        f.resize(2 * m_num_topics+1, 0.0);

    q[m_num_topics] = log(m_gamma) + f_new;

    k_old = d_state->m_table_to_topic[t];

    for (k = 0; k < m_num_topics; k ++)
    {
        if (k == k_old)
        {
            f[k] = lgamma(m_size_vocab*m_eta + m_word_counts_by_z[k] - count_sum) -
                   lgamma(m_size_vocab*m_eta+m_word_counts_by_z[k]);

            memcpy(counts, counts_copy, sizeof(int)*m_size_vocab);
            for (m = 0; m < d_state->m_word_counts_by_t[t]; m ++)
            {
                i = words[m];
                w = d_state->m_words[i].m_word_index;
                if (counts[w] > 0)
                {
                    f[k] += lgamma(m_eta + m_word_counts_by_zw[k][w]) -
                            lgamma(m_eta + m_word_counts_by_zw[k][w] - counts[w]);
                    counts[w] = 0;
                }
            }
            if (m_num_tables_by_z[k] == 1) q[k] = INF; // make it extremely small as log(0)
            else q[k] = log(m_num_tables_by_z[k]-1) + f[k];
        }
        else
        {
            f[k] = lgamma(m_size_vocab*m_eta + m_word_counts_by_z[k]) -
                   lgamma(m_size_vocab*m_eta+m_word_counts_by_z[k] + count_sum);

            memcpy(counts, counts_copy, sizeof(int)*m_size_vocab);
            for (m = 0; m < d_state->m_word_counts_by_t[t]; m ++)
            {
                i = words[m];
                w = d_state->m_words[i].m_word_index;
                if (counts[w] > 0)
                {
                    f[k] += lgamma(m_eta + m_word_counts_by_zw[k][w]+counts[w]) -
                            lgamma(m_eta + m_word_counts_by_zw[k][w]);
                    counts[w] = 0;
                }
            }
            q[k] = log(m_num_tables_by_z[k]) + f[k];
        }
    }
    //normalizing in log space for sampling
    log_normalize(q, m_num_topics+1);
    q[0] = exp(q[0]);
    double total_q = q[0];
    for (k = 1; k < m_num_topics+1; k++)
    {
        total_q += exp(q[k]);
        q[k] = total_q;
    }

    double u = runiform() * total_q;
    for (k = 0; k < m_num_topics+1; k ++)
        if (u < q[k]) break;

    if (k != k_old) // status doesn't change, but k could change
    {
        d = d_state->m_doc_id;

        /// reassign the topic to current table
        d_state->m_table_to_topic[t] = k;

        /// update the statistics by removing the table t from topic k_old
        m_num_tables_by_z[k_old] --;
        m_word_counts_by_z[k_old]     -= count_sum;
        m_word_counts_by_zd[k_old][d] -= count_sum;

        /// update the statistics by adding the table t to topic k
        m_num_tables_by_z[k] ++;
        m_word_counts_by_z[k]     += count_sum;
        m_word_counts_by_zd[k][d] += count_sum;

        for (int m = 0; m < d_state->m_word_counts_by_t[t]; m ++)
        {
            i = words[m];
            w = d_state->m_words[i].m_word_index;
            m_word_counts_by_zw[k_old][w] --;
            m_word_counts_by_zw[k][w] ++;
        }
        if (k == m_num_topics) // a new topic is created
        {
            m_num_topics ++; // create a new topic
            if ((int)m_num_tables_by_z.size() < m_num_topics+1)
            {
                m_num_tables_by_z.push_back(0);
                m_word_counts_by_z.push_back(0);

                int* p = new int [m_num_docs];
                memset(p, 0, sizeof(int)*m_num_docs);
                m_word_counts_by_zd.push_back(p);

                p = new int [m_size_vocab];
                memset(p, 0, sizeof(int)*m_size_vocab);
                m_word_counts_by_zw.push_back(p);
            }
        }
    }
    delete [] counts;
    delete [] counts_copy;
}

void hdp_state::sample_word_assignment(doc_state* d_state, int i, bool remove, double_vec & q, double_vec & f)
{
    if (remove) doc_state_update(d_state, i, -1);

    if ((int)q.size() < d_state->m_num_tables + 1)
        q.resize(2 * d_state->m_num_tables+1, 0.0);

    if ((int)f.size() < m_num_topics)
        f.resize(2 * m_num_topics+1, 0.0);

    int k, t, w;
    w = d_state->m_words[i].m_word_index;
    double f_new = m_gamma/m_size_vocab;
    for (k = 0; k < m_num_topics; k++)
    {
        f[k] = (m_word_counts_by_zw[k][w] + m_eta)/(m_word_counts_by_z[k] + m_size_vocab * m_eta);
        f_new += m_num_tables_by_z[k] * f[k];
    }
    f_new = f_new/(m_total_num_tables + m_gamma);

    double total_q = 0.0, f_k = 0.0;
    for (t = 0; t < d_state->m_num_tables; t++)
    {
        if (d_state->m_word_counts_by_t[t] > 0)
        {
            k = d_state->m_table_to_topic[t];
            f_k = f[k];
        }
        else f_k = 0.0;

        total_q += d_state->m_word_counts_by_t[t] * f_k;
        q[t] = total_q;
    }
    total_q += m_alpha * f_new;
    q[d_state->m_num_tables] = total_q;

    double u = runiform() * total_q;
    for (t = 0; t < d_state->m_num_tables+1; t++)
        if (u < q[t]) break;

    d_state->m_words[i].m_table_assignment = t; // assign the new table

    if (t == d_state->m_num_tables) // this is a new table, we need get its k
    {
        if ((int)q.size() < m_num_topics + 1)
            q.resize(2 * m_num_topics+1, 0.0);

        total_q = 0.0;
        for (k = 0; k < m_num_topics; k ++)
        {
            total_q += m_num_tables_by_z[k] * f[k];
            q[k] = total_q;
        }
        total_q += m_gamma/m_size_vocab;
        q[m_num_topics] = total_q;
        u = runiform() * total_q;
        for (k = 0; k < m_num_topics+1; k++)
            if (u < q[k]) break;
        doc_state_update(d_state, i, +1, k);
    }
    else
    {
        doc_state_update(d_state, i, +1);
    }
}

// k is only provided when m_table_to_topic doesn't have that
void hdp_state::doc_state_update(doc_state* d_state, int i, int update, int k)
{
    int d, w, t;
    d = d_state->m_doc_id;
    w = d_state->m_words[i].m_word_index;
    //k = d_state->m_words[i].m_topic_assignment;
    t = d_state->m_words[i].m_table_assignment;
    if (k < 0) k = d_state->m_table_to_topic[t];
    assert(k >= 0);


    d_state->m_word_counts_by_t[t] += update;

    m_word_counts_by_z[k]          += update;
    m_word_counts_by_zw[k][w]      += update;
    m_word_counts_by_zd[k][d]      += update;

    if (update == -1 && d_state->m_word_counts_by_t[t] == 0) /// this table becomes empty
    {
        m_total_num_tables --;
        m_num_tables_by_z[k] --;
        d_state->m_table_to_topic[t] = -1;
        /// m_num_topics, no need to change at this moment
    }

    if (update == 1 && d_state->m_word_counts_by_t[t] == 1) /// a new table is created
    {
        if (t == d_state->m_num_tables) d_state->m_num_tables ++; // create a new table
        d_state->m_table_to_topic[t] = k; // mapping the table

        m_num_tables_by_z[k] ++;          // adding the table to mixture k
        m_total_num_tables ++;

        if ((int)d_state->m_table_to_topic.size() < d_state->m_num_tables+1)
        {
            d_state->m_table_to_topic.push_back(-1);
            d_state->m_word_counts_by_t.push_back(0);
        }
        if (k == m_num_topics) // used to k == m_num_topics
        {
            assert(m_word_counts_by_z[k] == 1);
            if (k == m_num_topics) m_num_topics ++; // create a new topic
            if ((int)m_num_tables_by_z.size() < m_num_topics+1)
            {
                m_num_tables_by_z.push_back(0);
                m_word_counts_by_z.push_back(0);

                int* p = new int [m_num_docs];
                memset(p, 0, sizeof(int)*m_num_docs);
                m_word_counts_by_zd.push_back(p);

                p = new int [m_size_vocab];
                memset(p, 0, sizeof(int)*m_size_vocab);
                m_word_counts_by_zw.push_back(p);
            }
        }
    }
}

double hdp_state::doc_partition_likelihood(doc_state* d_state)
{
    double likelihood = d_state->m_num_tables * log(m_alpha)
                      - log_factorial(d_state->m_doc_length, m_alpha);
    /// use n! = Gamma(n+1), that is log(n!) = lgamma(n+1)
    for (int t = 0; t < d_state->m_num_tables; t++)
    {
        likelihood += lgamma(d_state->m_word_counts_by_t[t]);
    }
    return likelihood;
}

double hdp_state::table_partition_likelihood()
{
    double likelihood = m_num_topics * log(m_gamma)
                      - log_factorial(m_total_num_tables, m_gamma);
    /// use n! = Gamma(n+1), that is log(n!) = lgamma(n+1)
    for (int k = 0; k < m_num_topics; k++)
    {
        likelihood += lgamma(m_num_tables_by_z[k]);
    }
    return likelihood;
}

double hdp_state::data_likelihood()
{
    double likelihood = m_num_topics * lgamma(m_size_vocab * m_eta);
    double lgamma_eta = lgamma(m_eta);

    for (int k = 0; k < m_num_topics; k++)
    {
        likelihood -= lgamma(m_size_vocab * m_eta + m_word_counts_by_z[k]);
        for (int w = 0; w < m_size_vocab; w++)
        {
            if (m_word_counts_by_zw[k][w]>0)
            {
                likelihood += lgamma(m_word_counts_by_zw[k][w] + m_eta) - lgamma_eta;
            }
        }
    }
    return likelihood;
}

double hdp_state::joint_likelihood(hdp_hyperparameter * hdp_hyperparam)
{
    double likelihood = 0.0;
    for (int d = 0; d < m_num_docs; d++)
    {
        likelihood += doc_partition_likelihood(m_doc_states[d]);
    }
    likelihood += table_partition_likelihood();
    likelihood += data_likelihood();

    if (hdp_hyperparam->m_sample_hyperparameter) // counting the likelihood for gamma and alpha
    {
        double shape = hdp_hyperparam->m_gamma_a;
        double scale = hdp_hyperparam->m_gamma_b;
        likelihood += (shape-1)*log(m_gamma) - m_gamma/scale - shape*log(scale) - lgamma(shape);

        shape = hdp_hyperparam->m_alpha_a;
        scale = hdp_hyperparam->m_alpha_b;
        likelihood += (shape-1)*log(m_alpha) - m_alpha/scale - shape*log(scale) - lgamma(shape);
    }

    return likelihood;
}

void  hdp_state::save_state(char * name)
{
    char filename[500];

    // save the topic words counts
    sprintf(filename, "%s-topics.dat", name);
    FILE* file = fopen(filename, "w");

    for (int k = 0; k < m_num_topics; k ++)
    {
        for (int w = 0; w < m_size_vocab; w ++)
            fprintf(file, "%05d ", m_word_counts_by_zw[k][w]);
        fprintf(file, "\n");
    }
    fclose(file);

    sprintf(filename, "%s-word-assignments.dat", name);
    file = fopen(filename, "w");
    fprintf(file, "d w z t\n");
    int w, k, t;
    for (int d = 0; d < m_num_docs; d++)
    {
        doc_state* d_state = m_doc_states[d];
        int doc_id = d_state->m_doc_id;
        for (int i = 0; i < d_state->m_doc_length; i++)
        {
            w = d_state->m_words[i].m_word_index;
            t = d_state->m_words[i].m_table_assignment;
            k = d_state->m_table_to_topic[t];
            fprintf(file, "%d %d %d %d\n",
                    doc_id, w, k, t);
        }
    }
    fclose(file);
}

void hdp_state::save_state_ex(char * name)
{
    FILE * file = fopen(name, "wb");
    fwrite(&m_size_vocab, sizeof(int), 1, file);
    fwrite(&m_total_words, sizeof(int), 1, file);
    fwrite(&m_num_topics, sizeof(int), 1, file);
    fwrite(&m_total_num_tables, sizeof(int), 1, file);

    fwrite(&m_eta, sizeof(double), 1, file);
    fwrite(&m_gamma, sizeof(double), 1, file);
    fwrite(&m_alpha, sizeof(double), 1, file);
    
    for(int k = 0; k < m_num_topics; k ++)
    {
        fwrite(&(m_num_tables_by_z[k]), sizeof(int), 1, file);
        fwrite(&(m_word_counts_by_z[k]), sizeof(int), 1, file);
        fwrite(m_word_counts_by_zw[k], sizeof(int), m_size_vocab, file);
    }
    fclose(file);
}

void hdp_state::load_state_ex(char * name)
{
    FILE * file = fopen(name, "rb");
    fread(&m_size_vocab, sizeof(int), 1, file);
    fread(&m_total_words, sizeof(int), 1, file);
    fread(&m_num_topics, sizeof(int), 1, file);
    fread(&m_total_num_tables, sizeof(int), 1, file);

    fread(&m_eta, sizeof(double), 1, file);
    fread(&m_gamma, sizeof(double), 1, file);
    fread(&m_alpha, sizeof(double), 1, file);

    m_num_tables_by_z.resize(m_num_topics);
    m_word_counts_by_z.resize(m_num_topics);
    m_word_counts_by_zw.resize(m_num_topics);
    for(int k = 0; k < m_num_topics; k ++)
    {
        fread(&(m_num_tables_by_z[k]), sizeof(int), 1, file);
        fread(&(m_word_counts_by_z[k]), sizeof(int), 1, file);

        m_word_counts_by_zw[k] = new int [m_size_vocab];
        fread(m_word_counts_by_zw[k], sizeof(int), m_size_vocab, file);
    }
    fclose(file);
}

bool hdp_state::state_check_sum()
{
    bool status_OK = true;
    int sum = 0;
    for (int k = 0; k < m_num_topics; k ++)
    {
        sum += m_word_counts_by_z[k];
    }
    if (sum != m_total_words)
    {
        printf("\ntotal words does not match\n");
        status_OK = false;
    }
    for (int k = 0; k < m_num_topics; k ++)
    {
        sum = 0;
        for (int d = 0; d < m_num_docs; d ++)
        {
            sum += m_word_counts_by_zd[k][d];
        }
        if (sum != m_word_counts_by_z[k])
        {
            printf("\nin topic %d, total words does not match\n", k);
            status_OK = false;
        }
        sum = 0;
        for (int w = 0; w < m_size_vocab; w ++)
        {
            sum += m_word_counts_by_zw[k][w];
        }
        if (sum != m_word_counts_by_z[k])
        {
            printf("\nin topic %d, total words does not match\n", k);
            status_OK = false;
        }
    }

    for (int d = 0; d < m_num_docs; d++)
    {
        sum = 0;
        doc_state* d_state = m_doc_states[d];
        for (int t = 0; t < d_state->m_num_tables; t++)
        {
            sum += d_state->m_word_counts_by_t[t];
        }
        if (sum != d_state->m_doc_length)
        {
            printf("\nin doc %d, total words does not match\n", d);
            status_OK = false;
        }
    }
    return status_OK;
}

void hdp_state::sample_first_level_concentration(hdp_hyperparameter* hdp_hyperparam)
{
    /// (p 585 in escobar and west)
    double shape = hdp_hyperparam->m_gamma_a;
    double scale = hdp_hyperparam->m_gamma_b;
    int n = m_total_num_tables;
    int k = m_num_topics;

    double eta = rbeta(m_gamma + 1, n);
    double pi = shape + k - 1;
    double rate = 1.0 / scale - log(eta);
    pi = pi / (pi + rate * n);

    unsigned int cc = rbernoulli(pi);
    if (cc == 1)
        m_gamma = rgamma(shape + k, 1.0 / rate);
    else
        m_gamma = rgamma(shape + k - 1, 1.0 / rate);

    if (VERBOSE) printf("gamma=%f, ", m_gamma);

}
void hdp_state::sample_second_level_concentration(hdp_hyperparameter* hdp_hyperparam)
{
    double  shape = hdp_hyperparam->m_alpha_a;
    double  scale = hdp_hyperparam->m_alpha_b;

    int n = m_total_num_tables;
    double rate, sum_log_w, sum_s;

    for (int step = 0; step < SMALL_GIBBS_MAX_ITER; step++)
    {
        sum_log_w = 0.0;
        sum_s = 0.0;
        for (int d = 0; d < m_num_docs; d++)
        {
            sum_log_w += log(rbeta(m_alpha + 1, m_doc_states[d]->m_doc_length));
            sum_s += (double)rbernoulli(m_doc_states[d]->m_doc_length / (m_doc_states[d]->m_doc_length + m_alpha));
        }
        rate = 1.0 / scale - sum_log_w;
        m_alpha = rgamma(shape + n - sum_s, 1.0 / rate);
    }
    if (VERBOSE) printf("alpha=%f, ", m_alpha);
}

void hdp_state::copy_state(const hdp_state* state)
{
    m_eta   = state->m_eta;
    m_gamma = state->m_gamma;
    m_alpha = state->m_alpha;
    m_size_vocab  = state->m_size_vocab;
    m_total_words = state->m_total_words;
    m_num_docs    = state->m_num_docs;

    m_num_topics       = state->m_num_topics;
    m_total_num_tables = state->m_total_num_tables;

    m_num_tables_by_z  = state->m_num_tables_by_z;
    m_word_counts_by_z = state->m_word_counts_by_z;

    int size = state->m_word_counts_by_zd.size();
    m_word_counts_by_zd.resize(size, NULL);
    m_word_counts_by_zw.resize(size, NULL);
    for (int k = 0; k < size; k ++)
    {
        m_word_counts_by_zd[k] = new int [m_num_docs];
        memcpy(m_word_counts_by_zd[k], state->m_word_counts_by_zd[k], sizeof(int)*m_num_docs);
        m_word_counts_by_zw[k] = new int [m_size_vocab];
        memcpy(m_word_counts_by_zw[k], state->m_word_counts_by_zw[k], sizeof(int)*m_size_vocab);
    }

    m_doc_states = new doc_state* [m_num_docs];
    for (int d = 0; d < m_num_docs; d++)
    {
        doc_state* src_d_state = state->m_doc_states[d];
        doc_state* d_state = new doc_state();
        m_doc_states[d] = d_state;

        /// copy doc state
        d_state->m_doc_id     = src_d_state->m_doc_id;
        d_state->m_doc_length = src_d_state->m_doc_length;
        d_state->m_num_tables = src_d_state->m_num_tables;

        d_state->m_table_to_topic = src_d_state->m_table_to_topic;
        d_state->m_word_counts_by_t = src_d_state->m_word_counts_by_t;

        /// copy table statistics
        size = src_d_state->m_word_stats_by_t.size();
        d_state->m_word_stats_by_t.resize(size, word_stats());

        for (int t = 0; t < size; t ++)
            d_state->m_word_stats_by_t[t] = src_d_state->m_word_stats_by_t[t];

        d_state->m_words = new word_info [d_state->m_doc_length];
        memcpy(d_state->m_words, src_d_state->m_words, sizeof(word_info)*d_state->m_doc_length);
    }
}

ACTION hdp_state::select_mcmc_move(int& d0, int& d1, int& t0, int& t1)
{
 /*   bool seed_from_same_doc = (runiform() < 0.5);
    if (seed_from_same_doc)
    {
        do
        {
            d0 = runiform_int(m_num_docs);
        } while (m_doc_states[d0]->m_num_tables <= 1);


        t0 = runiform_int(m_doc_states[d0]->m_num_tables);
        t1 = runiform_int(m_doc_states[d0]->m_num_tables-1);
        if (t1 == t0) t1 = m_doc_states[d0]->m_num_tables-1;
        d1 = d0;
    }
    else
    {
        d0 = runiform_int(m_num_docs);
        d1 = runiform_int(m_num_docs-1);
        if (d1 == d0) d1 = m_num_docs-1;
        int i0 = runiform_int(m_doc_states[d0]->m_doc_length);
        int i1 = runiform_int(m_doc_states[d1]->m_doc_length);
        t0 = m_doc_states[d0]->m_words[i0].m_table_assignment;
        t1 = m_doc_states[d1]->m_words[i1].m_table_assignment;
    }
*/
 /*   if (seed_from_same_doc)
    {
        d0 = runiform_int(m_num_docs);
        d1 = d0;
        int i0 = runiform_int(m_doc_states[d0]->m_doc_length);
        int i1 = runiform_int(m_doc_states[d0]->m_doc_length-1);
        if (i1 == i0) i1 = m_doc_states[d0]->m_doc_length-1;
        t0 = m_doc_states[d0]->m_words[i0].m_table_assignment;
        t1 = m_doc_states[d1]->m_words[i1].m_table_assignment;
    }

    else
    {
        d0 = runiform_int(m_num_docs);
        d1 = runiform_int(m_num_docs-1);
        if (d1 == d0) d1 = m_num_docs-1;
        t0 = runiform_int(m_doc_states[d0]->m_num_tables);
        t1 = runiform_int(m_doc_states[d1]->m_num_tables);
    }


    int j0 = runiform_int(m_total_num_tables);
    int j1 = runiform_int(m_total_num_tables-1);
    */
    double* p = new double[m_total_num_tables];
    int j = 0;
    double tot_p = 0.0;
    for (int d = 0; d < m_num_docs; d++)
    {
        doc_state* d_state = m_doc_states[d];
        for (int t = 0; t < d_state->m_num_tables; t++)
        {
            p[j] = (double)d_state->m_word_counts_by_t[t];
            tot_p += p[j];
            j ++;
        }
    }
    int j0 = rmultinomial(p, m_total_num_tables, tot_p);
    tot_p -= p[j0];
    p[j0] = p[m_total_num_tables-1];
    int j1 = rmultinomial(p, m_total_num_tables-1, tot_p);
    if (j1 == j0) j1 = m_total_num_tables-1;

    int n = 0, old_n;
    bool do0 = true, do1 = true;
    for (int d = 0; d < m_num_docs; d ++)
    {
        old_n = n;
        n += m_doc_states[d]->m_num_tables;
        if (do0 && j0 < n)
        {
            d0 = d;
            t0 = j0 - old_n;
            do0 = false;
        }
        if (do1 && j1 < n)
        {
            d1 = d;
            t1 = j1 - old_n;
            do1 = false;
        }
        if (!do0 && !do1) break;
    }



    int k0 = m_doc_states[d0]->m_table_to_topic[t0];
    int k1 = m_doc_states[d1]->m_table_to_topic[t1];

    assert(k0 >= 0 && k1 >= 0);

    delete [] p;

    if (k1 == k0)
        return SPLIT;
    else
        return MERGE;
}

void hdp_state::doc_table_state_update(doc_state* d_state, int t, int update, int k)
{
    if (k < 0)  k = d_state->m_table_to_topic[t];
    else        d_state->m_table_to_topic[t] = k;

    assert(k >= 0);

    m_num_tables_by_z[k] += update;
    m_word_counts_by_z[k] += update * d_state->m_word_counts_by_t[t];

    int w, c, d = d_state->m_doc_id;

    word_stats::iterator iter = d_state->m_word_stats_by_t[t].begin();
    for (; iter != d_state->m_word_stats_by_t[t].end(); iter ++)
    {
        w = (*iter).first;
        c = (*iter).second;
        m_word_counts_by_zw[k][w] += update * c;
        m_word_counts_by_zd[k][d] += update * c;
    }

    if (update == -1) d_state->m_table_to_topic[t] = -1;

    if (update == 1 && k == m_num_topics)
    {
        m_num_topics ++;
        if ((int)m_num_tables_by_z.size() < m_num_topics+1)
        {
            m_num_tables_by_z.push_back(0);
            m_word_counts_by_z.push_back(0);

            int* p = new int [m_num_docs];
            memset(p, 0, sizeof(int)*m_num_docs);
            m_word_counts_by_zd.push_back(p);

            p = new int [m_size_vocab];
            memset(p, 0, sizeof(int)*m_size_vocab);
            m_word_counts_by_zw.push_back(p);
        }
    }
}

double hdp_state::sample_table_assignment_sm(doc_state* d_state, int t, bool remove,
        int k0, int k1, int target_k)
{
    // since we are deading with two tables, situations are much easier.
    if (remove) doc_table_state_update(d_state, t, -1);

    double p0 = log(m_num_tables_by_z[k0]);
    double p1 = log(m_num_tables_by_z[k1]);

    double v_eta = m_size_vocab * m_eta;
    int    count = d_state->m_word_counts_by_t[t];

    p0 += lgamma(v_eta + m_word_counts_by_z[k0]) - lgamma(v_eta + m_word_counts_by_z[k0] + count);
    p1 += lgamma(v_eta + m_word_counts_by_z[k1]) - lgamma(v_eta + m_word_counts_by_z[k1] + count);

    int w, c;
    word_stats::iterator iter = d_state->m_word_stats_by_t[t].begin();
    for (; iter != d_state->m_word_stats_by_t[t].end(); iter ++)
    {
        w = (*iter).first; c = (*iter).second;
        //assert(c > 0);
        p0 += lgamma(m_eta + m_word_counts_by_zw[k0][w] + c)
            - lgamma(m_eta + m_word_counts_by_zw[k0][w]);

        p1 += lgamma(m_eta + m_word_counts_by_zw[k1][w] + c)
            - lgamma(m_eta + m_word_counts_by_zw[k1][w]);
    }

    double p = log_sum(p0, p1);
    p0 -= p;
    p1 -= p;

    double prob = 0;
    int k;
    if (target_k == k0)
    {
        prob = p0;
        k = k0;
    }
    else if (target_k == k1)
    {
        prob = p1;
        k = k1;
    }
    else
    {
        double u = log(runiform());
        if (u < p0)
        {
            prob = p0;
            k = k0;
        }
        else
        {
            prob = p1;
            k = k1;
        }
    }
    doc_table_state_update(d_state, t, +1, k);
    return prob;
}

void hdp_state::merge_two_topics(int k0, int k1)
{
    /// merge two topics into one, k0, k1 --> k0
    assert(k0 != k1); // make sure they are not the same topic

    m_num_tables_by_z[k0] += m_num_tables_by_z[k1];
    m_num_tables_by_z[k1] = 0;
    m_word_counts_by_z[k0] += m_word_counts_by_z[k1];
    m_word_counts_by_z[k1] = 0;

    for (int w = 0; w < m_size_vocab; w ++)
    {
        m_word_counts_by_zw[k0][w] += m_word_counts_by_zw[k1][w];
        m_word_counts_by_zw[k1][w] = 0;
    }

    for (int d = 0; d < m_num_docs; d ++)
    {
        m_word_counts_by_zd[k0][d] += m_word_counts_by_zd[k1][d];
        m_word_counts_by_zd[k1][d] = 0;

        doc_state* d_state = m_doc_states[d];
        for (int t = 0; t < d_state->m_num_tables; t ++)
        {
            if (d_state->m_table_to_topic[t] == k1)
                d_state->m_table_to_topic[t] = k0;
        }
    }
}

double hdp_state::split_sampling(int num_scans, int d0, int d1, int t0, int t1,
                                 hdp_state * target_state)
{
    int k0 = m_doc_states[d0]->m_table_to_topic[t0];
    int k1 = m_doc_states[d1]->m_table_to_topic[t1];

    int_vec vec_docs, vec_tables;
    vec_docs.reserve(2 * m_num_docs);
    vec_tables.reserve(2 * m_num_docs);

    int d, t, i, j, target_k;
    if (target_state == NULL) // split to some state
    {
        assert(k0 == k1);
        k1 = m_num_topics;
        doc_table_state_update(m_doc_states[d1], t1, -1);     // detach table
        doc_table_state_update(m_doc_states[d1], t1, +1, k1); // attach table

        for (d = 0; d < m_num_docs; d ++)
        {
            doc_state* d_state = m_doc_states[d];
            for (t = 0; t < d_state->m_num_tables; t ++)
            {
                if (d_state->m_table_to_topic[t] == k0)
                {
                    if (d == d0 && t == t0) continue;
                    vec_docs.push_back(d);
                    vec_tables.push_back(t);
                    doc_table_state_update(d_state, t, -1);
                }
            }
        }
    }
    else   // split to target state
    {
        assert(k0 != k1);
        for (d = 0; d < m_num_docs; d ++)
        {
            doc_state* d_state = m_doc_states[d];
            for (t = 0; t < d_state->m_num_tables; t ++)
            {
                if (d_state->m_table_to_topic[t] == k0 || d_state->m_table_to_topic[t] == k1)
                {
                    if ((d == d0 && t == t0) || (d == d1 && t == t1)) continue;
                    vec_docs.push_back(d);
                    vec_tables.push_back(t);
                    doc_table_state_update(d_state, t, -1);
                }
            }
        }
    }

    size_t size = vec_docs.size();
    gsl_permutation* p = gsl_permutation_calloc(size);
    rshuffle(p->data, size, sizeof(size_t)); // shuffle the sequence

    double prob = 0.0;
    for (i = 0; i < (int)size; i ++)
    {
        j = gsl_permutation_get(p, i);
        d = vec_docs[j];
        t = vec_tables[j];
        doc_state* d_state = m_doc_states[d];

        // sequential updates
        if (target_state != NULL && num_scans == -1)
            target_k = target_state->m_doc_states[d]->m_table_to_topic[t];
        else
            target_k = -1;

        prob += sample_table_assignment_sm(d_state, t, false, k0, k1, target_k);
    }

    if (num_scans >= 0)
    {
        for (int num = 0; num < num_scans; num ++) // intermediat scans
        {
            for (i = 0; i < (int)size; i ++)
            {
                j = gsl_permutation_get(p, i);
                d = vec_docs[j];
                t = vec_tables[j];
                doc_state* d_state = m_doc_states[d];
                sample_table_assignment_sm(d_state, t, true, k0, k1, target_k);
            }
        }

        // final scan
        prob = 0.0;
        for (i = 0; i < (int)size; i ++)
        {
            j = gsl_permutation_get(p, i);
            d = vec_docs[j];
            t = vec_tables[j];
            doc_state* d_state = m_doc_states[d];
            if (target_state != NULL)
                target_k = target_state->m_doc_states[d]->m_table_to_topic[t];
            else
                target_k = -1;
            prob += sample_table_assignment_sm(d_state, t, true, k0, k1, target_k);
        }
    }

    if (p != NULL) gsl_permutation_free(p);

    return prob;
}

double compute_split_ratio(const hdp_state* split_state, const hdp_state* merge_state, int k0, int k1)
{
    double ratio = 0.0;
    double eta = split_state->m_eta;
    int size_vocab = split_state->m_size_vocab;

    double lgamma_eta = lgamma(eta);
    double v_eta = size_vocab * eta;
    double lgamma_v_eta = lgamma(v_eta);

    ratio += lgamma_v_eta - lgamma(v_eta + split_state->m_word_counts_by_z[k0]);
    ratio += lgamma_v_eta - lgamma(v_eta + split_state->m_word_counts_by_z[k1]);
    ratio -= lgamma_v_eta - lgamma(v_eta + merge_state->m_word_counts_by_z[k0]);

    for (int w = 0; w < size_vocab; w++)
    {
        if (split_state->m_word_counts_by_zw[k0][w] > 0)
            ratio += lgamma(split_state->m_word_counts_by_zw[k0][w] + eta) - lgamma_eta;

        if (split_state->m_word_counts_by_zw[k1][w] > 0)
            ratio += lgamma(split_state->m_word_counts_by_zw[k1][w] + eta) - lgamma_eta;

        if (merge_state->m_word_counts_by_zw[k0][w] > 0)
            ratio -= lgamma(merge_state->m_word_counts_by_zw[k0][w] + eta) - lgamma_eta;
    }

    ratio += log(split_state->m_gamma)
             +  lgamma(split_state->m_num_tables_by_z[k0]) + lgamma(split_state->m_num_tables_by_z[k1])
             -  lgamma(merge_state->m_num_tables_by_z[k0]);

    //printf("%d %d %d\n", split_state->m_num_tables_by_z[k0], split_state->m_num_tables_by_z[k1], merge_state->m_num_tables_by_z[k0]);

    /// all other table configurations are the same
    return ratio;
}
