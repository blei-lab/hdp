#include <assert.h>
#include <algorithm>
#include "state.h"

DocState::DocState() {
  words_ = NULL;
}

DocState::~DocState() {
  if (words_ != NULL) {
    delete [] words_;
    words_ = NULL;
  }
}

void DocState::setup_state_from_doc(const Document* doc) {
  int word, count;
  doc_length_ = doc->total_;
  words_ = new WordInfo[doc_length_];
  int m = 0;
  for (int n = 0; n < doc->length_; ++n) {
    word  = doc->words_[n];
    count = doc->counts_[n];
    for (int j = 0; j < count; ++j) {
      words_[m].word_ = word;
      words_[m].count_ = 1; // If we want approximate Gibbs, we could let count_ not 1.
      words_[m].topic_assignment_ = -1;
      ++m;
    }
  }
}

HDPState::HDPState() {
}

HDPState::~HDPState() {
  vct_ptr_free(&topic_lambda_);
}

void HDPState::init_hdp_state(double eta, double gamma, double alpha, int size_vocab) {
  eta_ = eta;
  gamma_ = gamma;
  alpha_ = alpha;
  size_vocab_ = size_vocab;

  num_topics_ = 0;
  vct_ptr_resize(&topic_lambda_, INIT_SIZE, size_vocab_); 
  word_counts_by_topic_.resize(INIT_SIZE, 0);
  beta_u_.resize(INIT_SIZE, 0);
  pi_.resize(INIT_SIZE, 0.0);
  pi_left_ = 1.0;
}

void HDPState::copy_hdp_state(const HDPState& src_state) {
  eta_ = src_state.eta_;
  gamma_ = src_state.gamma_;
  alpha_ = src_state.alpha_;
  size_vocab_ = src_state.size_vocab_;

  num_topics_ = src_state.num_topics_;
  
  if (topic_lambda_.size() < src_state.topic_lambda_.size()) {
    vct_ptr_resize(&topic_lambda_, src_state.topic_lambda_.size(), size_vocab_); 
  }
  for (int i = 0; i < num_topics_; ++i) {
    memcpy(topic_lambda_[i], src_state.topic_lambda_[i], size_vocab_ * sizeof(int));
  }

  word_counts_by_topic_ = src_state.word_counts_by_topic_;
  beta_u_ = src_state.beta_u_;
  //beta_v_ = src_state.beta_v_;
  pi_ = src_state.pi_;
  pi_left_ = src_state.pi_left_;
}

void HDPState::compact_hdp_state(vct_int* k_to_new_k) {
  int old_num_topics = num_topics_;
  k_to_new_k->resize(old_num_topics, -1);
  int k, new_k;
  for (k = 0, new_k = 0; k < old_num_topics; ++k) {
    if (word_counts_by_topic_[k] > 0) {
      k_to_new_k->at(k) = new_k;
      if (k != new_k) {
        vct_swap_elements(&word_counts_by_topic_, new_k, k);
        vct_swap_elements(&beta_u_, new_k, k);
        vct_swap_elements(&topic_lambda_, new_k, k);
        vct_swap_elements(&pi_, new_k, k);
      }
      ++new_k;
    }
  }
  num_topics_ = new_k;
}

void HDPState::load_hdp_state(const char* name) {
  char filename[500];

  sprintf(filename, "%s.info", name);
  FILE* info_file = fopen(filename, "r");
  fscanf(info_file, "eta: %lf\n", &eta_);
  fscanf(info_file, "gamma: %lf\n", &gamma_);
  fscanf(info_file, "alpha: %lf\n", &alpha_);
  fscanf(info_file, "size_vocab: %d\n", &size_vocab_);
  fscanf(info_file, "num_topics: %d\n", &num_topics_);
  fclose(info_file);

  word_counts_by_topic_.resize(num_topics_ + INIT_SIZE, 0.0); 
  sprintf(filename, "%s.counts", name);
  FILE* topic_count_file = fopen(filename, "r");
  for (int k = 0; k < num_topics_; ++k) {
    fscanf(topic_count_file, "%d", &word_counts_by_topic_[k]);
  }
  fclose(topic_count_file);

  vct_ptr_resize(&topic_lambda_, num_topics_ + INIT_SIZE, size_vocab_);
  sprintf(filename, "%s.topics", name);
  FILE* topic_file = fopen(filename, "r");
  for (int k = 0; k < num_topics_; ++k) {
    for (int w = 0; w < size_vocab_; ++w) {
      fscanf(topic_file, "%d", &topic_lambda_[k][w]);
    }
  }
  fclose(topic_file);

  beta_u_.resize(num_topics_ + INIT_SIZE, 0.0);
  //beta_v_.resize(num_topics_ + INIT_SIZE, 0.0);
  sprintf(filename, "%s.beta", name);
  FILE* stick_file = fopen(filename, "r");
  for (int k = 0; k < num_topics_; ++k) {
   // fscanf(stick_file, "%lf %lf", &(beta_u_[k]), &(beta_v_[k]));
   fscanf(stick_file, "%d", &(beta_u_[k]));
  }
  fclose(stick_file);

  pi_.resize(num_topics_ + INIT_SIZE, 0.0);
  sprintf(filename, "%s.pi", name);
  FILE* pi_file = fopen(filename, "r");
  for (int k = 0; k < num_topics_; ++k) {
    fscanf(pi_file, "%lf", &pi_[k]);
  }
  fscanf(pi_file, "%lf", &pi_left_);
  fclose(pi_file);
}

void HDPState::save_hdp_state(const char* name) {
  char filename[500];

  sprintf(filename, "%s.info", name);
  FILE* info_file = fopen(filename, "w");
  fprintf(info_file, "eta: %lf\n", eta_);
  fprintf(info_file, "gamma: %lf\n", gamma_);
  fprintf(info_file, "alpha: %lf\n", alpha_);
  fprintf(info_file, "size_vocab: %d\n", size_vocab_);
  fprintf(info_file, "num_topics: %d\n", num_topics_);
  fclose(info_file);

  sprintf(filename, "%s.counts", name);
  FILE* topic_count_file = fopen(filename, "w");
  for (int k = 0; k < num_topics_; ++k) {
    fprintf(topic_count_file, "%d\n", word_counts_by_topic_[k]);
  }
  fclose(topic_count_file);

  sprintf(filename, "%s.topics", name);
  FILE* topic_file = fopen(filename, "w");
  for (int k = 0; k < num_topics_; ++k) {
    fprintf(topic_file, "%d", topic_lambda_[k][0]);
    for (int w = 1; w < size_vocab_; ++w) {
      fprintf(topic_file, " %d", topic_lambda_[k][w]);
    }
    fprintf(topic_file, "\n");
  }
  fclose(topic_file);
  
  sprintf(filename, "%s.beta", name);
  FILE* stick_file = fopen(filename, "w");
  for (int k = 0; k < num_topics_; ++k) {
    fprintf(stick_file, "%d\n", beta_u_[k]);
  }
  fclose(stick_file);

  sprintf(filename, "%s.pi", name);
  FILE* pi_file = fopen(filename, "w");
  for (int k = 0; k < num_topics_; ++k) {
    fprintf(pi_file, "%f\n", pi_[k]);
  }
  fprintf(pi_file, "%f\n", pi_left_);
  fclose(pi_file);
}
 
HDP::HDP() {
  doc_states_ = NULL;
  hdp_state_ = NULL;
}

HDP::~HDP() {
  remove_doc_states();
  if (hdp_state_ != NULL) delete hdp_state_;
  hdp_state_ = NULL;
}

void HDP::remove_doc_states() {
  if (doc_states_ != NULL) {
    for (int d = 0; d < num_docs_; ++d) {
      DocState* doc_state = doc_states_[d];
      delete doc_state;
    }
    delete [] doc_states_;
    doc_states_ = NULL;
  }
  vct_ptr_free(&word_counts_by_topic_doc_);
  vct_ptr_free(&table_counts_by_topic_doc_);

  smoothing_prob_.clear();
  vct_ptr_free(&doc_prob_);
  doc_prob_sum_.clear();
  unique_topic_by_doc_.clear();
}

void HDP::init_hdp(double eta, double gamma, double alpha, int size_vocab) {
  hdp_state_ = new HDPState();
  hdp_state_->init_hdp_state(eta, gamma, alpha, size_vocab);
}

void HDP::setup_doc_states(const vector<Document* >& docs) {
  remove_doc_states();
  num_docs_ = docs.size();
  doc_states_ = new DocState* [num_docs_];
  for (int d = 0; d < num_docs_; ++d) {
    DocState* doc_state = new DocState();
    doc_state->doc_id_ = d;
    doc_state->setup_state_from_doc(docs[d]);
    doc_states_[d] = doc_state;
  }

  vct_ptr_resize(&word_counts_by_topic_doc_, hdp_state_->word_counts_by_topic_.size(), num_docs_);
  vct_ptr_resize(&table_counts_by_topic_doc_, hdp_state_->word_counts_by_topic_.size(), num_docs_);

  init_fast_gibbs_sampling_variables();
}

int HDP::iterate_gibbs_state(bool remove, bool permute) {
  if (permute) { // Permute data.
    rshuffle(doc_states_, num_docs_, sizeof(DocState*));
    for (int j = 0; j < num_docs_; ++j)
      rshuffle(doc_states_[j]->words_, doc_states_[j]->doc_length_, sizeof(WordInfo));
  }
  sample_top_level_proportions();
  vct p;
  int total_change = 0;
  for (int j = 0; j < num_docs_; ++j) {
    DocState* doc_state = doc_states_[j];
    for (int i = 0; i < doc_state->doc_length_; ++i) {
      total_change += sample_word_assignment(doc_state, i, remove, &p); 
    }
    sample_table_counts(doc_state, &p);
    if (j % 10 == 0) {
      sample_top_level_proportions();
    }
  }
  //sample_posterior_sticks();
  return total_change;
}

int HDP::sample_word_assignment(DocState* doc_state, int i, bool remove, vct* p) {
  int old_k = -1, k;
  if (remove) { 
    old_k = doc_state->words_[i].topic_assignment_;
    doc_state_update(doc_state, i, -1);
  }
  
  if ((int)p->size() < hdp_state_->num_topics_ + 1) {
    p->resize(2 * hdp_state_->num_topics_ + 1);
  }

  int d = doc_state->doc_id_;
  int w = doc_state->words_[i].word_;

  double p_w = 0.0;
  set<int>::iterator it = unique_topic_by_word_[w].begin(); 
  int j = 0;
  for (; it != unique_topic_by_word_[w].end(); ++it, ++j) {
    k = *it;
    p->at(j) = hdp_state_->topic_lambda_[k][w] * (smoothing_prob_[k] + doc_prob_[k][d]);
    p_w += p->at(j);
    p->at(j) = p_w;
  }
  double tail_prob = hdp_state_->alpha_ * hdp_state_->pi_left_ / hdp_state_->size_vocab_;
  double total_p = p_w + (doc_prob_sum_[d] + smoothing_prob_sum_) * hdp_state_->eta_ + tail_prob;
  double u = runiform() * total_p;
  if (u < p_w) { // in the word region.
    it = unique_topic_by_word_[w].begin();
    for (j = 0; it != unique_topic_by_word_[w].end(); ++it, ++j) {
      if (u < p->at(j)) {
        k = *it;
        break;
      }
    }
  } else {
    u = u - p_w;
    if (u < tail_prob) { // in the tail region.
      k = hdp_state_->num_topics_;
    } else { 
      u = (u - tail_prob) / hdp_state_->eta_; 
      if (u < doc_prob_sum_[d]) { // In the doc region,
        it = unique_topic_by_doc_[d].begin();
        total_p = 0.0;
        for (; it != unique_topic_by_doc_[d].end(); ++it) {
          k = *it;
          total_p += doc_prob_[k][d];
          if (u < total_p) break;
        }
      } else { // In the smoothing region.
        u = u - doc_prob_sum_[d];
        total_p = 0.0;
        for (k = 0; k < hdp_state_->num_topics_; ++k) {
          total_p += smoothing_prob_[k];
          if (u < total_p) break;
        }
      }
    }
  }

  doc_state->words_[i].topic_assignment_ = k;
  doc_state_update(doc_state, i, 1);
  return int(old_k != k);
}

void HDP::doc_state_update(DocState* doc_state, int i, int update) {
  int d = doc_state->doc_id_;
  int w = doc_state->words_[i].word_;
  int c = doc_state->words_[i].count_; 
  int k = doc_state->words_[i].topic_assignment_;
  //assert(k >= 0); // we must have it assigned before or assigned to a new one.

  if (update > 0)  {
    if (hdp_state_->topic_lambda_[k][w] == 0)
      unique_topic_by_word_[w].insert(k);
    if (word_counts_by_topic_doc_[k][d] == 0)
      unique_topic_by_doc_[d].insert(k);
  }

  update *= c;
  // Update HDP state
  smoothing_prob_sum_ -= smoothing_prob_[k];
  hdp_state_->word_counts_by_topic_[k] += update;
  hdp_state_->topic_lambda_[k][w] += update;

  doc_prob_sum_[d] -= doc_prob_[k][d];
  word_counts_by_topic_doc_[k][d] += update;

  if (update < 0 ) {
    if (hdp_state_->topic_lambda_[k][w] == 0)
      unique_topic_by_word_[w].erase(k);
    if (word_counts_by_topic_doc_[k][d] == 0) 
      unique_topic_by_doc_[d].erase(k);
  }

  if (update > 0 &&  k == hdp_state_->num_topics_) { // a new topic is generated.
    hdp_state_->num_topics_ ++; 
    double new_stick = rbeta(1.0, hdp_state_->gamma_) * hdp_state_->pi_left_;
    hdp_state_->pi_left_ = hdp_state_->pi_left_ - new_stick;
    hdp_state_->pi_[k] = new_stick;

    if ((int)hdp_state_->word_counts_by_topic_.size() < hdp_state_->num_topics_ + 1) {
      int new_size = 2 * hdp_state_->num_topics_ + 1;
      vct_ptr_resize(&hdp_state_->topic_lambda_, new_size, hdp_state_->size_vocab_);
      hdp_state_->word_counts_by_topic_.resize(new_size, 0);
      hdp_state_->beta_u_.resize(new_size, 0);
      //hdp_state_->beta_v_.resize(new_size, 0.0);
      hdp_state_->pi_.resize(new_size, 0.0);
      vct_ptr_resize(&word_counts_by_topic_doc_, new_size, num_docs_);
      vct_ptr_resize(&table_counts_by_topic_doc_, new_size, num_docs_);

      smoothing_prob_.resize(new_size, 0.0);
      vct_ptr_resize(&doc_prob_, new_size, num_docs_);
    }
  }

  double etaW = hdp_state_->size_vocab_ * hdp_state_->eta_;
  smoothing_prob_[k] = hdp_state_->alpha_ * hdp_state_->pi_[k] / (hdp_state_->word_counts_by_topic_[k] + etaW);
  smoothing_prob_sum_ += smoothing_prob_[k];
  doc_prob_[k][d] = word_counts_by_topic_doc_[k][d] / (hdp_state_->word_counts_by_topic_[k] + etaW);
  doc_prob_sum_[d] += doc_prob_[k][d];
}

void HDP::sample_table_counts(DocState* doc_state, vct* p) {
  int d = doc_state->doc_id_;
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    hdp_state_->beta_u_[k] -= table_counts_by_topic_doc_[k][d];
    int n = word_counts_by_topic_doc_[k][d];
    if (n < 2) {
      table_counts_by_topic_doc_[k][d] = n; 
    } else {
      if ((int)p->size() < n) {
        p->resize(2 * n + 1);
      }
      double alpha0 = hdp_state_->alpha_ * hdp_state_->pi_[k];
      double base = lgamma(alpha0) - lgamma(alpha0 + n);
      double total_p = 0.0;
      int i;
      for (i = 0; i < n; ++i) {
        p->at(i) = base + (i + 1) * log(alpha0) + stirling_.get_log_stirling_num(n, i+1);
        p->at(i) = exp(p->at(i));
        total_p += p->at(i);
        p->at(i) = total_p;
      }
      double u = runiform() * total_p;
      for (i = 0; i < n; ++i) {
        if (u < p->at(i))
          break;
      }
      table_counts_by_topic_doc_[k][d] = i + 1;
    }
    hdp_state_->beta_u_[k] += table_counts_by_topic_doc_[k][d];
  }
}

void HDP::sample_top_level_proportions() {
  double total = 0;
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    hdp_state_->pi_[k] = rgamma(hdp_state_->beta_u_[k], 1.0);
    total += hdp_state_->pi_[k];
  }
  hdp_state_->pi_left_ = rgamma(hdp_state_->gamma_, 1.0);
  total += hdp_state_->pi_left_;

  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    hdp_state_->pi_[k] /= total;
  }
  hdp_state_->pi_left_ /= total;

  double etaW = hdp_state_->size_vocab_ * hdp_state_->eta_;
  smoothing_prob_sum_ = 0.0;
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    smoothing_prob_[k] = hdp_state_->alpha_ * hdp_state_->pi_[k] / (hdp_state_->word_counts_by_topic_[k] + etaW);    
    smoothing_prob_sum_ += smoothing_prob_[k];
  }
}
/*
void HDP::sample_prior_sticks() {
  double a, b;
  hdp_state_->pi_left_ = 1.0;
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    a = 1.0 + hdp_state_->beta_u_[k];
    b = gamma_ + hdp_state_->beta_v_[k];
    hdp_state_->pi_[k] = rbeta(a, b) * hdp_state_->pi_left_;
    hdp_state_->pi_left_ -= hdp_state_->pi_[k];
  }
  

  double etaW = size_vocab_ * eta_;
  smoothing_prob_sum_ = 0.0;
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    smoothing_prob_[k] = alpha_ * hdp_state_->pi_[k] / (hdp_state_->word_counts_by_topic_[k] + etaW);    
    smoothing_prob_sum_ += smoothing_prob_[k];
  }
}

void HDP::sample_posterior_sticks() {
  
  double cum_count = 0;
  fill(hdp_state_->beta_v_.begin(), hdp_state_->beta_v_.end(), 0.0);
  for (int k = hdp_state_->num_topics_ - 1; k >= 0; --k) {
    hdp_state_->beta_v_[k] = cum_count;
    cum_count += hdp_state_->beta_u_[k];
  }
  
  sample_prior_sticks();
}
*/

void HDP::compact_hdp_state() {
  int old_num_topics = hdp_state_->num_topics_;
  vct_int k_to_new_k;
  hdp_state_->compact_hdp_state(&k_to_new_k);

  if (old_num_topics == hdp_state_->num_topics_) return;

  int new_k, k;
  for (k = 0; k < (int)k_to_new_k.size(); ++k) {
    new_k = k_to_new_k[k];
    if (new_k >= 0) {
      vct_swap_elements(&word_counts_by_topic_doc_, new_k, k);
      vct_swap_elements(&table_counts_by_topic_doc_, new_k, k);
      vct_swap_elements(&smoothing_prob_, new_k, k);
      vct_swap_elements(&doc_prob_, new_k, k);

      if (new_k != k) {
        for (int w = 0; w < hdp_state_->size_vocab_; ++w) {
          if (unique_topic_by_word_[w].erase(k) > 0) {
            unique_topic_by_word_[w].insert(new_k);        
          }
        }
        for (int d = 0; d < num_docs_; ++d) {
          if (unique_topic_by_doc_[d].erase(k) > 0) {
            unique_topic_by_doc_[d].insert(new_k);
          }
        }
      }
    }
  }

  // compact each docs
  for (int d = 0; d < num_docs_; ++d) {
    DocState* doc_state = doc_states_[d];
    for (int i = 0; i < doc_state->doc_length_; ++i) {
      k = doc_state->words_[i].topic_assignment_;
      doc_state->words_[i].topic_assignment_ = k_to_new_k[k];
    }
  }
}

double HDP::log_likelihood(const HDPState* old_hdp_state) {
  double likelihood = 0.0;

  likelihood += num_docs_ * lgamma(hdp_state_->alpha_);
  vct lg_alpha_pi(hdp_state_->num_topics_);
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    lg_alpha_pi[k] = lgamma(hdp_state_->alpha_ * hdp_state_->pi_[k]);
  }

  for (int i = 0; i < num_docs_; ++i) {
    int d = doc_states_[i]->doc_id_;
    likelihood -= lgamma(hdp_state_->alpha_ + doc_states_[d]->doc_length_);
    for (int k = 0; k < hdp_state_->num_topics_; ++k) {
      if (word_counts_by_topic_doc_[k][d] > 0) {
        likelihood += lgamma(hdp_state_->alpha_ * hdp_state_->pi_[k] + word_counts_by_topic_doc_[k][d]);        
        likelihood -= lg_alpha_pi[k];
      }
    }
  }

  int old_num_topics = 0;
  if (old_hdp_state != NULL) {
    old_num_topics = old_hdp_state->num_topics_;
  }
  double etaW = hdp_state_->size_vocab_ * hdp_state_->eta_;
  for (int k = 0; k < old_num_topics; ++k) {
    if (hdp_state_->word_counts_by_topic_[k] > old_hdp_state->word_counts_by_topic_[k]) {
      likelihood += lgamma(old_hdp_state->word_counts_by_topic_[k] + etaW);
      likelihood -= lgamma(hdp_state_->word_counts_by_topic_[k] + etaW);
      for (int w = 0; w < hdp_state_->size_vocab_; ++w) {
        if (hdp_state_->topic_lambda_[k][w] > old_hdp_state->topic_lambda_[k][w]) {
          likelihood -= lgamma(old_hdp_state->topic_lambda_[k][w] + hdp_state_->eta_);
          likelihood += lgamma(hdp_state_->topic_lambda_[k][w] + hdp_state_->eta_);
        }
      }
    }
  }

  double lg_eta = lgamma(hdp_state_->eta_);
  double lg_etaW = lgamma(etaW);
  for (int k = old_num_topics; k < hdp_state_->num_topics_; ++k) {
    if (hdp_state_->word_counts_by_topic_[k] > 0) {
      likelihood += lg_etaW;
      likelihood -= lgamma(hdp_state_->word_counts_by_topic_[k] + etaW);
      for (int w = 0; w < hdp_state_->size_vocab_; ++w) {
        if (hdp_state_->topic_lambda_[k][w] > 0) {
          likelihood -= lg_eta;
          likelihood += lgamma(hdp_state_->topic_lambda_[k][w] + hdp_state_->eta_);
        }
      }
    }
  }
  return likelihood;
}

void HDP::save_state(const char* name) {
  hdp_state_->save_hdp_state(name);
}

void HDP::load_state(const char* name) {
  hdp_state_ = new HDPState();
  hdp_state_->load_hdp_state(name);
}

void HDP::hyper_inference(double gamma_a, double gamma_b, double alpha_a, double alpha_b) {
  sample_first_level_concentration(gamma_a, gamma_b);
  sample_second_level_concentration(alpha_a, alpha_b);
}

void HDP::init_fast_gibbs_sampling_variables() {
  unique_topic_by_word_.resize(hdp_state_->size_vocab_);
  smoothing_prob_.resize(hdp_state_->word_counts_by_topic_.size(), 0);
  vct_ptr_resize(&doc_prob_, hdp_state_->word_counts_by_topic_.size(), num_docs_);
  doc_prob_sum_.resize(num_docs_, 0.0);
  unique_topic_by_doc_.resize(num_docs_);
}

void HDP::save_doc_states(const char* name) {
  char filename[500];
  sprintf(filename, "%s.doc.states", name);
  FILE* doc_state_file = fopen(filename, "w");
  for (int d = 0; d < num_docs_; ++d) {
    fprintf(doc_state_file, "%d", word_counts_by_topic_doc_[0][d]);
    for (int k = 1; k < hdp_state_->num_topics_; ++k) {
      fprintf(doc_state_file, " %d", word_counts_by_topic_doc_[k][d]);
    }
    fprintf(doc_state_file, "\n");
  }
  fclose(doc_state_file);
}

void HDP::sample_first_level_concentration(double gamma_a, double gamma_b) {
  /// (p 585 in escobar and west)
  double shape = gamma_a;
  double scale = gamma_b;
  int n = 0;
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    n += hdp_state_->beta_u_[k];
  }

  double eta = rbeta(hdp_state_->gamma_ + 1, n);
  double pi = shape + hdp_state_->num_topics_ - 1;
  double rate = 1.0 / scale - log(eta);
  pi = pi / (pi + rate * n);

  unsigned int cc = rbernoulli(pi);
  if (cc == 1)
    hdp_state_->gamma_ = rgamma(shape + hdp_state_->num_topics_, 1.0 / rate);
  else
    hdp_state_->gamma_ = rgamma(shape + hdp_state_->num_topics_ - 1, 1.0 / rate);
}

void HDP::sample_second_level_concentration(double alpha_a, double alpha_b) {
  double  shape = alpha_a;
  double  scale = alpha_b;

  int n = 0;
  for (int k = 0; k < hdp_state_->num_topics_; ++k) {
    n += hdp_state_->beta_u_[k];
  }
  double rate, sum_log_w, sum_s;

  for (int step = 0; step < 20; ++step) {
    sum_log_w = 0.0;
    sum_s = 0.0;
    for (int d = 0; d < num_docs_; ++d) {
      sum_log_w += log(rbeta(hdp_state_->alpha_ + 1, doc_states_[d]->doc_length_));
      sum_s += (double)rbernoulli(doc_states_[d]->doc_length_ / (doc_states_[d]->doc_length_ + hdp_state_->alpha_));
    }
    rate = 1.0 / scale - sum_log_w;
    hdp_state_->alpha_ = rgamma(shape + n - sum_s, 1.0 / rate);
  }
}

