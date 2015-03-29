#include "corpus.h"
#include <assert.h>
#include <stdio.h>

Corpus::Corpus() {
  num_docs_ = 0;
  size_vocab_ = 0;
  num_total_words_ = 0;
}

Corpus::~Corpus() {
  free_corpus();
}

void Corpus::free_corpus() {
  for (int i = 0; i < num_docs_; ++i) {
    Document* doc = docs_[i];
    delete doc;
  }
  docs_.clear();

  num_docs_ = 0;
  size_vocab_ = 0;
  num_total_words_ = 0;
}

int Corpus::read_data(FILE* fileptr, int buffer_size, int OFFSET) {
  free_corpus(); // Remove old documents.
  docs_.reserve(buffer_size);
  int length = 0, count = 0, word = 0, n = 0, nd = 0, nw = 0;
  while (nd < buffer_size) {
    if (fscanf(fileptr, "%10d", &length) == EOF) break;
    Document * doc = new Document(length);
    for (n = 0; n < length; ++n) {
      fscanf(fileptr, "%10d:%10d", &word, &count);
      word = word - OFFSET;
      doc->words_[n] = word;
      doc->counts_[n] = count;
      doc->total_ += count;
      if (word >= nw) 
        nw = word + 1;
    }
    num_total_words_ += doc->total_;
    doc->id_ = nd; 
    docs_.push_back(doc);
    nd++;
  }
  num_docs_ += nd;
  size_vocab_ = nw;
  return nd;
}

void Corpus::read_data(const char* data_filename, int OFFSET) {
  free_corpus();
  int length = 0, count = 0, word = 0, n = 0, nd = 0, nw = 0;

  FILE * fileptr;
  fileptr = fopen(data_filename, "r");

  printf("Reading data from %s.\n", data_filename);
  while ((fscanf(fileptr, "%10d", &length) != EOF)) {
    Document * doc = new Document(length);
    for (n = 0; n < length; ++n) {
      fscanf(fileptr, "%10d:%10d", &word, &count);
      word = word - OFFSET;
      doc->words_[n] = word;
      doc->counts_[n] = count;
      doc->total_ += count;
      if (word >= nw) 
        nw = word + 1;
    }
    num_total_words_ += doc->total_;
    doc->id_ = nd; 
    docs_.push_back(doc);
    nd++;
  }
  fclose(fileptr);
  num_docs_ += nd;
  size_vocab_ = nw;
  printf("number of docs  : %d\n", nd);
  printf("number of terms : %d\n", nw);
  printf("number of total words : %d\n", num_total_words_);
}

int Corpus::remove_and_fetch(FILE* fileptr, int size, int OFFSET) {
  size = (size < num_docs_) ? size : num_docs_;
  for (int i = 0; i < size; ++i) {
    Document* doc = docs_[i];
    num_total_words_ -= doc->total_;
    delete doc;
    docs_[i] = NULL;
  }
  docs_.erase(docs_.begin(), docs_.begin() + size);
  docs_.reserve(num_docs_);
  num_docs_ -= size;

  int length = 0, count = 0, word = 0, n = 0, nd = 0, nw = 0;
  while (nd < size) {
    if (fscanf(fileptr, "%10d", &length) == EOF) break;
    Document * doc = new Document(length);
    for (n = 0; n < length; ++n) {
      fscanf(fileptr, "%10d:%10d", &word, &count);
      word = word - OFFSET;
      doc->words_[n] = word;
      doc->counts_[n] = count;
      doc->total_ += count;
      if (word >= nw) 
        nw = word + 1;
    }
    num_total_words_ += doc->total_;
    docs_.push_back(doc);
    nd++;
  }
  num_docs_ += nd;

  // Relabel doc_id_
  for (int i = 0; i < num_docs_; ++i) {
    docs_[i]->id_ = i;
  }
  return nd;
}


int Corpus::max_corpus_length() const {
  int max_length = 0;

  for (int d = 0; d < num_docs_; d++) {
    if (docs_[d]->length_ > max_length)
        max_length = docs_[d]->length_;
  }
  return max_length;
}

