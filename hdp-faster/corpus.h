// Class for reading documents in lda-c format.
//
#ifndef CORPUS_H
#define CORPUS_H

#include <stdio.h>
#include <vector>
using namespace std;

class Document {
public:
  /*  for document itself */
  int  id_;
  int* words_;
  int* counts_;
  int  length_;
  int  total_;
public:
  Document() {
    words_ = NULL;
    counts_ = NULL;
    length_ = 0;
    total_ = 0;
    id_ = -1;
  }

  Document(int len) {
    length_ = len;
    words_ = new int [len];
    counts_ = new int [len];
    total_ = 0;
    id_ = -1;
  }

  ~Document() {
    if (words_ != NULL) {
      delete [] words_;
      delete [] counts_;
      length_ = 0;
      total_ = 0;
      id_ = -1;
    }
  }
};

class Corpus {
public:
  Corpus();
  ~Corpus();
  void read_data(const char* data_filename, int OFFSET=0);
  int read_data(FILE* fileptr, int buffer_size=10000, int OFFSET=0);
  int remove_and_fetch(FILE* fileptr, int size, int OFFSET=0);
  int max_corpus_length() const;
  void free_corpus();
public:
  int num_docs_;
  int size_vocab_;
  int num_total_words_;
  vector<Document*> docs_;
};

#endif // CORPUS_H
