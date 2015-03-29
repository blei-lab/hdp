#ifndef _CORPUS_H
#define	_CORPUS_H

#include <vector>
using namespace std;

class document
{
public:
    int * words;
    int * counts;
    int length;
    int total;
    int id;
public:
    document()
    {
        words = NULL;
        counts = NULL;
        length = 0;
        total = 0;
        id = -1;
    }
    document(int len)
    {
        length = len;
        words = new int [length];
        counts = new int [length];
        total = 0;
        id = -1;
    }
    ~ document()
    {
        if (words != NULL)
        {
            delete [] words;
            delete [] counts;
            length = 0;
            total = 0;
            id = -1;
        }
    }
};

class corpus
{
public:
    corpus();
    ~corpus();
    void read_data(const char * filename);
public:
    int size_vocab;
    int total_words;
    int num_docs;
    vector<document*> docs;
};

#endif	/* _CORPUS_H */

