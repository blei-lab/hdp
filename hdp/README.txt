**********************************************************************
Hierarchical Dirichlet Process (with Split-Merge Operations)
**********************************************************************

(C) Copyright 2010, Chong Wang and David Blei

written by Chong Wang, chongw@cs.princeton.edu.

This file is part of hdp.

hdp is free software; you can redistribute it and/or modify it under the terms
of the GNU General Public License as published by the Free Software Foundation;
either version 2 of the License, or (at your option) any later version.

hdp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE.  See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program; if not, write to the Free Software Foundation, Inc., 59 Temple
Place, Suite 330, Boston, MA 02111-1307 USA


-----------------------------------------------------------------------------------------

This is a C++ implementation of hierarchical Dirichlet process for topic modeling. 

The split-merge algorithm is preliminary.

Note that this code requires the Gnu Scientific Library, http://www.gnu.org/software/gsl/

-----------------------------------------------------------------------------------------


TABLE OF CONTENTS


A. COMPILING

B. POSTERIOR INFERENCE

C. INFERENCE ON NEW DATA

D. PARAMETER SETTINGS

E. PRINTING TOPICS

-----------------------------------------------------------------------------------------


A. COMPILING

Type "make" in a shell. Make sure the GSL is installed. You may need to change
the Makefile a bit.


B. POSTERIOR INFERENCE

The following shows an example of performing posterior inference on a set of documents,

hdp --algorithm train --data data --directory train_dir


Data format

--data points to a file where each line is of the form (the LDA-C format):

     [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

where [M] is the number of unique terms in the document, and the
[count] associated with each term is how many times that term appeared
in the document. 

The sampler will produce some files in the --directory,

*-topics.dat: the word counts for each topic, with each line as a topic

*-word-assignments.dat: print each word's assignment to the topic and the table,
which is in R-friendly format,
d w z t

d: document id
w: word id
z: topic index
t: table index (only for document level. If you only analyze the topics, this is irrelevant.)

*.bin: the binary model file used for inference on new data.

state.log: various information to monitor the Markov chain.

More parameter settings, run:
hdp --help

Note: some parameters for split-merge are hand coded at the beginning of hdp.cpp
file.

-----------------------------------------------------------------------------------------

C. INFERENCE ON NEW DATA

To perform inference on a different set of data (in the same format as before), run:

hdp --algorithm test --data data --saved_model saved_model --directory test_dir 

where --saved_model is the binary file from the posterior inference on training data.
     
The sampler will produce some files in the --directory,

test-*-topics.dat: the word counts for each topic, with each line as a topic

test*-word-assignments.dat: print each word's assignment to the topic and the table,
which is in R-friendly format.

test.log: various information to monitor the Markov chain.

test-*.bin: the binary model file used for inference on newer data.

More parameter settings, run:
hdp --help

-----------------------------------------------------------------------------------------


D. PARAMETER SETTINGS

The meaning of the parameters is the same as in the in the following paper

Y. Teh, M. Jordan, M. Beal, and D. Blei. Hierarchical Dirichlet processes.
Journal of the American Statistical Association, 2006. 101[476]:1566-1581

-----------------------------------------------------------------------------------------

E. PRINTING TOPICS

A R script (print.topics.R) is included to print topics. Make sure it is
executable. (chmod +x print.topics.R) For example,

print.topics.R mode-topics.dat vocab.dat topics.dat 10

will produce a topic list with top 10 words selected. For help, run,

print.topics.R

