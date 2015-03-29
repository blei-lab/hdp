#!/usr/bin/env Rscript

print.topics <- function(words.fn, vocab.fn, topics.fn, top.n=5) 
{
   words <- as.matrix(read.table(words.fn, header=FALSE))
   vocab <- readLines(vocab.fn, warn=FALSE)
   num.topics <- nrow(words) 
   topics <- NULL 
   head <- ""
   for (k in seq(num.topics))
   {
       prob <- words[k,]
       total <- sum(prob)
       prob <- prob/total
       s <- sort.int(x=prob, decreasing=TRUE, index.return=TRUE)
       top.idx <- s$ix[1:top.n]
       topic.prob <- prob[top.idx]
       topic.words <- vocab[top.idx]
       topics <- cbind(topics, topic.words) 
       head <- paste(head, sprintf("%20d", k), sep="")
   }

   write(x=head, file=topics.fn)

   for (i in seq(top.n))    
   {
       line <- paste(sprintf("%20s", topics[i,]), collapse="")
       write(x=line, file=topics.fn, append=TRUE)
   }
   
}

args <- commandArgs(TRUE)
if (length(args)<3)
{
    cat("./print.topics word.counts.file vocab.file topics.file [top.n, 5 default]\n")
    stop("too few parameters")
}

words.fn <- args[1]
vocab.fn <- args[2]
topics.fn <- args[3]
top.n <- 5
if (length(args) >= 4)  top.n <- as.integer(args[4])

print.topics(words.fn, vocab.fn, topics.fn, top.n)

