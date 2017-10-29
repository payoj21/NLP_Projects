# import nltk
# import keras
# import os, sys
from math import log

pos_rev = []
neg_rev = []
vocab = dict()
punctuations = [',','.','!','"',"'",'(',')','?','[',']','{','}']
def train_file_read():
    total_count = 0
    f = open('hotelNegT-train.txt','r',encoding='utf-8')
    neg_count = 0
    for line in f:
        line = line.strip('\n')
        content = line.split('\t')
        review = content[1]
        for punct in punctuations:
            review = review.strip(punct)
        review_split = review.split(' ')
        neg_rev.append(review_split)
        neg_count += 1
        total_count += 1
    f.close()

    f = open('hotelPosT-train.txt','r', encoding='utf-8')
    pos_count = 0
    for line in f:
        line = line.strip('\n')
        content = line.split('\t')
        review = content[1]
        for punct in punctuations:
            review = review.strip(punct)
        review_split = review.split(' ')
        pos_rev.append(review_split)
        pos_count += 1
        total_count += 0
    f.close()
    # Prior
    prob_pos = pos_count/total_count
    prob_neg = neg_count/total_count
    return neg_rev, pos_rev, prob_pos, prob_neg

def dictionaries(neg_rev, pos_rev):

    for review in neg_rev:
        review = set(review)                # BINARIZATION using document frequency instead of term frequency
        for word in review:
            if word in vocab:
                if 'Negative_count' in word:
                    vocab[word]['Negative_count'] += 1
                    vocab[word]['Total_count'] += 1
                else:
                    vocab[word].update({'Negative_count' : 1})
                    vocab[word]['Total_count'] += 1
            else:
                vocab.update({word:{'Negative_count' : 1, 'Positive_count': 0, 'Total_count' : 1}})

    for review in pos_rev:
        review = set(review)                # BINARIZATION using document frequency instead of term frequency
        for word in review:
            if word in vocab:
                if 'Positive_count' in word:
                    vocab[word]['Positive_count'] += 1
                    vocab[word]['Total_count'] += 1
                else:
                    vocab[word].update({'Positive_count' : 1})
                    vocab[word]['Total_count'] += 1
            else:
                vocab.update({word:{'Positive_count' : 1, 'Negative_count' : 0, 'Total_count' : 1}})
    vocab_size = len(vocab)

    return vocab, vocab_size

def words_in_class(vocab):
    pos_words = 0
    neg_words = 0
    for word in vocab:
        pos_words += vocab[word]['Positive_count']
        neg_words += vocab[word]['Negative_count']

    return pos_words, neg_words

def naive_bayes(review, vocab, vocab_size, prob_pos, prob_neg, pos_words, neg_words):
    log_prob_pos = 0
    log_prob_neg = 0
    review = set(review)              # BINARIZATION using document frequency instead of term frequency
    for word in review:
        if word in vocab:       #Likelihood including Laplace smoothing; also ignoring UNK_WORDS
            log_prob_pos += (log(vocab[word]['Positive_count'] + 1) - log(pos_words + vocab_size))
            log_prob_neg += (log(vocab[word]['Negative_count'] + 1) - log(neg_words + vocab_size))

    log_prob_pos = log_prob_pos + log(prob_pos)
    log_prob_neg = log_prob_neg + log(prob_neg)

    return log_prob_pos, log_prob_neg

def reviews_classification(log_prob_pos, log_prob_neg):

    if (log_prob_neg > log_prob_pos):
        return 'NEG'
    else:
        return 'POS'

def test_file_read(vocab, vocab_size,prob_pos, prob_neg, pos_words, neg_words):
    f = open('HW3-testset.txt','r',encoding='utf-8')
    result = []
    for line in f:
        line = line.strip('\n')
        content = line.split('\t')
        id = content[0]
        review = content[1]
        for punct in punctuations:
            review = review.strip(punct)
        words = review.split(' ')
        log_prob_pos, log_prob_neg = naive_bayes(words, vocab, vocab_size,prob_pos, prob_neg, pos_words, neg_words)
        sentiment = reviews_classification(log_prob_pos, log_prob_neg)

        result.append([id,sentiment])

    f.close()

    f = open('jain-payoj-assgn3-out.txt','w')

    for id in result:

        f.write(str(id[0])+'\t'+str(id[1])+'\n')

    f.close()

neg_rev, pos_rev, prob_pos, prob_neg = train_file_read()
vocab, vocab_size = dictionaries(neg_rev,pos_rev)
pos_words, neg_words = words_in_class(vocab)
test_file_read(vocab,vocab_size,prob_pos, prob_neg,pos_words, neg_words)