#!/usr/bin/env python
import math
from collections import namedtuple
from pprint import pprint
import random
from collections import Counter

random.seed(2011)

def read_and_write():

    # Reads the data from a text file and store it in a list 'list_of_sentences' in the form of list of lists
    f = open('gene-trainF17.txt', 'r')
    list_of_sentences = []
    doc = []
    for line in f:
        if(line != '\n'):
            list_of_sentences.append(line)
        else:
            doc.append(list_of_sentences)
            list_of_sentences = []
    random.shuffle(doc)
    return doc

def read_test_set():
    # Reads the data from a text file and store it in a list 'test_set' in the form of list of lists
    f = open('F17-assgn4-test.txt', 'r')
    list_of_sentences = []
    test_data = []
    count = 0
    for line in f:
        if (line != '\n'):
            line = line.strip('\n')
            content = line.split('\t')
            word = content[1]
            list_of_sentences.append(word)
            count = count+1
        else:
            test_data.append(list_of_sentences)
            list_of_sentences = []
            count = count+1
    test_data.append(list_of_sentences)
    return test_data, count


def baseline(doc):

    #Splits the list_of_sentences into train and development set data and Max Match the dev set data after
    # counting number of tags and words in the train set

    training_sentence = dict()
    tagged_dict = dict()
    tagged_dict.update({'UNK':{}})
    count = 0

    for sentence in doc:
        if (count <= int(0.9*len(doc))):
            for term in sentence:
                term = term.strip('\n')
                content = term.split('\t')
                word = content[1]
                tag = content[2]
                if word not in training_sentence:
                    training_sentence.update({word:{tag:1}})
                else:
                    if tag not in training_sentence[word]:
                        training_sentence[word].update({tag:1})
                    else:
                        training_sentence[word][tag] = training_sentence[word][tag]+1
            count=count+1

    for key in training_sentence:
        most_common_pos = Counter(training_sentence[key]).most_common()[0][0]
        tagged_dict.update({key:most_common_pos})
        if(Counter(training_sentence[key]).most_common()[0][1] == 1):
            if most_common_pos in tagged_dict['UNK']:
                tagged_dict['UNK'][most_common_pos] = tagged_dict['UNK'][most_common_pos]+1
            else:
                tagged_dict['UNK'].update({most_common_pos:1})


    fd = open('test_data_output.txt', 'w')
    fdr = open('result_data.txt','w')
    count=0

    for sentence in doc:
        if(count>int(0.9*len(doc))):
            for term in sentence:
                term = term.strip('\n')
                content = term.split('\t')
                index = content[0]
                word = content[1]
                result_tag = content[2]
                fdr.write(index+'\t'+word+'\t'+result_tag+'\n')
                if(word in tagged_dict):
                    tag = tagged_dict[word]
                    fd.write(index+'\t'+word+'\t'+tag+'\n')
                else:
                    tag = Counter(tagged_dict['UNK']).most_common()[0][0]
                    fd.write(index + '\t' + word + '\t' + tag + '\n')
            fd.write('\n')
            fdr.write('\n')
        else:
            count=count+1

def dictionaries(doc):

    # creates dictionaries for tags, words, bigram types
    # word_dictionary stores all the words seen in train set as keys and information about a particular word as values
    # Information like total count, tag types, count of word given a tag type, probabilities of starting a sentence,
    # ending a sentence etc.
    # Same for tag_dictionary with many counts and probabilities
    # bigram_types contains bigrams types seen in the train set and the count of a particular bugram type

    word_dictionary = dict()
    tag_dictionary = dict()
    bigrams_types = dict()
    dev_set = []
    result_set = []
    total_words = 0
    total_tags = 0
    training_count = 0
    fdr = open('dev_set_result_data.txt', 'w')

    for sentence in doc:
 #       if (training_count <= int(0.9 * len(doc))):
            count = 0
            for term in sentence:
                term = term.strip('\n')
                content = term.split('\t')
                word = content[1]
                tag = content[2]

#---------------------------------------------TAG DICTIONARY-------------------------------------------------------------

                if tag not in tag_dictionary:
                    total_tags += 1
                    if(count != 0):
                        if(count < len(sentence)-1):
                            tag_dictionary.update({tag:{'Transition_probability':{},'Total_count':1,'Count_in_2nd_place':0, 'Count_starting_sentence':0, 'Count_in_1st_place':0, 'Count_ending_sentence': 0}})
                        else:
                            tag_dictionary.update({tag: {'Transition_probability':{}, 'Total_count':1,'Count_in_2nd_place':0, 'Count_starting_sentence':0, 'Count_in_1st_place': 0, 'Count_ending_sentence': 1}})

                    else:
                        tag_dictionary.update({tag:{'Transition_probability':{}, 'Total_count':1, 'Count_in_2nd_place': 0,'Count_starting_sentence': 1,'Count_in_1st_place': 0,'Count_ending_sentence': 0}})
                else:
                    tag_dictionary[tag]['Total_count'] += 1
                    if(count ==0):
                        tag_dictionary[tag]['Count_starting_sentence'] += 1
                    if(count == len(sentence)-1):
                        tag_dictionary[tag]['Count_ending_sentence'] += 1

                if len(sentence) >1 and count > 0:
                    first_tag = sentence[count - 1].strip('\n').split('\t')[2]
                    bigram = (first_tag, tag)
                    if (bigram not in bigrams_types):
                        bigrams_types.update({bigram: 1})
                        tag_dictionary[tag]['Count_in_2nd_place'] += 1
                        tag_dictionary[first_tag]['Count_in_1st_place'] += 1
                    else:
                        bigrams_types[bigram] += 1


# ---------------------------------------------WORD DICTIONARY-------------------------------------------------------------

                if word not in word_dictionary:
                    total_words += 1
                    if(count != 0):
                        if(count < len(sentence)-1):
                            word_dictionary.update({word:{'Tags':{tag:{'Count':1}},'Total_count':1, 'Count_in_2nd_place': 1, 'Count_starting_sentence': 0, 'Count_in_1st_place':1}})
                        else:
                            word_dictionary.update({word: {'Tags':{tag:{'Count':1}},'Total_count':1, 'Count_in_2nd_place': 1, 'Count_starting_sentence': 0, 'Count_in_1st_place': 0}})
                    else:
                        word_dictionary.update({word: {'Tags': {tag:{'Count':1}}, 'Total_count': 1,'Count_in_2nd_place': 0, 'Count_starting_sentence': 1, 'Count_in_1st_place': 1}})

                else:
                    word_dictionary[word]['Total_count'] += 1
                    if tag not in word_dictionary[word]['Tags']:
                        word_dictionary[word]['Tags'].update({tag:{'Count':1}})
                    else:
                        word_dictionary[word]['Tags'][tag]['Count'] += 1
                    if(count != 0):
                        word_dictionary[word]['Count_in_2nd_place'] += 1
                        if(count<len(sentence)-1):
                            word_dictionary[word]['Count_in_1st_place'] += 1
                    else:
                        word_dictionary[word]['Count_starting_sentence'] += 1
                        word_dictionary[word]['Count_in_1st_place'] += 1
# -------------------------------------------------------------------------------------------------------------------------
                count += 1
            training_count += 1
        # else:
        #     test_sentence = []
        #     for term in sentence:
        #         fdr.write(term)
        #         term = term.strip('\n')
        #         content = term.split('\t')
        #         word = content[1]
        #         tag = content[2]
        #         test_sentence.append(word)
        #     fdr.write('\n')
        #     dev_set.append(test_sentence)

# Laplace smoothing in transmission probability since there is a chance of a tag of starting or ending a sentence even
# though it never appeared at the start or end of a sentence in train set
    for tag in tag_dictionary:
        probability_of_starting = ((tag_dictionary[tag]['Count_starting_sentence']+1)/(training_count+len(tag_dictionary)))
        probability_of_ending = ((tag_dictionary[tag]['Count_ending_sentence']+1)/(training_count+len(tag_dictionary)))
        tag_dictionary[tag].update({'Probability_of_starting':probability_of_starting,'Probability_of_ending':probability_of_ending})

    return word_dictionary, tag_dictionary, bigrams_types, dev_set, training_count

def unknown_handling(word_dictionary, cutoff):

    # Unknown words in dev/test set is handles as follows:
    # In the train set, words which rarely appear (less than or equal to cutoff) are considered as UNKNOWN WORDS
    # I counted the number of word occurrences for rare words and used that counts and POS tags to calculate probabilities
    # for any unseen word from test/dev set

    unknown_dictionary = dict()

    unknown_dictionary['UNK_WORD'] = {'Total_count':0,'Tags':{}}
    for word in word_dictionary:
        if word_dictionary[word]['Total_count'] > cutoff:
            unknown_dictionary[word] = word_dictionary[word]
        else:
            total_counts = word_dictionary[word]['Total_count']
            unknown_dictionary['UNK_WORD']['Total_count'] += total_counts
            for tag in word_dictionary[word]['Tags']:
                if tag not in unknown_dictionary['UNK_WORD']['Tags']:
                    unknown_dictionary['UNK_WORD']['Tags'][tag] = word_dictionary[word]['Tags'][tag]
                else:
                    unknown_dictionary['UNK_WORD']['Tags'][tag]['Count'] += word_dictionary[word]['Tags'][tag]['Count']
    return unknown_dictionary



def word_given_tag_probabilities(word_dictionary,tag_dictionary):

    # This function calculates the emission probabilities of each word in the train set
    # i.e. Probability of a word given tag =
    #               count of word and tag appear together / total number of times that tag occurred in the train set
    # print('Length: ',len(tag_dictionary))
    for word in word_dictionary:
        for tag in word_dictionary[word]['Tags']:
            emission_probability = (word_dictionary[word]['Tags'][tag]['Count'])/(tag_dictionary[tag]['Total_count'])
            if 'Probability' not in word_dictionary[word]['Tags'][tag]:
                word_dictionary[word]['Tags'][tag].update({'Probability':emission_probability})
            else:
                word_dictionary[word]['Tags'][tag]['Probability'] = emission_probability

    return word_dictionary


def transition_probability(tag_dictionary, bigrams_types):
    total_bigram_types = len(bigrams_types)
    bigram_permutation = []
    for tag_first in tag_dictionary:
        for tag_second in tag_dictionary:
            bigram_permutation.append((tag_first, tag_second))

    for bigram in bigram_permutation:
        tag_i = bigram[1]
        tag_i_1 = bigram[0]
        tag_dictionary[tag_i]['Transition_probability'].update({tag_i_1:0})
        if(tag_dictionary[tag_i_1]['Total_count'] == 0 or bigram not in bigrams_types):
            transition_prob = 0
        else:
            transition_prob = float(bigrams_types[bigram])/tag_dictionary[tag_i_1]['Total_count']
        tag_dictionary[tag_i]['Transition_probability'][tag_i_1] = transition_prob

    return tag_dictionary



def Kneyser_ney(tag_dictionary, bigrams_types):

    # Applying Kneyser Ney smoothing
    # To give some transition probabilities to those bigram tags which never appeared in the train set but may appear in
    # the test/dev set

    discount = 0.75
    total_bigram_types = len(bigrams_types)
    bigram_permutation = []
    for tag_first in tag_dictionary:
        for tag_second in tag_dictionary:
            bigram_permutation.append((tag_first,tag_second))

    for bigram in bigram_permutation:
        tag_i = bigram[1]
        tag_i_1 = bigram[0]

        probability_continuation = float(tag_dictionary[tag_i]['Count_in_2nd_place']/total_bigram_types)

        if 'Probability_continuation' not in tag_dictionary[tag_i]:
            tag_dictionary[tag_i].update({'Probability_continuation':probability_continuation})
        lambda_i_1 = float((discount/tag_dictionary[tag_i_1]['Total_count'])*tag_dictionary[tag_i_1]['Count_in_1st_place'])
        if(bigram in bigrams_types):
            probability_tag_i_given_tag_i_1 = (max((bigrams_types[bigram]-discount),0))/tag_dictionary[tag_i_1]['Total_count'] + (lambda_i_1*probability_continuation)
        else:
            probability_tag_i_given_tag_i_1 = lambda_i_1 * probability_continuation
        if 'Probability_after_smoothing' not in tag_dictionary[tag_i]:
            tag_dictionary[tag_i].update({'Probability_after_smoothing':{tag_i_1:probability_tag_i_given_tag_i_1}})
        else:
            tag_dictionary[tag_i]['Probability_after_smoothing'].update({tag_i_1:probability_tag_i_given_tag_i_1})

    return tag_dictionary

def Laplace(tag_dictionary, bigrams_types):

    # Another method to smooth the transition probabilities of tags. Also called add one smoothing

    bigram_permutation = []
    for tag_first in tag_dictionary:
        for tag_second in tag_dictionary:
            bigram_permutation.append((tag_first, tag_second))

    for bigram in bigram_permutation:
        tag_i = bigram[1]
        tag_i_1 = bigram[0]
        if (bigram in bigrams_types):
            transition_probability = float((bigrams_types[bigram]+0.000001)/(tag_dictionary[tag_i_1]['Total_count']+(0.000001*len(tag_dictionary))))
        else:
            transition_probability = float(1/(tag_dictionary[tag_i_1]['Total_count']+len(tag_dictionary)))

        if 'Probability_after_smoothing' not in tag_dictionary[tag_i]:
            tag_dictionary[tag_i].update({'Probability_after_smoothing':{tag_i_1:transition_probability}})
        else:
            tag_dictionary[tag_i]['Probability_after_smoothing'].update({tag_i_1:transition_probability})


    return tag_dictionary





def viterbi(original_sentence,word_dictionary,tag_dictionary):

    # Apply viterbi decoding to the test/dev set

    sentence =[]
    for word in original_sentence:
        if word in word_dictionary:
            sentence.append(word)
        else:
            sentence.append('UNK_WORD')

    for word in word_dictionary:
        for tag in tag_dictionary:
            if tag not in word_dictionary[word]['Tags']:
                word_dictionary[word]['Tags'].update({tag: {'Count': 0, 'Probability':0.0}})

# ------------------------------------------------------ DEFINING VITERBI_DICTIONARY --------------------------------------------------------------------
    # Creating viterbi_dictionary and backtrace dictionaries to store values and tags respectively

    viterbi_dictionary = dict()
    backtrace = dict()

    for tag in tag_dictionary:
        viterbi_dictionary.update({tag: {}})
        backtrace.update({tag:{}})
        for i, word in enumerate(sentence):
            viterbi_dictionary[tag].update({(i,word):0.0})
            backtrace[tag].update({(i,word):None})

#------------------------------------------------------ INITIALISATION --------------------------------------------------------------------

    for tag in tag_dictionary:
        start_state_probability = 1
        starting_probability = tag_dictionary[tag]['Probability_of_starting']
        if len(sentence) != 1:
            if starting_probability != 0:
                if word_dictionary[sentence[0]]['Tags'][tag]['Probability'] != 0.0:
                    word_given_tag_probability = word_dictionary[sentence[0]]['Tags'][tag]['Probability']
                    viterbi_start = math.log(start_state_probability * starting_probability) + math.log(
                        word_given_tag_probability)
                else:
                    viterbi_start = float('-inf')
            else:
                viterbi_start = float('-inf')
        else:
            ending_probability = tag_dictionary[tag]['Probability_of_ending']
            if starting_probability == 0:
                starting_probability = ending_probability
            if starting_probability != 0:
                if word_dictionary[sentence[0]]['Tags'][tag]['Probability'] != 0.0:
                    word_given_tag_probability = word_dictionary[sentence[0]]['Tags'][tag]['Probability']
                    viterbi_start = math.log(start_state_probability * starting_probability) + math.log(
                        word_given_tag_probability)
                else:
                    viterbi_start = float('-inf')
            else:
                viterbi_start = float('-inf')

        viterbi_dictionary[tag][(0,sentence[0])] = viterbi_start
        backtrace[tag][(0,sentence[0])] = '0'

# ------------------------------------------------------ RECURSION  --------------------------------------------------------------------

    for index in range(1,len(sentence)):
        for tag in tag_dictionary:
            probs = []
            backtrace_intermediate = []
            if word_dictionary[sentence[index]]['Tags'][tag]['Probability'] != 0.0:
                for prev_tag in tag_dictionary:
                    if viterbi_dictionary[prev_tag][(index-1,sentence[index-1])] != float('-inf'):
                        if(tag_dictionary[tag]['Probability_after_smoothing'][prev_tag] != 0.0):
                            probs.append(viterbi_dictionary[prev_tag][(index - 1,sentence[index - 1])] + math.log(
                                tag_dictionary[tag]['Probability_after_smoothing'][prev_tag]) + math.log(
                                word_dictionary[sentence[index]]['Tags'][tag]['Probability']))
                            backtrace_intermediate.append((prev_tag,viterbi_dictionary[prev_tag][(index - 1,sentence[index - 1])] + math.log(
                                tag_dictionary[tag]['Probability_after_smoothing'][prev_tag])))
                        else:
                            viterbi_dictionary[tag][(index, sentence[index])] = float('-inf')
                            backtrace[tag][(index, sentence[index])] = None
                if len(probs) == 0:
                    viterbi_dictionary[tag][(index, sentence[index])] = float('-inf')

                    backtrace[tag][(index, sentence[index])] = None
                else:
                    viterbi_dictionary[tag][(index,sentence[index])] = max(probs)
                    backtrace[tag][(index,sentence[index])] = max(backtrace_intermediate, key=lambda x: x[1])[0]

            else:
                viterbi_dictionary[tag][(index,sentence[index])] = float('-inf')

                backtrace[tag][(index,sentence[index])] = None

# ------------------------------------------------------ TERMINATION --------------------------------------------------------------------

    final_viterbi = []
    final_backtrace = []
    for tag in tag_dictionary:
        if viterbi_dictionary[tag][(len(sentence)-1,sentence[len(sentence)-1])] != float('-inf'):
            final_viterbi.append(viterbi_dictionary[tag][(len(sentence)-1,sentence[len(sentence)-1])] + math.log(tag_dictionary[tag]['Probability_of_ending']))
            final_backtrace.append((tag,viterbi_dictionary[tag][(len(sentence)-1,sentence[len(sentence)-1])] + math.log(
                        tag_dictionary[tag]['Probability_of_ending'])))
#        else:
#            final_backtrace.append(('O',0.0))
    end_state = max(final_backtrace, key=lambda x:x[1])[0]

    path = [end_state]

    for index in range(len(sentence)-1,0,-1):
        tag = backtrace[path[len(path)-1]][(index,sentence[index])]
        path.append(tag)
    final_path = []

    for p in range(len(path)-1,-1,-1):
        final_path.append(path[p])
    return final_path

# ------------------------------------------------------ MAIN --------------------------------------------------------------------

doc = read_and_write()

test_data, count = read_test_set()

#baseline(doc)

words_dictionary, tag_dictionary, bigrams_types, dev_set, training_count = dictionaries(doc)

word_dictionary = unknown_handling(words_dictionary,1)
word_dictionary = word_given_tag_probabilities(word_dictionary,tag_dictionary)
#tag_dictionary = transition_probability(tag_dictionary,bigrams_types)
#tag_dictionary = Kneyser_ney(tag_dictionary,bigrams_types)
tag_dictionary = Laplace(tag_dictionary,bigrams_types)

fd = open('jain-payoj-assgn4-out.txt','w')
#fd = open('dev_set_output.txt','w')

for sentence in test_data:
#    counting = 0
#for sentence in dev_set:
#    counting += 1
    ner_tagging = viterbi(sentence,word_dictionary,tag_dictionary)
    for index in range(len(sentence)):
        fd.write(str(index+1)+'\t'+sentence[index]+'\t'+ner_tagging[index]+'\n')
    fd.write('\n')

fd.close()
