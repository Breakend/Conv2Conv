from cornell_movie_dialogue import DEFAULT_DATA_DIR, CornellMovieData
import nltk
import os
import sys  
import random
import numpy as np

line_pairs = CornellMovieData.get_line_pairs()

ADD_CHARACTERS_TO_HARVARD = False

def write_to_files_with_indices(line_pairs, source_file, target_file, indices, FORMAT_FOR_HARVARD_SEQ2SEQ=False):
    for source, target, source_character, target_character in list(line_pairs[indices]):
        source = source.decode('utf-8','ignore').encode("utf-8")
        target = target.decode('utf-8','ignore').encode("utf-8")
        source = source.strip("<u>").strip("</u>")
        target = target.strip("<u>").strip("</u>")
        if len(source) >= 50 or len(target) >= 50:
            continue

        if FORMAT_FOR_HARVARD_SEQ2SEQ: 
            source = nltk.word_tokenize(source)
            target = nltk.word_tokenize(target) 
            if ADD_CHARACTERS_TO_HARVARD:
                target = target.lower()
                source = source.lower()            
                source = ["%s-|-%s " % (word, source_character) for word in source]
                target = ["%s-|-%s " % (word, target_character) for word in target]
            source = " ".join(source)
            target = " ".join(target)
            source += "\n"
            target += "\n"
        else:
            source = "%s +++$+++ %s" % (source_character, source)
            target = "%s +++$+++ %s" % (target_character, target)

        source_file.write(source)
        target_file.write(target)

all_indices = range(len(line_pairs))
random.shuffle(all_indices)
train_indices = all_indices[:int(len(line_pairs)*.8)]
val_indices = all_indices[int(len(line_pairs)*.8)+1:int(len(line_pairs)*.9)]
test_indices = all_indices[int(len(line_pairs)*.9)+1:len(line_pairs)-1]

for t in ['seq2seq','conv2conv']:
    source_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_sources_%s.txt' % t), 'w')
    target_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_targets_%s.txt'% t), 'w')
    target_test_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_targets_test_%s.txt' % t), 'w')
    target_val_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_targets_val_%s.txt' % t), 'w')
    source_test_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_sources_test_%s.txt' % t), 'w')
    source_val_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_sources_val_%s.txt' % t), 'w')
    use_harvard = True if t == 'seq2seq' else False
    write_to_files_with_indices(np.array(line_pairs), source_file, target_file, train_indices, use_harvard)
    write_to_files_with_indices(np.array(line_pairs), source_val_file, target_val_file, val_indices, use_harvard)
    write_to_files_with_indices(np.array(line_pairs), source_test_file, target_test_file, test_indices, use_harvard)

    target_file.close()
    target_test_file.close()
    target_val_file.close()

    source_file.close()
    source_test_file.close()
    source_val_file.close()
