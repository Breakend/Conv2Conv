from cornell_movie_dialogue import DEFAULT_DATA_DIR, CornellMovieData
import nltk
import os
import sys  
import random
import numpy as np

source_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_sources.txt'), 'w')
target_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_targets.txt'), 'w')
target_test_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_targets_test.txt'), 'w')
target_val_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_targets_val.txt'), 'w')
source_test_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_sources_test.txt'), 'w')
source_val_file = open(os.path.join(DEFAULT_DATA_DIR, 'processed_sources_val.txt'), 'w')

line_pairs = CornellMovieData.get_line_pairs()

def write_to_files_with_indices(line_pairs, source_file, target_file, indices):
    for source, target, source_character, target_character in list(line_pairs[indices]):
        source = source.lower()            
        source = source.decode('utf-8','ignore').encode("utf-8")
        target = target.decode('utf-8','ignore').encode("utf-8")
        source = source.strip("<u>").strip("</u>")
        target = target.strip("<u>").strip("</u>")
        try:
            source = nltk.word_tokenize(source)
            target = target.lower()
            target = nltk.word_tokenize(target) 
        except:
	    import pdb; pdb.set_trace()
        source = ["%s-|-%s " % (word, source_character) for word in source]
        target = ["%s-|-%s " % (word, target_character) for word in target]
	
        source = " ".join(source)
        target = " ".join(target)
        source_file.write(source + "\n")
        target_file.write(target + "\n")

all_indices = range(len(line_pairs))
random.shuffle(all_indices)
train_indices = all_indices[:int(len(line_pairs)*.8)]
val_indices = all_indices[int(len(line_pairs)*.8)+1:int(len(line_pairs)*.9)]
test_indices = all_indices[int(len(line_pairs)*.9)+1:len(line_pairs)-1]

write_to_files_with_indices(np.array(line_pairs), source_file, target_file, train_indices)
write_to_files_with_indices(np.array(line_pairs), source_val_file, target_val_file, val_indices)
write_to_files_with_indices(np.array(line_pairs), source_test_file, target_test_file, test_indices)

target_file.close()
target_test_file.close()
target_val_file.close()

source_file.close()
source_test_file.close()
source_val_file.close()
