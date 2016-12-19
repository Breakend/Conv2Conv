'''Reads the Cornell movie dialog dataset and creates training and test datasets
Note this was modified from: https://github.com/pralexa/SimpleBot/blob/master/cornell_movie_dialog.py
'''

import numpy as np
import os
import sys
from string import punctuation
from six.moves import urllib
import tarfile
import shutil
import zipfile
from tensorflow.models.rnn.translate import data_utils
import dataset


DEFAULT_DATA_DIR = os.path.join(os.getcwd(), "data", "cornell")
DATA_URL = "http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip"

class CornellMovieData(dataset.Dataset):

    @staticmethod
    def get_preprepared_line_pairs(source_path, target_path, data_dir=DEFAULT_DATA_DIR):
        source_lines, target_lines = [], []
        with open(os.path.join(data_dir, source_path), 'r') as sources:
            source_lines = sources.readlines()
            source_characters = [x.split(' +++$+++ ')[0] for x in source_lines]
            source_lines = [x.split(' +++$+++ ')[1] for x in source_lines]

        with open(os.path.join(data_dir, target_path), 'r') as targets:
            target_lines = targets.readlines()
            target_characters = [x.split(' +++$+++ ')[0] for x in target_lines]
            target_lines = [x.split(' +++$+++ ')[1] for x in target_lines]
        return zip(source_lines, target_lines, source_characters, target_characters)


    @staticmethod
    def get_line_pairs(data_dir=DEFAULT_DATA_DIR):
        lines_by_number = {}
        with open(os.path.join(data_dir, 'movie_lines.txt')) as lines:
            for line in lines:
                lines_by_number[line.split()[0]] = line.split('+++$+++ ')[-1]

        dialogue_tuples = []

        with open(os.path.join(data_dir, 'movie_conversations.txt')) as conversations:

            source = open(os.path.join(data_dir, 'source.txt'), 'w')
            target = open(os.path.join(data_dir, 'target.txt'), 'w')

            line_pairs = []
            character_id_tuples = []

            def strip_punctuation(s):
                return ''.join(c for c in s if c not in punctuation)

            for conversation in conversations:

                # Get the line nums (between [ and ]) and split by commma
                character_ids = [int(x.strip('u')) for x in conversation.split(' +++$+++ ')[:2]]
                conv_lines = conversation.split('[')[1].split(']')[0].split(',')

                # Strip quote marks
                conv_lines = [strip_punctuation(lines) for lines in conv_lines]
                conv_lines = [lines.strip() for lines in conv_lines]

                for i in range(0, len(conv_lines) - 1):
                    if conv_lines[i] in lines_by_number and conv_lines[i + 1] in lines_by_number:
                        source.write(lines_by_number[conv_lines[i]])
                        target.write(lines_by_number[conv_lines[i + 1]])
                        dialogue_tuples.append((lines_by_number[conv_lines[i]], lines_by_number[conv_lines[i + 1]], character_ids[i%2], character_ids[(i+1)%2]))
                    if conv_lines[i] not in lines_by_number:
                        print("Could not find " + conv_lines[i] + "in movie lines")
                    if conv_lines[i + 1] not in lines_by_number:
                        print("Could not find " + conv_lines[i + 1] + "in movie lines")

            source.close()
            target.close()
            return dialogue_tuples

    @staticmethod
    def maybe_download_and_extract(dest_directory=DEFAULT_DATA_DIR):
      """Download and extract the tarball from the Cornell website"""

      if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
      filename = DATA_URL.split('/')[-1]
      filepath = os.path.join(dest_directory, filename)
      if not os.path.exists(filepath):
        sys.stdout.write('\r>> Downloading to ' + dest_directory)
        def _progress(count, block_size, total_size):
          sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
              float(count * block_size) / float(total_size) * 100.0))
          sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        zipfile.ZipFile(filepath, 'r').extractall(dest_directory)
        # Now move them out of that folder for ease of access
        files = os.listdir(os.path.join(dest_directory, 'cornell movie-dialogs corpus'))
        for f in files:
            if f.endswith('.txt'):
                full_file = os.path.join(dest_directory, 'cornell movie-dialogs corpus', f)
                print("Moving " + f + " to " + dest_directory)
                shutil.move(full_file, dest_directory)
