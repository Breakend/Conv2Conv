# TODO: write script to download harvard-seq2seq and all commands to generate this
# TODO: maybe add questionnaire results?


To train the custom model, simply run:

python custom_net.py

This will download and parse the cornell movie dialogue corpus on the training set.

To run evaluations, use:

python custom_net_eval.py

This will output the predictions to a predictions.txt file. These can be run through a tokenizer (required if comparing against gold using multi-bleu.perl. For experiments, the way this was done was simply:

with open('tokenized.txt', 'w') as f2:
    with open('predictions.txt', 'r') as f1:
        for line in f1.readlines():
            f2.write(' '.join(nltk.word_tokenize(line)+ '\n'))

Hyperparameters are constants at the top of the custom_net.py custom_net_eval.py files.

For the Baseline LSTM model, we used the Harvard Seq2Seq-Attn library. It was not included due to size limits and attribution reasons.

$ git clone https://github.com/harvardnlp/seq2seq-attn.git
$ python preprocess_for_seq2seq.py # to Download and preprocess (tokenize) the data
$ cd seq2seq-attn
$ python preprocess.py --srcfile ../data/cornell/processed_sources_seq2seq.txt --targetfile ../data/cornell/processed_targets_seq2seq.txt  --srcvalfile ../data/cornell/processed_sources_val_seq2seq.txt --targetvalfile ../data/cornell/processed_targets_val_seq2seq.txt --outputfile data/cornell
$ th train.lua -data_file data/cornell-train.hdf5 -val_data_file data/cornell-val.hdf5 -savefile demo-cornell
$ th evaluate.lua -model cornell-model_final.t7 -src_file ../data/cornell/processed_sources_test_seq2seq.txt -output_file predictions.txt -src_dict data/cornell.src.dict -targ_dict data/cornell.targ.dict
$ perl multi-bleu.perl ../data/cornell/processed_targets_test_seq2seq.txt < predictions.txt

Note: the multi-bleu.perl script is from the Moses Translation project (linked in script), but was included here for convenience.

Sample responses for all models to the prepared questionnaire have been put in the Questionnaire/* folder.
