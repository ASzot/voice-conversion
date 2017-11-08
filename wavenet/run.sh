#!/bin/bash

# Training with Global Conditioning
python train.py --data_dir=corpus --gc_channels=32 --num_steps=5000 --silence_threshold=0.1


# The --gc_channels argument does two things:

# It tells the train.py script that it should build a model that includes global conditioning.
# It specifies the size of the embedding vector that is looked up based on the id of the speaker.



# Generate 
python generate.py --wav_out_path=gen_test.wav --samples 16000 speaker311.wav --gc_channels=32 --gc_cardinality=377 --gc_id=311 logdir/train/2017-11-03T16-45-34/model.ckpt-80000




# --gc_channels=32 specifies 32 is the size of the embedding vector, 
# and must match what was specified when training.

# --gc_cardinality=377 is required as 376 is the largest id of a speaker in the VCTK corpus. 
# If some other corpus is used, then this number should match what is automatically determined 
# and printed out by the train.py script at training time.

# --gc_id=311 specifies the id of speaker, speaker 311, for which a sample is to be generated.