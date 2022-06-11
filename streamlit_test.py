from __future__ import absolute_import, division, print_function
# from __future__ import division
# from __future__ import print_function

import numpy as np
import os
import time
import tensorflow as tf
from tensorflow.core.example import example_pb2
from nucleus.protos import variants_pb2
from nucleus.util import vis
import streamlit as st
import pandas as pd
from IPython.display import display, Image


# EASY_INPUT='gs://deepvariant/cybdv/cybdv_0.9.0_easy.tfrecord.gz'
# DIFFICULT_INPUT='gs://deepvariant/cybdv/cybdv_0.9.0_difficult.tfrecord.gz'

# easy_input='easy.tfrecord.gz'
# difficult_input='difficult.tfrecord.gz'

# gsutil cp 'gs://deepvariant/cybdv/cybdv_0.9.0_easy.tfrecord.gz' 'easy.tfrecord.gz'
# gsutil cp 'gs://deepvariant/cybdv/cybdv_0.9.0_difficult.tfrecord.gz' 'difficult.tfrecord.gz'
easy_input='./easy.tfrecord.gz'
# raw_difficult_dataset = tf.data.TFRecordDataset(difficult_input, compression_type="GZIP")
raw_easy_dataset = tf.data.TFRecordDataset(easy_input, compression_type="GZIP")

# We can inspect an example to see what is inside:
# for e in raw_easy_dataset.take(1):
#   example = tf.train.Example()
#   example.ParseFromString(e.numpy())
#   print(example)

# Describe the features. These were defined in the prep script and can also be inspected above.
consolidated_variant_description = {
  'locus_id': tf.io.FixedLenFeature([], tf.string),
  'multiallelic': tf.io.FixedLenFeature([], tf.int64),
  'difficulty': tf.io.FixedLenFeature([], tf.int64),
  'genotypes': tf.io.VarLenFeature(tf.string)
}

# Quick parsing at the top level
def quick_parse_locus(locus_proto):
  return tf.io.parse_single_example(locus_proto, consolidated_variant_description)

# difficult_dataset = raw_difficult_dataset.map(quick_parse_locus)
easy_dataset = raw_easy_dataset.map(quick_parse_locus)
# for e in easy_dataset.take(1):
#   print(e)

# Simple counter
def count_calls(calls):
  count = 0
  for c in calls:
    count += 1
  return count

# print('Number of easy examples:', count_calls(easy_dataset))
# print('Number of difficult examples:', count_calls(difficult_dataset))

# A few examples of easy variants
easy_biallelics = easy_dataset.filter(lambda x: tf.equal(x['multiallelic'], False))
easy_multiallelics = easy_dataset.filter(lambda x: tf.equal(x['multiallelic'], True))

# Where DeepVariant had less than 90% likelihood for its choice OR it chose wrong
# difficult_biallelics = difficult_dataset.filter(lambda x: tf.equal(x['multiallelic'], False))
# difficult_multiallelics = difficult_dataset.filter(lambda x: tf.equal(x['multiallelic'], True))

# In the prep script, we set difficult=100 when DeepVariant's top choice did not match the truth (i.e. DeepVariant got it wrong)
# dv_missed = difficult_dataset.filter(lambda x: tf.equal(x['difficulty'], 100))

# Optional counting of examples (commented out as these take several seconds to run)
# Uncomment these if you want to see the counts of different types of variants.
# print('easy_biallelics count:', count_calls(easy_biallelics))
# print('easy_multiallelics count:', count_calls(easy_multiallelics))
# print('difficult_biallelics count', count_calls(difficult_biallelics))
# print('difficult_multiallelics count:', count_calls(difficult_multiallelics))
# print('dv_missed count:', count_calls(dv_missed))

def bytes_to_str(b):
  if isinstance(b, type('')):
    return b
  elif isinstance(b, type(b'')):
    return b.decode()
  else:
    raise ValueError('Incompatible type: {}. Expected bytes or str.'.format(type(b)))

def fully_parse_locus(top_level_parsed):
  # where top_level_parsed = tf.io.parse_single_example(locus_proto, consolidated_variant_description)

  def _clean_locus(locus):
    return {
        'locus_id': bytes_to_str(locus['locus_id'].numpy()),
        'multiallelic': bool(locus['multiallelic'].numpy()),
        'genotypes': locus['genotypes'],
        'difficulty': locus['difficulty']
    }
  clean_locus = _clean_locus(top_level_parsed)
  genotype_description = {
    'example': tf.io.FixedLenFeature([], tf.string),
    'truth_label': tf.io.FixedLenFeature([], tf.int64),
    'genotype_probabilities': tf.io.VarLenFeature(tf.float32),
    'dv_correct': tf.io.FixedLenFeature([], tf.int64),
    'dv_label': tf.io.FixedLenFeature([], tf.int64),
    'alt': tf.io.FixedLenFeature([], tf.string)
  }
  def _parse_genotype(sub_example):
    return tf.io.parse_single_example(sub_example, genotype_description)

  def _clean_genotype(e):
    genotype_example = tf.train.Example()
    genotype_example.ParseFromString(e['example'].numpy())
    return {
        'genotype_probabilities': list(e['genotype_probabilities'].values.numpy()),
        'dv_correct': bool(e['dv_correct'].numpy()),
        'dv_label': e['dv_label'].numpy(),
        'truth_label': e['truth_label'].numpy(),
        'example': genotype_example,
        'alt': bytes_to_str(e['alt'].numpy())
    }


  genotypes = clean_locus['genotypes'].values
  parsed_genotypes = []
  for s in genotypes:
    genotype_dict = _parse_genotype(s)
    clean_genotype_dict = _clean_genotype(genotype_dict)
    parsed_genotypes.append(clean_genotype_dict)

  clean_locus['genotypes'] = parsed_genotypes
  return clean_locus

# for e in easy_biallelics.take(1):
#     example = fully_parse_locus(e)
#     print(example)

def pretty_print_locus(locus):
  # initial information:
  allelic_string = "Multiallelic" if locus['multiallelic'] else "Biallelic" 
  print("%s -- %s: %d example%s" % (locus['locus_id'], allelic_string, len(locus['genotypes']), "s" if locus['multiallelic'] else ""))

def show_all_genotypes(genotypes):
  # Show pileup images for all genotypes
  for g in genotypes:
    vis.draw_deepvariant_pileup(example=g['example'])
    # And here's how DeepVariant did:
    print("Truth: %d, DeepVariant said: %d, Correct: %s" % (g['truth_label'], g['dv_label'], g['dv_correct']))
    print("DeepVariant genotype likelihoods:", g['genotype_probabilities'])
    print("\n\n")

def show_loci(loci, show_pileups=True):
  for raw_locus in loci:
    if show_pileups:
      print("_____________________________________________________________")
    locus = fully_parse_locus(raw_locus)
    pretty_print_locus(locus)
    if show_pileups:
      show_all_genotypes(locus['genotypes'])

show_loci(easy_biallelics.take(1))

WRITE_NORMAL = "\x1b[0m"
WRITE_GREEN_BACKGROUND = "\x1b[102m"
WRITE_RED_BACKGROUND = "\x1b[101m"

def play_game(calls, pro_mode=False, put_results_here=None):
  """
  Args:
    put_results_here: a list, allows saving results along the way even if the player stops the loop
  """
  # for example, calls = dataset.take(5)
  print("Can you beat DeepVariant?: type 0, 1, or 2 just like DeepVariant's CNN would for each example.")
  results = []
  score = 0
  total = 0
  dv_score = 0
  for c in calls:
    locus = fully_parse_locus(c)
    st.write("___________________________________________________________")
    # st.write(locus['locus_id'])
    allelic_string = "Multiallelic" if locus['multiallelic'] else "Biallelic" 
    st.write("%s: %d example%s" % (allelic_string, len(locus['genotypes']), "s" if locus['multiallelic'] else ""))
    quit_early = False
    # For every genotype, show the user the images and ask for their guess
    for e in locus['genotypes']:
      # Draw pileups:
    #   st.image(vis.draw_deepvariant_pileup(example=e['example']))
      img = vis.draw_deepvariant_pileup(example=e['example'], path="./test.png")
    #   display(img)
      st.image("test.png",caption = "Check those images before give your answer") 
    #   with open("GoogleMap.png", "wb") as png:
    #     png.write(img)
    #   im = img.visualize()
    #   open('output.png', 'wb').write(im.data)

    #   st.image(img) 

      st.write('Genotype in question: ', e['alt'])
      # Ask user until we get a 0, 1, or 2
      st.write('Select your answer:')
      
      index = 0
      one_entry = True


      guess_0 = st.radio(label="What's your answer?", options=('2', '1', '0','exit'))
    #   guess = st.number_input('Insert a number')
      agree = st.checkbox(key=index, label= 'Submit')
      index += 1

      if not agree:
          st.stop()

      else:
            # guess = guess_0  
          st.write('The current number is ', guess_0)
          start_compare(guess_0, e, score, total, dv_score, locus)
        #   guess = input("Your answer (0, 1, or 2):")
          one_entry = False

  return None
  
def start_compare(guess, e, score, total, dv_score, locus):
    guess = int(guess)

    truth = e['truth_label']
    if truth == guess:
        st.success("You are correct!")  
        print(WRITE_GREEN_BACKGROUND + "You are correct!", WRITE_NORMAL)
        score += 1
    else:
        st.error("Nope! Sorry it's actually"+str(truth))  
        print(WRITE_RED_BACKGROUND + "Nope! Sorry it's actually", truth, WRITE_NORMAL)
        total += 1
    # And here's how DeepVariant did:
    if e['dv_correct']:
        dv_score += 1
        dv_result_string = WRITE_GREEN_BACKGROUND + "is correct" + WRITE_NORMAL if e['dv_correct'] else WRITE_RED_BACKGROUND + "failed" + WRITE_NORMAL
        print("DeepVariant %s with likelihoods:" % (dv_result_string), e['genotype_probabilities'])
        result = {'id': locus['locus_id'], 'truth': truth, 'guess': guess, 'dv_label': e['dv_label']}
    st.write("___________________________________________________________")
    #   st.write("___________________________________________________________")
    #   st.write("=============================================================")

    st.info("Your score is: "+ str(score)+ "/"+str(total))
    st.info("DeepVariant's score is: "+str(dv_score))
    return None

st.title('Deepvariant examples')
st.write("DeepVariant turns genomic data into a pileup image, and then uses a convolutional neural network (CNN) to classify these images. This app lets you try compete with Deepvariant")

st.write("These are examples that DeepVariant was at least 99 presents sure about and got correct. Think of this as a tutorial. Your job is to pick 0, 1, or 2, depending on how many copies of the given alternate allele you see in the pileup.")
easy_biallelic_results = play_game(easy_biallelics.shuffle(80).take(1))