def print_segregated_abstract(pred, sample_sentences):
  import numpy as np
  preds = pred.numpy().tolist()
  from collections import defaultdict

  ans = defaultdict(list)
  for k, v in zip(preds, sample_sentences):
    ans[k].append(v)

  BACKGROUND = OBJECTIVE = METHODS = RESULTS = CONCLUSION = []
  for key, value in ans.items():
    if key == 0:
      BACKGROUND = value
    elif key == 1:
      CONCLUSION = value
    elif key == 2:
      METHODS = value
    elif key == 3:
      OBJECTIVE = value
    elif key == 4:
      RESULTS = value

  final_abstract = ""

  def add_lines(array, text):
    for sentence in array:
      text += sentence
      text += ". "
    text += "\n\n"
    return text

  if BACKGROUND:
    final_abstract += "\nBACKGROUND:\n"
    final_abstract = add_lines(BACKGROUND, final_abstract)
  if OBJECTIVE:
    final_abstract += "\nOBJECTIVE:\n"
    final_abstract = add_lines(OBJECTIVE, final_abstract)
  if METHODS:
    final_abstract += "\nMETHODS:\n"
    final_abstract = add_lines(METHODS, final_abstract)
  if RESULTS:
    final_abstract += "\nRESULTS:\n"
    final_abstract = add_lines(RESULTS, final_abstract)
  if CONCLUSION:
    final_abstract += "\nCONCLUSION:\n"
    final_abstract = add_lines(CONCLUSION, final_abstract)


  return final_abstract

def make_predictions(model, inputs):
  import tensorflow as tf
  pred_probs = model.predict(inputs, verbose=0)
  preds = tf.argmax(pred_probs, axis=1)
  return preds

def preprocess_data_for_inference(data):
  import re
  import tensorflow as tf
  import pandas as pd
  # Breaking data into sentences
  temp_list_of_sentences = [x for x in re.split("[//.|//!|//?]", data) if x!=""]
  list_of_sentences = []
  # Removing the preceding whitespace
  for sentence in temp_list_of_sentences:
    if sentence[0] == " ":
      sentence = sentence[1:]
    list_of_sentences.append(sentence)

  # Creating total lines variable
  total_lines = len(list_of_sentences) - 1

  def split_chars(text):
    return " ".join(list(text))

  samples = []

  for abstract_line_number, abstract_line in enumerate(list_of_sentences):
    line_data = {}
    line_data["text"] = abstract_line
    line_data["line_number"] = abstract_line_number
    line_data["total_lines"] = total_lines
    samples.append(line_data)


  samples_df = pd.DataFrame(samples)
  samples_df

  sample_sentences = samples_df["text"].tolist()
  sample_line_numbers_one_hot = tf.one_hot(samples_df["line_number"].to_numpy(), depth=15)
  sample_total_lines_one_hot = tf.one_hot(samples_df["total_lines"].to_numpy(), depth=20)
  sample_chars = [split_chars(sentence) for sentence in sample_sentences]

  sample_inputs = tf.data.Dataset.from_tensor_slices((sample_line_numbers_one_hot,
                                                      sample_total_lines_one_hot,
                                                      sample_sentences,
                                                      sample_chars))
  sample_inputs = tf.data.Dataset.zip((sample_inputs, sample_inputs))
  sample_inputs = sample_inputs.batch(32).prefetch(tf.data.AUTOTUNE)

  return (sample_inputs, sample_sentences)