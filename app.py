
from flask import Flask, request, jsonify

import pandas as pd

from rapidfuzz.process import extractOne 
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio

from transformers import AutoModel, AutoTokenizer
import torch 

import logging, sys
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

app = Flask(__name__)

# load transactions and users data from csv 
transactions = pd.read_csv('data/transactions.csv')
users = pd.read_csv('data/users.csv')


def substr(org_str, sub1='from', sub2='for'):
  """ returns substring in between two strings """
  s = org_str.lower()
  idx1 = s.index(sub1)
  idx2 = s.index(sub2)
  return s[idx1 + len(sub1)+1 : idx2]
  
  # task 1 endpoint 
@app.route('/match_user', methods=['GET'])
def match_user():
  """
  Implements GET match_user. 
  Input: transaction id
  Uses RapidFuzz - a fast
  string matching library. match_score is calculated using a metric [0,1].
  Higher score indicates most similarity 
  
  Returns:
      total no of matches.
      json object with a list of matches orderd by desc match_score.
  """
  transaction_id = request.args.get('transaction_id')
  transaction = transactions[transactions['id'] == transaction_id]
  
  if transaction.empty:
      return jsonify({"error": "Transaction not found"}), 404
  
  description = transaction['description'].values[0]
  sub_str = substr(description)
  logging.debug('%s', description)
  logging.debug('%s', sub_str)
  
  matches = []
  for _, user in users.iterrows():
    match_score = extractOne(query=sub_str, choices=[user['name']], scorer=Levenshtein.normalized_similarity)
    # match_score = extractOne(query=sub_str, choices=[user['name']], scorer=Levenshtein.distance)
    # match_score = extractOne(query=sub_str, choices=[user['name']], scorer=ratio)
    logging.debug('score = %s', match_score)
    
    if match_score is not None:
      matches.append({"id": user['id'], "name": user['name'], "match_score": match_score[1]})
  
  matches.sort(key=lambda x: x['match_score'], reverse=True)
  
  return jsonify({"users": matches, "total_number_of_matches": len(matches)})

# ----- task 2 ------
# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def get_embeddings(text, tokenizer, model):
  """ Returns embedded text """
  inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
  with torch.no_grad():
      outputs = model(**inputs)
  # Mean pooling to get sentence embeddings
  embeddings = outputs.last_hidden_state.mean(dim=1)
  return embeddings

def find_best_similarity_slice(input_text, target_text, window_size):
  """ 
  Finds the best slice in transaction descriptions that matches input string 
  Returns best slice and its corresponding similarity score and embedding  
  """
  input_embedding = get_embeddings(input_text, tokenizer, model)
  
  max_similarity = -1
  best_slice = None
  best_slice_embedding = None

  for i in range(len(target_text) - window_size + 1):
      slice_text = target_text[i:i+window_size]
      slice_embedding = get_embeddings(slice_text, tokenizer, model)
      similarity = torch.nn.functional.cosine_similarity(input_embedding, slice_embedding)
      
      if similarity > max_similarity:
          max_similarity = similarity.item()
          best_slice = slice_text
          best_slice_embedding = slice_embedding
          best_slice_tokens = slice_embedding.shape[1]

  return best_slice, max_similarity, best_slice_tokens


# task 2 endpoint
@app.route('/similar_transactions', methods=['GET'])
def similar_transactions():
  """
  Implements GET similar_transactions.
  Input: string 
  
  Uses huggingface transformer architecture with a 
  pretrained Bert model for embedding. 
  Calculates similarity based on cosine similarity [0, 1]
  
  Returns:
      No of tokens used 
      json list of similar transactions ordered by desc embedding score
  """
  input_str = request.args.get('input_str')
  window_size = len(input_str)
  
  logging.debug('%s', input_str)
  
  if not input_str:
      return jsonify({"error": "No input string provided"}), 400
  
  
  # Calculate similarity with transaction descriptions
  similarities = []
  for _, transaction in transactions.iterrows():
      best_slice, max_similarity, best_slice_tokens = find_best_similarity_slice(input_str, transaction['description'], window_size)
      
      logging.debug('best slice = %s', best_slice)
      logging.debug('max_suimilarity = %s', max_similarity )
      
      similarities.append({"id": transaction['id'], "embedding": max_similarity})
  
  # Sort similarities in descending order
  similarities.sort(key=lambda x: x['embedding'], reverse=True)
  
  return jsonify({"transactions": similarities, "total_number_of_tokens_used": best_slice_tokens})
  
if __name__ == '__main__':
    app.run(debug=True)
