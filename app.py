
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

# Load model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2" #possibility of using other models
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


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
  logging.debug('%s', description)
  
  matches = []
  for _, user in users.iterrows():
    match_score = extractOne(query=description, choices=[user['name']], scorer=Levenshtein.normalized_similarity)
    logging.debug('score = ', match_score)
    if match_score is not None:
      matches.append({"id": user['id'], "name": user['name'], "match_score": match_score[1]})
  
  matches.sort(key=lambda x: x['match_score'], reverse=True)
  
  return jsonify({"users": matches, "total_number_of_matches": len(matches)})



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
  logging.debug('%s', input_str)
  if not input_str:
      return jsonify({"error": "No input string provided"}), 400
  
  # Tokenize input string and compute embedding
  inputs = tokenizer(input_str, return_tensors='pt')
  input_tokens = inputs['input_ids'].shape[1]
  logging.debug('%s', input_tokens)  
  with torch.no_grad():
      input_embedding = model(**inputs).last_hidden_state.mean(dim=1)
  
  # Calculate similarity with transaction descriptions
  similarities = []
  for _, transaction in transactions.iterrows():
      desc_inputs = tokenizer(transaction['description'], return_tensors='pt')
      with torch.no_grad():
          desc_embedding = model(**desc_inputs).last_hidden_state.mean(dim=1)
      
      similarity = torch.nn.functional.cosine_similarity(input_embedding, desc_embedding).item()
      similarities.append({"id": transaction['id'], "embedding": similarity})
  
  # Sort similarities in descending order
  similarities.sort(key=lambda x: x['embedding'], reverse=True)
  
  return jsonify({"transactions": similarities, "total_number_of_tokens_used": input_tokens})
  
if __name__ == '__main__':
    app.run(debug=True)
