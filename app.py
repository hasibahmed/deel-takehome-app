
from flask import Flask, request, jsonify

import pandas as pd

from rapidfuzz.process import extractOne 
from rapidfuzz.distance import Levenshtein
from rapidfuzz.fuzz import ratio

from transformers import AutoModel, AutoTokenizer
import torch 


app = Flask(__name__)

transactions = pd.read_csv('data/transactions.csv')
users = pd.read_csv('data/users.csv')

@app.route('/match_user', methods=['GET'])
def match_user():
    transaction_id = request.args.get('transaction_id')
    transaction = transactions[transactions['id'] == transaction_id]
    
    if transaction.empty:
        return jsonify({"error": "Transaction not found"}), 404
    
    description = transaction['description'].values[0]
    print(description)
    
    # Rapid fuzz for matching: https://rapidfuzz.github.io/RapidFuzz/Usage/process.html
    matches = []
    for _, user in users.iterrows():
      # print(user['id'])
      # print(user['name'])
      match_score = extractOne(query=description, choices=[user['name']], scorer=Levenshtein.normalized_similarity)
      # print(match_score)
      if match_score is not None:
        matches.append({"id": user['id'], "name": user['name'], "match_metric": match_score[1]})
    
    matches.sort(key=lambda x: x['match_metric'], reverse=True)
    
    return jsonify({"users": matches, "total_number_of_matches": len(matches)})

  
if __name__ == '__main__':
    app.run(debug=True)
