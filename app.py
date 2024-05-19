import json
from flask import Flask, jsonify, render_template
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import uuid
import requests

app = Flask(__name__)

# Load pre-trained model and tokenizer from transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# API URL
API_URL = 'https://devapi.beyondchats.com/api/get_message_with_sources'

def fetch_data_from_api():
    response = requests.get(API_URL)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2, dim=0)

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def find_citations(data):
    citations = []
    for item in data['data']['data']:
        response_text = item.get('response')
        sources = item.get('source', [])
        response_emb = encode_text(response_text)
        cited_sources = []
        for source in sources:
            context_emb = encode_text(source['context'])
            similarity = cosine_similarity(response_emb, context_emb).item()
            if similarity > 0.75:  # Threshold for considering it as a match
                cited_sources.append({
                    'id': int(source['id']),  # Use integer ID
                    'link': source.get('link', '')
                })
        citations.append({
            'response': response_text,
            'citations': cited_sources
        })
    return citations

def save_to_json(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Fetch, process, and save the data
data = fetch_data_from_api()
if data:
    citations_data = find_citations(data)
    save_to_json(citations_data)

@app.route('/citations', methods=['GET'])
def get_citations():
    with open('data.json', 'r') as f:
        citations = json.load(f)
    return jsonify(citations[:10])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/save', methods=['POST'])
def save_citations():
    data = fetch_data_from_api()
    if data:
        try:
            with open('data.json', 'w') as file:
                json.dump(data, file, indent=4)
            return jsonify({'status': 'success'})
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to fetch data from API'})

if __name__ == '__main__':
    app.run(debug=True)
