
import json
from flask import Flask, jsonify, render_template
from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import uuid

app = Flask(__name__)

# Load pre-trained model and tokenizer from transformers
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Example input data
example_input = [
    {
        "response": "Yes, we offer online delivery services through major platforms like Swiggy and Zomato. You can also reserve a table directly from our website if you are planning to have breakfast!",
        "sources": [
            {
                "id": "51",
                "context": "Order online Thank you for your trust in us! We are available on all major platforms like zomato, swiggy. You can also order directly from our website",
                "link": "https://orders.brikoven.com"
            },
            {
                "id": "75",
                "context": "Do you give franchise if the brand No, we currently don't offer franchise opportunities for BrikOven! Although do feel free to drop in an email at theteam@brikoven. com so we can get in touch with you at a later stage if we do decide to give out franchisees",
                "link": ""
            },
            {
                "id": "8",
                "context": "Breakfast Reservations For Breakfast, we recommend making reservations in advance. Reservation is only available through our website",
                "link": "https://www.brikoven.com/reservations"
            }
        ]
    }
]

def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2, dim=0)

def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze()

def find_citations(data):
    citations = []
    for item in data:
        response_text = item.get('response')
        sources = item.get('sources', [])
        response_emb = encode_text(response_text)
        cited_sources = []
        for source in sources:
            context_emb = encode_text(source['context'])
            similarity = cosine_similarity(response_emb, context_emb).item()
            if similarity > 0.75:  # Threshold for considering it as a match
                cited_sources.append({
                    'id': str(uuid.uuid4()),  # Generate a unique ID
                    'link': source.get('link', '')
                })
        citations.append({
            'citations': cited_sources
        })
    return citations

def save_to_json(data, filename='data.json'):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Fetch, process, and save the data
citations_data = find_citations(example_input)
save_to_json(citations_data)

@app.route('/citations', methods=['GET'])
def get_citations():
    with open('data.json', 'r') as f:
        citations = json.load(f)
    return jsonify(citations[:10])

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
