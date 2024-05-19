
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
example_data = {
    "status": "success",
    "status_code": 200,
    "message": "Sample Sources fetched successfully!",
    "data": {
        "current_page": 1,
        "data": [
            {
                "id": 1,
                "response": "Yes, we offer online delivery services through major platforms like Swiggy and Zomato. You can also order directly from our website!",
                "source": [
                    {"id": "71", "context": "Order online Thank you for your trust in us! We are available on all major platforms: [Order online Order directly from our website](https://orders.brikoven.com), [Order from Swiggy](https://www.swiggy.com/direct/brand/7389?source=swiggy-direct&subSource=generic), [Order from zomato](https://www.zomato.com/bangalore/delivery?chain=18224650)", "link": ""},
                    {"id": "8", "context": "Breakfast Reservations\r For Breakfast, we recommend making reservations in advance. \r For walk-ins, we only seat parties on a first come, first served basis. \r Your reservation is confirmed upon you filling the form. \r Done reserving? Check out the menu below! \r https://www.brikoven.com/breakfastmenu \r \r Did you know you can order your breakfast online? Head over to our online orders platforms, and get 10% back as loyalty embers, redeemable on your next order on our platform, or dine in / takeaway! \r [Order Online](https://brikoven.dotpe.in/)", "link": "https://www.brikoven.com/reservations"},
                    {"id": "159", "context": "Do you give franchise if the brand No, we currently don't offer franchise opportunities for BrikOven! Although do feel free to drop in an email at theteam@brikoven. com so we can get in touch with you at a later stage if we do decide to give out franchisees.", "link": ""},
                    {"id": "157", "context": "8105462986 I'm sorry, but I don't have access to check your loyalty points. You can visit our website or contact our team at [theteam@brikoven. com](mailto: jtheteam@brikoven. com) for assistance with your loyalty points.", "link": ""},
                    {"id": "73", "context": "Order online Thank you for your trust in us! We are available on all major platforms: [Order online Order directly from our website](https://orders.brikoven.com), [Order from Swiggy](https://www.swiggy.com/direct/brand/7389?source=swiggy-direct&subSource=generic), [Order from Zomato](https://www.zomato.com/bangalore/delivery?chain=18224650)", "link": ""},
                    {"id": "57", "context": "Order online Order directly from us - [Order from website](orders.brikoven.com), \n[Order from Swiggy](https://www.swiggy.com/direct/brand/7389?source=swiggy-direct&subSource=generic), \n[Order from zomato](https://www.zomato.com/bangalore/delivery?chain=18224650)", "link": ""},
                    {"id": "11", "context": "I would like to understand what the closest location for me would be Unfortunately, I don't have access to your current location! We have many stores all over Bangalore! Just scroll to the bottom of our homepage to find the exact locations. [Check it out](https://www.brikoven.com/)", "link": ""},
                    {"id": "75", "context": "Order online Thank you for your trust in us!  We are available on all major platforms like Swiggy, Zomato.  You can also order directly from our website!", "link": ""},
                    {"id": "9", "context": "LOOKING FOR OUR STORE? \r We are available at the following locations in Bangalore:\r BRIK OVEN PALACE ROAD (PRESTIGE TRADE TOWER)\r BRIK OVEN KANAKAPURA (FORUM MALL)\r BRIK OVEN CHURCH STREET\r BRIK OVEN KORAMANGALA\r BRIK OVEN INDIRANAGAR\r BRIK OVEN WHITEFEILD\r BRIK OVEN SARJAPUR", "link": "https://www.brikoven.com/stores-direction-info-copy"},
                    {"id": "2", "context": "Hand-Made Fresh Mozzarella cheese\r Made every morning with the freshest cows milk from a local farm in the outskirts of Bangalore.  Our chefs take extreme pride in the Mozzarella we make.  This is now available to you from any of our outlets.  You can also order through Dunzo, Swiggy & Zomato.", "link": "https://www.brikoven.com/deli"},
                    {"id": "62", "context": "is there a buy 1 get 1 offer going on Yes, we currently have a Buy One, Get One Free offer on all pizzas during our Knead for Cheese Festival! Message \"Knead\" to us on WhatsApp and find out more https://wa.me/message/I6HNGDKCG432F1", "link": ""},
                    {"id": "60", "context": "How can I sign up?  You can become a member by making purchases at any of our stores (Except Street 1522).  To know more visit www. brikoven. com/embers", "link": ""}
                ],
                "created_at": "2024-05-17T13:06:55.000000Z",
                "updated_at": "2024-05-17T13:06:55.000000Z"
            },
            {
                "id": 2,
                "response": "We're known for having the best pizza in town, with authentic wood-fired pizzas and chocolatey milkshakes. Our sourdough base and blend of cheeses make our pizzas a must-try! Customers rave about our quattro formaggi and Diavolo pizzas.",
                "source": [
                    {"id": "127", "context": "Salad: Panzanella Salad Burrata, arugula, toasted sourdough, roasted apples, basil, and red onions, for a refreshing salad experience.", "link": "https://www.brikoven.com/qr-menu"}
                ],
                "created_at": "2024-05-17T13:06:55.000000Z",
                "updated_at": "2024-05-17T13:06:55.000000Z"
            }
        ]
    }
}

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
citations_data = find_citations(example_data)
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
    try:
        save_to_json(example_data)
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
