from flask import Flask, render_template, request, jsonify
from transformers import BertTokenizer, BertForSequenceClassification  
import torch

path_to_model = './nlp_project_sentiment_analysis_bert-acc-1.0'

tokenizer = BertTokenizer.from_pretrained(path_to_model)
model = BertForSequenceClassification.from_pretrained(path_to_model)

device = torch.device('cpu')
model.to(device)

labels = ["Negative", "Neutral", "Positive"]

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    print("webpage loaded")
    return render_template('index.html', sentiment='')

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    user_input = data.get('text', '')

    print(f"Received input: {user_input}")

    model.eval()
    input_encodings = tokenizer(user_input, padding=True, truncation=True, return_tensors='pt', max_length=14)

    input_ids = input_encodings['input_ids'].to(device)
    attention_mask = input_encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)  # Passing the input to the model
        logits = outputs.logits
    
    predicted_class_index = torch.argmax(logits, dim=1).item()
    predicted_sentiment = labels[predicted_class_index]
    return jsonify({'sentiment' : predicted_sentiment})

if __name__ == '__main__':
    app.run(debug=True)
