from flask import Flask, render_template, request

from transformers import PegasusForConditionalGeneration, PegasusTokenizer,  pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

model_name = "vuiseng9/pegasus-arxiv"
tokenizer = AutoTokenizer.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/text-summarization', methods=["POST"])
def summarize():
    if request.method == "POST":
        inputtext = request.form["inputtext_"]

        input_text = "summarize: " + inputtext

        tokenized_text = tokenizer.encode(
            input_text, 
            return_tensors='pt', 
            max_length=512, 
            truncation=True
        ).to(device)
        
        summary_ = model.generate(tokenized_text, min_length=150, max_length=500)
        summary = tokenizer.decode(summary_[0], skip_special_tokens=True)
        
        # Clean up the summary by removing special markers
        summary = summary.replace("S>", "").replace("/S>", "").replace("*", "").strip()
        # Remove multiple spaces
        summary = " ".join(summary.split())

    return render_template("output.html", data = {"summary": summary})

if __name__ == '__main__': # It Allows You to Execute Code When the File Runs as a Script
    app.run()

