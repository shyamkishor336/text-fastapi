# -*- coding: utf-8 -*-
"""
Created on Mon Mar 04 21:40:41 2024
@author: Shyam Kishor Pandit(PUR076BEI038)
"""

# 1. Library imports
import uvicorn
from fastapi import FastAPI, File, UploadFile
import numpy as np
import pytesseract
from PIL import Image
from pydantic import BaseModel
import nltk
import io
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import networkx as nx
nltk.download('punkt')
nltk.download('stopwords')
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline


# Define the path to your checkpoint directory
model_checkpoint = "E:/Downloads/checkpoint/checkpoint"

# Load the tokenizer and model from the checkpoint directory
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

# Create a summarization pipeline using the loaded model
pipe = pipeline('summarization', model=model, tokenizer=tokenizer)

# Define generation arguments
gen_kwargs = {'length_penalty': 0.6, 'num_beams': 8, 'max_length': 128}
# 2. Create the app object



def sentence_similarity(sent1, sent2):
    stop_words = set(stopwords.words('english'))
    words1 = [word.lower() for word in sent1 if word.isalnum() and word.lower() not in stop_words]
    words2 = [word.lower() for word in sent2 if word.isalnum() and word.lower() not in stop_words]
    
    all_words = list(set(words1 + words2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for word in words1:
        vector1[all_words.index(word)] += 1
    
    for word in words2:
        vector2[all_words.index(word)] += 1

    return 1 - cosine_distance(vector1, vector2)

def build_similarity_matrix(sentences):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j])
    
    return similarity_matrix

def generate_impsent(document, sent_percentage=0.6):
    sentences = sent_tokenize(document)
    num_sentences = int(len(sentences) * sent_percentage)
    num_sentences = max(num_sentences, 1)  # Ensure at least one sentence is selected

    similarity_matrix = build_similarity_matrix(sentences)

    nx_graph = nx.from_numpy_array(similarity_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

    sent_sentences = [sentence for _, sentence in ranked_sentences[:num_sentences]]
    sent = ' '.join(sent_sentences)
    pipe_out=pipe(sent, **gen_kwargs)
    return pipe_out


app = FastAPI()


@app.get('/')
def index():
    return {'message': 'Hello, World'}

   
@app.get('/summarize')
def text_summarize(text: str):
    result = generate_impsent(text)
    summarized_text = result[0]['summary_text']
    percent =(len(summarized_text)/len(text))*100
    return {
        'summary': summarized_text,
        'percent': f"{percent:.2f}"
    }

@app.post('/extract')
async def text_summarize(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    extracted_text = pytesseract.image_to_string(image)
    return {
        'text': extracted_text,

    }

# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
#pip install -r requirements.txt     