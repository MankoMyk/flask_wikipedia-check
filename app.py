from flask import Flask, request, jsonify
import json
import os
import wikipedia
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def main():
    return "Text similarity web service"

def similar(str1, str2):
        documents = [str1, str2]

        # Create the Document Term Matrix
        count_vectorizer = CountVectorizer(stop_words='english')
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(documents)

        # OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
        doc_term_matrix = sparse_matrix.todense()
        df = pd.DataFrame(doc_term_matrix, 
                        columns=count_vectorizer.get_feature_names(), 
                        index=['str1', 'str2'])

        # Compute Cosine Similarity
        js = cosine_similarity(df, df)
        return js.tolist()[0][1]


@app.route('/generate', methods=['GET'])
def generate():

    input = request.values.get('input', False)
    text = request.values.get('text', False)

    page = wikipedia.summary(input)

    print(page)

    # return page

    result1 = similar(page, text)
    if result1 > 0.92:
      return 'True   '+str(result1)
    else:
      return 'False   '+str(result1)

    # return str(result1) 

    # return input # 'Hello'

    # http://localhost:5000/generate?input=Ukraine&text=blablabla