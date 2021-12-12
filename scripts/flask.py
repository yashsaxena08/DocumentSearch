from flask import Flask, request, redirect, url_for, flash, jsonify
from allClasses import Similarity
# import pickle as p

x = Similarity()

app = Flask(__name__)

@app.route('/api/', methods=['POST'])
def pred_answers():

    data = request.get_json()
    prediction = x.evaluation(data['query'], data['answer'])
#     prediction = x.pred_answers(data['query'])
    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)