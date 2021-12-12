import json
import pandas as pd
import os
import glob
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline, RobertaTokenizerFast, RobertaForQuestionAnswering
import numpy as np
import faiss
import pickle
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
from tika import parser
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from itertools import chain
from annoy import AnnoyIndex
import time
from text_file import PreprocessingFiles
nltk.download('punkt')
nltk.download('stopwords')

embed = hub.KerasLayer("/home/ubuntu/DocumentSearch/use_model")
faiss_model = pickle.load(open('/home/ubuntu/DocumentSearch/pkl_models/model.pickle','rb'))
faiss_index = pickle.load(open('/home/ubuntu/DocumentSearch/pkl_models/index.pickle','rb'))
faiss_df = pickle.load(open('/home/ubuntu/DocumentSearch/pkl_models/textDict.pickle','rb'))
qna_model = torch.load('/home/ubuntu/DocumentSearch/pkl_models/qnamodel')
tokenizer = torch.load('/home/ubuntu/DocumentSearch/pkl_models/tokenizer')
qna_pipeline = pipeline('question-answering', model = qna_model, tokenizer=tokenizer)

class Similarity(PreprocessingFiles):
        
    def vector_search(self, query, model, index, num_results=10):
        '''
            Searches for top 10 results in the indexer for user query
        '''
        vector = model.encode(list(query))
        D, I = index.search(np.array(vector).astype("float32"), k=num_results)
        return D, I

    def faissSimilarity(self, query):
        '''
            Calls vector_search funtion for top FAISS similarity results 
            and returns a list having results with the document name.
            Metric Used - L2 Distance
        '''
        model, index, df = faiss_model, faiss_index, faiss_df
        D, I = self.vector_search([query], model, index, num_results=3)
        faiss = [[df['Sentence'][i],df['doc_name'][i]] for i in I[0] if i <= max(df['sent_id'])]
        return faiss
    
    def ANN_similarity(self, query, text):
        '''
            ANN(Approximate Nearest Neighbour) fetches top 3 closest neighbours
            by and returns a list having results 
            with the document name.
            Metric Used - Euclidean Distance
        '''
        euclidean = AnnoyIndex(512, 'euclidean')
        euclidean.load('/home/ubuntu/DocumentSearch/pkl_models/ann_tree.ann')
        query_vector = tf.make_ndarray(tf.make_tensor_proto(embed([query]))).tolist()[0]
        euc_sim = euclidean.get_nns_by_vector(query_vector, 3, search_k=-1, include_distances=False)
        answer = [[text[t],faiss_df['doc_name'][t]] for t in euc_sim]
        return answer
    

    def bertSimilarity(self, question, processed_ans):
        '''
            Bert takes 2 arguments 
        '''
        faiss = []
        ann = []
        faiss_ans, ann_ans = processed_ans
        
        for i in faiss_ans:
            sentence = i[0]
            answer_json = qna_pipeline({'question':question,'context':sentence})
            faiss.append([answer_json['answer'], i[1]])
        
        for i in ann_ans:
            sentence = i[0]
            answer_json = qna_pipeline({'question':question,'context':sentence})
            ann.append([answer_json['answer'], i[1]])

        return faiss, ann
        
    
        
    def pred_answers(self, query):

        results = {}
        faiss = []
        ann_euc = []
        bert_faiss = []
        bert_ann = []

        for q in query:
            
            faiss_ans = self.faissSimilarity(q)
            faiss.append(faiss_ans)
            ann_ans = self.ANN_similarity(q, self.toText_())
            ann_euc.append(ann_ans)
            faiss_b, ann_b = self.bertSimilarity(q,[faiss_ans,ann_ans])
            bert_faiss.append(faiss_b)
            bert_ann.append(ann_b)
        
        results['User Query'] = query
        results['FAISS Search'] = faiss
        results['ANN Search'] = ann_euc
        results['FAISS Bert Similarity'] = bert_faiss
        results['ANN Bert Similarity'] = bert_ann

        return results
    
    def cosine_metric(self, original, predicted):
        cv = CountVectorizer(max_features=500, stop_words='english')
        answers = [original , predicted]
        vectors = cv.fit_transform(answers).toarray()
        similarity = cosine_similarity(vectors)
        cosine = float(similarity[1][0])
        return cosine

    def F1_metric(self, original, predicted):
        pred_tokens = predicted.split()
        truth_tokens = original.split()
        
        common_tokens = set(pred_tokens) & set(truth_tokens)
        
        if len(common_tokens) == 0:
            return 0.0
        
        else:
            prec = len(common_tokens) / len(pred_tokens)
            rec = len(common_tokens) / len(truth_tokens)
            f1 = float(2 * (prec * rec) / (prec + rec))
        return f1
    
    def recall_metric(self, original, predicted):
  
        answer_tokens = original.split()
        prediction_tokens = predicted.split()
        common_tokens = set(prediction_tokens) & set(answer_tokens)
        recall = len(common_tokens) / len(answer_tokens)
        return recall
    
    def cosine_stopwords(self, original, predicted):
        
        original_list = word_tokenize(original)
        predicted_list = word_tokenize(predicted)
        sw = stopwords.words('english') 
        l1 =[];l2 =[]

        # remove stop words from the string
        original_set = {w for w in original_list if not w in sw} 
        predicted_set = {w for w in predicted_list if not w in sw}
        rvector = original_set.union(predicted_set) 
        for w in rvector:
            if w in original_set: l1.append(1) # create a vector
            else: l1.append(0)
            if w in predicted_set: l2.append(1)
            else: l2.append(0)
        c = 0
        # cosine formula 
        for i in range(len(rvector)):
                c+= l1[i]*l2[i]
        cosine = c / float((sum(l1)*sum(l2))**0.5)
        return cosine
    
    def all_scores(self, original, predicted):
        cosine_score = self.cosine_metric(original, predicted)
        cosine_stopwords = self.cosine_stopwords(original, predicted)
        f1_score = self.F1_metric(original, predicted)
        recall_metric = self.recall_metric(original, predicted)
        return cosine_score, cosine_stopwords, f1_score, recall_metric
       
    def evaluation(self, query, original_answer):
        results = self.pred_answers(query)
        results['original_answer'] = original_answer
        mean_cosine, mean_cosineStop, mean_f1, mean_recall = [],[],[],[]
        for faiss_predict, ann_predict, original in zip(results['FAISS Search'], results['ANN Search'], original_answer):
            cosine_score, cosine_stopwords, f1_score, recall_score = [], [], [], []
            for faiss, ann in zip(faiss_predict, ann_predict):
                cs, cst, fs, re = self.all_scores(original, faiss[0])
                cosine_score.append(cs)
                cosine_stopwords.append(cst)
                f1_score.append(fs)
                recall_score.append(re)
            mean_cosine.append(sum(cosine_score)/ len(cosine_score))
            mean_cosineStop.append(sum(cosine_stopwords)/ len(cosine_stopwords))
            mean_f1.append(sum(f1_score)/ len(f1_score))
            mean_recall.append(sum(recall_score)/ len(recall_score))
        results['mean_cosine'] = mean_cosine
        results['mean_cosineStopwords'] = mean_cosineStop
        results['mean_f1'] = mean_f1
        results['mean_recall'] = mean_recall
        
#         for faiss_predict, ann_predict, original in zip(results['FAISS Search'], results['ANN Search'], original_answer):
#             for faiss, ann in zip(faiss_predict, ann_predict):
#                 faiss.append(self.all_scores(original, faiss[0]))
#                 ann.append(self.all_scores(original, ann[0]))
                
#         for bert_predict, original in zip(results['Bert Similarity'], original_answer):
#             for bert in bert_predict:
#                 bert.append(self.all_scores(original, bert[0]))
                
        return results