'''Copy for changes'''

import json
import pandas as pd
import os
import glob
import nltk
# Used to create the dense document vectors.
import torch
import transformers
from sentence_transformers import SentenceTransformer
from transformers import BertForQuestionAnswering, AutoTokenizer, pipeline
# Used to create and store the Faiss index.
import numpy as np
import faiss
from pathlib import Path
# Used to do vector searches and display the results.

from tika import parser
import re
from itertools import chain
import time
import pickle
nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from subprocess import Popen, PIPE
# from docx import Document
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO
import sys
import docx2txt

class PreprocessingFiles():
    def pdfToText(self, pdf_path):
        '''
        Description : A custom function to convert pdf to text(string) format.
        Aurguments:
        pdf_path = Path of the resume in pdf fromat.
        '''

        rsrcmgr = PDFResourceManager()
        retstr = StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        device = TextConverter(rsrcmgr, retstr, laparams=laparams)
        fp = open(pdf_path, 'rb')
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        password = ""
        maxpages = 0
        caching = True
        pagenos = set()
        for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching,
                                      check_extractable=True):
            interpreter.process_page(page)
        text = retstr.getvalue()
        text = text.replace("\n", ' ')
        fp.close()
        device.close()
        retstr.close()
        if len(text) < 10:
            return 'null'
        else:
            return text

    def docxToText(self, docx_path):
        '''
        Description : Custom function which converts docx file to text(string) format.
        Arguments:
        docx_path = Path of the resume in docxfile format.
        '''

        txt = docx2txt.process(docx_path)
        if txt:
            return txt.replace('\t', ' ')
        else:
            return 'null'
        return None

    def text_dict(self, path):
        text_dict = {}
        docs = glob.glob(path + "/*.*")
        files = list(map(lambda x:os.path.splitext(os.path.basename(x))[0],docs))
        text = []
        for idx,file_path in enumerate(docs):
            if file_path.split('/')[-1].split('.')[-1] == 'pdf':
                string = re.sub(' +', ' ', self.pdfToText(file_path).replace('\n', '').replace('...',''))
                text = sent_tokenize(string)
            elif file_path.split('/')[-1].split('.')[-1] == 'docx' or file_path.split('/')[-1].split('.')[-1] == 'doc':
                string = re.sub(' +', ' ', self.docToText(file_path).replace('\n', '').replace('...',''))
                text = sent_tokenize(string)
            text_dict[files[idx]] = text
        return text_dict

    def filesToJSON(self, folder_path):
        current_file_names = glob.glob(folder_path + "/*.*")
        pre_existing_files = pickle.load(open('/home/ubuntu/my_project_dir/Document Search/ Notebooks and Scripts/pre_existing_files.pkl','rb'))
        docs = list(set(current_file_names)-set(pre_existing_files))
        f = open('/home/ubuntu/my_project_dir/Document Search/ Notebooks and Scripts/doc_text.json',)
        data = json.load(f)
        if len(docs)!=0:
            print(f'{len(docs)} new files found')
            file_text = self.text_dict(folder_path)
            updated = {**file_text, **data}
            with open("doc_text.json", "w") as outfile:
                json.dump(updated, outfile)
            pre_existing_files = docs
            with open('pre_existing_files.pkl', 'wb') as f:
                pickle.dump(pre_existing_files, f)
            return updated    
        else:
            print('No new files found')
            return data
        
    def toText_(self):
        presaved_json_path = '/home/ubuntu/DocumentSearch/sample_text.json'
        f = open(presaved_json_path,)
        data = json.load(f)
        text = []
        for k in data.keys():
            for t in data[k]:
                text.append(t)  
        return text
    
    def toDict(self):
        
        presaved_json_path = '/home/ubuntu/DocumentSearch/sample_text.json'
        f = open(presaved_json_path,)        
        data = json.load(f)
        return data


text_doc = PreprocessingFiles()

def toDF_(text_dict):
    df = {}
    long_lst = chain.from_iterable(text_dict.values())
    df['Sentence'] = list(long_lst)
    df['sent_id'] = list(range(len(df['Sentence'])))
    sub_lst_len = map(len,text_dict.values())
    f_name = text_dict.keys()   
    df['doc_name'] = list(chain.from_iterable(map(lambda x:[x[0]]*x[1],zip(f_name,sub_lst_len))))
    return df
    
def embeddings():
    df = toDF_(text_doc.toDict())
    model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
    embeddings = model.encode(df['Sentence'], show_progress_bar=True)
    n = embeddings.shape[1]
    index = faiss.IndexFlatL2(n)
    index = faiss.IndexIDMap(index)
    index.add_with_ids(embeddings, np.array(df['sent_id']))
    return model, index, df

def ann_indexer():
    ann_dict = {}
    eucl_index = AnnoyIndex(512, 'euclidean')
    for idx, chunk in enumerate(text_doc.toText_()):
        vector = tf.make_ndarray(tf.make_tensor_proto(embed([chunk]))).tolist()[0]
        ann_dict[idx] = chunk[0]
        eucl_index.add_item(idx, vector)

    eucl_index.build(8, n_jobs=-1)
    
    return eucl_index, ann_dict