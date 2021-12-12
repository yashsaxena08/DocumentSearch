import os
from text_processing import PreprocessingFiles
from itertools import chain
from annoy import AnnoyIndex
import tensorflow as tf
import tensorflow_hub as hub

embed = hub.KerasLayer("/home/ubuntu/DocumentSearch/use_model")

text_doc = PreprocessingFiles()

def updated_dict(text_dict):
    updated_dict = {}
    long_lst = chain.from_iterable(text_dict.values())
    updated_dict['Sentence'] = list(long_lst)
    updated_dict['sent_id'] = list(range(len(updated_dict['Sentence'])))
    sub_lst_len = map(len,text_dict.values())
    f_name = text_dict.keys()   
    updated_dict['doc_name'] = list(chain.from_iterable(map(lambda x:[x[0]]*x[1],zip(f_name,sub_lst_len))))
    return updated_dict
    
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