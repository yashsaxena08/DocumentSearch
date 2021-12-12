from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import f1_score
from jiwer import wer
from sklearn.feature_extraction.text import CountVectorizer


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
    
    return results

