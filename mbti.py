from flask import Flask, render_template, request,url_for, redirect
from waitress import serve
import string
from string import digits
#from nltk.corpus import stopwords 
#from nltk.tokenize import word_tokenize 
from spacy.lang.en import English
import re
import joblib

app = Flask(__name__)

def remove_urls (vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(str(vTEXT))

def load_model():
	global tfidf_tf,tf_model,Tfidf_vect_sn,tsvd_sn,zscore_sn,mms_sn,ch2_sn,sn_model,Tfidf_vect_jp,tsvd_jp,zscore_jp,mms_jp,ch2_jp,jp_model,Tfidf_vect_ei,tsvd_ei,zscore_ei,mms_ei,ch2_ei,ei_model
	tfidf_tf = joblib.load('model/tfidf-tf.pkl')
	tf_model=joblib.load("model/TF.m")

	Tfidf_vect_sn=joblib.load('model/tfidf-sn.pkl')
	tsvd_sn=joblib.load('model/tsvd-sn.pkl')
	zscore_sn=joblib.load('model/zscore-sn.pkl')
	mms_sn=joblib.load('model/mms-sn.pkl')
	ch2_sn=joblib.load('model/ch2-sn.pkl')
	sn_model=joblib.load("model/SN.m")

	Tfidf_vect_jp=joblib.load('model/tfidf-jp.pkl')
	tsvd_jp=joblib.load('model/tsvd-jp.pkl')
	zscore_jp=joblib.load('model/zscore-jp.pkl')
	mms_jp=joblib.load('model/mms-jp.pkl')
	ch2_jp=joblib.load('model/ch2-jp.pkl')
	jp_model=joblib.load("model/JP.m")

	Tfidf_vect_ei=joblib.load('model/tfidf-ei.pkl')
	tsvd_ei=joblib.load('model/tsvd-ei.pkl')
	zscore_ei=joblib.load('model/zscore-ei.pkl')
	mms_ei=joblib.load('model/mms-ei.pkl')
	ch2_ei=joblib.load('model/ch2-ei.pkl')
	ei_model=joblib.load("model/EI.m")

def pre_processing(sentence):
	#stop_words = set(stopwords.words('english')) 
	stop_words = set(["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"])
	nlp = English()
	tokenizer = nlp.Defaults.create_tokenizer(nlp)
	punc=string.punctuation
	sentence=sentence.lower()
	sentence=remove_urls(sentence)
	sentence=sentence.translate(str.maketrans('', '', string.punctuation))
	sentence=sentence.translate(str.maketrans('', '', digits))
	#word_tokens = word_tokenize(sentence) 
	word_token=tokenizer(sentence)
	word_tokens=[x.text for x in word_token]
	preProcessedSentence = [w for w in word_tokens if not w in stop_words]
	preProcessedSentence=' '.join(preProcessedSentence) 
	return preProcessedSentence

def pred(preProcessedSentence):
	ans=''

	Test_X_Tfidf_ei = Tfidf_vect_ei.transform([preProcessedSentence])
	Test_X_Tfidf_ei = tsvd_ei.transform(Test_X_Tfidf_ei)
	Test_X_Tfidf_ei = zscore_ei.transform(Test_X_Tfidf_ei)
	Test_X_Tfidf_ei = mms_ei.transform(Test_X_Tfidf_ei)
	Test_X_Tfidf_ei = ch2_ei.transform(Test_X_Tfidf_ei)
	y_pred = ei_model.predict(Test_X_Tfidf_ei)
	if y_pred[0]==0:
		ans+='E'
	else:
		ans+='I'

	Test_X_Tfidf_sn = Tfidf_vect_sn.transform([preProcessedSentence])
	Test_X_Tfidf_sn = tsvd_sn.transform(Test_X_Tfidf_sn)
	Test_X_Tfidf_sn = zscore_sn.transform(Test_X_Tfidf_sn)
	Test_X_Tfidf_sn = mms_sn.transform(Test_X_Tfidf_sn)
	Test_X_Tfidf_sn = ch2_sn.transform(Test_X_Tfidf_sn)
	y_pred = sn_model.predict(Test_X_Tfidf_sn)
	if y_pred[0]==0:
		ans+='S'
	else:
		ans+='N'

	Test_X_Tfidf_tf = tfidf_tf.transform([preProcessedSentence])
	y_pred = tf_model.predict(Test_X_Tfidf_tf)
	if y_pred[0]==0:
		ans+='T'
	else:
		ans+='F'

	Test_X_Tfidf_jp = Tfidf_vect_jp.transform([preProcessedSentence])
	Test_X_Tfidf_jp = tsvd_jp.transform(Test_X_Tfidf_jp)
	Test_X_Tfidf_jp = zscore_jp.transform(Test_X_Tfidf_jp)
	Test_X_Tfidf_jp = mms_jp.transform(Test_X_Tfidf_jp)
	Test_X_Tfidf_jp = ch2_jp.transform(Test_X_Tfidf_jp)
	y_pred = jp_model.predict(Test_X_Tfidf_jp)
	if y_pred[0]==0:
		ans+='P'
	else:
		ans+='J'

	return ans


@app.route('/', methods=['get', 'post'])
def index():
	if request.form.get('sentence')!=None:
		image=''
		ptype=''
		sentence = request.form.get('sentence')
		preProcessedSentence=pre_processing(sentence)
		ans=pred(preProcessedSentence)
		if(ans=='ISFP'):
			image='static/image/ISFP.svg'
			ptype='Adventurer'
		elif(ans=='ENFJ'):
			image='static/image/ENFJ.svg'
			ptype='Protagonist'
		elif(ans=='ENFP'):
			image='static/image/ENFP.svg'
			ptype='Campaigner'
		elif(ans=='ENTJ'):
			image='static/image/ENTJ.svg'
			ptype='Commander'
		elif(ans=='ENTP'):
			image='static/image/ENTP.svg'
			ptype='Debater'
		elif(ans=='ESFJ'):
			image='static/image/ESFJ.svg'
			ptype='Consul'
		elif(ans=='ESFP'):
			image='static/image/ESFP.svg'
			ptype='Entertainer'
		elif(ans=='ESTJ'):
			image='static/image/ESTJ.svg'
			ptype='Executive'
		elif(ans=='ESTP'):
			image='static/image/ESTP.svg'
			ptype='Entrepreneur'
		elif(ans=='INFJ'):
			image='static/image/INFJ.svg'
			ptype='Advocate'
		elif(ans=='INFP'):
			image='static/image/INFP.svg'
			ptype='Mediator'
		elif(ans=='INTJ'):
			image='static/image/INTJ.svg'
			ptype='Architect'
		elif(ans=='INTP'):
			image='static/image/INTP.svg'
			ptype='Logician'
		elif(ans=='ISFJ'):
			image='static/image/ISFJ.svg'
			ptype='Defender'
		elif(ans=='ISTJ'):
			image='static/image/ISTJ.svg'
			ptype='Logistician'
		elif(ans=='ISTP'):
			image='static/image/ISTP.svg'
			ptype='Virtuoso'	
		return redirect(url_for('.result',result=ans,image=image,ptype=ptype))	
	else:
		return render_template("index.html")

@app.route('/result', methods=['get', 'post'])
def result():
	result = request.args.get('result')
	image = request.args.get('image')
	ptype = request.args.get('ptype')
	return render_template('result.html',result=result,image=image,ptype=ptype)


if __name__ == '__main__':
	load_model()
	#app.run(host='127.0.0.1', port=8080, debug=True)
	serve(app, host='0.0.0.0', port=80)