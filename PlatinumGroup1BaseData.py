import pandas as pd
import regex as re
import nltk

textdatabase_df = pd.read_csv('/Users/januardopanggabean/Challenge Platinum Binar/train_preprocess.tsv.txt', sep='\t', names=['text', 'sentimentlabel01'])

#print(textdatabase_df.isna())

#print(textdatabase_df)

def lowercase(text):
    return text.lower()

def remove_unnecessary_char(text):
    text = re.sub('\n','', text)
    text = re.sub('rt','', text)
    text = re.sub('user','', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))',' ',text)
    text = re.sub('  +',' ', text)
    return text

def remove_nonaplhanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    text = re.sub('  +',' ', text) 
    return text

def textcleansing(text):
    lowertext = lowercase(text)
    lowercharfix = remove_unnecessary_char(lowertext)
    lowercharalpha = remove_nonaplhanumeric(lowercharfix)
    return lowercharalpha

textdatabase_df['text-cleansed'] = textdatabase_df['text'].apply(lambda x:textcleansing(x))

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stopwordsindo = nltk.corpus.stopwords.words('indonesian')
def stopwordscleanse(text):
    tokenizedwords= word_tokenize(text)
    filteredtokens = [tokenwords for tokenwords in tokenizedwords if tokenwords.lower() not in stopwordsindo]
    filteredtext = ' '.join(filteredtokens)
    return filteredtext

textdatabase_df['text-stopwd-clean'] = textdatabase_df['text-cleansed'].apply(lambda x:stopwordscleanse(x))
processed_data = pd.DataFrame({
                'clean_text' : textdatabase_df['text-cleansed'],
                'stopwclean_text' : textdatabase_df['text-stopwd-clean'],
                'labels' : textdatabase_df['sentimentlabel01']
                })

processed_data.to_csv("/Users/januardopanggabean/Challenge Platinum Binar/data/processed_data.csv", index=False)

def fullcleanse(text):
    phase1 = textcleansing(text)
    phase2 = stopwordscleanse(phase1)
    return phase2


#print(processed_data.head(30))
#print('Modul ini telah selesai. Lanjut ke modul selanjutnya')

#textdatabase_df.to_csv('/Users/januardopanggabean/VSCE Platinum Challenge/data/textdatamk2.csv', index=False)
