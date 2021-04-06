# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:17:29 2020

@author: A
"""
#%%
# 파이썬 버전 확인
import sys
print(sys.version)


#%%
from sklearn.feature_extraction.text import CountVectorizer
corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()
#각단어의 빈도수
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
#각단어
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
#%%
from sklearn.feature_extraction.text import CountVectorizer

text=["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words="english") # 불용어가 english 무슨의미?
print(vect.fit_transform(text).toarray())
print(vect.vocabulary_)
#%%
'''
# nltk data install
import nltk
nltk.download() 
'''
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

text=["Family is not an important thing. It's everything."]
sw = stopwords.words("english")
vect = CountVectorizer(stop_words =sw)
print(vect.fit_transform(text).toarray()) 
print(vect.vocabulary_)
#%%
'''
tf(d,t) : 특정 문서 d에서의 특정 단어 t의 등장 횟수.

df(t) : 특정 단어 t가 등장한 문서의 수.

TF-IDF는 모든 문서에서 자주 등장하는 단어는 중요도가 낮다고 판단하며, 
특정 문서에서만 자주 등장하는 단어는 중요도가 높다고 판단합니다. 
TF-IDF 값이 낮으면 중요도가 낮은 것이며, TF-IDF 값이 크면 중요도가 큰 것입니다. 
'''
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록한다.
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
#%%
from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'you know I want your love',
    'I like you',
    'what should I do ',    
]
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray()) # 출력되는 수치는 TF? IDF?
print(tfidfv.vocabulary_)

#%%
corpus = [
    '이순신',
    '이순신 은 문반 가문 출신 으로  한성 에서 태어났 다 .',
    '삼도 수군 통제사 로 일본 해군 과 연전 연승 하 였 고 이후 성웅 으로 추앙 받 고 있 다  .',    
]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray())
print(tfidfv.vocabulary_)

from sklearn.feature_extraction.text import CountVectorizer
vector = CountVectorizer()
print(vector.fit_transform(corpus).toarray()) # '이순신'의 빈도수가 2가 아니라 왜 1임? -> corpus[0]은 '이순신' 입니다. 즉, 리스트의 요소별로 '이순신' 형태소는 1번만 나옵니다.
print(vector.vocabulary_)
#%%
from konlpy.tag import Okt
import re  
okt = Okt()  
token="정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다." 
corpus = re.sub("\.", '', token) # 정규표현식을 사용하여 '.' 제거
corpus = okt.morphs(corpus) # 형태소 추출
print(corpus)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidfv = TfidfVectorizer().fit(corpus)
print(tfidfv.transform(corpus).toarray()) # 왜 1이 출력됨?
print(tfidfv.vocabulary_)
#%%
from konlpy.tag import Okt
import re  
okt = Okt()  
token="정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다." 
corpus = re.sub("\.", '', token) # 정규표현식을 사용하여 '.' 제거

okt.nouns(corpus) # 명사 추출
#%%
import pandas as pd

train_df = pd.read_csv('ratings_train.txt', sep = '\t')
train_df.head(3)
train_df['label'].value_counts()
train_df.info()

import re
# null값을 공백으로 변경
train_df = train_df.fillna('')
# 정규표현식을 사용하여 숫자를 공백으로 변경(정규 표현식으로 \d는 숫자입니다.)
train_df['document'] = train_df['document'].apply(lambda x : re.sub(r'\d+', '', x))


test_df = pd.read_csv('ratings_test.txt', sep = '\t')
test_df.info()

# null값을 공백으로 변경
test_df = test_df.fillna('')
# 정규표현식을 사용하여 숫자를 공백으로 변경(정규 표현식으로 \d는 숫자입니다.)
test_df['document'] = test_df['document'].apply(lambda x : re.sub(r'\d+', '', x))

# id 컬럼 삭제
train_df.drop('id', axis = 1, inplace = True)
test_df.drop('id', axis = 1, inplace = True)


from konlpy.tag import Okt
okt = Okt()  
def tw_tokenizer(text): # 매개변수로 전달된 text를 형태소로 토근화하여 list 반환
    tokens_ko = okt.morphs(text)
    return tokens_ko

from sklearn.feature_extraction.text import TfidfVectorizer
# min_df = 3 3번 이상의 빈도수를 가지는 단어를 추출  
# max_df = 0.9 최대 빈도수(0.9 -> 90% 추출)
tfidf_vect = TfidfVectorizer(tokenizer = tw_tokenizer,  
                             ngram_range = (1, 2), 
                             min_df = 3,                 
                             max_df = 0.7)
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train = tfidf_vect.transform(train_df['document'])


# 모델 선택                             
from sklearn.linear_model import LogisticRegression
lg_clf = LogisticRegression(random_state=0) # Logistic Regression 을 이용하여 감성 분석 Classification 수행. 

from sklearn.model_selection import GridSearchCV
params = { 'C': [1 ,3.5, 4.5, 5.5, 10 ] }
grid_cv = GridSearchCV(lg_clf, 
                       param_grid=params, 
                       cv=3,
                       scoring='accuracy', 
                       verbose=1 ) # Parameter C 최적화를 위해 GridSearchCV 를 이용.
grid_cv.fit(tfidf_matrix_train , train_df['label'] )
print(grid_cv.best_params_ , round(grid_cv.best_score_,4))



# 학습 데이터를 적용한 TfidfVectorizer를 이용하여 테스트 데이터를 TF-IDF 값으로 Feature 변환함. 
tfidf_matrix_test = tfidf_vect.transform(test_df['document'])

# classifier 는 GridSearchCV에서 최적 파라미터로 학습된 classifier를 그대로 이용
best_estimator = grid_cv.best_estimator_
preds = best_estimator.predict(tfidf_matrix_test)

# 모델 예측
from sklearn.metrics import accuracy_score
print('Logistic Regression 정확도: ',accuracy_score(test_df['label'],preds))
#%%
'''
*20 뉴스그룹 분류

텍스트분류:특정문서의 분류학습하여 다른문서의 분류를 예측

scikit-learn을 이용하여 뉴스 그룹 메시지를 분류하는 예제를 실행한다. 
이용하고자 하는 데이터 셋은 정치, 종교, 스포츠, 과학과 같은 20개의 다른 주제로 
19,000개 뉴스 그룹 메시지를 포함한다.
20 newsgroup 데이터로 문서 분류 
'''

#텍스트 정규화
from sklearn.datasets import fetch_20newsgroups

# subset='train'으로 학습용(Train) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
'''
뉴스구룹제목 작성자,이메일 다양한 정보들이 타깃클래스와 유사정보포함되므로 예측 성능이 좋아지므로 
headers', 'footers' ,quotes제외하고 내용으로만 어떤 뉴스구룹에 속하는지 분석한다
topic:: Recommendation

  When evaluating text classifiers on the 20 Newsgroups data, you
  should strip newsgroup-related metadata. In scikit-learn, you can do this by
  setting ``remove=('headers', 'footers', 'quotes')``. The F-score will be
  lower because it is more realistic.
'''
train_news= fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'), random_state=156)
x_train = train_news.data
y_train = train_news.target
print(type(x_train))

# subset='test'으로 테스트(Test) 데이터만 추출, remove=('headers', 'footers', 'quotes')로 내용만 추출
test_news= fetch_20newsgroups(subset='test',remove=('headers', 'footers','quotes'),random_state=156)
x_test = test_news.data
y_test = test_news.target

import numpy as np
x_test[0]
y_test[0]
np.unique(y_test)
#array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
#       17, 18, 19])
print('학습 데이터 크기 {0} , 테스트 데이터 크기 {1}'.format(len(train_news.data) , len(test_news.data)))

'''
1. Count Vectorization 피처 벡터화 변환과 LogisticRegression머신러닝 모델 학습/예측/평가
'''
from sklearn.feature_extraction.text import CountVectorizer
# Count Vectorization으로 feature extraction 변환 수행. 
cnt_vect = CountVectorizer() # CountVectorizer -> 문자의 개수를 카운팅하여 숫자로 변
cnt_vect.fit(x_train)
x_train_cnt_vect = cnt_vect.transform(x_train)

# 학습 데이터로 fit( )된 CountVectorizer를 이용하여 테스트 데이터를 feature extraction 변환 수행. 
x_test_cnt_vect = cnt_vect.transform(x_test)
#Shape: (11314, 101631)
print('학습 데이터 Text의 CountVectorizer Shape:',x_train_cnt_vect.shape)

'''
# KMeans으로 x_train_cnt_vect을  20개 집합으로 군집화
from sklearn.cluster import KMeans
# 20개 집합으로 군집화, 예제를 위해 동일한 클러스터링 결과 도출용 random_state=0 
km_cluster = KMeans(n_clusters=20, max_iter=10000, random_state=0)
km_cluster.fit(x_train_cnt_vect)
cluster_label = km_cluster.labels_ #개별데이터 분류, 분류된 그룹 레이블 0에서 19
cluster_centers = km_cluster.cluster_centers_ # 중심점(그룹별 피처들 중심좌표 )
np.unique(cluster_label)

import pandas as pd
x_train_df=pd.DataFrame(x_train,columns=['text'])#x_train의 DataFrame 생성
x_train_df['cluster_label']=cluster_label  #x_train_df에레 이블 추가
x_train_df['cluster_label'].value_counts()
x_train_df[x_train_df['cluster_label']==16].sort_values(by='text') #레이블16 행들 

'''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# LogisticRegression을 이용하여 학습/예측/평가 수행. (max_iter 경고 꼭 확인)
lr_clf = LogisticRegression()
lr_clf.fit(x_train_cnt_vect , y_train)
pred = lr_clf.predict(x_test_cnt_vect)
#0.607
print('CountVectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test,pred)))


#CountVectorizer,TfidfVectorizer
# stop words 필터링을 추가하고 ngram을 기본(1,1)에서 (1,2)로 변경하여 Feature Vectorization 적용.
#전체문서에 걸쳐 300개 이하로 나타나는 단어만 피처로 추출
#너무 높은 빈도수 단어는 너무 낮은 빈도수 단어는 비중요 불용어로 간주 
#min_df :  3이면 3이하로 나타나는 단어는 피처로 미추출
#max_df=2000 :가장 높은 빈도수 단어 순으로 정렬해 2000개 까지만 피처로 추출
#ngram_range : (min_n, max_n)  n-그램 범위 
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), max_df=300 )
tfidf_vect.fit(x_train)
x_train_tfidf_vect = tfidf_vect.transform(x_train)
x_test_tfidf_vect = tfidf_vect.transform(x_test)

lr_clf = LogisticRegression()
lr_clf.fit(x_train_tfidf_vect , y_train)
pred = lr_clf.predict(x_test_tfidf_vect)
print('TF-IDF Vectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred)))

from sklearn.model_selection import GridSearchCV

#3. GridSearchCV
# 최적 C 값 도출 튜닝 수행. CV는 3 Fold셋으로 설정. 
params = { 'C':[0.01, 0.1, 1, 5, 10]}
grid_cv_lr = GridSearchCV(lr_clf ,param_grid=params , cv=3 , scoring='accuracy' , verbose=1 )
grid_cv_lr.fit(x_train_tfidf_vect , y_train)
print('Logistic Regression best C parameter :',grid_cv_lr.best_params_ )

# 최적 C 값으로 학습된 grid_cv로 예측 수행하고 정확도 평가. 
pred = grid_cv_lr.predict(x_test_tfidf_vect)
print('TF-IDF Vectorized Logistic Regression 의 예측 정확도는 {0:.3f}'.format(accuracy_score(y_test ,pred))) # TF-IDF Vectorized Logistic Regression 의 예측 정확도는 0.701
#%%
'''
문서의 유사도를 구합니다.
'''
import numpy as np

x_train_cnt_vect_A = None # 문서 1
x_train_cnt_vect_B = None # 문서 2

# 사용자 정의 함수 사용
from numpy import dot
from numpy.linalg import norm
import numpy as np
def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))

#cos_sim(x_train_cnt_vect_A, x_train_cnt_vect_B)

doc1=np.array([0,1,1,1])
doc2=np.array([1,0,1,1])
doc3=np.array([2,0,2,2])

print(cos_sim(doc1, doc2)) #문서1과 문서2의 코사인 유사도
print(cos_sim(doc1, doc3)) #문서1과 문서3의 코사인 유사도
print(cos_sim(doc2, doc3)) #문서2과 문서3의 코사인 유사도


# sklearn 사용
from sklearn.metrics.pairwise import cosine_similarity
#cosine_similarity(x_train_cnt_vect_A, x_train_cnt_vect_B)
print(cosine_similarity([doc1], [doc2])) # 2차 배열을 매개변수로 사용합니다.
print(cosine_similarity([doc1], [doc3]))
print(cosine_similarity([doc2], [doc3])) # 1이 출력되는 이유 -> 코사인 유사도는 유사도를 구할 때, 벡터의 크기가 아니라 벡터의 방향(패턴)에 초점을 두기 때문입니다.(같은 위치에 같은 단어가 존재 하는지 파악합니다.)
#%%
from lightgbm import LGBMClassifier # -> 분류 예측
from lightgbm import LGBMRegressor # -> 회귀 예























