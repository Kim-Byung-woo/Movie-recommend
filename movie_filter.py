# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 14:25:51 2020

@author: A
"""
#%%
'''
NLTK의 기능을 제대로 사용하기 위해서는 NLTK Data라는 여러 데이터를 추가적으로 설치해야 합니다.

# nltk data install
import nltk
nltk.download() 
'''
import nltk
nltk.download()



#%%
import pandas as pd
pd.set_option('display.max_colwidth', -1)
import numpy as np
import json     
import random

'''
total_literal 생성 및 tmdb_5000_movies.csv 데이터 전역변수 선언
'''
movies=pd.read_csv('tmdb_5000_movies.csv')

# 필요한 데이터 컬럼 추출
movies_df = movies[['id','title', 'genres', 'vote_average', 'vote_count',
                 'popularity', 'keywords', 'overview','production_countries','tagline', 'release_date']]

# literal_eval: # 컬럼의 데이터가 사전 형태([{'id': 28, 'name': 'Action'}....)로 되어 있어서 사전에서 특정 키값 부분(name, ..)만 추출 하기 위해 형변환 
from ast import literal_eval
movies_df['genres'] = movies_df['genres'].apply(literal_eval) 
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)
movies_df['production_countries'] = movies_df['production_countries'].apply(literal_eval)

# eval로 변환된 데이터에서 특정 키값 부분만 추출
movies_df['genres'] = movies_df['genres'].apply(lambda x : [ y['name'] for y in x])
movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [ y['name'] for y in x])
movies_df['production_countries'] = movies_df['production_countries'].apply(lambda x : [ y['iso_3166_1'] for y in x])

# # CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환. 
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x))
movies_df['keywords_literal'] = movies_df['keywords'].apply(lambda x : (' ').join(x))
movies_df['countries_literal'] = movies_df['production_countries'].apply(lambda x : (' ').join(x))

# 불용어 제거
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
stop_words = set(stopwords.words('english')) 
a = []
movies_df['tagline'].fillna('', inplace=True)
result=[]
for i in movies_df['tagline']: 
    result=[]
    word_tokens=word_tokenize(i) # ['Enter','the','World','of','Pandora','.]
    for w in word_tokens:
        if w not in stop_words:
            result.append(w)  # ['Enter','World','Pandora']
    a.append(' '.join(result))  #  a = ['Enter','World','Pandora]
movies_df['tagline'] = a

#타이틀+장르리터럴+키워드리터럴+국가+태그라인 = 토탈리터럴 컬럼 추가
movies_df['total_literal'] = movies_df['title']+' '+ \
    movies_df['genres_literal'] +' '+\
        movies_df['keywords_literal']+' '+movies_df['countries_literal']\
            +' '+movies_df['tagline']

# CountVectorizer: 단어들의 카운트(출현 빈도(frequency))로 여러 문서들을 벡터화
# 벡터화: 문자의 유사도를 컴퓨터가 인식하기 위해 문자를 좌표평면상의 숫자 형태로 바꿔주는 방법
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2)) # min_df: 최소 빈도, ngram_range: n-그램 범위로 단어를 몇 개로 토큰화 할지를 의미
#카운트벡터라이즈 군집화 개수 행렬
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.toarray())

keywords_mat = count_vect.fit_transform(movies_df['keywords_literal'])
total_mat = count_vect.fit_transform(movies_df['total_literal'])
tagline_mat = count_vect.fit_transform(movies_df['tagline'])

# 각각의 영화마다 다른 모든 영화피처의 코사인 유사도 행렬 생성
from sklearn.metrics.pairwise import cosine_similarity
genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape) # 모든영화와 비교 했으므로 shpae가 (4803, 4803)가 됩니다.

keywords_sim = cosine_similarity(keywords_mat, keywords_mat)
total_sim = cosine_similarity(total_mat, total_mat)
tagline_sim = cosine_similarity(tagline_mat, tagline_mat)

# 유사도 역정렬(유사성 높은순)
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1] # [:, ::-1]-> numpy 배열의 내림정렬입니다.
keywords_sim_sorted_ind = keywords_sim.argsort()[:, ::-1]
total_sim_sorted_ind = total_sim.argsort()[:, ::-1]


#평점방식 투표수에 가중치부여
percentile = 0.6
m = movies_df['vote_count'].quantile(percentile)
C = movies_df['vote_average'].mean()
def weighted_vote_average(record):
    v = record['vote_count']#개별영화에 대한 투표회수
    R = record['vote_average']##개별영화에 대한 평점
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )

#가중치부여된 평점 weighted_vote 컬럼 추가
movies_df['weighted_vote'] = movies_df.apply(weighted_vote_average, axis=1) 
#가중치부여된 평점 높은순
movies_df[['title','vote_average','weighted_vote',
           'vote_count']].sort_values('weighted_vote',ascending=False)[:10]
'''
	title	vote_average	weighted_vote	vote_count
1881	The Shawshank Redemption	8.5	8.396052	8205
3337	The Godfather	8.4	8.263591	5893
662	Fight Club	8.3	8.216455	9413
3232	Pulp Fiction	8.3	8.207102	8428
65	The Dark Knight	8.2	8.136930	12002
1818	Schindler's List	8.3	8.126069	4329
3865	Whiplash	8.3	8.123248	4254
809	Forrest Gump	8.2	8.105954	7927
2294	Spirited Away	8.3	8.105867	3840
2731	The Godfather: Part II	8.3	8.079586	3338
'''
#%%

def print_menu():
    print("\n1. 영화 검색")
    print('2. 통합 검색') 
    print("3. 연도별 추천")
    print("4. 최근 시청 목록") 
    print("5. 종료", end = '') 
    menu = input("메뉴 선택: ")
    return int(menu)

def print_recomand_menu():
    print('\n 보고 싶은 영화가 있으신가요?') 
    print("1. 예")
    print("2. 아니오", end = '') 
    menu = input("메뉴 선택: ")
    return int(menu)

def print_select_moive_menu():
    menu = input('1 ~ 10번의 영화 중 하나를 선택하세요: ')
    return int(menu)

# def create_total_literal():

'''
맞춤 추천 알고리즘
'''
def get_keywords(movie_id):     
    corpus = []
    # 매개변수로 최근 시청한 영화 id 가져와 해당 영화 정보를 불러와서 빈도수 측정할 문자열 생성
    for idx in range(len(movie_id)):
        df_total_literal = pd.DataFrame(movies_df['total_literal'][movies_df['id'] == movie_id[idx]])
        total_literal = df_total_literal.iloc[0, -1] # 문자열이 잘려서 출력되어서 데이터 프레임 인덱싱을 사용합니다.
        corpus.append(total_literal)
        
    # 단어 빈도수 측정
    corpus_s = ' '.join(corpus)
    
    # total literal 값을 구분자를 이용해서 list 형태로 변환
    list_corpus = []
    list_corpus = corpus_s.split(' ')
    
    # 불용어 리스트를 불러오기
    from nltk.corpus import stopwords  
    stopwords = stopwords.words('english')
    
    # 불용어 제거
    temp_X = [] 
    temp_X = [word for word in list_corpus if not word in str(stopwords)]
    list_corpus = temp_X # 원본 데이터에 적용
    
    # 특수문자 제거
    import re
    corpus_result = []
    for i in list_corpus:
        tokens = re.sub('[-=.#/?:$}]', '', i)        
        corpus_result.append(tokens)
        
    
    corpus_result = [x for x in corpus_result if x] # ''값으로 대체된 요소 삭제   
    list_corpus = corpus_result # 원본 데이터에 적용
    
    from collections import Counter
    counts = Counter(list_corpus) # 리스트의 요소들의 빈도수 확인
    most_morphs = counts.most_common(5) # 빈도수 상위 3개 추출
    
    df_morphs = pd.DataFrame(most_morphs, columns=['morphs', 'counts']) # 튜플 타입인 most_morphs를 Dataframe으로 형변환
    
    list_keywords = []
    for i in range(len(df_morphs)):
        list_keywords.append(df_morphs.iloc[i]['morphs'])
    
    print(list_keywords)
    recommand_keywords = list_keywords
    
        
    '''  
    # CountVectorizer 설정
    vector = CountVectorizer(stop_words="english") # 영어 불용어 삭제
    #print(vector.fit_transform([corpus_s]).toarray())
    #print(vector.vocabulary_)
    
    vec_arr = vector.fit_transform([corpus_s]).toarray()
    vec_list = vec_arr.tolist() # array -> list 변경
    vec_list = sum(vec_list, []) # 2차원 list -> 1차원 list 변경
    max_vec_list = sorted(vec_list, reverse = True)
    
    # 3개의 추천 키워드 생성
    recommand_keywords = []
    first_score = max_vec_list[0]
    first_voca = [vaca for vaca, index in vector.vocabulary_.items() if index == vec_list.index(first_score)]
    first_voca = first_voca[0] # list -> str로 형변환
    recommand_keywords.append(first_voca)
    
    second_score = max_vec_list[1]
    second_voca = [vaca for vaca, index in vector.vocabulary_.items() if index == vec_list.index(second_score)]
    second_voca = second_voca[0]
    recommand_keywords.append(second_voca)
    
    third_score = max_vec_list[2]
    thrid_voca = [vaca for vaca, index in vector.vocabulary_.items() if index == vec_list.index(third_score)]
    thrid_voca = thrid_voca[0]
    recommand_keywords.append(thrid_voca)
    '''

    return recommand_keywords



'''
종합 검색 알고리즘
'''
def searcher(keywords, top_n=10):  

    #검색전용 데이터프레임 생성 - 원본 손실 방지
    search_df = movies_df.iloc[0:, 0:]
    #검색어인자 받아옴
    search = keywords
    # 4803번째 행의 total_literal 컬럼에 검색어 인자 대입 
    # print('키워드 추가전 ' + '%d', len(search_df))
    search_df = search_df.append({'total_literal':search}, ignore_index=True)
    # print('키워드 추가후 ' + '%d', len(search_df))
       
    # CountVectorizer: 단어들의 카운트(출현 빈도(frequency))로 여러 문서들을 벡터화
    # 벡터화: 문자의 유사도를 컴퓨터가 인식하기 위해 문자를 좌표평면상의 숫자 형태로 바꿔주는 방법
    search_mat = count_vect.fit_transform(search_df['total_literal'])
    # 코사인유사도생성
    search_sim = cosine_similarity(search_mat, search_mat)
    '''
    test = [4, 1, 6, 19, 12, 39, 55, 13]
    np.sort(test)[::-1] # np.sort(arr)[::-1]은 내리참순으로 정렬
    '''
    # search_sim_sorted_ind: keyword값과 total_literal값이 유사한 영화의 인덱스 행렬, argosort()[;, ::-1]을 이용하면 유사도가 높은 순(내림 차순)으로 정리된 search_sim 객체의 비교 행 위치 인덱스 값을 추출.
    search_sim_sorted_ind = search_sim.argsort()[:, ::-1]
    
    # top_n의 2배에 해당하는 total_literal 유사성이 높은 index 추출 
    search_indexes = search_sim_sorted_ind[len(search_df) - 1, : (top_n * 2)] # len(search_df) - 1: 기준이 되는 total_literal을 컬럼에 검색어 인자를 제일 마지막행에 대입해서 len(search_df) - 1을 사용합니다.
    search_indexes = search_indexes.reshape(-1) # dataframe에서 index로 사용하기 위해서 1차원 array로 변경
    search_movie = search_df.iloc[search_indexes].sort_values('popularity', ascending=False)[:10]
    
    # 추천 영화 출력
    df = pd.DataFrame(search_movie[['title', 'vote_average', 'weighted_vote']])
    # 출력 하기 위해 인덱스 조정
    df = df.set_index( np.arange(1, len(df) + 1, 1))
    print(df)
    return search_movie['id']
    
'''
최근 영화 목록 출력 알고리즘
''' 
def get_recent_movie(movie_id):
    max_cnt = len(movie_id)
    max_cnt = max_cnt -1     
    df = pd.DataFrame(movies_df[['title', 'vote_average', 'weighted_vote']][movies_df['id'] == movie_id[max_cnt]])
    for idx in range(max_cnt - 1, -1, -1):        
        df = df.append(movies_df[['title', 'vote_average', 'weighted_vote']][movies_df['id'] == movie_id[idx]])      
    # 출력 하기 위해 인덱스 조정
    df = df.set_index( np.arange(1, len(df) + 1, 1))
    # 최근 영화 출력
    print(df)


# 0~4800 사이의 중복되지 않는 랜덤값 생성 하여 list에 삽입 - 영화 검색 알고리즘에 사용
def ran_num ():
    list=[]
    ran_num = random.randint(0,4800)
    for i in range(4800):
        while ran_num in list:
            ran_num = random.randint(0, 4800)
        list.append(ran_num)
    return list

'''
영화 검색 알고리즘
'''
def find_index(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name] # 매개변수로 받은 제목과 같은 영화를 비교기준으로 선정
    title_index = title_movie.index.values # 선정된 영화의 인덱스를 가져옵니다.
    
    # top_n의 2배에 해당하는 쟝르 유사성이 높은 index 추출 
    similar_indexes = sorted_ind[title_index, :(top_n*2)] # 왜 2배?
    similar_indexes = similar_indexes.reshape(-1)
    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출 
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

def year_find(year):

    # null 값 처리
    movies_df['release_date'].fillna('3000-06-01',inplace=True)  
    year_list=[]
    for i in movies_df['release_date']:
        year_list.append(int(i[:4])) # release_year 생성 

    movies_df['release_year']=year_list
    year_top10=movies_df.sort_values(by="vote_average", ascending=False).groupby("release_year").head(10) # release_year 별 vote_average에 따라 내림정렬
    
    return(year_top10[year_top10['release_year']==year][['release_year','title']])	
#%%
def run():
    input_id = input("계정 입력: ")
    fname = input_id + '.json'
     # 계정 존재 확인(첫방문/재방문 조회)
    import os.path 
    file_exists = os.path.isfile(fname)
    
    json_user_id = ''
    movie_id = []
    if file_exists == True: # 기존 고객 인경우 
        print('===== Log in Success =====')
        # 고객 정보 불러오기
        with open(fname) as json_file:
            json_data = json.load(json_file)
            json_user_id = json_data["user_id"]
            #print(f'===== Thanks for visit {json_user_id} =====')
            movie_id = json_data["movie_id"]
        
        keywords_list = get_keywords(movie_id) # 맞춤 출력 키워드 저장
        # print(keywords_list) # 맞춤 추천 키워드 출력
        keywords = ' '.join(keywords_list) # 맞춤 추천 키워드 string으로 변환
        search_list = searcher(keywords).tolist()
        while 1:   
            recom_menu = print_recomand_menu()
            if recom_menu == 2: # 시청하고 싶은 영화가 없는 경우
                print('New Recommand Moive')
                
                while 1:
                   menu = print_menu()
                   if menu == 5: # 추천 프로그램 종료
                       print('Movie Recommand Exit')
                       # 고객정보 업데이트 및 저장
                       data = {}
                       data['user_id'] = json_user_id
                       
                       ex_list = list(set(movie_id)) # 중복 제거
                       
                       # 최근 본 영화가 10개가 넘어갈 경우 가장 오래된 순으로 목록에서 삭제
                       recent_movie_cnt = len(ex_list)
                       while recent_movie_cnt > 10:
                           ex_list.pop(0)
                           recent_movie_cnt = len(ex_list)
                                              
                       data['movie_id'] = ex_list
                       with open(fname, 'w') as outfile:
                           json.dump(data, outfile, indent = 4)
                       break
                   elif menu == 4: # 최근 시청 영화 목록 출력
                       print('Recently Movie')
                       # 최근 본 영화가 10개가 넘어갈 경우 가장 오래된 순으로 목록에서 삭제
                       recent_movie_cnt = len(movie_id)
                       while recent_movie_cnt > 10:
                           movie_id.pop(0)
                           recent_movie_cnt = len(movie_id)
                           
                       get_recent_movie(movie_id)
                       pass
                   elif menu == 1: # 영화 검색
                       search_str = input('제목을 입력하시오 : ')
                       similar_movies = find_index(movies_df, genre_sim_sorted_ind, search_str)
                       print(similar_movies[['title', 'weighted_vote']])
                       
                       pass
                   elif menu == 2: # 통합 검색
                       search_str = input('검색어를 입력하시오 : ')
                       search_list = searcher(search_str).tolist()
                       
                       # 시청하고 싶은 영화 선택
                       select_moive_num = print_select_moive_menu()
                       print(f'{select_moive_num}번 영화 선택')
                       select_id = search_list[select_moive_num - 1]
                       movie_id.append(int(select_id)) # 최근 시청 영화 id 저장
                       
                       pass
                   elif menu == 3: # 연도별 추천 영화 출력
                       search_str = input('연도를 입력하시오 : ')
                       print(year_find(int(search_str)))
                       pass
                   
                   else:
                       print('잘못된 메뉴입니다. 다시 선택하세요')
                       pass
                break
            elif recom_menu == 1: # 시청하고 싶은 영화 선택
                select_moive_num = print_select_moive_menu()
                print(f'{select_moive_num}번 영화 선택')
                select_id = search_list[select_moive_num - 1]
                movie_id.append(int(select_id)) # 최근 시청 영화 id 저장
                
                while 1:
                    menu = print_menu()
                    if menu == 5: # 추천 프로그램 종료
                        print('Movie Recommand Exit')
                       # 고객정보 업데이트 및 저장
                        data = {}
                        data['user_id'] = json_user_id
                        
                        ex_list = list(set(movie_id)) # 중복 제거
                        
                        # 최근 본 영화가 10개가 넘어갈 경우 가장 오래된 순으로 목록에서 삭제
                        recent_movie_cnt = len(ex_list)
                        while recent_movie_cnt > 10:
                           ex_list.pop(0)
                           recent_movie_cnt = len(ex_list)
                        
                        data['movie_id'] = ex_list
                        with open(fname, 'w') as outfile:
                            json.dump(data, outfile, indent = 4)
                        break
                    elif menu == 4:  # 최근 시청 영화 목록 출력
                        print('Recently Movie')
                        
                        # 최근 본 영화가 10개가 넘어갈 경우 가장 오래된 순으로 목록에서 삭제
                        recent_movie_cnt = len(movie_id)
                        while recent_movie_cnt > 10:
                           movie_id.pop(0)
                           recent_movie_cnt = len(movie_id)
                        
                        get_recent_movie(movie_id)
                        pass
                    elif menu == 1: # 영화 검색
                        search_str = input('제목을 입력하시오 : ')
                        similar_movies = find_index(movies_df, genre_sim_sorted_ind, search_str)
                        print(similar_movies[['title', 'weighted_vote']])
                        pass
                    elif menu == 2: # 통합 검색
                        search_str = input('검색어를 입력하시오 : ')
                        search_list = searcher(searcher).tolist()
                       
                        # 시청하고 싶은 영화 선택
                        select_moive_num = print_select_moive_menu()
                        print(f'{select_moive_num}번 영화 선택')
                        select_id = search_list[select_moive_num - 1]
                        movie_id.append(int(select_id)) # 최근 시청 영화 id 저장
                        
                        pass
                    elif menu == 3: # 연도별 추천 영화 출력
                        search_str = input('연도를 입력하시오 : ')
                        print(year_find(int(search_str)))
                        pass
                    
                    else:
                        print('잘못된 메뉴입니다. 다시 선택하세요')
                        pass
                break
            else:
                print('잘못된 메뉴입니다. 다시 선택하세요')
                pass
    else: # 신규 고객인 경우
        print('===== Sign in Success =====')
        while 1:
            menu = print_menu()
            if menu == 5: # 프로그램 종료
                print('Movie Recommand Exit')
                # 고객정보 업데이트 및 저장
                data = {}
                data['user_id'] = json_user_id
                
                ex_list = list(set(movie_id)) # 중복 제거
                # 최근 본 영화가 10개가 넘어갈 경우 가장 오래된 순으로 목록에서 삭제
                recent_movie_cnt = len(ex_list)
                while recent_movie_cnt > 10:
                   ex_list.pop(0)
                   recent_movie_cnt = len(ex_list)
                           
                data['movie_id'] = ex_list
                with open(fname, 'w') as outfile:
                    json.dump(data, outfile, indent = 4)
                break
            elif menu == 4: # 최근 시청 영화 목록 출력
                print('최근 시청 목록이 없습니다.')
                pass
            elif menu == 2: # 통합 검색
                search_str = input('검색어를 입력하시오 : ')
                search_list = searcher(search_str).tolist()
                
                recom_menu = print_recomand_menu()
                if recom_menu == 2:
                    print('New Recommand Moive')
                    pass
                elif recom_menu == 1:  # 시청하고 싶은 영화 선택
                    select_moive_num = print_select_moive_menu()
                    print(f'{select_moive_num}번 영화 선택')
                    select_id = search_list[select_moive_num - 1]
                    movie_id.append(int(select_id)) # 최근 시청 영화 id 저장
                pass
            elif menu == 1: # 영화 검색
                search_str = input('제목을 입력하시오 : ')
                similar_movies = find_index(movies_df, genre_sim_sorted_ind, search_str)
                print(similar_movies[['title', 'weighted_vote']])
                pass
            elif menu == 3:  # 연도별 추천 영화 출력
                search_str = input('연도를 입력하시오 : ')
                print(year_find(int(search_str)))
                pass
            else:
                print('잘못된 메뉴입니다. 다시 선택하세요')
                pass

    
        
        
        
#%%
run()

 





















