# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:22:09 2020

@author: A
"""
#%%
# 영화 추천 모델 (콘텐츠 기반의 필터링 )
# 장르, 평점, 투표수 
import pandas as pd
import numpy as np

movies =pd.read_csv('tmdb_5000_movies.csv')

movies.info()
movies.head(1)
#print(movies.shape)

'''
	237000000	[{"id": 28, "name": "Action"}, {"id": 12, "nam...	http://www.avatarmovie.com/	19995	[{"id": 1463, "name": "culture clash"}, {"id":...	en	Avatar	In the 22nd century, a paraplegic Marine is di...	150.437577	[{"name": "Ingenious Film Partners", "id": 289...	[{"iso_3166_1": "US", "name": "United States o...	2009-12-10	2787965087	162.0	[{"iso_639_1": "en", "name": "English"}, {"iso...	Released	Enter the World of Pandora.	Avatar	7.2	11800
'''
movies_df = movies[['id','title', 'genres', 'vote_average', 'vote_count',
                 'popularity', 'keywords', 'overview']]

pd.set_option('max_colwidth', 500) # 행 넓이를 확장하여 콘솔창에 출력되는 정보를 확장합니다.
movies_df[['genres','keywords']][:1]


from ast import literal_eval

l='[{"name": "Adventure"}]'
type(l)
lst= literal_eval(l) # literal_eval 무엇?
type(lst)
lst[0]

movies_df['genres'] = movies_df['genres'].apply(literal_eval)
movies_df['keywords'] = movies_df['keywords'].apply(literal_eval)


# x -> movies_df['genres'], 
# y -> {"id": 28, "name": "Action"}
movies_df['genres'] = movies_df['genres'].apply(lambda x : [ y['name'] for y in x]) # 영화 모든 장르명들을 저장합니다.
movies_df['keywords'] = movies_df['keywords'].apply(lambda x : [ y['name'] for y in x])
movies_df[['genres', 'keywords']][:1]

from sklearn.feature_extraction.text import CountVectorizer
# CountVectorizer를 적용하기 위해 공백문자로 word 단위가 구분되는 문자열로 변환. 
movies_df['genres_literal'] = movies_df['genres'].apply(lambda x : (' ').join(x)) # 장르의 배열의 요소를 문자열로 묶습니다.
# count_vect -> 장르명의 개수
count_vect = CountVectorizer(min_df=0, ngram_range=(1,2)) # 장르는 unique라서 min_df=0
genre_mat = count_vect.fit_transform(movies_df['genres_literal'])
print(genre_mat.toarray())
print(genre_mat.shape)

# 각각의 영화마다 다른 모든 영화피처의 코사인 유사도 행렬 
from sklearn.metrics.pairwise import cosine_similarity
genre_sim = cosine_similarity(genre_mat, genre_mat)
print(genre_sim.shape) # 모든영화와 비교 했으므로 shpae가 (4803, 4803)가 됩니다.
print(genre_sim[:2])

# 정렬
genre_sim_sorted_ind = genre_sim.argsort()[:, ::-1] #genre_sim_sorted_ind: 장르가 유사한 영화의 인덱스 행렬, argosort()[;, ::-1]을 이용하면 유사도가 높은 순(내림 차순)으로 정리된 genre_sim 객체의 비교 행 위치 인덱스 값을 추출.
print(genre_sim_sorted_ind[:1])

def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    
    # 인자로 입력된 movies_df DataFrame에서 'title' 컬럼이 입력된 title_name 값인 DataFrame추출
    title_movie = df[df['title'] == title_name]
    
    # title_named을 가진 DataFrame의 index 객체를 ndarray로 반환하고 
    # sorted_ind 인자로 입력된 genre_sim_sorted_ind 객체에서 유사도 순으로 top_n 개의 index 추출
    title_index = title_movie.index.values
    similar_indexes = sorted_ind[title_index, :(top_n)]
    
    # 추출된 top_n index들 출력. top_n index는 2차원 데이터 임. 
    # dataframe에서 index로 사용하기 위해서 1차원 array로 변경
    print(similar_indexes)
    similar_indexes = similar_indexes.reshape(-1)
    
    return df.iloc[similar_indexes]

# 장르의 유사성은 있으나 추출된 영화 목록들이 대중성이 없다.
similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
similar_movies[['title', 'vote_average']]
'''
	title	vote_average
2731	The Godfather: Part II	8.3
1243	Mean Streets	7.2
3636	Light Sleeper	5.7
1946	The Bad Lieutenant: Port of Call - New Orleans	6.0
2640	Things to Do in Denver When You're Dead	6.7
4065	Mi America	0.0
1847	GoodFellas	8.2
4217	Kids	6.8
883	Catch Me If You Can	7.7
3866	City of God	8.1
'''

# 평점이 좋은 영화 10개 추천 - 점수는 높으나 vote_count가 낮기 때문에 객관성이 없다.
movies_df[['title','vote_average','vote_count']].sort_values('vote_average', ascending=False)[:10]
'''

title	vote_average	vote_count
3519	Stiff Upper Lips	10.0	1
4247	Me You and Five Bucks	10.0	2
4045	Dancer, Texas Pop. 81	10.0	1
4662	Little Big Top	10.0	1
3992	Sardaarji	9.5	2
2386	One Man's Hero	9.3	2
2970	There Goes My Baby	8.5	2
1881	The Shawshank Redemption	8.5	8205
2796	The Prisoner of Zenda	8.4	11
3337	The Godfather	8.4	5893
'''

'''
가중 평점(Weighted Rating) = (V/(V+M)) * R + (M/(V+M)) * C
v: 개별 영화에 평점을 토표한 횟수
m: 평점을 부여하기 위한 최소 투표 회수
R: 개별 영화에 대한 평균 평점
C: 전체 영화에 대한 평균 평점
'''
# 가중치가 부여된 평점 방식
C = movies_df['vote_average'].mean() # 전체 영화의 평균 평점
m = movies_df['vote_count'].quantile(0.6) # 투표 회수에 따른 가중치 높을 수록 투표 회수가 높은 영화에 더 많은 가중 평점을 부여 
print('C:',round(C,3), 'm:',round(m,3))

percentile = 0.6
m = movies_df['vote_count'].quantile(percentile)
C = movies_df['vote_average'].mean()

def weighted_vote_average(record):
    v = record['vote_count']
    R = record['vote_average']
    
    return ( (v/(v+m)) * R ) + ( (m/(m+v)) * C )   

movies_df['weighted_vote'] = movies_df.apply(weighted_vote_average, axis=1) 


movies_df[['title','vote_average','weighted_vote','vote_count']].sort_values('weighted_vote',
                                                                          ascending=False)[:10]

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

def find_sim_movie(df, sorted_ind, title_name, top_n=10):
    title_movie = df[df['title'] == title_name] # 매개변수로 받은 제목과 같은 영화를 비교기준으로 선정
    title_index = title_movie.index.values # 선정된 영화의 인덱스를 가져옵니다.
    
    # top_n의 2배에 해당하는 쟝르 유사성이 높은 index 추출 
    similar_indexes = sorted_ind[title_index, :(top_n*2)] # 왜 2배?
    similar_indexes = similar_indexes.reshape(-1)
    # 기준 영화 index는 제외
    similar_indexes = similar_indexes[similar_indexes != title_index]
    
    # top_n의 2배에 해당하는 후보군에서 weighted_vote 높은 순으로 top_n 만큼 추출 
    return df.iloc[similar_indexes].sort_values('weighted_vote', ascending=False)[:top_n]

similar_movies = find_sim_movie(movies_df, genre_sim_sorted_ind, 'The Godfather',10)
similar_movies[['title', 'vote_average', 'weighted_vote']]

'''
	title	vote_average	weighted_vote
2731	The Godfather: Part II	8.3	8.079586
1847	GoodFellas	8.2	7.976937
3866	City of God	8.1	7.759693
1663	Once Upon a Time in America	8.2	7.657811
883	Catch Me If You Can	7.7	7.557097
281	American Gangster	7.4	7.141396
4041	This Is England	7.4	6.739664
1149	American Hustle	6.8	6.717525
1243	Mean Streets	7.2	6.626569
2839	Rounders	6.9	6.530427
'''

