import pandas as pd
import numpy as np
import random
from random import randint

TOP_RATE = 5
r_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings = pd.read_csv("u1.base", sep='\t', names=r_cols, encoding='latin-1')

m_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url']
movies = pd.read_csv("u.item", sep='|', names=m_cols, usecols=range(5), encoding='latin-1')

u_cols = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
users = pd.read_csv('u.user', sep='|', names=u_cols, encoding='latin-1')

movie_ratings = pd.merge(movies, ratings, on='movie_id', how='inner')
movie_stats = movie_ratings.groupby('movie_id', as_index=False)['rating'].mean()
# print(movie_stats)
ratings_sorted = movie_stats.sort_values('rating', ascending=False)
# print(ratings_sorted)

movie_ratings_users = pd.merge(movie_ratings, users, on='user_id')
# print(movie_ratings_users.columns)
ratings_by_gender = movie_ratings_users.pivot_table('rating', index=['movie_id'], columns='gender', aggfunc='mean')
female_ratings_sorted = ratings_by_gender.sort_values('F', ascending=False)
male_ratings_sorted = ratings_by_gender.sort_values('M', ascending=False)

movie_mean_rating = movie_ratings.groupby('movie_id', as_index=False)['rating'].mean()
# print(movie_mean_rating)
movie_count_ratings = movie_ratings.groupby('movie_id', as_index=False)['rating'].count()
# print(movie_count_ratings)

movie_count_ratings.columns = ['movie_id', 'num_of_ratings']
movie_merge_data = pd.merge(movie_mean_rating, movie_count_ratings, on='movie_id')
movie_merge_data['popularity'] = (movie_merge_data['rating'] * movie_merge_data['num_of_ratings']) / (
        users.shape[0] * TOP_RATE)

movies_sorted_by_popularity = movie_merge_data.sort_values('popularity', ascending=False)
popularity_and_movies_names = pd.merge(movies_sorted_by_popularity, movies, on='movie_id', how='inner')
# print(popularity_with_names.columns)

r_cols = ["user_id", "movie_id", "rating", "timestamp"]
ratings_test = pd.read_csv("u1.test", sep='\t', names=r_cols, encoding='latin-1')
movie_mean_rating_test = ratings_test.groupby('movie_id', as_index=False)['rating'].mean()

""" Ex. 2.a """
train_and_test_results = pd.merge(movie_mean_rating_test, popularity_and_movies_names, on='movie_id', how='inner')
# print(train_and_test_results.columns)

MAE = np.sum(abs(train_and_test_results['rating_x'] - train_and_test_results['rating_y'])) / \
      train_and_test_results.shape[0]
print("MAE for Ex 2.a = " + str(MAE))

user_watched_movies_train = (ratings.groupby('user_id')['movie_id'].apply(list)).to_dict()
# print(user_watched_movies_train)
list_recommended_by_rank = ratings_sorted['movie_id'].tolist()
rating_recommend = pd.DataFrame(columns=["user_id", "movie_id"])
random_recommend = pd.DataFrame(columns=["user_id", "movie_id"])
counter_by_rating = 0
counter_by_random = 0
AMOUNT_OF_RECOMMENDATIONS = 20
for user in ratings_test['user_id'].unique():
    num_of_recommends_by_rating = 0
    for recommend in list_recommended_by_rank:
        if recommend not in user_watched_movies_train[user]:
            rating_recommend.loc[counter_by_rating] = [user, recommend]
            counter_by_rating += 1
            num_of_recommends_by_rating += 1
            if num_of_recommends_by_rating == AMOUNT_OF_RECOMMENDATIONS:
                break
    num_of_recommends_by_random = 0
    while num_of_recommends_by_random != AMOUNT_OF_RECOMMENDATIONS:
        random_movie = randint(1, movies.shape[0])
        if random_movie not in user_watched_movies_train[user]:
            random_recommend.loc[counter_by_random] = [user, random_movie]
            counter_by_random += 1
            num_of_recommends_by_random += 1

user_watched_movies_test = (ratings_test.groupby('user_id')['movie_id'].apply(list)).to_dict()
users_recommended_rating = (rating_recommend.groupby('user_id')['movie_id'].apply(list)).to_dict()
users_recommended_random = (random_recommend.groupby('user_id')['movie_id'].apply(list)).to_dict()
sum_all_watched_and_recommend_by_rating = 0
sum_all_watched_and_recommend_by_random = 0
sum_movies_watched = 0
for user in ratings_test['user_id'].unique():
    sum_all_watched_and_recommend_by_rating += len(
        set(user_watched_movies_test[user]) & set(users_recommended_rating[user]))
    sum_all_watched_and_recommend_by_random += len(
        set(user_watched_movies_test[user]) & set(users_recommended_random[user]))
    sum_movies_watched += len((set(user_watched_movies_test[user])))
recall_rec_by_rating = sum_all_watched_and_recommend_by_rating / sum_movies_watched
recall_rec_by_random = sum_all_watched_and_recommend_by_random / sum_movies_watched
num_of_rec = len(ratings_test['user_id'].unique())
precision_rec_by_rating = sum_all_watched_and_recommend_by_rating / (num_of_rec * AMOUNT_OF_RECOMMENDATIONS)
precision_rec_by_random = sum_all_watched_and_recommend_by_random / (num_of_rec * AMOUNT_OF_RECOMMENDATIONS)

""" Ex. 2.b """
print("Recall by rating: " + str(round(recall_rec_by_rating, 4)))
print("Recall by random: " + str(round(recall_rec_by_random, 4)))
print("Precision by rating: " + str(round(precision_rec_by_rating, 4)))
print("Precision by random: " + str(round(precision_rec_by_random, 4)))

""" Ex. 2.c """
""" Female """
ratings_and_users_test = pd.merge(ratings_test, users, on='user_id')
female_rows = ratings_and_users_test['gender'] == 'F'
female_ratings_and_users = ratings_and_users_test[female_rows]
female_ratings = pd.DataFrame(female_ratings_sorted.to_records())
recommended_by_female_sorted_lst = female_ratings['movie_id'].tolist()
recommend_by_rating_female = pd.DataFrame(columns=["user_id", "movie_id"])
recommend_by_random_female = pd.DataFrame(columns=["user_id", "movie_id"])
counter_by_rating_female = 0
counter_by_random_female = 0
for user in female_ratings_and_users['user_id'].unique():
    num_of_recommends_by_rating_female = 0
    for recommend in recommended_by_female_sorted_lst:
        if recommend not in user_watched_movies_train[user]:
            recommend_by_rating_female.loc[counter_by_rating_female] = [user, recommend]
            counter_by_rating_female += 1
            num_of_recommends_by_rating_female += 1
            if num_of_recommends_by_rating_female == AMOUNT_OF_RECOMMENDATIONS:
                break
    num_of_recommends_by_random_female = 0
    while num_of_recommends_by_random_female != AMOUNT_OF_RECOMMENDATIONS:
        random_movie = random.choice(recommended_by_female_sorted_lst)
        if random_movie not in user_watched_movies_train[user]:
            recommend_by_random_female.loc[counter_by_random_female] = [user, random_movie]
            counter_by_random_female += 1
            num_of_recommends_by_random_female += 1

users_recommended_rating_female = (recommend_by_rating_female.groupby('user_id')['movie_id'].apply(list)).to_dict()
users_recommended_random_female = (recommend_by_random_female.groupby('user_id')['movie_id'].apply(list)).to_dict()
sum_all_watched_and_recommend_by_rating_female = 0
sum_all_watched_and_recommend_by_random_female = 0
sum_movies_watched_female = 0
for user in female_ratings_and_users['user_id'].unique():
    sum_all_watched_and_recommend_by_rating_female += len(
        set(user_watched_movies_test[user]) & set(users_recommended_rating_female[user]))
    sum_all_watched_and_recommend_by_random_female += len(
        set(user_watched_movies_test[user]) & set(users_recommended_random_female[user]))
    sum_movies_watched_female += len((set(user_watched_movies_test[user])))
recall_rec_by_rating_female = sum_all_watched_and_recommend_by_rating_female / sum_movies_watched_female
recall_rec_by_random_female = sum_all_watched_and_recommend_by_random_female / sum_movies_watched_female
num_of_rec_female = len(female_ratings_and_users['user_id'].unique())
precision_rec_by_rating_female = sum_all_watched_and_recommend_by_rating_female / (
        num_of_rec_female * AMOUNT_OF_RECOMMENDATIONS)
precision_rec_by_random_female = sum_all_watched_and_recommend_by_random_female / (
        num_of_rec_female * AMOUNT_OF_RECOMMENDATIONS)

print("Recall by rating for females: " + str(round(recall_rec_by_rating_female, 4)))
print("Recall by random for females: " + str(round(recall_rec_by_random_female, 4)))
print("Precision by rating for females: " + str(round(precision_rec_by_rating_female, 4)))
print("Precision by random for females: " + str(round(precision_rec_by_random_female, 4)))

""" male """

ratings_and_users_test = pd.merge(ratings_test, users, on='user_id')
male_rows = ratings_and_users_test['gender'] == 'M'
male_ratings_and_users = ratings_and_users_test[male_rows]
male_ratings = pd.DataFrame(male_ratings_sorted.to_records())
recommended_by_male_sorted_lst = male_ratings['movie_id'].tolist()
recommend_by_rating_male = pd.DataFrame(columns=["user_id", "movie_id"])
recommend_by_random_male = pd.DataFrame(columns=["user_id", "movie_id"])
counter_by_rating_male = 0
counter_by_random_male = 0
for user in male_ratings_and_users['user_id'].unique():
    num_of_recommends_by_rating_male = 0
    for recommend in recommended_by_male_sorted_lst:
        if recommend not in user_watched_movies_train[user]:
            recommend_by_rating_male.loc[counter_by_rating_male] = [user, recommend]
            counter_by_rating_male += 1
            num_of_recommends_by_rating_male += 1
            if num_of_recommends_by_rating_male == AMOUNT_OF_RECOMMENDATIONS:
                break
    num_of_recommends_by_random_male = 0
    while num_of_recommends_by_random_male != AMOUNT_OF_RECOMMENDATIONS:
        random_movie = random.choice(recommended_by_male_sorted_lst)
        if random_movie not in user_watched_movies_train[user]:
            recommend_by_random_male.loc[counter_by_random_male] = [user, random_movie]
            counter_by_random_male += 1
            num_of_recommends_by_random_male += 1

users_recommended_rating_male = (recommend_by_rating_male.groupby('user_id')['movie_id'].apply(list)).to_dict()
users_recommended_random_male = (recommend_by_random_male.groupby('user_id')['movie_id'].apply(list)).to_dict()
sum_all_watched_and_recommend_by_rating_male = 0
sum_all_watched_and_recommend_by_random_male = 0
sum_movies_watched_male = 0
for user in male_ratings_and_users['user_id'].unique():
    sum_all_watched_and_recommend_by_rating_male += len(
        set(user_watched_movies_test[user]) & set(users_recommended_rating_male[user]))
    sum_all_watched_and_recommend_by_random_male += len(
        set(user_watched_movies_test[user]) & set(users_recommended_random_male[user]))
    sum_movies_watched_male += len((set(user_watched_movies_test[user])))
recall_rec_by_rating_male = sum_all_watched_and_recommend_by_rating_male / sum_movies_watched_male
recall_rec_by_random_male = sum_all_watched_and_recommend_by_random_male / sum_movies_watched_male
num_of_rec_male = len(male_ratings_and_users['user_id'].unique())
precision_rec_by_rating_male = sum_all_watched_and_recommend_by_rating_male / (
        num_of_rec_male * AMOUNT_OF_RECOMMENDATIONS)
precision_rec_by_random_male = sum_all_watched_and_recommend_by_random_male / (
        num_of_rec_male * AMOUNT_OF_RECOMMENDATIONS)

print("Recall by rating for males: " + str(round(recall_rec_by_rating_male, 4)))
print("Recall by random for males: " + str(round(recall_rec_by_random_male, 4)))
print("Precision by rating for males: " + str(round(precision_rec_by_rating_male, 4)))
print("Precision by random for males: " + str(round(precision_rec_by_random_male, 4)))