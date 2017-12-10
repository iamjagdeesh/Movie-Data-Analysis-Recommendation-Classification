import argparse
import operator
from collections import Counter

import numpy
from util import Util


class UserMovieRecommendation(object):
    def __init__(self, user_id):
        self.util = Util()
        self.genre_data = self.util.genre_data
        self.user_id = user_id
        self.watched_movies = self.util.get_all_movies_for_user(self.user_id)
        self.model_movies_dict = {}

    def get_movie_movie_matrix(self, model):
        """
        Finds movie_tag matrix and returns movie_movie_similarity matrix
        :param model:
        :return: movie_movie_similarity matrix
        """
        movie_latent_matrix = None
        movies = None
        if model == "LDA":
            movie_tag_data_frame = self.genre_data
            tag_df = movie_tag_data_frame.groupby(['movieid'])['tag_string'].apply(list).reset_index()
            movies = tag_df.movieid.tolist()
            movies_tags_list = list(tag_df.tag_string)
            (U, Vh) = self.util.LDA(movies_tags_list, num_topics=10, num_features=len(self.genre_data.tag_string.unique()))
            movie_latent_matrix = self.util.get_doc_topic_matrix(U, num_docs=len(movies), num_topics=10)
        elif model == "SVD" or model == "PCA":
            movie_tag_frame = self.util.get_movie_tag_matrix()
            movie_tag_matrix = movie_tag_frame.values
            movies = list(movie_tag_frame.index.values)
            if model == "SVD":
                (U, s, Vh) = self.util.SVD(movie_tag_matrix)
                movie_latent_matrix = U[:, :10]
            else:
                (U, s, Vh) = self.util.PCA(movie_tag_matrix)
                tag_latent_matrix = U[:, :10]
                movie_latent_matrix = numpy.dot(movie_tag_matrix, tag_latent_matrix)
        elif model == "TD":
            tensor = self.fetch_movie_genre_tag_tensor()
            factors = self.util.CPDecomposition(tensor, 10)
            movies = self.genre_data["movieid"].unique()
            movies.sort()
            movie_latent_matrix = factors[0]
        elif model == "PageRank":
            movie_tag_frame = self.util.get_movie_tag_matrix()
            movie_tag_matrix = movie_tag_frame.values
            movies = list(movie_tag_frame.index.values)
            movie_latent_matrix = movie_tag_matrix
        latent_movie_matrix = movie_latent_matrix.transpose()
        movie_movie_matrix = numpy.dot(movie_latent_matrix, latent_movie_matrix)

        return movies, movie_movie_matrix

    def compute_pagerank(self):
        """
        Function to prepare data for pageRank and calling pageRank method
        :return: list of (movie,weight) tuple
        """
        (movies, movie_movie_matrix) = self.get_movie_movie_matrix("PageRank")
        seed_movies = self.watched_movies

        return self.util.compute_pagerank(seed_movies, movie_movie_matrix, movies)

    def get_recommendation(self, model):
        """
        Function to recommend movies for a given user_id based on the given model
        :param user_id:
        :param model:
        :return: list of movies for the given user as a recommendation
        """
        recommended_movies = []
        if len(self.watched_movies) == 0:
            print("THIS USER HAS NOT WATCHED ANY MOVIE.\nAborting...")
            exit(1)
        if model == "PageRank":
            recommended_dict = self.compute_pagerank()
            for movie_p, weight_p in recommended_dict:
                if len(recommended_movies) == 5:
                    break
                if movie_p not in self.watched_movies:
                    recommended_movies.append(movie_p)
        elif model == "Combination":
            return self.get_combined_recommendation()
        elif model == "SVD" or model == "PCA" or model == "LDA" or model == "TD":
            (movies, movie_movie_matrix) = self.get_movie_movie_matrix(model)
            movie_row_dict = {}
            for i in range(0, len(movies)):
                if movies[i] in self.watched_movies:
                    movie_row_dict[movies[i]] = movie_movie_matrix[i]
            distribution_list = self.util.get_distribution_count(self.watched_movies, 5)
            index = 0
            for movie in self.watched_movies:
                movie_row = movie_row_dict[movie]
                labelled_movie_row = dict(zip(movies, movie_row))
                num_of_movies_to_pick = distribution_list[index]
                for each in self.watched_movies:
                    del labelled_movie_row[each]
                for each in recommended_movies:
                    del labelled_movie_row[each]
                labelled_movie_row_sorted = sorted(labelled_movie_row.items(), key=operator.itemgetter(1), reverse=True)
                labelled_movie_row_sorted = labelled_movie_row_sorted[0:num_of_movies_to_pick]
                for (m, v) in labelled_movie_row_sorted:
                    recommended_movies.append(m)
                if len(recommended_movies) == 5:
                    break
                index += 1

        return recommended_movies

    def fetch_movie_genre_tag_tensor(self):
        """
        Create Movie Genre Tag tensor
        :return: tensor
        """
        movie_list = self.genre_data["movieid"].unique()
        movie_list.sort()
        movie_count = 0
        movie_dict = {}
        for element in movie_list:
            movie_dict[element] = movie_count
            movie_count += 1

        genre_list = self.genre_data["genre"].unique()
        genre_list.sort()
        genre_count = 0
        genre_dict = {}
        for element in genre_list:
            genre_dict[element] = genre_count
            genre_count += 1

        user_df = self.genre_data[self.genre_data['movieid'].isin(self.watched_movies)]
        tag_list = user_df["tag_string"].unique()
        tag_list.sort()
        tag_count = 0
        tag_dict = {}
        for element in tag_list:
            tag_dict[element] = tag_count
            tag_count += 1

        tensor = numpy.zeros((movie_count, genre_count, tag_count))

        for index, row in self.genre_data.iterrows():
            movie = row["movieid"]
            genre = row["genre"]
            tag = row["tag_string"]
            if genre not in genre_list or tag not in tag_list:
                continue
            movie_id = movie_dict[movie]
            genre_name = genre_dict[genre]
            tag_name = tag_dict[tag]
            tensor[movie_id][genre_name][tag_name] = 1

        return tensor

    def get_combined_recommendation(self):
        """
        Function to combine recommendations from all models based on frequency of appearance and order
        :param user_id:
        :return: list of recommended movies
        """
        recommended_movies = []
        model_list = ["SVD","LDA","PCA","PageRank","TD"]
        models_present = self.model_movies_dict.keys()
        models_absent = list(set(model_list) - set(models_present))
        for model in models_absent:
            self.model_movies_dict[model] = self.get_recommendation(model)
        model_movies_list = list(self.model_movies_dict.values())
        movie_dict = Counter()
        for movie_list in model_movies_list:
            for i in range(0, len(movie_list)):
                movie_dict[movie_list[i]] += 1 + (len(movie_list) - i) * 0.2
        movie_dict_sorted = sorted(movie_dict.items(), key=operator.itemgetter(1), reverse=True)
        movie_dict_sorted = movie_dict_sorted[0:5]
        for (m, v) in movie_dict_sorted:
            recommended_movies.append(m)

        return recommended_movies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='phase_3_task_1.py user_id model',
    )
    parser.add_argument('user_id', action="store", type=int)
    parser.add_argument('model', action="store", choices=['SVD', 'PCA', 'LDA', 'TD', 'PageRank', 'Combination'])
    ip = vars(parser.parse_args())
    user_id = ip['user_id']
    model = ip['model']
    obj = UserMovieRecommendation(user_id=user_id)
    if model not in obj.model_movies_dict.keys():
        recommended_movies = obj.get_recommendation(model)
        obj.model_movies_dict[model] = recommended_movies
    else:
        recommended_movies = obj.model_movies_dict[model]
    obj.util.print_movie_recommendations_and_collect_feedback(recommended_movies, 1, user_id)
    while True:
        confirmation = input("\n\nAre you done checking recommendation for all models? (y/Y/n/N): ")
        if confirmation == "y" or confirmation == "Y":
            break
        model = input("\n\nPlease enter the next model you want to use for recommendation: ")
        if model not in obj.model_movies_dict.keys():
            recommended_movies = obj.get_recommendation(model)
            obj.model_movies_dict[model] = recommended_movies
        else:
            recommended_movies = obj.model_movies_dict[model]
        obj.util.print_movie_recommendations_and_collect_feedback(recommended_movies, 1, user_id)
