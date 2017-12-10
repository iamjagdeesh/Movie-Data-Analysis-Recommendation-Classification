import argparse
import math
import operator

import config_parser
import data_extractor
from util import Util


class ProbabilisticRelevanceFeedbackUserMovieRecommendation(object):
    def __init__(self, user_id):
        self.user_id = user_id
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.feedback_data = self.get_feedback_data()
        self.util = Util()
        self.movies_dict = {}
        self.tag_dict = {}
        self.movie_tag_matrix = self.get_movie_tag_matrix()
        self.feedback_metadata_dict = {}

    def get_feedback_data(self):
        """
        Get the relevance feedback csv
        :return: dataframe containing the relevance feedback
        """
        data = None
        try:
            data = self.data_extractor.get_task2_feedback_data()
        except:
            print("Relevance feedback information missing.\nAborting...")
            exit(1)

        user_relevancy_info = data[data["user-id"] == self.user_id]
        movies = user_relevancy_info["movie-id"].unique()
        if len(movies) == 0:
            print("No relevance feedback data available for the user " + str(self.user_id))
            print("Aborting...")
            exit(1)

        return data

    def get_movie_tag_matrix(self):
        """
        Compute the movie tag matrix
        :return: movie tag matix of 0s and 1s
        """
        movie_tag_matrix = self.util.get_movie_tag_matrix()
        movie_tag_matrix[movie_tag_matrix > 0] = 1

        movie_index = 0
        movies_list = list(movie_tag_matrix.index.values)
        for movie in movies_list:
            self.movies_dict[movie] = movie_index
            movie_index += 1

        tag_index = 0
        tags_list = list(movie_tag_matrix.columns.values)
        for tag in tags_list:
            self.tag_dict[tag] = tag_index
            tag_index += 1

        return movie_tag_matrix.values

    def get_movie_similarity(self, movie_id):
        """
        Calculate the similarity for the movie id passed as input
        :param movie_id:
        :return: similarity value
        """
        movie_index = self.movies_dict[movie_id]
        movie_tag_values = self.movie_tag_matrix[movie_index]

        similarity = 0
        for tag in self.tag_dict.keys():
            if tag in self.feedback_metadata_dict.keys():
                (p_i, u_i) = self.feedback_metadata_dict[tag]
            else:
                (p_i, u_i) = self.get_feedback_metadata(self.tag_dict[tag])
                self.feedback_metadata_dict[tag] = (p_i, u_i)
            numerator = p_i * (1 - u_i)
            denominator = u_i * (1 - p_i)
            temp = movie_tag_values[self.tag_dict[tag]] * (math.log(numerator / denominator))
            similarity += temp

        return similarity

    def get_feedback_metadata(self, tag_index):
        """
        get p_i and u_i values for the tag index passed as input
        :param tag_index:
        :return: (p_i, u_i)
        """
        user_feedback_data = self.feedback_data[self.feedback_data['user-id'] == self.user_id]
        user_relevant_data = user_feedback_data[user_feedback_data['relevancy'] == 'relevant']
        user_relevant_movies = user_relevant_data['movie-id'].unique()
        user_irrelevant_data = user_feedback_data[user_feedback_data['relevancy'] == 'irrelevant']
        user_irrelevant_movies = user_irrelevant_data['movie-id'].unique()

        R = len(user_relevant_movies)
        N = R + len(user_irrelevant_movies)

        count = 0
        for movie in user_relevant_movies:
            movie_index = self.movies_dict[movie]
            if self.movie_tag_matrix[movie_index][tag_index] == 1:
                count += 1
        r_i = count

        count = 0
        for movie in user_feedback_data['movie-id'].unique():
            movie_index = self.movies_dict[movie]
            if self.movie_tag_matrix[movie_index][tag_index] == 1:
                count += 1
        n_i = count

        numerator = r_i + 0.5
        denominator = R + 1
        p_i = numerator / float(denominator)

        numerator = n_i - r_i + 0.5
        denominator = N - R + 1
        u_i = numerator / float(denominator)

        return p_i, u_i

    def get_movie_recommendations(self):
        """
        Get top 5 movies movie recommendations based on probabilistic relevance feedback
        :return: 5 movie recommendations
        """
        movie_similarity = {}

        for movie in self.movies_dict.keys():
            movie_similarity[movie] = self.get_movie_similarity(movie)

        movie_recommendations = []
        for movie, value in sorted(movie_similarity.items(), key=operator.itemgetter(1), reverse=True):
            if len(movie_recommendations) == 5:
                break
            movie_recommendations.append(movie)

        return movie_recommendations

    def print_movie_recommendations_and_collect_feedback(self):
        """
        Display movie recommendations and take user relevance feedback
        :return:
        """
        movies = self.get_movie_recommendations()
        self.util.print_movie_recommendations_and_collect_feedback(movies, 2, self.user_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='phase_3_task_2.py user_id',
    )
    parser.add_argument('user_id', action="store", type=int)
    input = vars(parser.parse_args())
    user_id = input['user_id']
    prop_rel_feed_rec = ProbabilisticRelevanceFeedbackUserMovieRecommendation(user_id)
    prop_rel_feed_rec.print_movie_recommendations_and_collect_feedback()
