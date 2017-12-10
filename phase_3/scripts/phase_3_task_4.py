import os

import config_parser
import data_extractor
import pandas as pd
from phase_3_task_3 import MovieLSH
from util import Util


class NearestNeighborBasedRelevanceFeedback(object):
    def __init__(self):
        self.conf = config_parser.ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = data_extractor.DataExtractor(self.data_set_loc)
        self.util = Util()
        self.movies_dict = {}
        self.movie_tag_matrix = self.get_movie_tag_matrix()
        self.task_3_input = self.data_extractor.get_lsh_details()
        self.movieLSH = MovieLSH(self.task_3_input["num_layers"], self.task_3_input["num_hashs"])
        (self.query_df, self.query_vector) = self.fetch_query_vector_from_csv()
        self.movieLSH.create_index_structure(self.task_3_input["movie_list"])

    def fetch_query_vector_from_csv(self):
        """
        fetch the previous query vector
        :return: query vector
        """
        if os.path.isfile(self.data_set_loc + "/relevance-feedback-query-vector.csv"):
            df = self.data_extractor.get_relevance_feedback_query_vector()
        else:
            df = pd.DataFrame(columns=["latent-semantic-number-" + str(num) for num in range(1, 501)])
            zero_query_vector = {}
            for num in range(1, 501):
                zero_query_vector["latent-semantic-number-" + str(num)] = 0
            df = df.append(zero_query_vector, ignore_index=True)

        return df, df.values[-1]

    def save_query_vector_to_csv(self):
        """
        append query vector to csv
        """
        new_query_point_dict = {}
        for num in range(1, 501):
            new_query_point_dict["latent-semantic-number-" + str(num)] = self.query_vector[num - 1]

        self.query_df = self.query_df.append(new_query_point_dict, ignore_index=True)
        self.query_df.to_csv(self.data_set_loc + "/relevance-feedback-query-vector.csv", index=False)

    def get_movie_tag_matrix(self):
        """
        Movie tag matrix in latent space
        :return: movie tag matrix in latent space
        """
        movie_tag_df = None
        try:
            movie_tag_df = self.data_extractor.get_movie_latent_semantics_data()
        except:
            print("Unable to find movie matrix for movies in latent space.\nAborting...")
            exit(1)

        movie_index = 0
        movie_ids_list = movie_tag_df.movieid
        for movie_id in movie_ids_list:
            self.movies_dict[movie_id] = movie_index
            movie_index += 1

        return movie_tag_df.values

    def get_feedback_data(self):
        """
        fetch relevance feedback data
        :return: relevance feedback data
        """
        data = None
        try:
            data = self.data_extractor.get_task4_feedback_data()
        except:
            print("Relevance feedback file is missing.\nAborting...")
            exit(1)

        return data

    def update_query_point(self):
        """
        update query point based on the relevance feedback
        :return: new query point
        """
        previous_query_vector = self.query_vector

        rel_query_vector = [0 for _ in range(1, 501)]
        irrel_query_vector = [0 for _ in range(1, 501)]

        feedback_data = self.get_feedback_data()
        movies_vector_length = {}
        for index, row in feedback_data.iterrows():
            movie_id = row['movie-id']
            relevancy = row['relevancy']
            if movie_id in movies_vector_length.keys():
                vector_magnitude = movies_vector_length[movie_id]
            else:
                vector_magnitude = self.util.get_vector_magnitude(self.movie_tag_matrix[self.movies_dict[movie_id]])
                movies_vector_length[movie_id] = vector_magnitude
            if relevancy == 'relevant':
                for i in range(0, 500):
                    result = self.movie_tag_matrix[self.movies_dict[movie_id]][i] / float(vector_magnitude)
                    rel_query_vector[i] += result
            elif relevancy == 'irrelevant':
                for i in range(0, 500):
                    result = self.movie_tag_matrix[self.movies_dict[movie_id]][i] / float(vector_magnitude)
                    irrel_query_vector[i] += result

        relevant_data = feedback_data[feedback_data['relevancy'] == 'relevant']
        num_of_rel_movie_records = len(relevant_data['relevancy'])
        irrelevant_data = feedback_data[feedback_data['relevancy'] == 'irrelevant']
        num_of_irrel_movie_records = len(irrelevant_data['relevancy'])

        new_query_vector = []
        for i in range(0, 500):
            result = previous_query_vector[i]
            if num_of_rel_movie_records != 0:
                result += (rel_query_vector[i] / float(num_of_rel_movie_records))
            if num_of_irrel_movie_records != 0:
                result -= (irrel_query_vector[i] / float(num_of_irrel_movie_records))
            new_query_vector.append(result)

        self.query_vector = new_query_vector
        self.save_query_vector_to_csv()

    def get_nearest_neighbours(self, n):
        """
        Obtain the nearest neighbors based on the relevance feedback
        :param n:
        :return: n nearest neighbors
        """
        self.update_query_point()
        movie_ids = self.movieLSH.query_for_nearest_neighbours(self.query_vector, n)
        return movie_ids

    def print_movie_recommendations_and_collect_feedback(self, n):
        """
        Print nearest movies and collect relevance feedback
        :param n:
        :return:
        """
        nearest_movie_ids = self.get_nearest_neighbours(n)
        self.util.print_movie_recommendations_and_collect_feedback(nearest_movie_ids, 4, None)


if __name__ == "__main__":
    nn_rel_feed = NearestNeighborBasedRelevanceFeedback()
    while True:
        n = int(input("\n\nEnter value of 'r' : "))
        nn_rel_feed.print_movie_recommendations_and_collect_feedback(n)
        confirmation = input("\n\nDo you want to continue? (y/Y/n/N): ")
        if confirmation != "y" and confirmation != "Y":
            break
