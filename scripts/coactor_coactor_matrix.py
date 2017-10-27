import os

import numpy
from config_parser import ParseConfig
from data_extractor import DataExtractor


class CoactorCoactorMatrix(object):
    """
    Class to compute the Coactor Matrix which represents the number of movies each pair of actors have acted in, together
    """
    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = os.path.join(os.path.abspath(os.path.dirname(__file__)),self.conf.config_section_mapper("filePath").get("data_set_loc"))
        self.data_extractor = DataExtractor(self.data_set_loc)

    def fetchCoactorCoactorSimilarityMatrix(self):
        """
        Creates the coactor matrix with all the actors in a given set
        :return: coactor matrix
        """
        movie_actor_df = self.data_extractor.get_movie_actor_data()
        movie_actor_set_df = movie_actor_df.groupby(['actorid'])["movieid"].apply(set).reset_index()
        num_of_actors = len(movie_actor_df.actorid.unique())
        coactor_matrix = [[0] * num_of_actors for i in range(num_of_actors)]
        for index, movie_set in zip(movie_actor_set_df.index, movie_actor_set_df.movieid):
            for index_2, movie_set_2 in zip(movie_actor_set_df.index, movie_actor_set_df.movieid):
                if index != index_2:
                    coactor_matrix[index][index_2] = len(movie_set.intersection(movie_set_2))

        numpy.savetxt("coactor_coactor_matrix.csv", coactor_matrix, delimiter=",")
        return coactor_matrix, movie_actor_set_df.actorid.unique()


if __name__ == "__main__":
    obj = CoactorCoactorMatrix()
    obj.fetchCoactorCoactorSimilarityMatrix()
