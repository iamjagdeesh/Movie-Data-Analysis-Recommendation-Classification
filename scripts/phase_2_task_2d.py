import numpy as np

from config_parser import ParseConfig
from data_extractor import DataExtractor
from util import Util


class TagMovieRatingTensor(object):
    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.max_ratings = 5
        self.ordered_ratings = [0, 1, 2, 3, 4, 5]
        self.ordered_movie_names = []
        self.ordered_tag_names = []
        self.print_list = ["\n\nFor Tags:", "\n\nFor Movies:", "\n\nFor Ratings:"]
        self.util = Util()
        self.tensor = self.fetchTagMovieRatingTensor()
        self.factors = self.util.CPDecomposition(self.tensor, 5)

    def fetchTagMovieRatingTensor(self):
        """
        Create tag movie rating tensor
        :return: tensor
        """
        mltags_df = self.data_extractor.get_mltags_data()

        tag_id_list = mltags_df["tagid"]
        tag_id_count = 0
        tag_id_dict = {}
        for element in tag_id_list:
            if element in tag_id_dict.keys():
                continue
            tag_id_dict[element] = tag_id_count
            tag_id_count += 1
            name = self.util.get_tag_name_for_id(element)
            self.ordered_tag_names.append(name)

        movieid_list = mltags_df["movieid"]
        movieid_count = 0
        movieid_dict = {}
        for element in movieid_list:
            if element in movieid_dict.keys():
                continue
            movieid_dict[element] = movieid_count
            movieid_count += 1
            name = self.util.get_movie_name_for_id(element)
            self.ordered_movie_names.append(name)

        tensor = np.zeros((tag_id_count, movieid_count, self.max_ratings + 1))

        for index, row in mltags_df.iterrows():
            tagid = row["tagid"]
            movieid = row["movieid"]
            avg_movie_rating = self.util.get_average_ratings_for_movie(movieid)
            for rating in range(0, int(avg_movie_rating) + 1):
                tagid_id = tag_id_dict[tagid]
                movieid_id = movieid_dict[movieid]
                tensor[tagid_id][movieid_id][rating] = 1

        return tensor

    def print_latent_semantics(self, r):
        """
                Pretty print latent semantics
                :param r:
        """
        i = 0
        for factor in self.factors:
            print(self.print_list[i])
            latent_semantics = self.util.get_latent_semantics(r, factor.transpose())
            self.util.print_latent_semantics(latent_semantics, self.get_factor_names(i))
            i += 1

    def get_factor_names(self, i):
        """
                Obtain factor names
                :param i:
                :return: factor names
        """
        if i == 0:
            return self.ordered_tag_names
        elif i == 1:
            return self.ordered_movie_names
        elif i == 2:
            return self.ordered_ratings

    def get_partitions(self, no_of_partitions):
        """
                Partition factor matrices
                :param no_of_partitions:
                :return: list of groupings
        """
        i = 0
        groupings_list = []
        for factor in self.factors:
            groupings = self.util.partition_factor_matrix(factor, no_of_partitions, self.get_factor_names(i))
            groupings_list.append(groupings)
            i += 1

        return groupings_list

    def print_partitioned_entities(self, no_of_partitions):
        """
                Pretty print groupings
                :param no_of_partitions:
        """
        groupings_list = self.get_partitions(no_of_partitions)
        i = 0
        for groupings in groupings_list:
            print(self.print_list[i])
            self.util.print_partitioned_entities(groupings)
            i += 1


if __name__ == "__main__":
    obj = TagMovieRatingTensor()
    obj.print_latent_semantics(5)
    obj.print_partitioned_entities(5)
