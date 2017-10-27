import numpy as np

from config_parser import ParseConfig
from data_extractor import DataExtractor
from util import Util


class ActorMovieYearTensor(object):

    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.ordered_years = []
        self.ordered_movie_names = []
        self.ordered_actor_names = []
        self.print_list = ["\n\nFor Years:", "\n\nFor Movies:", "\n\nFor Actors:"]
        self.util = Util()
        self.tensor = self.fetchActorMovieYearTensor()
        self.factors = self.util.CPDecomposition(self.tensor, 5)

    def fetchActorMovieYearTensor(self):
        """
        Create actor movie year tensor
        :return: tensor
        """
        movies_df = self.data_extractor.get_mlmovies_data()
        actor_df = self.data_extractor.get_movie_actor_data()

        movie_actor_df = actor_df.merge(movies_df, how="left", on="movieid")
        year_list = movie_actor_df["year"]
        year_count = 0
        year_dict = {}
        for element in year_list:
            if element in year_dict.keys():
                continue
            year_dict[element] = year_count
            year_count += 1
            self.ordered_years.append(element)

        movieid_list = movie_actor_df["movieid"]
        movieid_count = 0
        movieid_dict = {}
        for element in movieid_list:
            if element in movieid_dict.keys():
                continue
            movieid_dict[element] = movieid_count
            movieid_count += 1
            name = self.util.get_movie_name_for_id(element)
            self.ordered_movie_names.append(name)

        actorid_list = movie_actor_df["actorid"]
        actorid_count = 0
        actorid_dict = {}
        for element in actorid_list:
            if element in actorid_dict.keys():
                continue
            actorid_dict[element] = actorid_count
            actorid_count += 1
            name = self.util.get_actor_name_for_id(element)
            self.ordered_actor_names.append(name)

        tensor = np.zeros((year_count, movieid_count, actorid_count))

        for index, row in movie_actor_df.iterrows():
            year = row["year"]
            movieid = row["movieid"]
            actorid = row["actorid"]
            year_id = year_dict[year]
            movieid_id = movieid_dict[movieid]
            actorid_id = actorid_dict[actorid]
            tensor[year_id][movieid_id][actorid_id] = 1

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
            return self.ordered_years
        elif i == 1:
            return self.ordered_movie_names
        elif i == 2:
            return self.ordered_actor_names

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
    obj = ActorMovieYearTensor()
    obj.print_latent_semantics(5)
    obj.print_partitioned_entities(5)
