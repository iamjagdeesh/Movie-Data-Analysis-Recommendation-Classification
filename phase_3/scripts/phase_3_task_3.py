import json
import math
import os
import random
import numpy
import pandas as pd
from config_parser import ParseConfig
from scipy.spatial import distance
from util import Util
import argparse
conf = ParseConfig()

class MovieLSH():
    def __init__(self, num_layers, num_hashs):
        self.util = Util()
        self.movie_tag_df = self.util.get_movie_tag_matrix()
        self.num_layers = num_layers
        self.num_hashs = num_hashs
        self.latent_range_dict = {}
        self.lsh_points_dict = {}
        self.lsh_range_dict = {}
        self.column_groups = []
        self.U_matrix = []
        self.total_movies_considered = []
        self.movie_bucket_df = pd.DataFrame()
        self.movie_latent_df = pd.DataFrame()
        self.w_length = 0.0
        (self.U, self.s, self.Vt) = self.util.SVD(self.movie_tag_df.values)
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def assign_group(self, value):
        """
        Assigns bucket
        :param value:
        :return: bucket
        """
        if value < 0:
            return math.floor(value/self.w_length)
        else:
            return math.ceil(value / self.w_length)

    def init_lsh_vectors(self, U_dataframe):
        """
        initialize lsh vectors
        :param U_dataframe:
        """
        origin = list(numpy.zeros(shape=(1, 500)))
        for column in U_dataframe:
            self.latent_range_dict[column] = (U_dataframe[column].min(), U_dataframe[column].max())
        for i in range(0, self.num_layers * self.num_hashs):
            cur_vector_list = []
            for column in U_dataframe:
                cur_vector_list.append(random.uniform(self.latent_range_dict[column][0], self.latent_range_dict[column][1]))
            self.lsh_points_dict[i] = cur_vector_list
            self.lsh_range_dict[i] = distance.euclidean(origin, cur_vector_list)

    def project_on_hash_function(self, movie_vector, lsh_vector):
        """
        projection of movie vector on the hash fn
        :param movie_vector:
        :param lsh_vector:
        :return: projection value
        """
        movie_lsh_dot_product = numpy.dot(movie_vector, lsh_vector)
        if movie_lsh_dot_product == 0.0:
            return 0
        lsh_vector_dot_product = numpy.dot(lsh_vector, lsh_vector)
        projection = movie_lsh_dot_product/lsh_vector_dot_product*lsh_vector
        projection_magnitude = numpy.linalg.norm(projection)
        return projection_magnitude


    def LSH(self, vector):
        """
        list of buckets for the vector
        :param vector:
        :return:
        """
        bucket_list = []
        for lsh_vector in range(0, len(self.lsh_points_dict)):
            bucket_list.append(self.assign_group(self.project_on_hash_function(numpy.array(vector), numpy.array(self.lsh_points_dict[lsh_vector]))))
        return bucket_list

    def group_data(self):
        """
        groups all movies into buckets
        :return:
        """
        U_dataframe = pd.DataFrame(self.U)
        U_dataframe = U_dataframe[U_dataframe.columns[0:500]]
        self.init_lsh_vectors(U_dataframe)
        self.w_length = min(self.lsh_range_dict.values()) / float(100)
        self.column_groups = {vector: [] for vector in self.lsh_range_dict.keys()}
        bucket_matrix = numpy.zeros(shape=(len(self.U), len(self.lsh_points_dict)))
        self.U_matrix = U_dataframe.values

        for movie in range(0, len(self.U_matrix)):
            bucket_matrix[movie] = self.LSH(self.U_matrix[movie])

        movie_df = self.movie_tag_df.reset_index()
        movie_id_df = pd.DataFrame(movie_df["movieid"])
        self.movie_latent_df = U_dataframe.join(movie_id_df, how="left")
        self.movie_latent_df.to_csv(os.path.join(self.data_set_loc, "movie_latent_semantic.csv"), index=False)
        return pd.DataFrame(bucket_matrix).join(movie_id_df, how="left")

    def index_data(self, df):
        """
        Assigns buckets to movies in the dataframe
        :param df:
        :return:
        """
        index_structure_dict = {}
        for index, row in df.iterrows():
            movie_id = row["movieid"]
            column = 0
            for i in range(0, self.num_layers):
                bucket = ""
                for j in range(0, self.num_hashs):
                    interval = row[column]
                    bucket = bucket + str(int(interval)) + "."
                    column += 1
                    if bucket.strip(".") in index_structure_dict:
                        index_structure_dict[bucket.strip(".")].add(movie_id)
                    else:
                        movie_set = set()
                        movie_set.add(movie_id)
                        index_structure_dict[bucket.strip(".")] = movie_set
        return index_structure_dict

    def fetch_hash_keys(self, bucket_list):
        """
        Obtain the hash keys for the bucket list
        :param bucket_list:
        :return:
        """
        column = 0
        hash_key_list = []
        for i in range(0, self.num_layers):
            bucket = ""
            for j in range(0, self.num_hashs):
                interval = bucket_list[column]
                if(j != self.num_hashs-1):
                    bucket = bucket + str(int(interval)) + "."
                else:
                    bucket = bucket + str(int(interval))
                column += 1
            hash_key_list.append(bucket)
        return hash_key_list

    def create_index_structure(self, movie_list):
        """
        Creates index structure for search
        :param movie_list:
        """
        self.movie_bucket_df = self.group_data()
        movie_list_bucket_df = self.movie_bucket_df[self.movie_bucket_df["movieid"].isin(movie_list)] if movie_list  else self.movie_bucket_df
        self.index_structure = self.index_data(movie_list_bucket_df)

    def query_for_nearest_neighbours_for_movie(self, query_movie_id, no_of_nearest_neighbours):
        """
        Nearest neighbors for the the movie passed as input
        :param query_movie_id:
        :param no_of_nearest_neighbours:
        :return: list of r nearest movies
        """
        query_movie_name = self.util.get_movie_name_for_id(query_movie_id)
        print("\nQuery Movie Name : " + query_movie_name + " - " + str(int(query_movie_id)) + "\n")
        query_vector = self.movie_latent_df[self.movie_latent_df["movieid"] == query_movie_id]
        query_vector = query_vector.iloc[0].tolist()[0:-1]
        return self.query_for_nearest_neighbours(query_vector, no_of_nearest_neighbours)

    def query_for_nearest_neighbours(self, query_vector, no_of_nearest_neighbours):
        """
        Nearest neighbor for the query vector
        :param query_vector:
        :param no_of_nearest_neighbours:
        :return: list of r nearest movies
        """
        query_bucket_list = self.LSH(query_vector)
        query_hash_key_list = self.fetch_hash_keys(query_bucket_list)
        query_hash_key_set = set(query_hash_key_list)
        selected_movie_set = set()
        nearest_neighbour_list = {}
        flag = False
        for j in range(0, self.num_hashs):
            for bucket in query_hash_key_set:
                movies_in_current_bucket = self.index_structure.get(bucket.rsplit(".", j)[0], '')
                movies_in_current_bucket.discard('')
                selected_movie_set.update(movies_in_current_bucket)
                self.total_movies_considered.extend(list(movies_in_current_bucket))
                selected_movie_vectors = self.movie_latent_df[self.movie_latent_df["movieid"].isin(selected_movie_set)]
                distance_from_query_list = []
                for k in range(0, len(selected_movie_vectors.index)):
                    row_list = selected_movie_vectors.iloc[k].tolist()
                    euclidean_distance = distance.euclidean(row_list[0:-1], query_vector)
                    if(euclidean_distance != 0):
                        distance_from_query_list.append((row_list[-1], euclidean_distance))
                distance_from_query_list = sorted(distance_from_query_list, key=lambda x: x[1])
                nearest_neighbour_list = ([each[0] for each in distance_from_query_list[0:no_of_nearest_neighbours]])
                if (len(nearest_neighbour_list) >= no_of_nearest_neighbours):
                    flag = True
                    break
            if flag:
                break
        nearest_neighbours = [int(each) for each in nearest_neighbour_list]
        return nearest_neighbours


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='phase_3_task_3.py 4 3',
    )
    parser.add_argument("num_layers", action="store", type=int)
    parser.add_argument("num_hashs_per_layer", action="store", type=int)
    parser.add_argument("movie_str", action="store", type=str)
    arg_input = vars(parser.parse_args())
    num_layers = arg_input["num_layers"]
    num_hashs = arg_input["num_hashs_per_layer"]
    if arg_input["movie_str"] == "":
        movie_list = []
    else:
        movie_list = arg_input["movie_str"].split(",")
    movie_lsh = MovieLSH(num_layers, num_hashs)
    with open(os.path.join(movie_lsh.data_set_loc, 'task_3_details.json'), 'w') as outfile:
        outfile.write(json.dumps({"num_layers": num_layers,
                                  'num_hashs': num_hashs,
                                  "movie_list": movie_list},
                                 sort_keys=True, indent=4, separators=(',', ': ')))
    print("Creating Index Structure for Movies...")
    movie_lsh.create_index_structure(movie_list)
    while True:
        query_movie = int(input("\nEnter Query Movie ID : "))
        no_of_nearest_neighbours = int(input("\nEnter No. of Nearest Neighbours : "))
        nearest_neighbours = movie_lsh.query_for_nearest_neighbours_for_movie(query_movie, no_of_nearest_neighbours)
        print("\nTotal number of movie considered: %s" % len(movie_lsh.total_movies_considered))
        print("Total number of unique movies considered: %s" % len(set(movie_lsh.total_movies_considered)))
        movie_lsh.util.print_movie_recommendations_and_collect_feedback(nearest_neighbours, 3, None)
        confirmation = input("\n\nDo you want to continue? (y/Y/n/N): ")
        if confirmation not in ("y", "Y"):
            break
