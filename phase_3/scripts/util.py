import logging
import math
import operator
import os

import gensim
import numpy
import pandas as pd
import tensorly.tensorly.decomposition as decomp
from config_parser import ParseConfig
from data_extractor import DataExtractor
from phase1_task_2 import GenreTag

logging.getLogger("gensim").setLevel(logging.CRITICAL)


class Util(object):
    """
    Class containing all the common utilities used across the entire code base
    """
    def __init__(self):
        self.conf = ParseConfig()
        self.data_set_loc = self.conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.mlmovies = self.data_extractor.get_mlmovies_data()
        self.genre_tag = GenreTag()
        self.genre_data = self.genre_tag.get_genre_data()

    def get_movie_id(self, movie):
        """
        Obtain name ID for the name passed as input
        :param movie:
        :return: movie id
        """
        all_movie_data = self.mlmovies
        movie_data = all_movie_data[all_movie_data['moviename'] == movie]
        movie_id = movie_data['movieid'].unique()

        return movie_id[0]

    def CPDecomposition(self, tensor, rank):
        """
        Perform CP Decomposition
        :param tensor:
        :param rank:
        :return: factor matrices obtained after decomposition
        """
        (movie_count, genre_count, tag_count) = tensor.shape
        rank = min(rank, movie_count-1, genre_count-1, tag_count-1)
        factors = decomp.parafac(tensor, rank)

        return factors

    def SVD(self, matrix):
        """
        Perform SVD
        :param matrix:
        :return: factor matrices and the core matrix
        """
        U, s, Vh = numpy.linalg.svd(matrix, full_matrices=False)

        return U, s, Vh

    def PCA(self, matrix):
        """
        Perform PCA
        :param matrix:
        :return: factor matrices and the core matrix
        """
        cov_df = numpy.cov(matrix, rowvar=False)
        U, s, Vh = numpy.linalg.svd(cov_df)

        return U, s, Vh

    def LDA(self, input_compound_list, num_topics, num_features):
        """
        Perform LDA
        :param input_compound_list:
        :param num_topics:
        :param num_features:
        :return: topics and object topic distribution
        """
        dictionary = gensim.corpora.Dictionary(input_compound_list)
        corpus = [dictionary.doc2bow(text) for text in input_compound_list]
        lda = gensim.models.ldamodel.LdaModel(corpus, num_topics, id2word=dictionary, passes=20)
        latent_semantics = lda.print_topics(num_topics, num_features)
        corpus = lda[corpus]

        return corpus, latent_semantics

    def get_doc_topic_matrix(self, u, num_docs, num_topics):
        """
        Reconstructing data
        :param u:
        :param num_docs:
        :param num_topics:
        :return: reconstructed data
        """
        u_matrix = numpy.zeros(shape=(num_docs, num_topics))

        for i in range(0, len(u)):
            doc = u[i]
            for j in range(0, len(doc)):
                (topic_no, prob) = doc[j]
                u_matrix[i, topic_no] = prob

        return u_matrix

    def get_transition_dataframe(self, data_frame):
        """
        Function to get the transition matrix for Random walk
        :param data_frame:
        :return: transition matrix
        """
        for column in data_frame:
            data_frame[column] = pd.Series(
                [0 if ind == int(column) else each for ind, each in zip(data_frame.index, data_frame[column])],
                index=data_frame.index)
        data_frame["row_sum"] = data_frame.sum(axis=1)
        for column in data_frame:
            data_frame[column] = pd.Series(
                [each / sum if (column != "row_sum" and each > 0 and ind != int(column) and sum!=0) else each for ind, each, sum in
                 zip(data_frame.index, data_frame[column], data_frame.row_sum)],
                index=data_frame.index)
        data_frame = data_frame.drop(["row_sum"], axis=1)
        data_frame.loc[(data_frame.T == 0).all()] = float(1 / (len(data_frame.columns)))
        data_frame = data_frame.transpose()

        return data_frame

    def get_seed_matrix(self, transition_df, seed_nodes, nodes):
        """
        Function to get the Restart matrix for entries in the seed list
        :param transition_df:
        :param seed_nodes:
        :param nodeids:
        :return: seed_matrix
        """
        seed_matrix = [0.0 for each in range(len(transition_df.columns))]
        seed_value_list = self.distribute(seed_nodes, num_of_seeds_to_recommend=1)
        for each in seed_nodes:
            seed_matrix[list(nodes).index(each)] = seed_value_list[list(seed_nodes).index(each)]

        return seed_matrix

    def compute_pagerank(self, seed_nodes, node_matrix, nodes):
        """
        Function to compute the Personalised Pagerank for the given input
        :param seed_actors:
        :param actor_matrix:
        :param actorids:
        :return:
        """
        data_frame = pd.DataFrame(node_matrix)
        transition_df = self.get_transition_dataframe(data_frame)
        seed_matrix = self.get_seed_matrix(transition_df, seed_nodes, nodes)
        result_list = seed_matrix
        temp_list = []
        num_of_iter = 0
        while temp_list != result_list and num_of_iter <= 1000:
            num_of_iter += 1
            temp_list = result_list
            result_list = list(
                0.85 * numpy.matmul(numpy.array(transition_df.values), numpy.array(result_list)) + 0.15 * numpy.array(
                    seed_matrix))
        page_rank_dict = {i: j for i, j in zip(nodes, result_list)}
        sorted_rank = sorted(page_rank_dict.items(), key=operator.itemgetter(1), reverse=True)

        return sorted_rank[0:len(seed_nodes)+5]

    def print_movie_recommendations_and_collect_feedback(self, movie_ids, task_no, user_id):
        """
        Interface to obtain relevance feedback
        :param movie_ids: List of movies
        :param task_no: Task from which the interface is called
        :param user_id: user for which the movies are displayed
        """
        if len(movie_ids) == 0:
            print("No movies found.")
            exit(1)

        if task_no in [1, 2]:
            print("Movie recommendations: ")
        elif task_no in [3, 4]:
            print("Nearest movies: ")
        else:
            print("Incorrect task number - " + task_no + "\nAborting...")
            exit(1)
            
        count = 1
        movie_dict = {}
        for movie_id in movie_ids:
            movie_name = self.get_movie_name_for_id(movie_id)
            print(str(count) + ". " + str(movie_name) + " - " + str(movie_id))
            movie_dict[count] = (movie_name, movie_id)
            count += 1

        done = False
        rel_movies = []
        irrel_movies = []
        while not done:
            movies_list = input("\nPlease enter comma separated ids of the relevant movies: ")
            rel_ids = set(movies_list.strip(" ").strip(",").replace(" ", "").split(","))
            while '' in rel_ids:
                rel_ids.remove('')

            incorrect = False
            for item in rel_ids:
                if int(item) not in [num for num in range(1, len(movie_ids) + 1)]:
                    print("Incorrect movie ID selected.")
                    incorrect = True
                    break
            if incorrect:
                continue

            confirmation = input("Are you sure these are the relevant movies? " + str(list(rel_ids)) + " (y/Y/n/N): ")
            if confirmation != "y" and confirmation != "Y":
                continue

            movies_list = input("\nPlease enter comma separated ids of the irrelevant movies: ")
            irrel_ids = set(movies_list.strip(" ").strip(",").replace(" ", "").split(","))
            while '' in irrel_ids:
                irrel_ids.remove('')

            incorrect = False
            for item in irrel_ids:
                if int(item) not in list(set(list([num for num in range(1, len(movie_ids) + 1)])) - set(
                        int(num) for num in rel_ids)):
                    print("Incorrect movie ID selected.")
                    incorrect = True
                    break
            if incorrect:
                continue

            confirmation = input(
                "Are you sure these are the irrelevant movies? " + str(list(irrel_ids)) + " (y/Y/n/N): ")
            if confirmation != "y" and confirmation != "Y":
                continue

            done = True
            for item in rel_ids:
                rel_movies.append(movie_dict[int(item)])
            for item in irrel_ids:
                irrel_movies.append(movie_dict[int(item)])

        if task_no == 1 or task_no == 2:
            if not os.path.isfile(self.data_set_loc + "/task2-feedback.csv"):
                df = pd.DataFrame(columns=['movie-name', 'movie-id', 'relevancy', 'user-id'])
            else:
                df = self.data_extractor.get_task2_feedback_data()

            for movie in rel_movies:
                df = df.append(
                    {'movie-name': movie[0], 'movie-id': movie[1], 'relevancy': 'relevant', 'user-id': user_id},
                    ignore_index=True)
            for movie in irrel_movies:
                df = df.append(
                    {'movie-name': movie[0], 'movie-id': movie[1], 'relevancy': 'irrelevant', 'user-id': user_id},
                    ignore_index=True)

            df.to_csv(self.data_set_loc + "/task2-feedback.csv", index=False)
        elif task_no == 3 or task_no == 4:
            if not os.path.isfile(self.data_set_loc + "/task4-feedback.csv"):
                df = pd.DataFrame(columns=['movie-name', 'movie-id', 'relevancy'])
            else:
                df = self.data_extractor.get_task4_feedback_data()

            for movie in rel_movies:
                df = df.append({'movie-name': movie[0], 'movie-id': movie[1], 'relevancy': 'relevant'},
                               ignore_index=True)
            for movie in irrel_movies:
                df = df.append({'movie-name': movie[0], 'movie-id': movie[1], 'relevancy': 'irrelevant'},
                               ignore_index=True)

            df.to_csv(self.data_set_loc + "/task4-feedback.csv", index=False)

    def get_distribution_count(self, seed_nodes, num_of_seeds_to_recommend):
        """
        Given the number of seeds to be recommended and the seed_nodes,
        returns the distribution for each seed_node considering order
        :param seed_nodes:
        :param num_of_seeds_to_recommend:
        :return: distribution_list
        """
        seed_value_list = self.distribute(seed_nodes, num_of_seeds_to_recommend)
        seed_value_list = [round(each) for each in seed_value_list]
        total_count = sum(seed_value_list)
        difference = num_of_seeds_to_recommend - total_count
        if difference > 0:
            for i in range(0, len(seed_value_list)):
                if seed_value_list[i] == 0:
                    seed_value_list[i] = 1
                    difference -= 1
                    if difference == 0:
                        return seed_value_list
            for i in range(0, len(seed_value_list)):
                seed_value_list[i] += 1
                difference -= 1
                if difference == 0:
                    return seed_value_list
        elif difference < 0:
            for i in range(0, len(seed_value_list)):
                if seed_value_list[len(seed_value_list) - 1 - i] != 0:
                    seed_value_list[len(seed_value_list) - 1 - i] -= 1
                    difference += 1
                if difference == 0:
                    return seed_value_list

        return seed_value_list

    def get_movie_tag_matrix(self):
        """
        Function to get movie_tag matrix containing list of tags in each movie
        :return: movie_tag_matrix
        """
        tag_df = self.genre_data
        unique_tags = tag_df.tag_string.unique()
        idf_data = tag_df.groupby(['movieid'])['tag_string'].apply(set)
        tf_df = tag_df.groupby(['movieid'])['tag_string'].apply(list).reset_index()
        movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag_string))
        tf_weight_dict = {movie: self.genre_tag.assign_tf_weight(tags) for movie, tags in
                          list(movie_tag_dict.items())}
        idf_weight_dict = self.genre_tag.assign_idf_weight(idf_data, unique_tags)
        tag_df = self.genre_tag.get_model_weight(tf_weight_dict, idf_weight_dict, tag_df, 'tfidf')
        tag_df["total"] = tag_df.groupby(['movieid','tag_string'])['value'].transform('sum')
        temp_df = tag_df[["movieid", "tag_string", "total"]].drop_duplicates().reset_index()
        genre_tag_tfidf_df = temp_df.pivot_table('total', 'movieid', 'tag_string')
        genre_tag_tfidf_df = genre_tag_tfidf_df.fillna(0)

        return genre_tag_tfidf_df

    def distribute(self, seed_nodes, num_of_seeds_to_recommend):
        """
        Distributes importance among seed_nodes based on order of occurrence
        :param seed_nodes:
        :param num_of_seeds_to_recommend:
        :return: list of size num_of_seeds_to_recommend with distributed values
        """
        seed_value = float(num_of_seeds_to_recommend) / len(seed_nodes)
        seed_value_list = [seed_value for seed in seed_nodes]
        delta = seed_value / len(seed_nodes)
        for i in range(0, len(seed_nodes) - 1):
            seed_value_list[i] = seed_value_list[i] + (len(seed_nodes) - 1 - i) * delta
            for j in range(i + 1, len(seed_nodes)):
                seed_value_list[j] = seed_value_list[j] - delta

        return seed_value_list

    def get_movie_name_for_id(self, movieid):
        """
        movie name for movie id
        :param movieid:
        :return: movie name
        """
        all_movie_data = self.mlmovies
        movie_data = all_movie_data[all_movie_data['movieid'] == movieid]
        movie_name = movie_data['moviename'].unique()

        return movie_name[0]

    def get_tag_list_for_movie(self, movie):
        """
        Get a tag list for the movie
        :param movie: movie id
        :return: list of tags
        """
        movie_specific_data = self.genre_data[self.genre_data["movieid"] == movie]
        tags_list = movie_specific_data["tag_string"].unique()

        return tags_list

    def get_movies_for_tag(self, tag):
        """
        Get the list of movies containing the tag
        :param tag: tag string
        :return: list of movies
        """
        tag_specific_data = self.genre_data[self.genre_data["tag_string"] == tag]
        movies_list = tag_specific_data["movieid"].unique()

        return movies_list

    def get_all_movies_for_user(self, user_id):
        """
        Obtain all movies watched by the user
        :param user_id:
        :return: list of movies watched by the user
        """
        user_data = self.genre_data[self.genre_data['userid'] == user_id]
        user_data = user_data.sort_values('timestamp', ascending=False)
        movies = user_data['movieid'].unique()

        return movies

    def get_movies_after_year(self, year):
        all_movie_data = self.mlmovies
        movie_data = all_movie_data[all_movie_data['year'] >= year]
        movie_id_list = movie_data['movieid'].unique()

        return movie_id_list

    def get_vector_magnitude(self, vector):
        """
        Calculate the magnitude of the vector
        :param vector:
        :return: length of the vector
        """
        result = 0
        for i in vector:
            result += (i * i)

        return math.sqrt(result)

def rbf_kernel(gamma, **kwargs):
    def f(x1, x2):
        distance = numpy.linalg.norm(x1 - x2) ** 2
        return numpy.exp(-gamma * distance)

    return f


if __name__ == "__main__":
    obj = Util()
