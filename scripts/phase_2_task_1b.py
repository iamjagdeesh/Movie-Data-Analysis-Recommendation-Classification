import argparse
import logging
import math
from collections import Counter

import pandas as pd
from config_parser import ParseConfig
from data_extractor import DataExtractor
from phase1_task_2 import GenreTag
from util import Util

logging.basicConfig(level=logging.INFO)

log = logging.getLogger(__name__)
conf = ParseConfig()
util = Util()

class LdaGenreActor(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def get_lda_data(self, genre):
        """
        Does LDA on movie-actor counts and outputs movies in terms of latent semantics as U
        and actor in terms of latent semantics as Vh
        :param genre:
        :return: returns U and Vh
        """

        # Getting movie_genre_data
        movie_genre_data_frame = self.data_extractor.get_mlmovies_data()
        movie_genre_data_frame = self.split_genres(movie_genre_data_frame)

        # Getting actor_movie_data
        movie_actor_data_frame = self.data_extractor.get_movie_actor_data()

        genre_actor_frame = movie_genre_data_frame.merge(movie_actor_data_frame, how="left", left_on="movieid",
                                                         right_on="movieid")
        # genre_actor_frame = genre_actor_frame[genre_actor_frame['year'].notnull()].reset_index()
        genre_actor_frame = genre_actor_frame[["movieid", "year", "genre", "actorid", "actor_movie_rank"]]

        genre_actor_frame["actorid_string"] = pd.Series(
            [str(id) for id in genre_actor_frame.actorid],
            index=genre_actor_frame.index)

        genre_data_frame = genre_actor_frame[genre_actor_frame["genre"]==genre]
        actor_df = genre_data_frame.groupby(['movieid'])['actorid_string'].apply(list).reset_index()
        actor_df = actor_df.sort_values('movieid')
        actor_df.to_csv('movie_actor_lda.csv', index=True, encoding='utf-8')

        actor_df = list(actor_df.iloc[:,1])

        (U, Vh) = util.LDA(actor_df, num_topics=4, num_features=1000)

        for latent in Vh:
            print(latent)

class SvdGenreActor(GenreTag):
    """
            Class to relate Genre and Actor, inherits the ActorTag to use the common weighing functons
    """

    def __init__(self):
        """
        Initialiazing the data extractor object to get data from the csv files
        """
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def split_genres(self, data_frame):
        """
        This function extractors genres from each row and converts into independent rows
        :param data_frame:
        :return: data frame with multiple genres split into different rows
        """
        genre_data_frame = data_frame['genres'].str.split('|', expand=True).stack()
        genre_data_frame.name = "genre"
        genre_data_frame.index = genre_data_frame.index.droplevel(-1)
        genre_data_frame = genre_data_frame.reset_index()
        data_frame = data_frame.drop("genres", axis=1)
        data_frame = data_frame.reset_index()
        data_frame = genre_data_frame.merge(data_frame, how="left", on="index")
        return data_frame

    def assign_rank_weight(self, data_frame):
        """
        This function assigns a value for all the actors in a movie on a scale of 100,
         based on their rank in the movie.
        :param tag_series:
        :return: dictionary of (movieid, actor_rank) to the computed rank_weight
        """
        groupby_movies = data_frame.groupby("movieid")
        movie_rank_weight_dict = {}
        for movieid, info_df in groupby_movies:
           max_rank = info_df.actor_movie_rank.max()
           for rank in info_df.actor_movie_rank.unique():
             movie_rank_weight_dict[(movieid, rank)] = (max_rank - rank + 1)/max_rank*100
        return movie_rank_weight_dict

    def assign_idf_weight(self, data_frame, unique_actors):
        """
        This function computes the idf weight for all tags in a data frame,
        considering each movie as a document
        :param data_frame:
        :param unique_tags:
        :return: dictionary of tags and idf weights
        """
        idf_counter = {actorid_string: 0 for actorid_string in unique_actors}
        data_frame.actorid_string = pd.Series([set(actors.split(',')) for actors in data_frame.actorid_string], index=data_frame.index)
        for actor_list in data_frame.actorid_string:
            for actorid_string in actor_list:
                idf_counter[actorid_string] += 1
        for actorid_string, count in list(idf_counter.items()):
            idf_counter[actorid_string] = math.log(len(data_frame.index)/count)
        return idf_counter

    def assign_tf_weight(self, actor_series):
        """
        This function computes the tf weight for all tags for a movie
        :param tag_series:
        :return: dictionary of tags and tf weights
        """
        counter = Counter()
        for each in actor_series:
            counter[each] += 1
        total = sum(counter.values())
        for each in counter:
            counter[each] = (counter[each]/total)
        return dict(counter)

    def get_model_weight(self, tf_weight_dict, idf_weight_dict, rank_weight_dict, actor_df, model):
        """
               This function combines tf_weight on a scale of 100, idf_weight on a scale of 100,
               and timestamp_weight on a scale of 10 , based on the model.
               :param tf_weight_dict, idf_weight_dict, rank_weight_dict, tag_df, model
               :return: data_frame with column of the combined weight
        """
        if model == "TF":
            actor_df["value"] = pd.Series(
                [(ts_weight + (tf_weight_dict.get(movieid, 0).get(actorid_string, 0)*100) + rank_weight_dict.get((movieid, rank), 0)) for
                 index, ts_weight, actorid_string, movieid, rank
                 in zip(actor_df.index, actor_df.year_weight, actor_df.actorid_string, actor_df.movieid, actor_df.actor_movie_rank)],
                index=actor_df.index)
        else:
            actor_df["value"] = pd.Series(
                [(ts_weight + (tf_weight_dict.get(movieid, 0).get(actorid_string, 0)*(idf_weight_dict.get(actorid_string, 0))*100) + rank_weight_dict.get((movieid, rank), 0)) for
                 index, ts_weight, actorid_string, movieid, rank
                 in zip(actor_df.index, actor_df.year_weight, actor_df.actorid_string, actor_df.movieid, actor_df.actor_movie_rank)],
                index=actor_df.index)
        return actor_df

    def combine_computed_weights(self, data_frame, rank_weight_dict, model, genre):
        """
                Triggers the weighing process and sums up all the calculated weights for each tag
                :param data_frame:
                :param rank_weight_dict:
                :param model:
                :return: dictionary of tags and weights
        """
        actor_df = data_frame.reset_index()
        temp_df = data_frame[data_frame["genre"]==genre]
        unique_actors = actor_df.actorid_string.unique()
        idf_data = actor_df.groupby(['movieid'])['actorid_string'].apply(lambda x: ','.join(x)).reset_index()
        tf_df = temp_df.groupby(['movieid'])['actorid_string'].apply(lambda x: ','.join(x)).reset_index()
        movie_actor_dict = dict(zip(tf_df.movieid, tf_df.actorid_string))
        tf_weight_dict = {movie: self.assign_tf_weight(actorid_string.split(',')) for movie, actorid_string in
                          list(movie_actor_dict.items())}
        idf_weight_dict = {}
        if model != 'TF':
            idf_weight_dict = self.assign_idf_weight(idf_data, unique_actors)
        actor_df = self.get_model_weight(tf_weight_dict, idf_weight_dict, rank_weight_dict, temp_df, model)
        actor_df["total"] = actor_df.groupby(['actorid_string'])['value'].transform('sum')
        actor_df = actor_df.drop_duplicates("actorid_string").sort_values("total", ascending=False)
        #actor_tag_dict = dict(zip(tag_df.tag, tag_df.total))
        return actor_df

    def get_genre_actor_data_frame(self):
        """
        Function to merge mutiple tables and get the required dataframe for tf-idf calculation
        :return: dataframe
        """
        # Getting movie_genre_data
        movie_genre_data_frame = self.data_extractor.get_mlmovies_data()
        movie_genre_data_frame = self.split_genres(movie_genre_data_frame)

        # Getting actor_movie_data
        movie_actor_data_frame = self.data_extractor.get_movie_actor_data()

        genre_actor_frame = movie_genre_data_frame.merge(movie_actor_data_frame, how="left", left_on="movieid", right_on="movieid")
        #genre_actor_frame = genre_actor_frame[genre_actor_frame['year'].notnull()].reset_index()
        genre_actor_frame = genre_actor_frame[["movieid", "year", "genre", "actorid", "actor_movie_rank"]]
        genre_actor_frame = genre_actor_frame.sort_values("year", ascending=True)

        data_frame_len = len(genre_actor_frame.index)
        genre_actor_frame["year_weight"] = pd.Series(
            [(index + 1) / data_frame_len * 10 for index in genre_actor_frame.index],
            index=genre_actor_frame.index)

        genre_actor_frame["actorid_string"] = pd.Series(
            [str(id) for id in genre_actor_frame.actorid],
            index = genre_actor_frame.index)

        return genre_actor_frame

    def svd_genre_actor(self, genre):
        """
        Does SVD on movie-actor matrix and outputs movies in terms of latent semantics as U
        and actors in terms of latent semantics as Vh
        :param genre:
        :return: returns U and Vh
        """
        genre_actor_frame = self.get_genre_actor_data_frame()
        rank_weight_dict = self.assign_rank_weight(genre_actor_frame[['movieid', 'actor_movie_rank']])
        genre_actor_frame = self.combine_computed_weights(genre_actor_frame, rank_weight_dict, "TFIDF", genre)
        temp_df = genre_actor_frame[["movieid", "actorid_string", "total"]].drop_duplicates()
        genre_actor_tfidf_df = temp_df.pivot(index='movieid', columns='actorid_string', values='total')
        genre_actor_tfidf_df = genre_actor_tfidf_df.fillna(0)

        genre_actor_tfidf_df.to_csv('genre_actor_matrix.csv', index=True, encoding='utf-8')

        df = pd.DataFrame(pd.read_csv('genre_actor_matrix.csv'))
        df1 = genre_actor_tfidf_df.values[:, :]
        row_headers = list(df["movieid"])
        column_headers = list(df)
        del column_headers[0]

        column_headers_names = []

        for col_head in column_headers:
            col_head_name = util.get_actor_name_for_id(int(col_head))
            column_headers_names = column_headers_names + [col_head_name]

        (U, s, Vh) = util.SVD(df1)

        # To print latent semantics
        latents = util.get_latent_semantics(4, Vh)
        util.print_latent_semantics(latents, column_headers_names)

        u_frame = pd.DataFrame(U[:, :4], index=row_headers)
        v_frame = pd.DataFrame(Vh[:4, :], columns=column_headers)
        u_frame.to_csv('u_1b_svd.csv', index=True, encoding='utf-8')
        v_frame.to_csv('vh_1b_svd.csv', index=True, encoding='utf-8')
        return (u_frame, v_frame, s)

class PcaGenreActor(SvdGenreActor):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def pca_genre_actor(self, genre):
        """
        Does PCA on movie-actor matrix and outputs movies in terms of latent semantics as U
        and actors in terms of latent semantics as Vh
        :param genre:
        :return: returns U and Vh
        """

        genre_actor_frame = self.get_genre_actor_data_frame()
        rank_weight_dict = self.assign_rank_weight(genre_actor_frame[['movieid', 'actor_movie_rank']])
        genre_actor_frame = self.combine_computed_weights(genre_actor_frame, rank_weight_dict, "TFIDF", genre)
        temp_df = genre_actor_frame[["movieid", "actorid_string", "total"]].drop_duplicates()
        genre_actor_tfidf_df = temp_df.pivot(index='movieid', columns='actorid_string', values='total')
        genre_actor_tfidf_df = genre_actor_tfidf_df.fillna(0)
        genre_actor_tfidf_df.to_csv('genre_actor_matrix.csv', index = True , encoding='utf-8')

        df = pd.DataFrame(pd.read_csv('genre_actor_matrix.csv'))
        df1 = genre_actor_tfidf_df.values[:, :]
        column_headers = list(df)
        del column_headers[0]

        column_headers_names = []

        for col_head in column_headers:
            col_head_name = util.get_actor_name_for_id(int(col_head))
            column_headers_names = column_headers_names + [col_head_name]

        (U, s, Vh) = util.PCA(df1)

        # To print latent semantics
        latents = util.get_latent_semantics(4, Vh)
        util.print_latent_semantics(latents, column_headers_names)

        u_frame = pd.DataFrame(U[:, :4], index=column_headers)
        v_frame = pd.DataFrame(Vh[:4, :], columns=column_headers)
        u_frame.to_csv('u_1b_pca.csv', index=True, encoding='utf-8')
        v_frame.to_csv('vh_1b_pca.csv', index=True, encoding='utf-8')
        return (u_frame, v_frame, s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='phase_2_task_1b.py Action pca',
    )
    parser.add_argument('genre', action="store", type=str)
    parser.add_argument('model', action="store", choices=set(('pca', 'svd', 'lda')))
    input = vars(parser.parse_args())
    genre = input['genre']
    model = input['model']

    if model == 'pca':
        obj_pca = PcaGenreActor()
        obj_pca.pca_genre_actor(genre)
    elif model == 'svd':
        obj_svd = SvdGenreActor()
        obj_svd.svd_genre_actor(genre)
    else:
        obj_lda = LdaGenreActor()
        obj_lda.get_lda_data(genre)