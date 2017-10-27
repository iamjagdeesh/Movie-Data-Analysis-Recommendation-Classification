import argparse
import logging

import pandas as pd
from config_parser import ParseConfig
from phase1_task_2 import GenreTag
from util import Util

logging.basicConfig(level=logging.ERROR)

log = logging.getLogger(__name__)
conf = ParseConfig()
# util = Util()

class LdaGenreTag(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.util = Util()

    def get_lda_data(self, genre):
        """
        Does LDA on movie-tag counts and outputs movies in terms of latent semantics as U
        and tags in terms of latent semantics as Vh
        :param genre:
        :return: returns U and Vh
        """

        data_frame = self.get_genre_data().reset_index()
        genre_data_frame = data_frame[data_frame["genre"]==genre]
        tag_df = genre_data_frame.groupby(['movieid'])['tag'].apply(list).reset_index()
        tag_df = tag_df.sort_values('movieid')
        tag_df = list(tag_df.iloc[:,1])

        (U, Vh) = self.util.LDA(tag_df, num_topics=4, num_features=1000)

        for latent in Vh:
            print(latent)

class SvdGenreTag(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.util = Util()

    def genre_tag(self, genre):
        """
        Does SVD on movie-tag matrix and outputs movies in terms of latent semantics as U
        and tags in terms of latent semantics as Vh
        :param genre:
        :return: returns U and Vh
        """

        genre_tag_frame = self.get_genre_data()
        given_genre_frame = self.combine_computed_weights(genre_tag_frame, "TFIDF", genre)
        temp_df = given_genre_frame[["moviename", "tag", "total"]].drop_duplicates()
        genre_tag_tfidf_df = temp_df.pivot(index='moviename', columns='tag', values='total')
        genre_tag_tfidf_df = genre_tag_tfidf_df.fillna(0)

        genre_tag_tfidf_df.to_csv('genre_tag_matrix.csv', index=True, encoding='utf-8')

        df = pd.DataFrame(pd.read_csv('genre_tag_matrix.csv'))
        df1 = genre_tag_tfidf_df.values[:, :]
        row_headers = list(df["moviename"])
        column_headers = list(df)
        del column_headers[0]

        (U, s, Vh) = self.util.SVD(df1)

        # To print latent semantics
        latents = self.util.get_latent_semantics(4, Vh)
        self.util.print_latent_semantics(latents, column_headers)

        u_frame = pd.DataFrame(U[:,:4], index=row_headers)
        v_frame = pd.DataFrame(Vh[:4,:], columns=column_headers)
        u_frame.to_csv('u_1a_svd.csv', index=True, encoding='utf-8')
        v_frame.to_csv('vh_1a_svd.csv', index=True, encoding='utf-8')
        return (u_frame, v_frame, s)

class PcaGenreTag(GenreTag):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.util = Util()

    def genre_tag(self, genre):
        """
        Does PCA on movie-tag matrix and outputs movies in terms of latent semantics as U
        and tags in terms of latent semantics as Vh
        :param genre:
        :return: returns U and Vh
        """

        genre_tag_frame = self.get_genre_data()
        given_genre_frame = self.combine_computed_weights(genre_tag_frame, "TFIDF", genre)
        temp_df = given_genre_frame[["moviename", "tag", "total"]].drop_duplicates()
        genre_tag_tfidf_df = temp_df.pivot(index='moviename', columns='tag', values='total')
        genre_tag_tfidf_df = genre_tag_tfidf_df.fillna(0)
        genre_tag_tfidf_df.to_csv('genre_tag_matrix.csv', index = True , encoding='utf-8')

        df = pd.DataFrame(pd.read_csv('genre_tag_matrix.csv'))
        df1 = genre_tag_tfidf_df.values[:, :]
        column_headers = list(df)
        del column_headers[0]

        (U, s, Vh) = self.util.PCA(df1)

        # To print latent semantics
        latents = self.util.get_latent_semantics(4, Vh)
        self.util.print_latent_semantics(latents, column_headers)

        u_frame = pd.DataFrame(U[:, :4], index=column_headers)
        v_frame = pd.DataFrame(Vh[:4, :], columns=column_headers)
        u_frame.to_csv('u_1a_pca.csv', index=True, encoding='utf-8')
        v_frame.to_csv('vh_1a_pca.csv', index=True, encoding='utf-8')
        return (u_frame, v_frame, s)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='phase_2_task_1a.py Action pca',
    )
    parser.add_argument('genre', action="store", type=str)
    parser.add_argument('model', action="store", choices=set(('pca','svd','lda')))
    input = vars(parser.parse_args())
    genre = input['genre']
    model = input['model']

    if model == 'pca':
        obj_pca = PcaGenreTag()
        obj_pca.genre_tag(genre)
    elif model == 'svd':
        obj_svd = SvdGenreTag()
        obj_svd.genre_tag(genre)
    else :
        obj_lda = LdaGenreTag()
        obj_lda.get_lda_data(genre)