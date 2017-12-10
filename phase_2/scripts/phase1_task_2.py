import logging

import pandas as pd
from config_parser import ParseConfig
from phase1_task_1 import ActorTag

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
conf = ParseConfig()

class GenreTag(ActorTag):
    """
        Class to relate Genre and tags, inherits the ActorTag to use the common weighing functons
    """
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def get_model_weight(self, tf_weight_dict, idf_weight_dict, tag_df, model):
        """
               This function combines tf_weight on a scale of 100, idf_weight on a scale of 100,
               and timestamp_weight on a scale of 10 , based on the model.
               :param tf_weight_dict, idf_weight_dict, rank_weight_dict, tag_df, model
               :return: data_frame with column of the combined weight
        """
        if model == "TF":
            tag_df["value"] = pd.Series(
                [(ts_weight + tf_weight_dict.get(movieid, 0).get(tag, 0)) for
                 index, ts_weight, tag, movieid
                 in zip(tag_df.index, tag_df.timestamp_weight, tag_df.tag, tag_df.movieid)],
                index=tag_df.index)
        else:
            tag_df["value"] = pd.Series(
                [(ts_weight + (tf_weight_dict.get(movieid, 0).get(tag, 0)*(idf_weight_dict.get(tag, 0)))) for
                 index, ts_weight, tag, movieid
                 in zip(tag_df.index, tag_df.timestamp_weight, tag_df.tag, tag_df.movieid)],
                index=tag_df.index)
        return tag_df

    def combine_computed_weights(self, data_frame, model, genre):
        """
                Triggers the weighing process and sums up all the calculated weights for each tag
                :param data_frame:
                :param rank_weight_dict:
                :param model:
                :return: dictionary of tags and weights
        """
        tag_df = data_frame.reset_index()
        temp_df = data_frame[data_frame["genre"]==genre]
        unique_tags = tag_df.tag.unique()
        idf_data = tag_df.groupby(['movieid'])['tag'].apply(set)
        tf_df = temp_df.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        movie_tag_dict = dict(zip(tf_df.movieid, tf_df.tag))
        tf_weight_dict = {movie: self.assign_tf_weight(tags.split(',')) for movie, tags in
                          list(movie_tag_dict.items())}
        idf_weight_dict = {}
        if model != 'TF':
            idf_weight_dict = self.assign_idf_weight(idf_data, unique_tags)
        tag_df = self.get_model_weight(tf_weight_dict, idf_weight_dict, temp_df, model)
        tag_df["total"] = tag_df.groupby(['movieid', 'tag'])['value'].transform('sum')
        tag_df = tag_df.drop_duplicates(['movieid',"tag"]).sort_values("total", ascending=False)
        #actor_tag_dict = dict(zip(tag_df.tag, tag_df.total))
        return tag_df

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

    def get_genre_data(self):
        """
                Merges data from different csv files necessary to compute the tag weights for each genre,
                assigns weights to timestamp.
                :return: data frame
        """
        data_frame = self.data_extractor.get_mlmovies_data()
        tag_data_frame = self.data_extractor.get_genome_tags_data()
        movie_data_frame = self.data_extractor.get_mltags_data()
        data_frame = self.split_genres(data_frame)
        movie_tag_data_frame = movie_data_frame.merge(tag_data_frame, how="left", left_on="tagid", right_on="tagId")
        genre_tag_frame = data_frame.merge(movie_tag_data_frame, how="left", on="movieid")
        genre_tag_frame = genre_tag_frame[genre_tag_frame['timestamp'].notnull()].reset_index()
        genre_tag_frame = genre_tag_frame[["movieid", "moviename", "genre", "timestamp", "tagid", "tag"]]
        genre_tag_frame = genre_tag_frame.sort_values("timestamp", ascending=True)
        data_frame_len = len(genre_tag_frame.index)
        genre_tag_frame["timestamp_weight"] = pd.Series(
            [(index + 1) / data_frame_len * 10 for index in genre_tag_frame.index],
            index=genre_tag_frame.index)
        return genre_tag_frame

    def merge_genre_tag(self, genre, model):
        """
        Triggers the compute function and outputs the result tag vector
        :param genre:
        :param model:
        :return: returns a dictionary of Genres to dictionary of tags and weights.
        """
        genre_tag_frame = self.get_genre_data()
        tag_dict = self.combine_computed_weights(genre_tag_frame, model, genre)
        print({genre: tag_dict})


if __name__ == "__main__":
    obj = GenreTag()
    # parser = argparse.ArgumentParser(description='task2.py genre model')
    # parser.add_argument('genre', action="store", type=str)
    # parser.add_argument('model', action="store", type=str, choices=set(('TF', 'TFIDF')))
    # input = vars(parser.parse_args())
    # genre = input['genre']
    # model = input['model']
    #obj.merge_genre_tag(genre=genre, model=model)
    obj.merge_genre_tag(genre="Sci-Fi", model="TFIDF")


