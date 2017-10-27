import logging
import math
from collections import Counter

import pandas as pd
from config_parser import ParseConfig
from data_extractor import DataExtractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()


class ActorTag(object):

    """
    Class to relate actors and tags.
    """

    def __init__(self):
        """
        Initializing the data extractor object to get data from the csv files
        """
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)

    def assign_idf_weight(self, data_series, unique_tags):
        """
        This function computes the idf weight for all tags in a data frame,
        considering each movie as a document
        :param data_frame:
        :param unique_tags:
        :return: dictionary of tags and idf weights
        """
        idf_counter = {tag: 0 for tag in unique_tags}
        for tag_list in data_series:
            for tag in tag_list:
                idf_counter[tag] += 1
        for tag, count in list(idf_counter.items()):
            idf_counter[tag] = math.log(len(data_series.index)/count, 2)
        return idf_counter

    def assign_tf_weight(self, tag_series):
        """
        This function computes the tf weight for all tags for a movie
        :param tag_series:
        :return: dictionary of tags and tf weights
        """
        counter = Counter()
        for each in tag_series:
            counter[each] += 1
        total = sum(counter.values())
        for each in counter:
            counter[each] = (counter[each]/total)
        return dict(counter)

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

    def get_model_weight(self, tf_weight_dict, idf_weight_dict, rank_weight_dict, tag_df, model):
        """
        This function combines tf_weight on a scale of 100, idf_weight on a scale of 100,
        actor_rank for each tag on scale of 100 and timestamp_weight on a scale of 10 , based on the model.
        :param tf_weight_dict, idf_weight_dict, rank_weight_dict, tag_df, model
        :return: data_frame with column of the combined weight
        """
        if model == "TF":
            tag_df["value"] = pd.Series(
                [(tf_weight_dict.get(movieid, 0).get(tag, 0)*100) + rank_weight_dict.get((movieid, rank), 0) for
                 index, ts_weight, tag, movieid, rank
                 in zip(tag_df.index, tag_df.timestamp_weight, tag_df.tag, tag_df.movieid, tag_df.actor_movie_rank)],
                index=tag_df.index)
        else:
            tag_df["value"] = pd.Series(
                [(ts_weight + (tf_weight_dict.get(movieid, 0).get(tag, 0)*(idf_weight_dict.get(tag, 0))*100) + rank_weight_dict.get((movieid, rank), 0))  for
                 index, ts_weight, tag, movieid, rank
                 in zip(tag_df.index, tag_df.timestamp_weight, tag_df.tag, tag_df.movieid, tag_df.actor_movie_rank)],
                index=tag_df.index)
        return tag_df

    def combine_computed_weights(self, data_frame, rank_weight_dict, idf_weight_dict, model):
        """
        Triggers the weighing process and sums up all the calculated weights for each tag
        :param data_frame:
        :param rank_weight_dict:
        :param model:
        :return: dictionary of tags and weights
        """
        tag_df = data_frame.reset_index()
        temp_df = tag_df.groupby(['movieid'])['tag'].apply(lambda x: ','.join(x)).reset_index()
        movie_tag_dict = dict(zip(temp_df.movieid, temp_df.tag))
        tf_weight_dict = {movie: self.assign_tf_weight(tags.split(',')) for movie, tags in list(movie_tag_dict.items())}
        tag_df = self.get_model_weight(tf_weight_dict, idf_weight_dict, rank_weight_dict, tag_df, model)
        tag_df["total"] = tag_df.groupby(['tag'])['value'].transform('sum')
        tag_df = tag_df.drop_duplicates("tag").sort_values("total", ascending=False)
        actor_tag_dict = dict(zip(tag_df.tag, tag_df.total))
        return actor_tag_dict

    def merge_movie_actor_and_tag(self, actorid, model):
        """
        Merges data from different csv files necessary to compute the tag weights for each actor,
        assigns weights to timestamp.
        :param actorid:
        :param model:
        :return: returns a dictionary of Actors to dictionary of tags and weights.
        """
        mov_act = self.data_extractor.get_movie_actor_data()
        ml_tag = self.data_extractor.get_mltags_data()
        genome_tag = self.data_extractor.get_genome_tags_data()
        actor_info = self.data_extractor.get_imdb_actor_info_data()
        actor_movie_info = mov_act.merge(actor_info, how="left", left_on="actorid", right_on="id")
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        merged_data_frame = actor_movie_info.merge(tag_data_frame, how="left", on="movieid")
        merged_data_frame = merged_data_frame[merged_data_frame['timestamp'].notnull()]
        merged_data_frame = merged_data_frame.drop(["userid"], axis=1)
        rank_weight_dict = self.assign_rank_weight(merged_data_frame[['movieid', 'actor_movie_rank']])
        merged_data_frame = merged_data_frame.sort_values("timestamp", ascending=True).reset_index()
        data_frame_len = len(merged_data_frame.index)
        merged_data_frame["timestamp_weight"] = pd.Series([(index + 1) / data_frame_len * 10 for index in merged_data_frame.index],
                                                   index=merged_data_frame.index)
        if model == 'TFIDF':
            idf_weight_dict = self.assign_idf_weight(merged_data_frame.groupby('movieid')['tag'].apply(set), merged_data_frame.tag.unique())
            tag_dict = self.combine_computed_weights(merged_data_frame[merged_data_frame['actorid'] == actorid], rank_weight_dict, idf_weight_dict, model)
        else:
            tag_dict = self.combine_computed_weights(merged_data_frame[merged_data_frame['actorid'] == actorid], rank_weight_dict, {},model)

        return tag_dict


if __name__ == "__main__":
    obj = ActorTag()
    actor_id = 17838
    model = "TFIDF"
    print("TFIDF tag values for actor:\n")
    print(obj.merge_movie_actor_and_tag(actor_id, model))
