import json
import os

import pandas as pd
from config_parser import ParseConfig


class DataExtractor(object):
    """
    Class to return resources from the disk
    """

    def __init__(self, file_path):
        self.file_path = file_path

    def data_extractor(self, file_name):  # return the data frame with respect to the csv file in 'resources' directory
        file_loc = os.path.join(self.file_path, file_name)
        data_frame = pd.read_csv(file_loc, low_memory=False)
        return data_frame

    def get_movie_actor_data(self):
        return self.data_extractor("movie-actor.csv")

    def get_mltags_data(self):
        return self.data_extractor("mltags.csv")

    def get_genome_tags_data(self):
        return self.data_extractor("genome-tags.csv")

    def get_mlmovies_data(self):
        return self.data_extractor("mlmovies.csv")

    def get_imdb_actor_info_data(self):
        return self.data_extractor("imdb-actor-info.csv")

    def get_mlratings_data(self):
        return self.data_extractor("mlratings.csv")

    def get_mlusers_data(self):
        return self.data_extractor("mlusers.csv")

    def get_task2_feedback_data(self):
        return self.data_extractor("task2-feedback.csv")

    def get_task4_feedback_data(self):
        return self.data_extractor("task4-feedback.csv")

    def get_movie_latent_semantics_data(self):
        return self.data_extractor("movie_latent_semantic.csv")

    def get_json(self):
        file_loc = os.path.join(self.file_path, "label_movies.json")
        json_movie_label_dict = json.load(open(file_loc))

        return json_movie_label_dict

    def get_relevance_feedback_query_vector(self):
        return self.data_extractor("relevance-feedback-query-vector.csv")

    def get_lsh_details(self):
        return json.load(open(os.path.join(self.file_path, 'task_3_details.json')))


if __name__ == "__main__":
    conf = ParseConfig()
    data_set_location = conf.config_section_mapper("filePath").get("data_set_loc")
    extract_data = DataExtractor(data_set_location)
    data_frame = extract_data.data_extractor("mlmovies.csv")
    print("File columns for mlmovies.csv")
    print("Columns = %s" % (data_frame.columns.values))
