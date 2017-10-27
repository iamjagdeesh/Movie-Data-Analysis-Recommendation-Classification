import logging
import os

import pandas as pd
from config_parser import ParseConfig

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class DataExtractor(object):
    def __init__(self, file_path):
        self.file_path = file_path

    def data_extractor(self, file_name):  # return the data frame with respect to the csv file in 'resources' directory
        file_loc = os.path.join(self.file_path, file_name)
        data_frame = pd.read_csv(file_loc)
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


if __name__ == "__main__":
    conf = ParseConfig()
    data_set_location = conf.config_section_mapper("filePath").get("data_set_loc")
    extract_data = DataExtractor(data_set_location)
    data_frame = extract_data.data_extractor("mlmovies.csv")
    log.info("File columns for mlmovies.csv")
    log.info("Columns = %s" % (data_frame.columns.values))
