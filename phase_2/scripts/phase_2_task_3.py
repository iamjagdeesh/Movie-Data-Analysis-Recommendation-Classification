import argparse
import logging
import operator
from util import Util
import numpy
import pandas as pd
from actor_actor_similarity_matrix import ActorActorMatrix
from coactor_coactor_matrix import CoactorCoactorMatrix
from config_parser import ParseConfig
from data_extractor import DataExtractor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
log.disabled = True

conf = ParseConfig()

class PageRankActor(ActorActorMatrix):
    """Class to calculate Personalised PageRank"""
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.actor_matrix, self.actorids = self.fetchActorActorSimilarityMatrix()
        self.coactor_obj = CoactorCoactorMatrix()
        self.coactor_matrix, self.coactorids = self.coactor_obj.fetchCoactorCoactorSimilarityMatrix()
        self.util = Util()

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

    def get_seed_matrix(self, transition_df, seed_actors, actorids):
        """
        Function to get the Restart matrix for entries in the seed list
        :param transition_df:
        :param seed_actors:
        :param actorids:
        :return: seed_matrix
        """
        seed_matrix = [0.0 for each in range(len(transition_df.columns))]
        seed_value = float(1 / len(seed_actors))
        for each in seed_actors:
            seed_matrix[list(actorids).index(each)] = seed_value
        return seed_matrix

    def print_actors_and_pageranks(self, page_rank_tuple):
        for first, second in page_rank_tuple:
            print("%s[%s]: %s" % (self.util.get_actor_name_for_id(first), first, second))

    def compute_pagerank(self, seed_actors, actor_matrix, actorids):
        """
        Function to compute the Personalised Pagerank for the given input
        :param seed_actors:
        :param actor_matrix:
        :param actorids:
        :return:
        """
        data_frame = pd.DataFrame(actor_matrix)
        transition_df = self.get_transition_dataframe(data_frame)
        seed_matrix = self.get_seed_matrix(transition_df, seed_actors, actorids)
        result_list = seed_matrix
        temp_list = []
        while(temp_list!=result_list):
            temp_list = result_list
            result_list = list(0.85*numpy.matmul(numpy.array(transition_df.values), numpy.array(result_list))+ 0.15*numpy.array(seed_matrix))
        page_rank_dict = {i: j for i, j in zip(actorids, result_list)}
        sorted_rank = sorted(page_rank_dict.items(), key=operator.itemgetter(1), reverse=True)
        self.print_actors_and_pageranks(sorted_rank[0:len(seed_actors)+10])

    def compute_actors_pagerank(self, seed_list):
        """
        function to trigger Personalized Pagerank calculation for Actor-Actor-Similarity matrix - Task-3a
        :param seed_list:
        :return:
        """
        self.compute_pagerank(seed_list, self.actor_matrix, self.actorids)

    def compute_coactors_pagerank(self, seed_list):
        """
         function to trigger Personalized Pagerank calculation for Coactor-Coactor matrix - Task-3b
         :param seed_list:
         :return:
        """
        self.compute_pagerank(seed_list, self.coactor_matrix, self.coactorids)

if __name__ == "__main__":
    PRA = PageRankActor()
    parser = argparse.ArgumentParser(description='phase_2_task_3.py Actor/Coactor seed_actors')
    parser.add_argument('type', action="store", type=str, choices=set(("actor", "coactor")))
    parser.add_argument('seed_actors', action="store", type=str)
    input = vars(parser.parse_args())
    type = input['type']
    seed_actors = input['seed_actors']
    seed_actor_list = [int(each) for each in seed_actors.split(",")]
    if type == "actor":
        PRA.compute_actors_pagerank(seed_actor_list)
    else:
        PRA.compute_coactors_pagerank(seed_actor_list)
    # PRA.compute_coactors_pagerank([1860883])#3619702, 3426176])  # 2055016])

