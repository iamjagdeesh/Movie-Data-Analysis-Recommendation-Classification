import argparse
import logging
import operator

import numpy
import pandas as pd
from actor_actor_similarity_matrix import ActorActorMatrix
from config_parser import ParseConfig
from data_extractor import DataExtractor
from phase_2_task_1b import SvdGenreActor
from util import Util

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

conf = ParseConfig()
util = Util()
actor_actor_matrix_obj = ActorActorMatrix()


class SimilarActors(ActorActorMatrix):

    def __init__(self):
        """
        Initialiazing the data extractor object to get data from the csv files

        """
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        actor_actor_matrix_obj.fetchActorActorSimilarityMatrix()

    def get_actor_actor_vector(self, actorid):
        """
        Function to find similarity between actors using actor-actor similarity vector in tag space using tf-idf
        :param actorid:
        :return:
        """
        (matrix, actorids) = self.fetchActorActorSimilarityMatrix()
        #In the pre-processing task above command should be run so that actor_actor_similarity matrix will be generated
        #and saved as csv which can be used multiple number of times. Will comment the above line, when its done.

        # Loading the required actor_actor_similarity matrix from csv
        df = pd.DataFrame(pd.read_csv('actor_actor_matrix.csv', header=None))
        matrix = df.values

        actorids = util.get_sorted_actor_ids()

        index_actor = None
        for i,j in enumerate(actorids):
            if j == actorid:
                index_actor = i
                break

        if index_actor==None:
            print ("Actor Id not found.")
            return None

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actorids, actor_row))
        del actor_actor_dict[actorid]

        actor_actor_name_dict = {}
        for key in actor_actor_dict.keys():
            actor_actor_name_dict[util.get_actor_name_for_id(int(key))] = actor_actor_dict[key]

        actor_actor_name_dict = sorted(actor_actor_name_dict.items(), key=operator.itemgetter(1), reverse=True)
        print(actor_actor_name_dict[0:10])
        return actor_actor_name_dict

class RelatedActorsSvd(SvdGenreActor):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def get_related_actors_svd(self, actorid):
        """
        Function to find similarity between actors using actor-actor similarity vector in tag space using svd
        :param actorid:
        :return:
        """

        actor_actor_matrix = actor_actor_matrix_obj.fetchActorActorSimilarityMatrix()


        # Loading the required dataset
        df = pd.DataFrame(pd.read_csv('actor_tag_matrix.csv', header=None))
        df1 = df.values

        (U, s, Vh) = util.SVD(df1)

        actor_latent_matrix = U[:, :5]

        latent_actor_matrix = actor_latent_matrix.transpose()
        actor_actor_matrix = numpy.dot(actor_latent_matrix, latent_actor_matrix)
        numpy.savetxt("actor_actor_matrix_with_svd_latent_values.csv", actor_actor_matrix, delimiter=",")

        df = pd.DataFrame(pd.read_csv('actor_actor_matrix_with_svd_latent_values.csv', header=None))
        matrix = df.values

        actorids = util.get_sorted_actor_ids()

        index_actor = None
        for i, j in enumerate(actorids):
            if j == actorid:
                index_actor = i
                break

        if index_actor == None:
            print("Actor Id not found.")
            return None

        actor_names = []
        for actor_id in actorids:
            actor_name = util.get_actor_name_for_id(int(actor_id))
            actor_names = actor_names + [actor_name]

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actor_names, actor_row))
        del actor_actor_dict[util.get_actor_name_for_id(int(actorid))]

        # for key in actor_actor_dict.keys():
        #      actor_actor_dict[key] = abs(actor_actor_dict[key])

        actor_actor_dict = sorted(actor_actor_dict.items(), key=operator.itemgetter(1), reverse=True)
        print(actor_actor_dict[0:10])
        return actor_actor_dict

class RelatedActorsPca():
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")

    def get_related_actors_pca(self, actorid):
        """
        Function to find similarity between actors using actor-actor similarity vector in tag space using pca
        :param actorid:
        :return:
        """

        # Loading the required dataset
        df = pd.DataFrame(pd.read_csv('actor_tag_matrix.csv', header=None))
        df1 = df.values

        (U, s, Vh) = util.PCA(df1)

        tag_latent_matrix = U[:, :5]
        actor_tag_matrix = df1
        actor_latent_matrix = numpy.dot(actor_tag_matrix, tag_latent_matrix)
        actorids = util.get_sorted_actor_ids()

        latent_actor_matrix = actor_latent_matrix.transpose()
        actor_actor_matrix = numpy.dot(actor_latent_matrix, latent_actor_matrix)
        numpy.savetxt("actor_actor_matrix_with_pca_latent_values.csv", actor_actor_matrix, delimiter=",")

        df = pd.DataFrame(pd.read_csv('actor_actor_matrix_with_pca_latent_values.csv', header=None))
        matrix = df.values

        actorids = util.get_sorted_actor_ids()

        index_actor = None
        for i, j in enumerate(actorids):
            if j == actorid:
                index_actor = i
                break

        if index_actor == None:
            print("Actor Id not found.")
            return None

        actor_names = []
        for actor_id in actorids:
            actor_name = util.get_actor_name_for_id(int(actor_id))
            actor_names = actor_names + [actor_name]

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actor_names, actor_row))
        del actor_actor_dict[util.get_actor_name_for_id(int(actorid))]

        # for key in actor_actor_dict.keys():
        #     actor_actor_dict[key] = abs(actor_actor_dict[key])

        actor_actor_dict = sorted(actor_actor_dict.items(), key=operator.itemgetter(1), reverse=True)
        print(actor_actor_dict[0:10])
        return actor_actor_dict

class LdaActorTag(object):
    def __init__(self):
        super().__init__()
        self.data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
        self.data_extractor = DataExtractor(self.data_set_loc)
        self.util = Util()

    def get_related_actors_lda(self, actorid):
        """
        Function to find similarity between actors using actor-actor similarity vector in tag space using lda
        :param actorid:
        :return:
        """
        mov_act = self.data_extractor.get_movie_actor_data()
        ml_tag = self.data_extractor.get_mltags_data()
        genome_tag = self.data_extractor.get_genome_tags_data()
        actor_info = self.data_extractor.get_imdb_actor_info_data()
        actor_movie_info = mov_act.merge(actor_info, how="left", left_on="actorid", right_on="id")
        tag_data_frame = ml_tag.merge(genome_tag, how="left", left_on="tagid", right_on="tagId")
        merged_data_frame = tag_data_frame.merge(actor_movie_info, how="left", on="movieid")

        merged_data_frame = merged_data_frame.fillna('')
        tag_df = merged_data_frame.groupby(['actorid'])['tag'].apply(list).reset_index()

        tag_df = tag_df.sort_values('actorid')
        actorid_list = tag_df.actorid.tolist()
        tag_df = list(tag_df.iloc[:,1])

        (U, Vh) = self.util.LDA(tag_df, num_topics=5, num_features=1000)

        actor_topic_matrix = self.util.get_doc_topic_matrix(U, num_docs=len(actorid_list), num_topics=5)
        topic_actor_matrix = actor_topic_matrix.transpose()
        actor_actor_matrix = numpy.dot(actor_topic_matrix,topic_actor_matrix)

        numpy.savetxt("actor_actor_matrix_with_svd_latent_values.csv", actor_actor_matrix, delimiter=",")

        df = pd.DataFrame(pd.read_csv('actor_actor_matrix_with_svd_latent_values.csv', header=None))
        matrix = df.values

        actorids = self.util.get_sorted_actor_ids()

        index_actor = None
        for i, j in enumerate(actorids):
            if j == actorid:
                index_actor = i
                break

        if index_actor == None:
            print("Actor Id not found.")
            return None

        actor_names = []
        for actor_id in actorids:
            actor_name = self.util.get_actor_name_for_id(int(actor_id))
            actor_names = actor_names + [actor_name]

        actor_row = matrix[index_actor].tolist()
        actor_actor_dict = dict(zip(actor_names, actor_row))
        del actor_actor_dict[self.util.get_actor_name_for_id(int(actorid))]

        # for key in actor_actor_dict.keys():
        #     actor_actor_dict[key] = abs(actor_actor_dict[key])

        actor_actor_dict = sorted(actor_actor_dict.items(), key=operator.itemgetter(1), reverse=True)
        print(actor_actor_dict[0:10])
        return actor_actor_dict[0:10]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='phase_2_task_1a.py Action pca',
    )
    parser.add_argument('actorid', action="store", type=int)
    parser.add_argument('model', action="store", choices=set(('tfidf', 'pca', 'svd', 'lda')))
    input = vars(parser.parse_args())
    actorid = input['actorid']
    model = input['model']

    if model == 'pca':
        obj_pca = RelatedActorsPca()
        actor_actor_dict = obj_pca.get_related_actors_pca(actorid)
    elif model == 'svd':
        obj_svd = RelatedActorsSvd()
        actor_actor_dict = obj_svd.get_related_actors_svd(actorid)
    elif model == 'lda':
        obj_lda = LdaActorTag()
        actor_actor_dict = obj_lda.get_related_actors_lda(actorid)
    else:
        obj_tfidf = SimilarActors()
        actor_actor_dict = obj_tfidf.get_actor_actor_vector(actorid)