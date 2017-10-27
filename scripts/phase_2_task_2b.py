from coactor_coactor_matrix import CoactorCoactorMatrix
from util import Util


class CoactorCoactorSVD(object):
    def __init__(self):
        self.util = Util()
        self.coactor_coactor_matrix_object = CoactorCoactorMatrix()
        self.coactor_coactor_similarity_matrix, self.actor_ids = self.coactor_coactor_matrix_object.fetchCoactorCoactorSimilarityMatrix()
        self.u, self.s, self.vt = self.util.SVD(self.coactor_coactor_similarity_matrix)


    def get_actor_names_list(self):
        """
                Fetches List of Actors Names
                :return: List of Actor Names
        """
        actor_names_list = []
        for actor in self.actor_ids:
            actor_names_list.append(self.util.get_actor_name_for_id(actor))

        return actor_names_list

    def get_partitions(self, no_of_partitions):
        """
                Fetches all the non-overlapping partitions
                :param no_of_partitions: no of partitions the space needs to be partitioned to
        """
        actor_names_list = self.get_actor_names_list()
        groupings = self.util.partition_factor_matrix(self.u, no_of_partitions, actor_names_list)

        return groupings

    def print_partitioned_actors(self, no_of_partitions):
        """
                Prints the partitions and the actors in each partition
                :param no_of_partitions: no of partitions the space needs to be partitioned to
        """
        groupings = self.get_partitions(no_of_partitions)
        self.util.print_partitioned_entities(groupings)

    def print_latent_semantics(self, r):
        """
                Pretty print latent semantics
                :param r:
        """
        latent_semantics = self.util.get_latent_semantics(r, self.vt)
        actor_names_list = self.get_actor_names_list()
        self.util.print_latent_semantics(latent_semantics, actor_names_list)


if __name__ == "__main__":
    obj = CoactorCoactorSVD()
    obj.print_latent_semantics(3)
    print("\n\n\n")
    obj.print_partitioned_actors(3)
