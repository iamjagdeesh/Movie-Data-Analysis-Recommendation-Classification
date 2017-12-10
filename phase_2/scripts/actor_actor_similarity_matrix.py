import numpy
import pandas as pd

from phase1_task_1 import ActorTag


class ActorActorMatrix(ActorTag):
    def fetchActorActorSimilarityMatrix(self):
        actor_info = self.data_extractor.get_imdb_actor_info_data()
        actor_ids = actor_info.id
        actor_ids = actor_ids.sort_values()
        list_of_tag_vectors = []
        for actorid in actor_ids:
            tag_vector = self.merge_movie_actor_and_tag(actorid, 'TFIDF')
            list_of_tag_vectors.append(tag_vector)
        actor_tag_matrix = pd.DataFrame(list_of_tag_vectors).values
        where_are_NaNs = numpy.isnan(actor_tag_matrix)
        actor_tag_matrix[where_are_NaNs] = 0

        tag_actor_matrix = actor_tag_matrix.transpose()

        actor_actor_matrix = numpy.dot(actor_tag_matrix, tag_actor_matrix)

        #matrix.to_csv('actor_tag_matrix.csv', index=False, encoding='utf-8')
        numpy.savetxt("actor_tag_matrix.csv", actor_tag_matrix, delimiter=",")
        numpy.savetxt("tag_actor_matrix.csv", tag_actor_matrix, delimiter=",")
        numpy.savetxt("actor_actor_matrix.csv", actor_actor_matrix, delimiter=",")
        return actor_actor_matrix, actor_ids


if __name__ == "__main__":
    obj = ActorActorMatrix()
    matrix = obj.fetchActorActorSimilarityMatrix()