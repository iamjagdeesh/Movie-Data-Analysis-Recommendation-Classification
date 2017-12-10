import argparse
import operator
from collections import Counter
import cvxopt

cvxopt.solvers.options['show_progress'] = False

import config_parser
import data_extractor
import numpy
from scipy.spatial import distance as dist
from util import *

util = Util()
conf = config_parser.ParseConfig()
data_set_loc = conf.config_section_mapper("filePath").get("data_set_loc")
data_extractor_obj = data_extractor.DataExtractor(data_set_loc)
movie_tag_frame = util.get_movie_tag_matrix()

movie_tag_matrix_value = movie_tag_frame.values
(U, s, Vh) = util.SVD(movie_tag_matrix_value)
movie_latent_matrix = U[:, :10]

movies = list(movie_tag_frame.index.values)
tags = list(movie_tag_frame)
label_movies_json_data = data_extractor_obj.get_json()


class Classifier(object):
    def __init__(self, r=0):
        self.util = util
        self.r = r
        self.movie_latent_matrix = movie_latent_matrix
        self.movies = movies
        self.label_movies_json_data = label_movies_json_data
        self.movie_label_dict = self.get_labelled_movies()

    def get_labelled_movies(self):
        movie_label_dict = {}
        for label in self.label_movies_json_data.keys():
            for movieid in self.label_movies_json_data[label]:
                if int(movieid) not in self.movies:
                    print("Invalid movie ID \'" + str(movieid) + "\' entered. Skipping this movie")
                    continue
                movie_label_dict[int(movieid)] = label

        return movie_label_dict

    def find_label_RNN(self, query_movie_id):
        movie_tag_matrix = self.movie_latent_matrix
        query_movie_index = self.movies.index(query_movie_id)
        query_movie_vector = movie_tag_matrix[query_movie_index]

        distance_to_labelled_movies = {}
        for labelled_movie in self.movie_label_dict.keys():
            labelled_movie_index = self.movies.index(labelled_movie)
            labelled_movie_vector = movie_tag_matrix[labelled_movie_index]
            distance_to_labelled_movies[labelled_movie] = dist.euclidean(query_movie_vector, labelled_movie_vector)

        distance_to_labelled_movies_sorted = sorted(distance_to_labelled_movies.items(), key=operator.itemgetter(1))
        distance_to_labelled_movies_sorted = distance_to_labelled_movies_sorted[0:self.r]

        label_count_dict = Counter()
        for (labelled_movie, distance) in distance_to_labelled_movies_sorted:
            label_count_dict[self.movie_label_dict[labelled_movie]] += 1

        label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        (label, count) = label_count_dict_sorted[0]

        return label

    def predict_using_DTC(self, tree, movie):
        if tree.dominant_label:
            return tree.dominant_label
        movie_value_for_tag = self.movie_latent_matrix[movies.index(movie)][tree.feature_index]
        if movie_value_for_tag > tree.mean_value:
            return self.predict_using_DTC(tree.right, movie)
        else:
            return self.predict_using_DTC(tree.left, movie)

    def demo_output(self, model):
        predicted_label = None
        tree = None
        if model == "DTC":
            node = Node(self.movie_label_dict)
            tree = node.construct_tree()
        while True:
            query_movie_id = input("Enter the movie ID for prediction: ")
            query_movie_id = int(query_movie_id)
            if query_movie_id not in self.movies:
                print("Invalid movie ID entered. Skipping this movie!")
                continue
            if model == "RNN":
                predicted_label = self.find_label_RNN(query_movie_id)
            elif model == "DTC":
                predicted_label = self.predict_using_DTC(tree, query_movie_id)
            print("Entered movie: " + str(query_movie_id) + " - " + self.util.get_movie_name_for_id(query_movie_id))
            print("Predicted label: " + str(predicted_label))
            confirmation = input("Are you done querying? (y/Y/n/N): ")
            if confirmation == "y" or confirmation == "Y":
                break


class Node(object):
    def __init__(self, movie_label_dict):
        self.left = None
        self.right = None
        self.data = movie_label_dict
        self.dominant_label = False
        self.parent = None
        self.feature_index = self.find_best_feature()
        self.mean_value = self.find_mean()

    def find_mean(self):
        if self.feature_index is None:
            return None
        indexes_of_movies = [movies.index(each) for each in self.data.keys()]
        matrix_of_movies = movie_latent_matrix[indexes_of_movies, self.feature_index]
        mean_of_movies = numpy.nanmean(matrix_of_movies)

        return mean_of_movies

    def calculate_fsr(self, feature_index, movies_of_class1, movies_of_class2):
        indexes_of_class1_movies = [movies.index(each) for each in movies_of_class1]
        indexes_of_class2_movies = [movies.index(each) for each in movies_of_class2]
        matrix_of_class1_movies = movie_latent_matrix[indexes_of_class1_movies, feature_index]
        matrix_of_class2_movies = movie_latent_matrix[indexes_of_class2_movies, feature_index]
        mean_of_class1 = numpy.nanmean(matrix_of_class1_movies)
        mean_of_class2 = numpy.nanmean(matrix_of_class2_movies)
        variance_of_class1 = numpy.nanvar(matrix_of_class1_movies)
        variance_of_class2 = numpy.nanvar(matrix_of_class2_movies)

        try:
            fsr = ((mean_of_class1 - mean_of_class2)**2) / ((variance_of_class1**2) + (variance_of_class2**2))
        except ZeroDivisionError:
            fsr = float("-inf")

        return fsr

    def find_best_feature(self):
        labels = list(set(self.data.values()))
        if len(labels) < 2:
            return None
        movieids = list(self.data.keys())

        label_movie_dict = {}
        for movieid in movieids:
            label = self.data[movieid]
            if label in label_movie_dict.keys():
                label_movie_dict[label].append(movieid)
            else:
                label_movie_dict[label] = [movieid]

        fsr_list = []
        for feature_index in range(0, 10):
            fsr = 0
            count = 0
            for i in range(0, len(labels) - 1):
                for j in range(i+1, len(labels)):
                    fsr += self.calculate_fsr(feature_index, label_movie_dict[labels[i]], label_movie_dict[labels[j]])
                    count += 1
            avg_fsr = fsr / float(count)
            fsr_list.append(avg_fsr)
        best_feature_index, value = max(enumerate(fsr_list), key=operator.itemgetter(1))

        return best_feature_index

    def check_dominancy(self, movie_label_dict_values):
        label_count_dict = Counter(movie_label_dict_values)
        label_count_dict_sorted = sorted(label_count_dict.items(), key=operator.itemgetter(1), reverse=True)
        (dominant_label, dominant_count) = label_count_dict_sorted[0]
        dominancy = float(dominant_count) / len(movie_label_dict_values)
        if dominancy > 0.85:
            return dominant_label
        else:
            return False

    def construct_tree(self):
        movie_label_dict_values = self.data.values()
        if len(movie_label_dict_values) == 0:
            return None
        dominant_label = self.check_dominancy(movie_label_dict_values)
        if dominant_label:
            self.dominant_label = dominant_label
            return self
        left_movie_label_dict = {}
        right_movie_label_dict = {}
        for (movie, label) in self.data.items():
            movie_value_for_tag = movie_latent_matrix[movies.index(movie)][self.feature_index]
            if movie_value_for_tag > self.mean_value:
                right_movie_label_dict[movie] = label
            else:
                left_movie_label_dict[movie] = label
        self.left = Node(left_movie_label_dict).construct_tree()
        self.right = Node(right_movie_label_dict).construct_tree()

        return self

class SupportVectorMachine(object):
    """The Support Vector Machine classifier.
    """
    def __init__(self, C=1, kernel=rbf_kernel, power=4, gamma=None, coef=4):
        self.C = C
        self.kernel = kernel
        self.power = power
        self.gamma = gamma
        self.coef = coef
        self.lagr_multipliers = None
        self.support_vectors = None
        self.support_vector_labels = None
        self.intercept = None

    def fit(self, X, y):

        n_samples, n_features = numpy.shape(X)

        # Set gamma to 1/n_features by default
        if not self.gamma:
            self.gamma = 1 / n_features

        # Initialize kernel method with parameters
        self.kernel = self.kernel(
            power=self.power,
            gamma=self.gamma,
            coef=self.coef)

        # Calculate kernel matrix
        kernel_matrix = numpy.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                kernel_matrix[i, j] = self.kernel(X[i], X[j])

        # Define the quadratic optimization problem
        P = cvxopt.matrix(numpy.outer(y, y) * kernel_matrix, tc='d')
        q = cvxopt.matrix(numpy.ones(n_samples) * -1)
        A = cvxopt.matrix(y, (1, n_samples), tc='d')
        b = cvxopt.matrix(0, tc='d')

        if not self.C:
            G = cvxopt.matrix(numpy.identity(n_samples) * -1)
            h = cvxopt.matrix(numpy.zeros(n_samples))
        else:
            G_max = numpy.identity(n_samples) * -1
            G_min = numpy.identity(n_samples)
            G = cvxopt.matrix(numpy.vstack((G_max, G_min)))
            h_max = cvxopt.matrix(numpy.zeros(n_samples))
            h_min = cvxopt.matrix(numpy.ones(n_samples) * self.C)
            h = cvxopt.matrix(numpy.vstack((h_max, h_min)))

        # Solve the quadratic optimization problem using cvxopt
        minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

        # Lagrange multipliers
        lagr_mult = numpy.ravel(minimization['x'])

        # Extract support vectors
        # Get indexes of non-zero lagr. multipiers
        idx = lagr_mult > 1e-10
        # Get the corresponding lagr. multipliers
        self.lagr_multipliers = lagr_mult[idx]
        # Get the samples that will act as support vectors
        self.support_vectors = X[idx]
        # Get the corresponding labels
        self.support_vector_labels = y[idx]

        # Calculate intercept with first support vector
        self.intercept = self.support_vector_labels[0]
        for i in range(len(self.lagr_multipliers)):
            self.intercept -= self.lagr_multipliers[i] * self.support_vector_labels[
                i] * self.kernel(self.support_vectors[i], self.support_vectors[0])

    def predict(self, X):
        y_pred = []
        # Iterate through list of samples and make predictions
        for sample in X:
            prediction = 0
            # Determine the label of the sample by the support vectors
            for i in range(len(self.lagr_multipliers)):
                prediction += self.lagr_multipliers[i] * self.support_vector_labels[
                    i] * self.kernel(self.support_vectors[i], sample)
            prediction += self.intercept
            y_pred.append(numpy.sign(prediction))
        return numpy.array(y_pred)


class RunSvm(object):

    def get_all_pairs_of_labels(self):
        labels = list(label_movies_json_data.keys())
        labels.sort()
        label_pairs = []
        for i in range(0, len(labels) - 1):
            for j in range(i + 1, len(labels)):
                label_pairs.append((labels[i], labels[j]))

        return label_pairs

    def get_label_vectors_dict(self, label0, label1):
        indexes_of_label0_movies = [movies.index(each) for each in label_movies_json_data[label0]]
        indexes_of_label1_movies = [movies.index(each) for each in label_movies_json_data[label1]]
        matrix_of_label0_movies = movie_latent_matrix[indexes_of_label0_movies, :]
        matrix_of_label1_movies = movie_latent_matrix[indexes_of_label1_movies, :]

        label_vectors_dict = {}
        label_vectors_dict[-1] = matrix_of_label0_movies
        label_vectors_dict[1] = matrix_of_label1_movies

        return label_vectors_dict

    def get_labelpair_clf_dict(self):
        label_pairs = self.get_all_pairs_of_labels()
        labelpair_clf_dict = {}
        for label0, label1 in label_pairs:
            label_vectors_dict = self.get_label_vectors_dict(label0, label1)
            concatenated_data = numpy.concatenate((label_vectors_dict[-1], label_vectors_dict[1]), axis=0)
            concatenated_labels = []
            for i in range(0, len(label_vectors_dict[-1])):
                concatenated_labels.append(-1)
            for j in range(0, len(label_vectors_dict[1])):
                concatenated_labels.append(1)
            concatenated_labels = numpy.array(concatenated_labels)
            clf = SupportVectorMachine(kernel=rbf_kernel, power=4, coef=1)
            clf.fit(concatenated_data, concatenated_labels)
            labelpair_clf_dict[(label0, label1)] = clf

            return labelpair_clf_dict

    def get_label(self, labelpair_clf_dict, query_movie_id):
        movie_vector = movie_latent_matrix[movies.index(query_movie_id)]
        label_count = Counter()
        for label0, label1 in labelpair_clf_dict.keys():
            clf = labelpair_clf_dict[(label0, label1)]
            label_sign = clf.predict([movie_vector])
            if label_sign[0] == -1:
                label_count[label0] += 1
            else:
                label_count[label1] += 1
        label_count_dict_sorted = sorted(label_count.items(), key=operator.itemgetter(1), reverse=True)
        (label, max) = label_count_dict_sorted[0]

        return label

    def starting_function(self):
        labelpair_clf_dict = self.get_labelpair_clf_dict()
        while True:
            query_movie_id = input("Enter the movie ID you want to know the predicted label: ")
            query_movie_id = int(query_movie_id)
            if query_movie_id not in movies:
                print("Invalid movie ID entered, hence skipping this movie!")
                continue
            predicted_label = self.get_label(labelpair_clf_dict, query_movie_id)
            print("Entered movie: " + str(query_movie_id) + " - " + util.get_movie_name_for_id(query_movie_id))
            print("Predicted label: " + str(predicted_label))
            confirmation = input("Are you done querying? (y/Y/n/N): ")
            if confirmation == "y" or confirmation == "Y":
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='phase_3_task_5.py model',
    )
    parser.add_argument('model', action="store", choices=['RNN', 'SVM', 'DTC'])
    ip = vars(parser.parse_args())
    model = ip['model']
    r = 1
    if model == "SVM":
        run_svm_obj = RunSvm()
        run_svm_obj.starting_function()
    elif model == "RNN":
        r = int(input("Enter the value for r: "))
        obj = Classifier(r)
        obj.demo_output(model)
    else:
        obj = Classifier(r)
        obj.demo_output(model)