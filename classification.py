import numpy
from sklearn import svm, pipeline
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler
from sklearn.decomposition import PCA

# this class essentially implements an enum.
# For example, x = Labeling.BOXING (1)
class Labeling:
	BOXING = 1
	HANDCLAPPING = 2
	WALKING = 3
	JOGGING = 4
	RUNNING = 5
	HANDWAVING = 6

def convertLabel(s):
	if s == "boxing":
		return Labeling.BOXING
	elif s == "handclapping":
		return Labeling.HANDCLAPPING
	elif s == "walking":
		return Labeling.WALKING
	elif s == "jogging":
		return Labeling.JOGGING
	elif s == "running":
		return Labeling.RUNNING
	elif s == "handwaving":
		return Labeling.HANDWAVING
	else:
		print s
		exit()
		return -1 # error. unseen class.


# reprocess data so that the first column is a numeric value
# this value corresponds to the class.
def reprocessData(file_path):
	DEPTH_OF_LABELING = 4

	f = open(file_path, 'r')
	out = open(file_path + ".reprocessed.txt", 'w')

	for line in f.readlines():
		# first column is the label, remainder are features.
		label_and_features = line.split(",")
		# label is currently a path, split that path.
		# we will reprint the file with the label as a numeric
		words = label_and_features[0].split("\\")
		
		new_line = ""
		new_line += str(convertLabel(words[DEPTH_OF_LABELING]))
		new_line += ","
		new_line += ",".join(label_and_features[1:])
		
		out.write(new_line)

	out.close()
	f.close()

# load and parse data for SVM
def loadTrainingAndTestData(training_path, testing_path):
	training_data = numpy.genfromtxt(training_path, delimiter = ',')
	training_labels = training_data[:, 0]
	training_features = training_data[:, 1:]

	testing_data = numpy.genfromtxt(testing_path, delimiter = ',')
	testing_labels = testing_data[:, 0]
	testing_features = testing_data[:, 1:]

	return training_labels, training_features, testing_labels, testing_features

# Build an SVM with a chi-squared kernel for accurate recognition.
def buildClassifiers(training_labels, training_features, testing_labels, testing_features):
	kernel_svm = svm.SVC(gamma = .2, degree = 100)
	linear_svm = svm.LinearSVC()

	# create a pipeline for kernel approximation
	#feature_map = RBFSampler(gamma = .2, random_state = 1)
	feature_map = AdditiveChi2Sampler()
	approx_kernel_svm = pipeline.Pipeline([("feature_map", feature_map), ("svm", svm.LinearSVC())])

	# fit and predict using linear and kernel svm.
	kernel_svm.fit(training_features, training_labels)
	kernel_svm_score = kernel_svm.score(testing_features, testing_labels)

	linear_svm.fit(training_features, training_labels)
	linear_svm_score = linear_svm.score(testing_features, testing_labels)

	print "linear score: ", linear_svm_score
	print "kernel score: ", kernel_svm_score

	sample_sizes = 50 * numpy.arange(1, 10)
	approx_kernel_scores = []
	"""
	for D in sample_sizes:
		approx_kernel_svm.set_params(feature_map__sample_steps = D)
		approx_kernel_svm.set_params(feature_map__sample_interval = D)
		#approx_kernel_svm.set_params(feature_map__n_components = D)
		approx_kernel_svm.fit(training_features, training_labels)
		score = approx_kernel_svm.score(testing_features, testing_labels)
		print "approx score: ", score
		approx_kernel_scores.append(score)
	"""

	approx_kernel_svm.fit(training_features, training_labels)
	score = approx_kernel_svm.score(testing_features, testing_labels)
	print "approx score: ", score
	return

# entry point
if __name__ == '__main__':

	# Step 1: Reprocess the data into the desired format.
	#file_path = "C:/data/kth/histogramsDev.txt"
	#reprocessData(file_path)

	# Step 2: Load new data into label/feature arrays, with a train and test set.
	training_path = "C:/data/kth/histogramsDev.txt.reprocessed.txt"
	testing_path = "C:/data/kth/histogramsEval.txt.reprocessed.txt"
	training_labels, training_features, testing_labels, testing_features = loadTrainingAndTestData(training_path, testing_path)

	# Step 3: Build classifiers.
	buildClassifiers(training_labels, training_features, testing_labels, testing_features)