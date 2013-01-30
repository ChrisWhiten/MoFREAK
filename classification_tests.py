# This script is used to test different classifiers on the mofreak data.  Quick and dirty script, not necessarily extensible.
import numpy
import pylab
from sklearn import svm, pipeline
from sklearn.kernel_approximation import RBFSampler, AdditiveChi2Sampler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Constants
NUMBER_OF_ACTIONS = 6
NUMBER_OF_PEOPLE = 25
NUMBER_OF_VIDEOS_PER_PERSON_PER_ACTION = 4
NUMBER_OF_CLUSTERS = 600
NUMBER_OF_VIDEOS = 599

FENG = True
EHSAN = False

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

def histogram_intersection(A, B):
  ret_val = 0
  for i in xrange(A.shape[0]):
    ret_val += min(A[i], B[i])

  return ret_val

def intersection(A,B):
  B = B.T
  rows = A.shape[0]
  cols = B.shape[1]
  kernel = numpy.zeros(shape = (rows, cols))

  for row in xrange(rows):
    for col in xrange(cols):
      kernel[row, col] = histogram_intersection(A[row, :], B[:, col])

  return kernel

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
def loadTrainingAndTestData(features_file, labels_file):
	# group by person.
	grouped_data = []
	grouped_labels = []
	current_indices = []

	label_data = numpy.genfromtxt(labels_file, delimiter = ',')
	training_data = numpy.genfromtxt(features_file, delimiter = ',')

	# group data by people, so we can easily leave-one-out.
	for i in xrange(NUMBER_OF_PEOPLE):
		# person 13 is missing one video...
		MISSING_VIDEO_OFFSET = 0
		if i == 12: 
			MISSING_VIDEO_OFFSET = -1

		#data = numpy.zeros(shape = (NUMBER_OF_PEOPLE * NUMBER_OF_VIDEOS_PER_PERSON_PER_ACTION + MISSING_VIDEO_OFFSET, NUMBER_OF_CLUSTERS))
		#labels = numpy.zeros(shape = (NUMBER_OF_PEOPLE * NUMBER_OF_VIDEOS_PER_PERSON_PER_ACTION + MISSING_VIDEO_OFFSET, 3))
		data = numpy.zeros(shape = (NUMBER_OF_ACTIONS * NUMBER_OF_VIDEOS_PER_PERSON_PER_ACTION + MISSING_VIDEO_OFFSET, NUMBER_OF_CLUSTERS))
		labels = numpy.zeros(shape = (NUMBER_OF_ACTIONS * NUMBER_OF_VIDEOS_PER_PERSON_PER_ACTION + MISSING_VIDEO_OFFSET, 3))

		grouped_data.append(data)
		grouped_labels.append(labels)
		current_indices.append(0) # track current row in each group

	i = 0
	STEP = NUMBER_OF_VIDEOS_PER_PERSON_PER_ACTION
	MISSING_VIDEO_I = 148 

	if EHSAN:
		STEP = NUMBER_OF_VIDEOS_PER_PERSON_PER_ACTION 
		MISSING_VIDEO_I = 148

	# to account for the missing video.  Odd that it's missing!
	# i == 148 or 288 corresponds to the location of the missing video.
	while i < NUMBER_OF_VIDEOS:
		person_index = int(label_data[i, 1])
		current_index = current_indices[person_index - 1]

		# slice corresponding piece of matrix from training data into grouping.
		if i == MISSING_VIDEO_I:
			grouped_data[person_index - 1][current_index : current_index + STEP - 1, :] = training_data[i : i + STEP - 1, :]
			grouped_labels[person_index - 1][current_index : current_index + STEP - 1, :] = label_data[i : i + STEP - 1, :]
			
			current_indices[person_index - 1] += STEP - 1
			i += STEP - 1

		else:
			print i
			#print label_data[i : i + STEP, :]
			#print label_data[i : i + STEP, :].shape

			grouped_labels[person_index - 1][current_index : current_index + STEP, :] = label_data[i : i + STEP, :]
			grouped_data[person_index - 1][current_index : current_index + STEP, :] = training_data[i : i + STEP, :]


			current_indices[person_index - 1] += STEP

			i += STEP

	#print grouped_data[0]
	#print grouped_labels[13]
	#print "exiting"
	#exit()

	return grouped_data, grouped_labels

def generateAllPossibleLeaveOneOutCombosForLibSVM(grouped_data, grouped_labels):
	
	for left_out_person in xrange(NUMBER_OF_PEOPLE):
		rows = NUMBER_OF_VIDEOS - grouped_data[left_out_person].shape[0]
		cols = NUMBER_OF_CLUSTERS

		# the testing data is simply the data from the left out person.
		testing_data = grouped_data[left_out_person]
		testing_labels = grouped_labels[left_out_person]

		# build the training data by concatenating all of the data from each person except the left out person.
		training_data = numpy.zeros(shape = (rows, cols))
		training_labels = numpy.zeros(shape = (rows, 3))
		current_index = 0
		for training_person in xrange(NUMBER_OF_PEOPLE):
			# don't add the left out person to the training set, clearly..
			if training_person == left_out_person:
				continue

			new_rows = grouped_data[training_person].shape[0]
			training_data[current_index : current_index + new_rows, :] = grouped_data[training_person]
			training_labels[current_index : current_index + new_rows, :] = grouped_labels[training_person]

			current_index += new_rows

		# write data file.
		training_filename = "C:/data/kth/chris/run1/left_out_" + str(left_out_person + 1) + ".train"
		setupInLibsvmFormat(training_data, training_labels, training_filename)

		testing_filename = "C:/data/kth/chris/run1/left_out_" + str(left_out_person + 1) + ".test"
		setupInLibsvmFormat(testing_data, testing_labels, testing_filename)


# Build an SVM with a chi-squared kernel for accurate recognition.
def buildClassifiers(grouped_data, grouped_labels):
	scores = []

	for left_out_person in xrange(NUMBER_OF_PEOPLE):
		rows = NUMBER_OF_VIDEOS - grouped_data[left_out_person].shape[0]
		cols = NUMBER_OF_CLUSTERS

		# the testing data is simply the data from the left out person.
		testing_data = grouped_data[left_out_person]
		testing_labels = grouped_labels[left_out_person]

		# build the training data by concatenating all of the data from each person except the left out person.
		training_data = numpy.zeros(shape = (rows, cols))
		training_labels = numpy.zeros(shape = (rows, 3))
		current_index = 0
		for training_person in xrange(NUMBER_OF_PEOPLE):
			# don't add the left out person to the training set, clearly..
			if training_person == left_out_person:
				continue

			new_rows = grouped_data[training_person].shape[0]
			training_data[current_index : current_index + new_rows, :] = grouped_data[training_person]
			training_labels[current_index : current_index + new_rows, :] = grouped_labels[training_person]

			current_index += new_rows

		# for now, remove all columns from labels except first.
		training_labels = training_labels[:, 0]
		testing_labels = testing_labels[:, 0]
		
		print "made it to training"
		#kernel_svm = svm.SVC(gamma = .2, degree = 100)
		#linear_svm = svm.LinearSVC()
		new_svm = svm.SVC(kernel = intersection)
		rf = RandomForestClassifier(n_estimators = 300, min_samples_split = 2, n_jobs = -1, oob_score = True)

		# create a pipeline for kernel approximation
		#feature_map = RBFSampler(gamma = .2, random_state = 1)
		#feature_map = AdditiveChi2Sampler()
		#approx_kernel_svm = pipeline.Pipeline([("feature_map", feature_map), ("svm", svm.LinearSVC())])

		# fit and predict using linear and kernel svm.
		new_svm.fit(training_data, training_labels)
		new_svm_score = new_svm.score(testing_data, testing_labels)

		rf.fit(training_data, training_labels)
		rf_score = rf.score(testing_data, testing_labels)

		#kernel_svm.fit(training_data, training_labels)
		#kernel_svm_score = kernel_svm.score(testing_data, testing_labels)

		#linear_svm.fit(training_data, training_labels)
		#linear_svm_score = linear_svm.score(testing_data, testing_labels)

		#approx_kernel_svm.fit(training_data, training_labels)
		#cs_score = approx_kernel_svm.score(testing_data, testing_labels)

		#score_set = [new_svm_score, kernel_svm_score, linear_svm_score, score]
		score_set = [new_svm_score, rf_score]#, cs_score, rf_score]
		scores.append(score_set)

		#print "linear score: ", linear_svm_score
		#print "kernel score: ", kernel_svm_score
		print "histogram intersection score: ", new_svm_score
		#print "approx chi-squared score: ", cs_score
		print "RF score: ", rf_score

	# for now, return this for plotting.
	print scores
	print "done."
	print "length of scores: ", len(scores)
	summed_chisquared_score = 0
	summed_hi_score = 0
	summed_rf_score = 0
	for i in xrange(NUMBER_OF_PEOPLE):
		summed_hi_score += scores[i][0]
		#summed_chisquared_score += scores[i][0]
		#summed_hi_score += scores[i][1]
		summed_rf_score += scores[i][1]

	#avg_cs_score = summed_chisquared_score/float(NUMBER_OF_PEOPLE)
	avg_hi_score = summed_hi_score/float(NUMBER_OF_PEOPLE)
	#avg_rf_score = summed_rf_score/float(NUMBER_OF_PEOPLE)
	#print "Chi-squared average: ", avg_cs_score
	print "HI average: ", avg_hi_score
	print "RF average: ", avg_rf_score
	return linear_svm

# visualization based on code from 
# http://scikit-learn.org/dev/auto_examples/plot_kernel_approximation.html#example-plot-kernel-approximation-py
def visualize(training_features, training_labels, linear_svm):
	# project the decision surface down to the 2 principal components of the dataset
	# enables us to visualize the dataset in 2D.
	pca = PCA(n_components = 2).fit(training_features)
	X = pca.transform(training_features)

	# generate grid along first 2 princ comps
	multiples = numpy.arange(-2, 2, 0.1) # from -2 to 2, on intervals of size 0.1
	# steps along first component
	first = multiples[:, numpy.newaxis] * pca.components_[0, :]
	# 2nd.
	second = multiples[:, numpy.newaxis] * pca.components_[1, :]
	# combine them
	grid = first[numpy.newaxis, :, :] + second[:, numpy.newaxis, :]
	flat_grid = grid.reshape(-1, training_features.shape[1]) # this was data, not training_features

	# title for the plots
	titles = ['Linear SVM']

	pylab.figure(figsize = (12, 5))

	# predict and plot
	pylab.subplot(1, 2, 1)
	Z = linear_svm.predict(flat_grid)

	# put hte result into a colour plot
	Z = Z.reshape(grid.shape[:-1])
	pylab.contourf(multiples, multiples, Z, cmap = pylab.cm.Paired)
	pylab.axis('off')

	# plot the training points.
	pylab.scatter(X[:, 0], X[:, 1], c = training_labels, cmap = pylab.cm.Paired)

	pylab.title(titles[0])
	pylab.show()
	return

def setupInLibsvmFormat(training_data, label_data, output_filename):
	f = open(output_filename, "w")

	for line in xrange(label_data.shape[0]):
		libsvm_line = ""

		# first, the label.
		libsvm_line += str(int(label_data[line, 0]))
		libsvm_line += " "

		# now the features.
		for feature in xrange(training_data.shape[1]):
			libsvm_line += str(feature + 1)
			libsvm_line += ":"
			libsvm_line += str(training_data[line, feature])
			libsvm_line += " "

		# finally, end the line.
		libsvm_line += "\n"
		f.write(libsvm_line)

	f.close()




# entry point
if __name__ == '__main__':

	data = "C:/data/kth/chris/run1/hist.txt"
	labels = "C:/data/kth/chris/run1/label.txt"
	
	# Step 1: Reprocess the data into the desired format.
	label_data = numpy.genfromtxt(labels, delimiter = ',')
	training_data = numpy.genfromtxt(data, delimiter = ',')
	#setupInLibsvmFormat(training_data, label_data, "entire_dataset_libsvm.txt")

	#file_path = "C:/data/kth/histogramsDev.txt"
	#reprocessData(file_path)

	# Step 2: Load new data into label/feature arrays, with a train and test set.
	grouped_data, grouped_labels = loadTrainingAndTestData(data, labels)

	# Step 2.5: Export all possible leave-one-out combos to libsvm format.
	#generateAllPossibleLeaveOneOutCombosForLibSVM(grouped_data, grouped_labels)

	# Step 3: Build classifiers.
	linear_svm = buildClassifiers(grouped_data, grouped_labels)

	# Step 4: Visualize. [broken] [todo]
	#visualize(training_features, training_labels, linear_svm)