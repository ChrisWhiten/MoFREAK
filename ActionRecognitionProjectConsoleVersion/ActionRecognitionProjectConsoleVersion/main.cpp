#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>

#include <boost\filesystem.hpp>

#include <opencv2\opencv.hpp>

#include "MoFREAKUtilities.h"
#include "Clustering.h"
#include "BagOfWordsRepresentation.h"
#include "SVMInterface.h"

using namespace std;
using namespace boost::filesystem;

bool DISTRIBUTED = false;

string MOFREAK_PATH = "C:/data/TRECVID/mosift/"; // because converting mosift to mofreak just puts them in the same folder as the mosift points. That's fine.
string MOSIFT_DIR = "C:/data/TRECVID/mosift/";
//string MOSIFT_FILE = "C:/data/LGW_20071101_E1_CAM1.mpeg.Pointing.txt.mosift.0";
//string VIDEO_PATH = "C:/data/TRECVID/gatwick_dev08/dev/LGW_20071101_E1_CAM1.mpeg/LGW_20071101_E1_CAM1.mpeg";
string VIDEO_PATH = "C:/data/TRECVID/videos/";
string SVM_PATH = "C:/data/TRECVID/svm/";

// for clustering, separate mofreak into pos and neg examples.
string MOFREAK_NEG_PATH = "C:/data/TRECVID/negative_mofreak_examples/";
string MOFREAK_POS_PATH = "C:/data/TRECVID/positive_mofreak_examples/";

string DETECTION_TARGET_VIDEO_FILE = "C:/data/TRECVID/something.mpeg";


const int FEATURE_DIMENSIONALITY = 8;
const int NUM_CLUSTERS = 1000;
const int NUM_CLASSES = 2;//1;//2;
const int ALPHA = 20; // for sliding window...

vector<int> possible_classes;

MoFREAKUtilities mofreak;
vector<MoFREAKFeature> mofreak_ftrs;
SVMInterface svm_interface;

enum states {CLUSTER, CLASSIFY, CONVERT, PICK_CLUSTERS, COMPUTE_BOW_HISTOGRAMS, DETECT, TRAIN, GET_SVM_RESPONSES};

struct Detection
{
	int start_frame;
	int end_frame;
	float score;
	string video_name;

	// override the < operator for sorting.
	bool operator < (const Detection &det) const
	{
		return (score < det.score);
	};
};

void clusterMoFREAKPoints()
{
	// gather mofreak files...
	cout << "Gathering MoFREAK Features..." << endl;
	vector<std::string> mofreak_files;

	directory_iterator end_iter;

	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				mofreak_files.push_back(current_file.string());
				mofreak.readMoFREAKFeatures(mofreak_files.back());
			}
		}
	}

	mofreak_ftrs = mofreak.getMoFREAKFeatures();
	cout << "MoFREAK features gathered." << endl;

	// organize pts into a cv::Mat.
	const int FEATURE_DIMENSIONALITY = 32;//32;//80;//192;//65;//192;
	const int NUM_CLUSTERS = 600;
	const int POINTS_TO_SAMPLE = 12000;
	const int NUM_CLASSES = 6;
	const int NUMBER_OF_PEOPLE = 25;
	cv::Mat data_pts(mofreak_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, POINTS_TO_SAMPLE, NUM_CLASSES, possible_classes);
	clustering.setAppearanceDescriptor(16, true);
	clustering.setMotionDescriptor(16, true);

	cout << "Formatting features..." << endl;

	clustering.buildDataFromMoFREAK(mofreak_ftrs, false, false);

	cout << "Clustering..." << endl;

	//clustering.clusterWithKMeans();
	clustering.randomClusters();

	// print clusters to file
	cout << "Writing clusters to file..." << endl;
	clustering.writeClusters();

	cout << "Computing BOW Representation..." << endl;

	BagOfWordsRepresentation bow_rep(mofreak_files, NUM_CLUSTERS, FEATURE_DIMENSIONALITY, NUMBER_OF_PEOPLE, true, true);
	bow_rep.setAppearanceDescriptor(16, true);
	bow_rep.setMotionDescriptor(16, true);
	bow_rep.computeBagOfWords();

	cout << "BOW Representation computed." << endl;
}


void evaluateSVMWithLeaveOneOut()
{
	// gather testing and training files...
	vector<std::string> testing_files;
	vector<std::string> training_files;

	directory_iterator end_iter;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 5, 5) == "train")
			{
				training_files.push_back(current_file.string());
			}
			else if (filename.substr(filename.length() - 4, 4) == "test")
			{
				testing_files.push_back(current_file.string());
			}
		}
	}


	// evaluate the SVM with leave-one-out.
	ofstream output_file("C:/data/kth/chris/svm_results.txt");

	double summed_accuracy = 0.0;
	for (unsigned i = 0; i < training_files.size(); ++i)
	{
		SVMInterface svm_guy;
		// tell GUI where we're at in the l-o-o process
		cout << "Leaving out person " << i + 1 << endl;

		// build model.
		svm_guy.trainModelProb(training_files[i]);

		// get accuracy.
		string test_filename = testing_files[i];
		double accuracy = svm_guy.testModelProb(test_filename);
		summed_accuracy += accuracy;

		// debugging...print to testing file.
		output_file << training_files[i] <<", " << testing_files[i] << ", " << accuracy << std::endl;
	}	

	output_file.close();

	// output average accuracy.
	double denominator = (double)training_files.size();
	double average_accuracy = summed_accuracy/denominator;

	cout << "Averaged accuracy: " << average_accuracy << "%" << endl;
}

void convertMoSIFTToMoFREAK()
{
	// iterate mosift directory.
	directory_iterator end_iter;

	if (DISTRIBUTED)
	{
		MOSIFT_DIR = "mosift/";
		VIDEO_PATH = "videos/";
	}
	cout << "Step 1..." << endl;
	for (directory_iterator dir_iter(MOSIFT_DIR); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			// parse mosift files so first x characters gets us the video name.
			path current_file = dir_iter->path();
			string mosift_path = current_file.generic_string();
			string mosift_filename = current_file.filename().generic_string();
			cout << "filename: " << mosift_filename << endl;

			if (mosift_filename.substr(mosift_filename.length() - 3, 3) == "svm")
			{
				continue;
			}

			string video_file = VIDEO_PATH;
			video_file.append(mosift_filename.substr(0, 25));
			cout << "vid: " << video_file << endl;

			
			string mofreak_path = mosift_path;
			mofreak_path.append(".mofreak");
			// compute. write.
			MoFREAKUtilities mofreak;
			
			cout << "Buidling " << mofreak_path << " from " << mosift_path << " with video " << video_file << endl;
			mofreak.buildMoFREAKFeaturesFromMoSIFT(mosift_path, video_file, mofreak_path);
			mosift_path.append(".mofreak");
			cout << "writing mofreak features to file..." << endl;
			mofreak.writeMoFREAKFeaturesToFile(mosift_path);
			cout << "Completed " << mosift_path << endl;
		}
	}
}

void pickClusters()
{
	// load all MoFREAK files.
	// So, we will have one folder with all MoFREAK files in it.  Simple...
	cout << "Gathering MoFREAK Features..." << endl;
	vector<std::string> mofreak_files;

	// POSITIVE EXAMPLES
	directory_iterator end_iter;

	for (directory_iterator dir_iter(MOFREAK_POS_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				mofreak.readMoFREAKFeatures(current_file.string());
			}
		}
	}

	mofreak.setAllFeaturesToLabel(1);
	mofreak_ftrs = mofreak.getMoFREAKFeatures();

	// NEGATIVE EXAMPLES
	MoFREAKUtilities negative_mofreak;

	for (directory_iterator dir_iter(MOFREAK_NEG_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				negative_mofreak.readMoFREAKFeatures(current_file.string());
			}
		}
	}

	negative_mofreak.setAllFeaturesToLabel(-1);
	std::vector<MoFREAKFeature> negative_ftrs = negative_mofreak.getMoFREAKFeatures();

	// append the negative features to the end of the positive ones.
	mofreak_ftrs.insert(mofreak_ftrs.end(), negative_ftrs.begin(), negative_ftrs.end());
	cout << "MoFREAK features gathered." << endl;

	// Do random cluster selection.
	cv::Mat data_pts(mofreak_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 1, NUM_CLASSES, possible_classes);
	clustering.setAppearanceDescriptor(0, true);
	clustering.setMotionDescriptor(8, true);

	cout << "Formatting features..." << endl;
	// [TODO]
	// MAKE FIXED_CLASS 1 WHEN WE ADD POSITIVE EXAMPLES.
	// -1 WHEN WE ADD NEGATIVE EXAMPLES.
	// MAKE THIS CLASS ADAPTABLE TO INCREMENTALLY ADD POINTS (on a per-file basis maybe... I'm not sure.)
	//int fixed_class = 1;
	//clustering.buildDataFromMoFREAK(mofreak_ftrs, false, false, true, fixed_class);
	clustering.buildDataFromMoFREAK(mofreak_ftrs, false, false, false);

	cout << "Clustering..." << endl;
	clustering.randomClusters();

	// print clusters to file
	cout << "Writing clusters to file..." << endl;
	clustering.writeClusters();
	cout << "Clusters written." << endl;
}

// for this, organize mofreak files into pos + neg folders and do them separately.
// use openmp to parallelize each file's BOW stuff.
// give each one it's own libsvm file to output ot, so we don't get any conflicts.
// we will merge at the end with python.

// so, this function will give us sliding window BOW features.
// We can also use this to get our SVM responses to mean-shift away.
void computeBOWHistograms(bool positive_examples)
{
	// gather all files int vector<string> mofreak_files
	cout << "Gathering MoFREAK Features..." << endl;
	vector<std::string> mofreak_files;

	directory_iterator end_iter;

	if (DISTRIBUTED)
	{
		MOFREAK_PATH = "mosift/";
	}

	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				mofreak_files.push_back(current_file.string());
			}
		}
	}

	//mofreak_ftrs = mofreak.getMoFREAKFeatures();
	cout << "MoFREAK features gathered." << endl;

	// load clusters.
	BagOfWordsRepresentation bow_rep(NUM_CLUSTERS, FEATURE_DIMENSIONALITY);

	// for each file....
	// slide window of length alpha and use those pts to create a BOW feature.
#pragma omp parallel for
	for (int i = 0; i < mofreak_files.size(); ++i)
	{
		cout << "Computing on " << mofreak_files[i] << endl;
		std::string bow_file = mofreak_files[i];
		bow_file.append(".test");
		ofstream bow_out(bow_file);

		int label = positive_examples ? 1 : -1;
		bow_rep.computeSlidingBagOfWords(mofreak_files[i], ALPHA, label, bow_out);
		bow_out.close();
		cout << "Done " << mofreak_files[i] << endl;
	}

	// write each feature to libsvm file with 1 if pos, -1 if neg.
	// [that's done in computeSlidingBag...]
}

void detectEvents()
{
	vector<std::string> response_files;
	directory_iterator end_iter;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.length() > 9)
			{
				if (filename.substr(filename.length() - 13, 13) == "responses.txt")
				{
					response_files.push_back(current_file.string());
				}
			}
		}
	}

	for (auto it = response_files.begin(); it != response_files.end(); ++it)
	{
		cout << "filename: " << *it << endl;
		// read in libsvm output.
		ifstream svm_in(*it);

		// store each value in a list that we can reference.
		vector<double> svm_responses;
		while (!svm_in.eof())
		{
			float response;
			svm_in >> response;
			svm_responses.push_back(response);
		}

		cout << svm_responses.size() << " total SVM responses." << endl;


		// get peaks. [val(x) > val(x - 1) & val(x) > val(x + 1)]
		vector<int> peak_indices;
		for (unsigned int i = 1; i < svm_responses.size() - 1; ++i)
		{
			float response_x = svm_responses[i];
			float response_x_minus_1 = svm_responses[i - 1];
			float response_x_plus_1 = svm_responses[i + 1];

			if ((response_x > response_x_minus_1) && (response_x > response_x_plus_1))
			{
				peak_indices.push_back(i);
			}
		}

		cout << peak_indices.size() << " total detected peaks" << endl;

		// For each of those peaks, run the meanshift-like process to determine if its a window-wise local maxima in the response space.
		// that is, check the alpha/2 points before it and alpha/2 points after it.  If it is the largest value in that window,
		// then this is a candidate detection.
		vector<int> candidate_indices;
		for (auto peak = peak_indices.begin(); peak != peak_indices.end(); ++peak)
		{
			double value = svm_responses[*peak];
			int start_index = max((*peak) - ALPHA, 0);
			int end_index = min((*peak) + ALPHA, (int)svm_responses.size() - 1);
			bool is_local_max = true;

			for (int i = start_index; i < end_index; ++i)
			{
				if (svm_responses[i] > value)
				{
					is_local_max = false;
					break;
				}
			}

			if (is_local_max)
			{
				candidate_indices.push_back(*peak);
			}
		}

		cout << candidate_indices.size() << " detected candidates" << endl;


		// finally, if the detection's response is above our defined threshold, it's a real detection.
		float THRESHOLD = 0;
		int MAX_DETECTIONS = 30;
		int MIN_DETECTIONS = 1;
		float STEP = 0.05;
		bool PREVIOUSLY_LOWERED = true;
		bool FIRST_TRY = true;
		// trying an optimization metric for the THRESHOLD.  Say...we want 50 detections per video,
		// we will optimize until that's right.
		
		vector<Detection> detections;
		while (true)
		{
			for (auto candidate = candidate_indices.begin(); candidate != candidate_indices.end(); ++candidate)
			{
				if (svm_responses[*candidate] > THRESHOLD)
				{
					// the BOW feature stretches from the root point (*it) to alpha away.  So if alpha is 10 and it's the first response,
					// it would be keyframes 0 to 10 (or frames 0 to 50).
					int end_index = (*candidate) + ALPHA;
				
					Detection detection;
					detection.start_frame = (*candidate) * 5;
					detection.end_frame = end_index * 5;
					detection.score = svm_responses[*candidate];
					detection.video_name = DETECTION_TARGET_VIDEO_FILE;

					detections.push_back(detection);
				}
			}

			unsigned int num_detections = detections.size();
			cout << num_detections << " detections" << endl;

			if (num_detections < MIN_DETECTIONS)
			{
				// maybe there aren't enough candidates.
				if (candidate_indices.size() < MIN_DETECTIONS)
				{
					break;
					//MIN_DETECTIONS = 3 * candidate_indices.size()/4;
				}
				// too few detections, lower the threshold to allow for more.
				if (FIRST_TRY || PREVIOUSLY_LOWERED)
				{
					THRESHOLD -= STEP;
					PREVIOUSLY_LOWERED = true;
					FIRST_TRY = false;
				}
				else
				{
					// we raised it last time to allow less, but not not enough.
					// shrink the step size to get a finer grain.
					STEP -= 0.005;
					THRESHOLD -= STEP;
					PREVIOUSLY_LOWERED = true;
				}
				cout << "STEP:" << STEP << ", THRESHOLD: " << THRESHOLD << endl;
				detections.clear();
			}
			else if (num_detections > MAX_DETECTIONS)
			{
				// too many detections, raise threshold to allow less.
				if (FIRST_TRY || !PREVIOUSLY_LOWERED)
				{
					THRESHOLD += STEP;
					FIRST_TRY = false;
					PREVIOUSLY_LOWERED = false;
				}
				else
				{
					// we lowered it last time to allow more, but now we have too many.
					// shrink the step size grain for finer detail and raise the threshold by this new amount.
					STEP += 0.005;
					THRESHOLD += STEP;
					PREVIOUSLY_LOWERED = false;
				}
				detections.clear();
				cout << "STEP:" << STEP << ", THRESHOLD: " << THRESHOLD << endl;
			}
			else
			{
				// we are in the desired detection range.  happy days.
				// now we can sort and print them.
				cout << "Accepting a threshold of " << THRESHOLD << " that permits " << num_detections << " events." << endl;
				break;
			}
		}
		// sort by likelihood
		std::sort(detections.begin(), detections.end());
		std::reverse(detections.begin(), detections.end());

		// print to file
		ofstream detection_stream(*it + ".detections");
		for (auto det = detections.begin(); det != detections.end(); ++det)
		{
			detection_stream << *it << ", " << det->start_frame << ", " << det->end_frame << ", " << det->score << endl;
		}
		detection_stream.close();
		cout << "-----------------------------------" << endl << endl;
	}
}

void trainTRECVID()
{
	SVMInterface svm;
	svm.trainModel(SVM_PATH + "/training.train");
}

void computeSVMResponses()
{
	SVMInterface svm;
	directory_iterator end_iter;
	string model_file = SVM_PATH + "/model.svm";

	if (DISTRIBUTED)
	{
		SVM_PATH = "mosift/";
		model_file = "C:/data/model.svm";
	}

	cout << "SVM_PATH: " << SVM_PATH << endl;

	for (directory_iterator dir_iter(SVM_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 4, 4) == "test")
			{
				string test_file = SVM_PATH + "/" + filename;
				cout << "Testing SVM with file " << test_file << " with model " << model_file << endl;
				svm.testModelTRECVID(test_file,  model_file);
			}
		}
	}
}

void main()
{
	//int state = DETECT;
	//int state = GET_SVM_RESPONSES;
	int state = DETECT;
	
	possible_classes.push_back(-1);
	possible_classes.push_back(1);

	clock_t start, end;

	if (state == CLUSTER)
	{
		start = clock();
		clusterMoFREAKPoints();
		end = clock();
	}
	else if (state == CLASSIFY)
	{
		start = clock();
		evaluateSVMWithLeaveOneOut();
		end = clock();
	}
	else if (state == CONVERT)
	{
		start = clock();
		convertMoSIFTToMoFREAK();
		end = clock();
	}
	else if (state == PICK_CLUSTERS)
	{
		start = clock();
		pickClusters();
		end = clock();
	}
	else if (state == COMPUTE_BOW_HISTOGRAMS)
	{
		start = clock();
		const bool POSITIVE_EXAMPLES = false;
		computeBOWHistograms(POSITIVE_EXAMPLES);
		end = clock();
	}
	else if (state == DETECT)
	{
		start = clock();
		detectEvents();
		end = clock();
	}

	else if (state == TRAIN)
	{
		start = clock();
		trainTRECVID();
		end = clock();
	}

	else if (state == GET_SVM_RESPONSES)
	{
		start = clock();
		computeSVMResponses();
		end = clock();
	}

	cout << "Took this long: " << (end - start)/(double)CLOCKS_PER_SEC << " seconds! " << endl;
	cout << "All done.  Press any key to continue..." << endl;
	system("PAUSE");
}