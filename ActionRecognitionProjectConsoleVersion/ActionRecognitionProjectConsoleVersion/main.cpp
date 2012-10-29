#include <string>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <exception>

#include <boost\filesystem.hpp>

#include <opencv2\opencv.hpp>

#include "MoFREAKUtilities.h"
#include "Clustering.h"
#include "BagOfWordsRepresentation.h"
#include "SVMInterface.h"

using namespace std;
using namespace boost::filesystem;

bool DISTRIBUTED = false;

string MOSIFT_DIR = "C:/data/TRECVID/mosift/testing/";
string MOFREAK_PATH = MOSIFT_DIR; // because converting mosift to mofreak just puts them in the same folder as the mosift points. That's fine.
string VIDEO_PATH = "C:/data/TRECVID/mosift/testing/videos/";//"C:/data/TRECVID/videos/";
string SVM_PATH = "C:/data/TRECVID/svm/";
string METADATA_PATH;

// for clustering, separate mofreak into pos and neg examples.
string MOFREAK_NEG_PATH = "C:/data/TRECVID/negative_mofreak_examples/";
string MOFREAK_POS_PATH = "C:/data/TRECVID/positive_mofreak_examples/";

int NUM_MOTION_BYTES = 8;
int NUM_APPEARANCE_BYTES = 8;
int FEATURE_DIMENSIONALITY = NUM_MOTION_BYTES + NUM_APPEARANCE_BYTES;
int NUM_CLUSTERS, NUMBER_OF_PEOPLE, NUM_CLASSES, ALPHA;

vector<int> possible_classes;
std::deque<MoFREAKFeature> mofreak_ftrs;

enum states {CLASSIFY, CONVERT, PICK_CLUSTERS, COMPUTE_BOW_HISTOGRAMS, DETECT, TRAIN, GET_SVM_RESPONSES,
			MOSIFT_TO_DETECTION, POINT_DETECTION};

enum datasets {KTH, TRECVID, HOLLYWOOD, UTI1, UTI2, HMDB51};

int dataset = HMDB51;
int state = MOSIFT_TO_DETECTION;

MoFREAKUtilities mofreak(dataset);
SVMInterface svm_interface;

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

void setParameters()
{
	if (dataset == TRECVID)
	{
		// event information for TRECVID videos.
		NUM_CLUSTERS = 1000;
		NUM_CLASSES = 2;
		possible_classes.push_back(-1);
		possible_classes.push_back(1);
		ALPHA = 12;

		// structural folder information.
		MOSIFT_DIR = "C:/data/TRECVID/mosift/testing/";
		MOFREAK_PATH = MOSIFT_DIR; // because converting mosift to mofreak just puts them in the same folder as the mosift points. That's fine.
		VIDEO_PATH = "C:/data/TRECVID/mosift/testing/videos/";//"C:/data/TRECVID/videos/";
		SVM_PATH = "C:/data/TRECVID/svm/";

		// for clustering, separate mofreak into pos and neg examples.
		MOFREAK_NEG_PATH = "C:/data/TRECVID/negative_mofreak_examples/";
		MOFREAK_POS_PATH = "C:/data/TRECVID/positive_mofreak_examples/";
	}

	// KTH
	else if (dataset == KTH)
	{
		NUM_CLUSTERS = 600;
		NUM_CLASSES = 6;
		NUMBER_OF_PEOPLE = 25;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info.
		MOSIFT_DIR = "C:/data/kth/mosift/";
		MOFREAK_PATH = "C:/data/kth/mofreak/"; 
		VIDEO_PATH = "C:/data/kth/all_in_one/videos/";
		SVM_PATH = "C:/data/kth/svm/";
		METADATA_PATH = "";
	}

	else if (dataset == HMDB51)
	{
		NUM_CLUSTERS = 5100;
		NUM_CLASSES = 51;

		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info
		MOFREAK_PATH = "C:/data/hmdb51/mofreak/";
		VIDEO_PATH = "C:/data/hmdb51/videos/";
		SVM_PATH = "C:/data/hmdb51/svm/";
		METADATA_PATH = "C:/data/hmdb51/metadata/";
	}

	else if (dataset == UTI2)
	{
		NUM_CLUSTERS = 600;
		NUM_CLASSES = 6;
		NUMBER_OF_PEOPLE = 10;
		for (unsigned i = 0; i < NUM_CLASSES; ++i)
		{
			possible_classes.push_back(i);
		}

		// structural folder info.
		MOFREAK_PATH = "C:/data/UTI/segmented/mofreak/";
		VIDEO_PATH = "C:/data/UTI/segmented/videos/";
		SVM_PATH = "C:/data/UTI/segmented/svm/";
	}
}

/*
Cluster 1 class at a time, releasing memory in between.
We are having huge memory issues, 
so we can't load it all in at once.
*/
void clusterHMDB51()
{
	cout << "Gathering MoFREAK Features..." << endl;
	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 0, NUM_CLASSES, possible_classes);
	clustering.setAppearanceDescriptor(8, true);
	clustering.setMotionDescriptor(8, true);

	/*cout << "Formatting features..." << endl;
	clustering.buildDataFromMoFREAK(mofreak_ftrs, false, false);

	cout << "Clustering..." << endl;
	//clustering.clusterWithKMeans();
	clustering.randomClusters();*/

	directory_iterator end_iter;

	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
		// new action.
		if (is_directory(dir_iter->status()))
		{
			std::vector<string> mofreak_files;
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;

			// set the mofreak object's action to that folder name.
			mofreak.setCurrentAction(video_action);

			// compute mofreak on all files on that folder.
			string action_mofreak_path = MOFREAK_PATH + "/" + video_action;
			cout << "action mofreak path: " << action_mofreak_path << endl;

			for (directory_iterator video_iter(action_mofreak_path); 
				video_iter != end_iter; ++video_iter)
			{
				string filename = video_iter->path().filename().generic_string();
				if (filename.substr(filename.length() - 7, 7) == "mofreak")
				{
					cout << video_iter->path().string() << endl;

					try
					{
						mofreak_files.push_back(video_iter->path().string());
						mofreak.readMoFREAKFeatures(mofreak_files.back());
					}
					catch (exception &e)
					{
						cout << e.what() << endl;
						system("PAUSE");
						exit(1);
					}
				}
			}

			// cluster this set, then drop these mofreak features.
			cout << "assign data pts" << endl;
			std::deque<MoFREAKFeature> ftrs = mofreak.getMoFREAKFeatures();
			cout << "got mofreak features" << endl;
			try
			{
				cv::Mat data_pts(ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				system("PAUSE");
				exit(1);
			}
			cout << "build data" << endl;
			clustering.buildDataFromMoFREAK(ftrs, false, false);
			cout << "rand clust" << endl;
			clustering.randomClusters(true);
			cout << "clear" << endl;
			mofreak.clearFeatures();
		}
	}

	clustering.writeClusters();
}
void clusterKTH()
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
		else if (is_directory(dir_iter->status()))
		{
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;

			// set the mofreak object's action to that folder name.
			mofreak.setCurrentAction(video_action);

			// compute mofreak on all files on that folder.
			string action_mofreak_path = MOFREAK_PATH + "/" + video_action;
			cout << "action mofreak path: " << action_mofreak_path << endl;

			for (directory_iterator video_iter(action_mofreak_path); 
				video_iter != end_iter; ++video_iter)
			{
				string filename = video_iter->path().filename().generic_string();
				if (filename.substr(filename.length() - 7, 7) == "mofreak")
				{
					cout << video_iter->path().string() << endl;

					try
					{
						mofreak_files.push_back(video_iter->path().string());
						mofreak.readMoFREAKFeatures(mofreak_files.back());
					}
					catch (exception &e)
					{
						cout << e.what() << endl;
						system("PAUSE");
						exit(1);
					}
				}
			}
		}
	}

	mofreak_ftrs = mofreak.getMoFREAKFeatures();
	cout << "MoFREAK features gathered." << endl;

	// organize pts into a cv::Mat.
	cv::Mat data_pts(mofreak_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 0, NUM_CLASSES, possible_classes);
	clustering.setAppearanceDescriptor(8, true);
	//clustering.setMotionDescriptor(8, true);
	clustering.setMotionDescriptor(8, true);

	cout << "Formatting features..." << endl;
	clustering.buildDataFromMoFREAK(mofreak_ftrs, false, false);

	cout << "Clustering..." << endl;
	//clustering.clusterWithKMeans();
	clustering.randomClusters();

	// print clusters to file
	cout << "Writing clusters to file..." << endl;
	clustering.writeClusters();
	cout << "Finished writing." << endl;
}

void computeBOWKTH()
{
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
			}
		}

		// for hmdb51
		else if (is_directory(dir_iter->status()))
		{
			// get folder name.
			string video_action = dir_iter->path().filename().generic_string();
			string action_video_path = VIDEO_PATH + "/" + video_action;

			for (directory_iterator video_iter(action_video_path); 
				video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string video_filename = video_iter->path().filename().generic_string();
					if (video_filename.substr(video_filename.length() - 7, 7) == "mofreak")
					{
						mofreak_files.push_back(video_iter->path().string());
					}
				}
			}
		}
	}

	cout << "Computing BOW Representation..." << endl;
	BagOfWordsRepresentation bow_rep(mofreak_files, NUM_CLUSTERS, FEATURE_DIMENSIONALITY, NUMBER_OF_PEOPLE, true, true, dataset);

	bow_rep.setAppearanceDescriptor(NUM_APPEARANCE_BYTES, true);
	bow_rep.setMotionDescriptor(NUM_MOTION_BYTES, true);

	bow_rep.computeBagOfWords(SVM_PATH, MOFREAK_PATH, METADATA_PATH);
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
	std::string results_file = SVM_PATH;
	results_file.append("/svm_results.txt");
	ofstream output_file(results_file);

	string model_file_name = SVM_PATH;
	model_file_name.append("/model.svm");

	string svm_out = SVM_PATH;
	svm_out.append("/responses.txt");

	// confusion matrix.
	// 6x6, since we have 6 action in KTH/UTI2.
	cv::Mat confusion_matrix = cv::Mat::zeros(6, 6, CV_32F);

	double summed_accuracy = 0.0;
	for (unsigned i = 0; i < training_files.size(); ++i)
	{
		SVMInterface svm_guy;
		// tell GUI where we're at in the l-o-o process
		cout << "Leaving out person " << i + 1 << endl;

		// build model.
		svm_guy.trainModel(training_files[i], model_file_name);

		// get accuracy.
		string test_filename = testing_files[i];
		double accuracy = svm_guy.testModel(test_filename, model_file_name, svm_out);
		summed_accuracy += accuracy;

		// update confusion matrix.
		// get svm responses.
		vector<int> responses;

		ifstream response_file(svm_out);
		string line;
		while (std::getline(response_file, line))
		{
			int response;
			istringstream(line) >> response;
			//response_file >> (int)response;
			responses.push_back(response);
		}
		response_file.close();

		// now get expected output.
		vector<int> ground_truth;

		ifstream truth_file(test_filename);
		while (std::getline(truth_file, line))
		{
			int truth;
			int first_space = line.find(" ");
			if (first_space != string::npos)
			{
				istringstream (line.substr(0, first_space)) >> truth;
				ground_truth.push_back(truth);
			}
		}

		// now add that info to the confusion matrix.
		// row = ground truth, col = predicted..
		for (int i = 0; i < responses.size(); ++i)
		{
			int row = ground_truth[i] - 1;
			int col = responses[i] - 1;

			confusion_matrix.at<float>(row, col) += 1;
		}
		
		

		// debugging...print to testing file.
		output_file << training_files[i] <<", " << testing_files[i] << ", " << accuracy << std::endl;
	}	

	// normalize each row.
	// 6 rows/cols (1 per action)
	for (int row = 0; row < 6; ++row)
	{
		float normalizer = 0.0;
		for (int col = 0; col < 6; ++col)
		{
			normalizer += confusion_matrix.at<float>(row, col);
		}

		for (int col = 0; col < 6; ++col)
		{
			confusion_matrix.at<float>(row, col) /= normalizer;
		}
	}

	cout << "Confusion matrix" << endl << "---------------------" << endl;
	for (int row = 0; row < confusion_matrix.rows; ++row)
	{
		for (int col = 0; col < confusion_matrix.cols; ++col)
		{
			cout << confusion_matrix.at<float>(row, col) << ", ";
		}
		cout << endl << endl;
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
			if (TRECVID)
			{
				video_file.append(mosift_filename.substr(0, 25));
			}
			else
			{
				video_file.append(mosift_filename.substr(0, mosift_filename.length() - 3));
				video_file.append("avi");
			}
			cout << "vid: " << video_file << endl;

			
			string mofreak_path = MOFREAK_PATH + mosift_filename + ".mofreak";//mosift_path;
			//mofreak_path.append(".mofreak");
			// compute. write.
			MoFREAKUtilities mofreak(dataset);
			
			cout << "Buidling " << mofreak_path << " from " << mosift_path << " with video " << video_file << endl;
			mofreak.buildMoFREAKFeaturesFromMoSIFT(mosift_path, video_file, mofreak_path);
			mosift_path.append(".mofreak");
			cout << "writing mofreak features to file..." << endl;
			mofreak.writeMoFREAKFeaturesToFile(mofreak_path);
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
	MoFREAKUtilities negative_mofreak(dataset);

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
	std::deque<MoFREAKFeature> negative_ftrs = negative_mofreak.getMoFREAKFeatures();

	// append the negative features to the end of the positive ones.
	mofreak_ftrs.insert(mofreak_ftrs.end(), negative_ftrs.begin(), negative_ftrs.end());
	cout << "MoFREAK features gathered." << endl;

	// Do random cluster selection.
	cv::Mat data_pts(mofreak_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, 1, NUM_CLASSES, possible_classes);
	clustering.setAppearanceDescriptor(8, true);
	clustering.setMotionDescriptor(8, true);

	cout << "Formatting features..." << endl;
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
					detection.video_name = "GenericVideoName.mpg"; // [TODO]

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
	string model_file_name = "C:/data/TRECVID/svm/model.svm";
	svm.trainModel(SVM_PATH + "/training.train", model_file_name);
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

void computeMoFREAKFiles()
{
	directory_iterator end_iter;

	cout << "Here are the videos: " << VIDEO_PATH << endl;
	cout << "MoFREAK files will go here: " << MOFREAK_PATH << endl;
	for (directory_iterator dir_iter(VIDEO_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			// parse mosift files so first x characters gets us the video name.
			path current_file = dir_iter->path();
			string video_path = current_file.generic_string();
			string video_filename = current_file.filename().generic_string();

			if ((video_filename.substr(video_filename.length() - 3, 3) == "avi"))
			{

				cout << "filename: " << video_filename << endl;
				cout << "AVI: " << VIDEO_PATH << "/" << video_filename << endl;

				string video = VIDEO_PATH + "/" + video_filename;
				string mofreak_path = MOFREAK_PATH + "/" + video_filename + ".mofreak";

				mofreak.computeMoFREAKFromFile(video, mofreak_path, true);
			}
		}
		else if (is_directory(dir_iter->status()))
		{
			// get folder name.
			string video_action = dir_iter->path().filename().generic_string();
			cout << "action: " << video_action << endl;

			// set the mofreak object's action to that folder name.
			mofreak.setCurrentAction(video_action);

			// compute mofreak on all files on that folder.
			string action_video_path = VIDEO_PATH + "/" + video_action;
			cout << "action video path: " << action_video_path << endl;

			for (directory_iterator video_iter(action_video_path); 
				video_iter != end_iter; ++video_iter)
			{
				if (is_regular_file(video_iter->status()))
				{
					string video_filename = video_iter->path().filename().generic_string();
					if (video_filename.substr(video_filename.length() - 3, 3) == "avi")
					{
						cout << "filename: " << video_filename << endl;
						cout << "AVI: " << action_video_path << video_filename << endl;

						string mofreak_path = MOFREAK_PATH + "/" + video_action + "/" + video_filename + ".mofreak";
						cout << "mofreak path: " << mofreak_path << endl;
						mofreak.computeMoFREAKFromFile(action_video_path + "/" + video_filename, mofreak_path, true);
					}
				}
			}
		}
	}
}

void main()
{
	//int state = MOSIFT_TO_DETECTION;///PICK_CLUSTERS;//MOSIFT_TO_DETECTION;
	//int dataset = KTH;
	setParameters();

	clock_t start, end;

	if (state == CLASSIFY)
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
	
	else if (state == POINT_DETECTION)
	{
		start = clock();
		computeMoFREAKFiles();
		//mofreak.computeMoFREAKFromFile("C:/data/kth/all_in_one/videos/person13_jogging_d3_uncomp.avi", true);
		end = clock();
	}

	else if (state == MOSIFT_TO_DETECTION)
	{
		start = clock();
		//computeMoFREAKFiles();
		//convertMoSIFTToMoFREAK();
		if (dataset == TRECVID)
		{
			pickClusters();
			computeBOWHistograms(false);
			computeSVMResponses();
			detectEvents();
		}
		
		else if (dataset == KTH)
		{
			clusterKTH();
			computeBOWKTH();
			evaluateSVMWithLeaveOneOut();
		}

		else if (dataset == UTI2)
		{
			//clusterUTI2();
			clusterKTH(); // same number of classes here... why not?
			//computeBOWUTI2();
			computeBOWKTH(); 
			evaluateSVMWithLeaveOneOut();// same as above.  Why not?
		}

		else if (dataset == HMDB51)
		{
			//clusterHMDB51();
			computeBOWKTH();
		}

		end = clock();
	}

	cout << "Took this long: " << (end - start)/(double)CLOCKS_PER_SEC << " seconds! " << endl;
	cout << "All done.  Press any key to continue..." << endl;
	system("PAUSE");
}