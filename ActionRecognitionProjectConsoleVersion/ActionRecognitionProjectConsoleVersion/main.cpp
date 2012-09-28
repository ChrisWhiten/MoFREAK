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


string SVM_PATH = "C:/data/kth/svm/";
string MOFREAK_PATH = "C:/data/TRECVID/positive_mofreak_examples/";//"C:/data/TRECVID/mofreak/";
string MOSIFT_DIR = "C:/data/TRECVID/mosift/eval_personruns_mosift/";
//string MOSIFT_FILE = "C:/data/LGW_20071101_E1_CAM1.mpeg.Pointing.txt.mosift.0";
//string VIDEO_PATH = "C:/data/TRECVID/gatwick_dev08/dev/LGW_20071101_E1_CAM1.mpeg/LGW_20071101_E1_CAM1.mpeg";
string VIDEO_PATH = "C:/data/TRECVID/eval/";

// for clustering, separate mofreak into pos and neg examples.
string MOFREAK_NEG_PATH = "C:/data/TRECVID/negative_mofreak_examples/";
string MOFREAK_POS_PATH = "C:/data/TRECVID/positive_mofreak_examples/";


const int FEATURE_DIMENSIONALITY = 8;
const int NUM_CLUSTERS = 1000;
const int NUM_CLASSES = 2;//1;//2;
const int ALPHA = 12; // for sliding window...

vector<int> possible_classes;

MoFREAKUtilities mofreak;
vector<MoFREAKFeature> mofreak_ftrs;
SVMInterface svm_interface;

enum states {CLUSTER, CLASSIFY, CONVERT, PICK_CLUSTERS, COMPUTE_BOW_HISTOGRAMS};

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

	for (directory_iterator dir_iter(MOSIFT_DIR); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			// parse mosift files so first x characters gets us the video name.
			path current_file = dir_iter->path();
			string mosift_path = current_file.generic_string();
			string mosift_filename = current_file.filename().generic_string();
			string video_file = VIDEO_PATH;
			video_file.append(mosift_filename.substr(0, 25));

			
			string mofreak_path = mosift_path;
			mofreak_path.append(".mofreak");
			// compute. write.
			MoFREAKUtilities mofreak;
			mofreak.buildMoFREAKFeaturesFromMoSIFT(mosift_path, video_file, mofreak_path);
			mosift_path.append(".mofreak");
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

	for (directory_iterator dir_iter(MOFREAK_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 7, 7) == "mofreak")
			{
				mofreak_files.push_back(current_file.string());
				//mofreak.readMoFREAKFeatures(mofreak_files.back());
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
		bow_file.append(".bow");
		ofstream bow_out(bow_file);

		int label = positive_examples ? 1 : -1;
		bow_rep.computeSlidingBagOfWords(mofreak_files[i], ALPHA, label, bow_out);
		bow_out.close();
		cout << "Done " << mofreak_files[i] << endl;
	}

	// write each feature to libsvm file with 1 if pos, -1 if neg.
	// [that's done in computeSlidingBag...]
}

void main()
{
	int state = COMPUTE_BOW_HISTOGRAMS;
	
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
		const bool POSITIVE_EXAMPLES = true;
		computeBOWHistograms(POSITIVE_EXAMPLES);
		end = clock();
	}

	cout << "Took this long: " << (end - start)/(double)CLOCKS_PER_SEC << " seconds! " << endl;
	cout << "All done.  Press any key to continue..." << endl;
	system("PAUSE");
}