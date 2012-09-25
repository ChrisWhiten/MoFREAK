#include "BagOfWordsRepresentation.h"

// vanilla string split operation.  Direct copy-paste from stack overflow
// source: http://stackoverflow.com/questions/236129/splitting-a-string-in-c
std::vector<std::string> &BagOfWordsRepresentation::split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> BagOfWordsRepresentation::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}


BagOfWordsFeature::BagOfWordsFeature() : bag_of_words(cv::Mat()), _ft(NULL), number_of_words(0), matcher_type(BF_L2)
{
	matcher = NULL;
}

BagOfWordsFeature::BagOfWordsFeature(const BagOfWordsFeature &b)
{}


void BagOfWordsRepresentation::normalizeClusters()
{
	for (unsigned int clust = 0; clust < clusters->rows; ++clust)
	{
		normalizeAppearanceOfFeature((*clusters)(cv::Range(clust, clust + 1), cv::Range(0, clusters->cols)));
		normalizeMotionOfFeature((*clusters)(cv::Range(clust, clust + 1), cv::Range(0, clusters->cols)));
	}
}

void BagOfWordsRepresentation::normalizeAppearanceOfFeature(cv::Mat &ftr)
{
	// only normalize euclidean motion spaces
	if (appearance_is_binary)
		return;

	const int APPEARANCE_START = 0;
	const int APPEARANCE_END = appearance_descriptor_size;

	float normalizer = 0.0;

	// compute normalizer
	for (unsigned col = APPEARANCE_START; col < APPEARANCE_END; ++col)
	{
		normalizer += ftr.at<float>(0, col);
	}

	// divide each val by the normalizer.
	for (unsigned col = APPEARANCE_START; col < APPEARANCE_END; ++col)
	{
		ftr.at<float>(0, col) = ftr.at<float>(0, col)/normalizer;
	}
}

void BagOfWordsRepresentation::normalizeMotionOfFeature(cv::Mat &ftr)
{
	// only normalize euclidean motion spaces
	if (motion_is_binary)
		return;

	const int MOTION_START = appearance_descriptor_size;//64;
	const int MOTION_END = appearance_descriptor_size + motion_descriptor_size;//192;

	float normalizer = 0.0;

	// compute normalizer
	for (unsigned col = MOTION_START; col < MOTION_END; ++col)
	{
		normalizer += ftr.at<float>(0, col);
	}

	// divide each val by the normalizer.
	for (unsigned col = MOTION_START; col < MOTION_END; ++col)
	{
		ftr.at<float>(0, col) = ftr.at<float>(0, col)/normalizer;
	}
}

float BagOfWordsRepresentation::standardEuclideanDistance(cv::Mat &a, cv::Mat &b) const
{
	float distance = 0.0;

	// compute distance between the two vectors.
	for (unsigned int i = 0; i < a.cols; ++i)
	{
		float a_i = a.at<float>(0, i);
		float b_i = b.at<float>(0, i);
		distance += ((a_i - b_i) * (a_i - b_i));
	}

	distance = sqrt(distance);
	return distance;
}

unsigned int BagOfWordsRepresentation::hammingDistance(cv::Mat &a, cv::Mat &b)
{
	unsigned int hamming_distance = 0;

	for (unsigned i = 0; i < a.cols; ++i)
	{
		unsigned int a_bits = (unsigned int)a.at<float>(0, i);
		unsigned int b_bits = (unsigned int)b.at<float>(0, i);

		// start as 0000 0001
		unsigned int bit = 1;

		// get the xor of a and b, each 1 in the xor adds to the hamming distance...
		unsigned int xor_result = a_bits ^ b_bits;

		// now count the bits, using 'bit' as a mask and performing a bitwise AND
		for (bit = 1; bit != 0; bit <<= 1)
		{
			if ((xor_result & bit) > 0)
			{
				hamming_distance++;
			}
		}
	}

	return hamming_distance;
}

void BagOfWordsRepresentation::findBestMatchFREAKAndFrameDifference(cv::Mat &feature_vector, cv::Mat &clusters, int &best_cluster_index, float &best_cluster_score)
{
	// constants
	const float MOTION_EUCLID_DIST_NORM = 255; //?
	const int FREAK_HAMMING_DIST_NORM = 512;

	const int FREAK_START_INDEX = 0;
	const int FREAK_END_INDEX = 64;

	const int MOTION_START_INDEX = 64;
	const int MOTION_END_INDEX = 65;//?

	// base case: initialize with the score/index of the 0th cluster.
	best_cluster_index = 0;

	// start with the hamming distance over FREAK points.
	float FREAK_distance = 0.0;

	cv::Mat query_FREAK_descriptor = feature_vector(cv::Range(0, 1), cv::Range(FREAK_START_INDEX, FREAK_END_INDEX));
	cv::Mat cluster_FREAK_descriptor = clusters(cv::Range(0, 1), cv::Range(FREAK_START_INDEX, FREAK_END_INDEX));

	FREAK_distance = hammingDistance(query_FREAK_descriptor, cluster_FREAK_descriptor);
	FREAK_distance /= FREAK_HAMMING_DIST_NORM;

	cv::Mat query_motion_descriptor = feature_vector(cv::Range(0, 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));
	cv::Mat cluster_motion_descriptor = clusters(cv::Range(0, 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));

	float euclidean_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
	euclidean_distance /= MOTION_EUCLID_DIST_NORM;

	float final_dist = euclidean_distance + FREAK_distance;
	best_cluster_score = final_dist;

	// now the remaining points.
	for (unsigned cluster = 1; cluster < clusters.rows; ++cluster)
	{
		final_dist = 0.0;
		euclidean_distance = 0.0;
		FREAK_distance = 0.0;

		cluster_FREAK_descriptor = clusters(cv::Range(cluster, cluster + 1), cv::Range(FREAK_START_INDEX, FREAK_END_INDEX));
		cluster_motion_descriptor = clusters(cv::Range(cluster, cluster + 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));

		FREAK_distance = hammingDistance(query_FREAK_descriptor, cluster_FREAK_descriptor);
		FREAK_distance /= FREAK_HAMMING_DIST_NORM;

		euclidean_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
		euclidean_distance /= MOTION_EUCLID_DIST_NORM;

		final_dist = euclidean_distance + FREAK_distance;

		// compare to best.
		if (final_dist < best_cluster_score)
		{
			best_cluster_score = final_dist;
			best_cluster_index = cluster;
		}
	}
}

void BagOfWordsRepresentation::findBestMatchDescriptorInvariant(cv::Mat &feature_vector, cv::Mat &clusters, int &best_cluster_index, float &best_cluster_score, ofstream &file)
{
	// constants
	const unsigned int APPEARANCE_HAMMING_DIST_NORM = 99999999;//1024;//= appearance_descriptor_size * 8;
	const unsigned int APPEARANCE_START_INDEX = 0;
	const unsigned int APPEARNCE_END_INDEX = appearance_descriptor_size;
	const unsigned int MOTION_HAMMING_DIST_NORM = motion_descriptor_size * 8;
	const unsigned int MOTION_START_INDEX = appearance_descriptor_size;
	const unsigned int MOTION_END_INDEX = appearance_descriptor_size + motion_descriptor_size;


	// clusters a pre-normalized.  normalize feature vector.
	normalizeMotionOfFeature(feature_vector);
	normalizeAppearanceOfFeature(feature_vector);

	// base case: initialize with the score/index of the 0th cluster.
	best_cluster_index = 0;
	float appearance_distance = 0.0;
	float motion_distance = 0.0;

	// Compute the distance between the appearance components
	
	cv::Mat query_appearance_descriptor, cluster_appearance_descriptor;
	query_appearance_descriptor = feature_vector(cv::Range(0, 1), cv::Range(APPEARANCE_START_INDEX, APPEARNCE_END_INDEX));
	cluster_appearance_descriptor = clusters(cv::Range(0, 1), cv::Range(APPEARANCE_START_INDEX, APPEARNCE_END_INDEX));

	if (appearance_is_binary && (appearance_descriptor_size > 0))
	{
		appearance_distance = hammingDistance(query_appearance_descriptor, cluster_appearance_descriptor);
		appearance_distance /= APPEARANCE_HAMMING_DIST_NORM;
	}
	else
	{
		appearance_distance = standardEuclideanDistance(query_appearance_descriptor, cluster_appearance_descriptor);
	}
	

	// Compute the distance between the motion components
	cv::Mat query_motion_descriptor, cluster_motion_descriptor;
	query_motion_descriptor = feature_vector(cv::Range(0, 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));
	cluster_motion_descriptor = clusters(cv::Range(0, 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));

	if (motion_is_binary)
	{
		motion_distance = hammingDistance(query_motion_descriptor, cluster_motion_descriptor);
		motion_distance /= MOTION_HAMMING_DIST_NORM;
	}
	else
	{
		motion_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
	}

	float final_dist = appearance_distance + motion_distance;
	file << final_dist << endl;

	best_cluster_score = final_dist;

	// Previous computations were just so everything was initialized (for the base case)
	// Now run those computations against all other clusters.
	for (unsigned cluster = 1; cluster < clusters.rows; ++cluster)
	{
		final_dist = 0.0;
		motion_distance = 0.0;
		appearance_distance = 0.0;

		query_appearance_descriptor = clusters(cv::Range(cluster, cluster + 1), cv::Range(APPEARANCE_START_INDEX, APPEARNCE_END_INDEX));
		cluster_motion_descriptor = clusters(cv::Range(cluster, cluster + 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));

		// Appearance distance.
		
		if (appearance_is_binary && (appearance_descriptor_size > 0))
		{
			appearance_distance = hammingDistance(query_appearance_descriptor, cluster_appearance_descriptor);
			appearance_distance /= APPEARANCE_HAMMING_DIST_NORM;
		}
		else
		{
			appearance_distance = standardEuclideanDistance(query_appearance_descriptor, cluster_appearance_descriptor);
		}
		

		// Motion distance.
		if (motion_is_binary)
		{
			motion_distance = hammingDistance(query_motion_descriptor, cluster_motion_descriptor);
			motion_distance /= MOTION_HAMMING_DIST_NORM;
		}
		else
		{
			motion_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
		}

		final_dist = appearance_distance + motion_distance;
		file << final_dist << endl;

		// If we have a new shortest distance, store that.
		if (final_dist < best_cluster_score)
		{
			best_cluster_score = final_dist;
			best_cluster_index = cluster;
		}
	}
}
void BagOfWordsRepresentation::findBestMatch(cv::Mat &feature_vector, cv::Mat &clusters, int &best_cluster_index, float &best_cluster_score)
{
	// base case: initialize with the score/index of the 0th cluster.
	best_cluster_index = 0;

	// euclidean distance.
	float euc_distance = 0.0;

	cv::Mat query_SIFT_descriptor = feature_vector(cv::Range(0, 1), cv::Range(0, 128));
	cv::Mat cluster_SIFT_descriptor = clusters(cv::Range(0, 1), cv::Range(0, 128));

	euc_distance = standardEuclideanDistance(query_SIFT_descriptor, cluster_SIFT_descriptor);

	best_cluster_score = euc_distance;

	// now the remaining points.
	for (unsigned cluster = 1; cluster < clusters.rows; ++cluster)
	{
		euc_distance = 0.0;
		for (unsigned int i = 0; i < feature_vector.cols; ++i)
		{
			float a_i = feature_vector.at<float>(0, i);
			float b_i = clusters.at<float>(cluster, i);
			euc_distance += ((a_i - b_i) * (a_i - b_i));
		}
		euc_distance = sqrt(euc_distance);

		// compare to best.
		if (euc_distance < best_cluster_score)
		{
			best_cluster_score = euc_distance;
			best_cluster_index = cluster;
		}
	}
}

cv::Mat BagOfWordsRepresentation::buildHistogram(std::string &file, bool &success)
{
	success = false;

	cv::Mat histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
	for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		histogram.at<float>(0, col) = 0;

	// open file.
	ifstream input_file(file);
	string line;

	ofstream distances("distances.txt");
	while (std::getline(input_file, line))
	{
		// discard first 6 values.
		std::istringstream iss(line);
		double discard;

		for (unsigned i = 0; i < 6; ++i)
		{
			iss >> discard;
		}

		// read line and parse into FEATURE_DIMENSIONALITY-dim vector.
		cv::Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_32FC1);
		float elem;
		
		for (unsigned int i = 0; i < FEATURE_DIMENSIONALITY; ++i)
		{
			iss >> elem;
			feature_vector.at<float>(0, i) = elem;
		}

		// match that vector against centroids to assign to correct codeword.
		// brute force match each mosift point against all clusters to find best match.
		int best_cluster_index;
		float best_cluster_score;
		
		//findBestMatch(feature_vector, *clusters, best_cluster_index, best_cluster_score);
		findBestMatchDescriptorInvariant(feature_vector, *clusters, best_cluster_index, best_cluster_score, distances);
		//findBestMatchFREAKAndFrameDifference(feature_vector, *clusters, best_cluster_index, best_cluster_score);


		// + 1 to that codeword
		histogram.at<float>(0, best_cluster_index) = histogram.at<float>(0, best_cluster_index) + 1;
		success = true;
	}

	distances.close();

	if (!success)
		return histogram;

	// after doing that for all lines in file, normalize.
	float histogram_sum = 0;
	for (unsigned col = 0; col < histogram.cols; ++col)
	{
		histogram_sum += histogram.at<float>(0, col);
	}

	for (unsigned col = 0; col < histogram.cols; ++col)
	{
		histogram.at<float>(0, col) = histogram.at<float>(0, col)/histogram_sum;
	}
	
	return histogram;
}

void BagOfWordsRepresentation::loadClusters()
{
	clusters = new cv::Mat(NUMBER_OF_CLUSTERS, FEATURE_DIMENSIONALITY, CV_32FC1);

	ifstream cluster_file("clusters.txt");
	string line;

	unsigned int row = 0;
	while (std::getline(cluster_file, line))
	{
		float elem;
		for (unsigned int col = 0; col < FEATURE_DIMENSIONALITY; ++col)
		{
			cluster_file >> elem;
			clusters->at<float>(row, col) = elem;
		}
		++row;
	}
}

void BagOfWordsRepresentation::computeBagOfWords()
{
	ofstream test("test.txt");
	test << "computing" << endl;
	test.close();
	// open file streams to write data for SVM
	ofstream hist_file("hist.txt");
	ofstream label_file("label.txt");

	vector<ofstream *> training_files;
	vector<ofstream *> testing_files;
	vector<vector<string> > training_file_lines;
	vector<vector<string> > testing_file_lines;

	for (unsigned int i = 0; i < NUMBER_OF_PEOPLE; ++i)
	{
		stringstream training_string;
		stringstream testing_string;

		training_string << "C:/data/kth/svm/left_out_" << i + 1 << ".train";
		testing_string << "C:/data/kth/svm/left_out_" << i + 1 << ".test";

		ofstream *training_file = new ofstream(training_string.str());
		ofstream *testing_file = new ofstream(testing_string.str());

		training_string.str("");
		training_string.clear();
		testing_string.str("");
		testing_string.clear();

		training_files.push_back(training_file);
		testing_files.push_back(testing_file);

		vector<string> training_lines;
		vector<string> testing_lines;
		training_file_lines.push_back(training_lines);
		testing_file_lines.push_back(testing_lines);

	}

	// for each file, find the action + person number + video number
	// person01_boxing_d2_uncomp.txt....
	// So, split it on the underscore.
	// word_1[-2:] is the person.
	// word_2[:] should match one of the strings...
	// word_3[1:] is the video number
#pragma omp parallel for
	for (int i = 0; i < files.size(); ++i)// = qsl.begin(); it != qsl.end(); ++it)
	{
		boost::filesystem::path file_path(files[i]);
		boost::filesystem::path file_name = file_path.filename();
		std::string file_name_str = file_name.generic_string();

		int action, person, video_number;

		// get the action.
		if (boost::contains(file_name_str, "boxing"))
		{
			action = BOXING;
		}
		else if (boost::contains(file_name_str, "walking"))
		{
			action = WALKING;
		}
		else if (boost::contains(file_name_str, "jogging"))
		{
			action = JOGGING;
		}
		else if (boost::contains(file_name_str, "running"))
		{
			action = RUNNING;
		}
		else if (boost::contains(file_name_str, "handclapping"))
		{
			action = HANDCLAPPING;
		}
		else if (boost::contains(file_name_str, "handwaving"))
		{
			action = HANDWAVING;
		}
		else
		{
			action = HANDWAVING; // hopefully we never miss this?  Just giving a default value. 
		}

		// parse the filename...
		std::vector<std::string> filename_parts = split(file_name_str, '_');

		// the person is the last 2 characters of the first section of the filename.
		std::stringstream(filename_parts[0].substr(filename_parts[0].length() - 2, 2)) >> person;

		// the video number is the last character of the 3rd section of the filename.
		std::stringstream(filename_parts[2].substr(filename_parts[2].length() - 1, 1)) >> video_number;

		// now extract each mosift point and assign it to the correct codeword.
		bool success;
		cv::Mat hist = buildHistogram(files[i], success);

		if (!success)
			continue;

		// prepare line to be written to file.
		stringstream ss;
		string current_line;

		ss << (action + 1) << " ";
		for (unsigned col = 0; col < hist.cols; ++col)
		{
			ss << (col + 1) << ":" << hist.at<float>(0, col) << " ";

			if (!(hist.at<float>(0, col) >= 0))
			{
				int zz = 0;
				zz++;
			}
		}
		current_line = ss.str();
		ss.str("");
		ss.clear();

		// print output to correct files. libsvm-ready
		for (unsigned int p = 0; p < NUMBER_OF_PEOPLE; ++p)
		{
			if ((p + 1) == person)
			{
				// to print histogram to testing file.  leave this one out!
				testing_file_lines[p].push_back(current_line);
			}

			// this shouldn't be left out. print to training file.
			else
			{
				training_file_lines[p].push_back(current_line);
			}
		}

		// now write to old style files.
		label_file << action + 1 << "," << person << "," << video_number << std::endl;

		for (unsigned int col = 0; col < hist.cols; ++col)
		{
			hist_file << hist.at<float>(0, col);
			if (col < hist.cols - 1)
				hist_file << ",";
		}
		hist_file << std::endl;
	}
	hist_file.close();
	label_file.close();

	for (unsigned int i = 0; i < NUMBER_OF_PEOPLE; ++i)
	{
		for (unsigned line = 0; line < training_file_lines[i].size(); ++line)
		{
			*training_files[i] << training_file_lines[i][line] << endl;
		}
		
		for (unsigned line = 0; line < testing_file_lines[i].size(); ++line)
		{
			*testing_files[i] << testing_file_lines[i][line] << endl;
		}
	}

	// close the libsvm training and testing files.
	for (unsigned int i = 0; i < NUMBER_OF_PEOPLE; ++i)
	{
		training_files[i]->close();
		testing_files[i]->close();
	}
}

BagOfWordsRepresentation::BagOfWordsRepresentation(std::vector<std::string> &file_list, int num_clust, int ftr_dim, int num_people, bool appearance_is_bin, bool motion_is_bin) : NUMBER_OF_CLUSTERS(num_clust), 
	FEATURE_DIMENSIONALITY(ftr_dim), NUMBER_OF_PEOPLE(num_people), motion_is_binary(motion_is_bin), appearance_is_binary(appearance_is_bin)
{
	files = file_list;
	loadClusters();
	normalizeClusters();

	// default values. MoSIFT.
	motion_descriptor_size = 128; 
	appearance_descriptor_size = 128;
}

void BagOfWordsRepresentation::setMotionDescriptor(unsigned int size, bool binary)
{
	motion_is_binary = binary;
	motion_descriptor_size = size;
}

void BagOfWordsRepresentation::setAppearanceDescriptor(unsigned int size, bool binary)
{
	appearance_is_binary = binary;
	appearance_descriptor_size = size;
}