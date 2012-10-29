#include "BagOfWordsRepresentation.h"
#include <exception>

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


void BagOfWordsRepresentation::findBestMatch(cv::Mat &feature_vector, cv::Mat &clusters, int &best_cluster_index, float &best_cluster_score)
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

	if (motion_is_binary && (motion_descriptor_size > 0))
	{
		motion_distance = hammingDistance(query_motion_descriptor, cluster_motion_descriptor);
		motion_distance /= MOTION_HAMMING_DIST_NORM;
	}
	else
	{
		motion_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
	}

	appearance_distance = 0;
	float final_dist = appearance_distance + motion_distance;
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
		if (motion_is_binary && (motion_descriptor_size > 0))
		{
			motion_distance = hammingDistance(query_motion_descriptor, cluster_motion_descriptor);
			motion_distance /= MOTION_HAMMING_DIST_NORM;
		}
		else
		{
			motion_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
		}

		appearance_distance = 0;
		final_dist = appearance_distance + motion_distance;

		// If we have a new shortest distance, store that.
		if (final_dist < best_cluster_score)
		{
			best_cluster_score = final_dist;
			best_cluster_index = cluster;
		}
	}

	query_appearance_descriptor.release();
	cluster_appearance_descriptor.release();
	query_motion_descriptor.release();
	cluster_motion_descriptor.release();
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
		
		findBestMatch(feature_vector, *clusters, best_cluster_index, best_cluster_score);


		// + 1 to that codeword
		histogram.at<float>(0, best_cluster_index) = histogram.at<float>(0, best_cluster_index) + 1;
		success = true;
		feature_vector.release();
	}

	input_file.close();

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

	string cluster_path = "clusters.txt";

	bool DISTRIBUTED = false;
	if (DISTRIBUTED)
	{
		cluster_path = "C:/TRECVID/clusters.txt";
	}
	ifstream cluster_file(cluster_path);
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

	cluster_file.close();
}

// when doing this, make sure all mofreak points for the video are in ONE file, to avoid missing cuts.
void BagOfWordsRepresentation::computeSlidingBagOfWords(std::string &file, int alpha, int label, ofstream &out)
{
	bool over_alpha_frames = false;
	string distance_file = file;

	std::list<cv::Mat> histograms_per_frame;
	std::vector<cv::Mat> feature_list; // for just the current frame.

	// get start frame and start loading mofreak features...
	ifstream input_file(file);
	string line;
	int current_frame;

	std::getline(input_file, line);
	std::istringstream init(line);
	double discard;
	
	// ignore the first two values. the 3rd is the frame.
	init >> discard;
	init >> discard;
	init >> current_frame;

	// 4 5 and 6 aren't relevant either.
	init >> discard;
	init >> discard;
	init >> discard;

	// load the first feature to the list.  Now we're rolling.
	cv::Mat ftr_vector(1, FEATURE_DIMENSIONALITY, CV_32FC1);
	float elem;

	for (unsigned int i = 0; i < FEATURE_DIMENSIONALITY; ++i)
	{
		init >> elem;
		ftr_vector.at<float>(0, i) = elem;
	}
	feature_list.push_back(ftr_vector);

	// now the first line is out of the way.  Load them all
	// ---------------------------------------------------

	while (std::getline(input_file, line))
	{
		std::istringstream iss(line);
		// first two still aren't relevant... x,y position.
		iss >> discard;
		iss >> discard;

		int frame;
		iss >> frame;
		// next 3 aren't relevant.
		iss >> discard;
		iss >> discard;
		iss >> discard;

		// still on the same frame.  Just add to our feature list.
		if (frame == current_frame)
		{
			cv::Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_32FC1);
			for (unsigned i = 0; i < FEATURE_DIMENSIONALITY; ++i)
			{
				iss >> elem;
				feature_vector.at<float>(0, i) = elem;
			}
			
			feature_list.push_back(feature_vector);
		}

		// new frame.  need to compute the hist on that frame and possibly compute a new BOW feature.
		else
		{
			// 1: compute the histogram...
			cv::Mat new_histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
			for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				new_histogram.at<float>(0, col) = 0;

			for (auto it = feature_list.begin(); it != feature_list.end(); ++it)
			{
				// match that vector against centroids to assign to correct codeword.
				// brute force match each mosift point against all clusters to find best match.
				int best_cluster_index;
				float best_cluster_score;
		
				findBestMatch(*it, *clusters, best_cluster_index, best_cluster_score);

				// + 1 to that codeword
				new_histogram.at<float>(0, best_cluster_index) = new_histogram.at<float>(0, best_cluster_index) + 1;
			}

			// 2: add the histogram to our list.
			histograms_per_frame.push_back(new_histogram);

			// histogram list is at capacity!
			// compute summed histogram over all histograms as BOW feature.
			// then pop.
			// finally, write this new libsvm-worthy feature to file
			if (histograms_per_frame.size() == alpha)
			{
				over_alpha_frames = true;

				cv::Mat histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
				for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
					histogram.at<float>(0, col) = 0;

				// sum over the histograms we have.
				for (auto it = histograms_per_frame.begin(); it != histograms_per_frame.end(); ++it)
				{
					for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
					{
						histogram.at<float>(0, col) += it->at<float>(0, col);
					}
				}

				// remove oldest histogram.
				histograms_per_frame.pop_front();

				// normalize the histogram.
				double normalizer = 0.0;
				for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				{
					normalizer += histogram.at<float>(0, col);
				}

				for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				{
					histogram.at<float>(0, col) = histogram.at<float>(0, col) / normalizer;
				}

				// write to libsvm...
				stringstream ss;
				string current_line;

				ss << label << " ";
				for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
				{
					ss << (col + 1) << ":" << histogram.at<float>(0, col) << " ";
				}
				current_line = ss.str();
				ss.str("");
				ss.clear();

				out << current_line << endl;
			}

			// reset the feature list for the new frame.
			feature_list.clear();

			// add current line to the _new_ feature list.
			cv::Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_32FC1);
			for (int i = 0; i < FEATURE_DIMENSIONALITY; ++i)
			{
				iss >> elem;
				feature_vector.at<float>(0, i) = elem;
			}
			
			feature_list.push_back(feature_vector);
			current_frame = frame;
		}
	}

	// if we didn't get to write for this file, write it here.
	if (!over_alpha_frames)
	{
		cv::Mat histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
		for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
			histogram.at<float>(0, col) = 0;

		// sum over the histograms we have.
		for (auto it = histograms_per_frame.begin(); it != histograms_per_frame.end(); ++it)
		{
			for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
			{
				histogram.at<float>(0, col) += it->at<float>(0, col);
			}
		}

		// normalize the histogram.
		double normalizer = 0.0;
		for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		{
			normalizer += histogram.at<float>(0, col);
		}

		for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		{
			histogram.at<float>(0, col) = histogram.at<float>(0, col) / normalizer;
		}

		// write to libsvm...
		stringstream ss;
		string current_line;

		ss << label << " ";
		for (int col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		{
			ss << (col + 1) << ":" << histogram.at<float>(0, col) << " ";
		}
		current_line = ss.str();
		ss.str("");
		ss.clear();

		out << current_line << endl;
	}
}

void BagOfWordsRepresentation::computeBagOfWords()
{
	/*
	Open file streams for all leave-one-out
	possibilities for SVM evaluation.
	*/

	vector<ofstream *> training_files;
	vector<ofstream *> testing_files;
	vector<vector<string> > training_file_lines;
	vector<vector<string> > testing_file_lines;

	for (int i = 0; i < NUMBER_OF_PEOPLE; ++i)
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

	/*
	For each file, find the action + person number + video number
	person01_boxing_d2_uncomp.txt....
	So, split it on the underscore.
	word_1[-2:] is the person.
	word_2[:] should match one of the strings...
	word_3[1:] is the video number
	*/
	
	cout << "parallel time" << endl;
#pragma omp parallel for
	for (int i = 0; i < files.size(); ++i)
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

		std::vector<std::string> filename_parts = split(file_name_str, '_');
		std::stringstream(filename_parts[0].substr(filename_parts[0].length() - 2, 2)) >> person;
		std::stringstream(filename_parts[2].substr(filename_parts[2].length() - 1, 1)) >> video_number;
		filename_parts.clear();

		/*
		Iterate over each feature and assign it to the closest codeword.
		These assignments will build a histogram to be used
		as an SVM feature.
		*/

		//cout << "Building histogram..." << endl;
		bool success;
		cv::Mat hist;
		try
		{
			hist = buildHistogram(files[i], success);
		}
		catch (cv::Exception &e)
		{
			cout << "Error: " << e.what() << endl;
			exit(1);
		}

		if (!success)
			continue;

		/*
		Prepare each histogram to be written as a line to multiple files.
		It gets assigned to each training file, except for the training
		file where the person id matches that leave-one-out iteration
		*/

		stringstream ss;
		string current_line;

		ss << (action + 1) << " ";
		for (int col = 0; col < hist.cols; ++col)
		{
			ss << (int)(col + 1) << ":" << (float)hist.at<float>(0, col) << " ";
		}
		current_line = ss.str();
		ss.str("");
		ss.clear();

		for (int p = 0; p < NUMBER_OF_PEOPLE; ++p)
		{
			if ((p + 1) == person)
			{
				testing_file_lines[p].push_back(current_line);
			}
			else
			{
				training_file_lines[p].push_back(current_line);
			}
		}
	}

	cout << "done being parallel" << endl;

	/*
	Finally, after all of the BOW features have been computed,
	we write them to the corresponding files.
	This is outside of the parallelized loop,
	since writing to a file isn't thread-safe.
	*/

	if (training_file_lines.size() != NUMBER_OF_PEOPLE || testing_file_lines.size() != NUMBER_OF_PEOPLE)
	{
		cout << "Why on earth?" << endl;
		exit(1);
	}
	for (int i = 0; i < NUMBER_OF_PEOPLE; ++i)
	{
		cout << "num lines: " << training_file_lines[i].size() << endl;
		for (unsigned line = 0; line < training_file_lines[i].size(); ++line)
		{
			try
			{
				*training_files[i] << training_file_lines[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
		
		cout << "num lines: " << testing_file_lines[i].size() << endl;
		for (unsigned line = 0; line < testing_file_lines[i].size(); ++line)
		{
			try
			{
				*testing_files[i] << testing_file_lines[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
	}

	cout << "Wrote to files." << endl;

	// close the libsvm training and testing files.
	for (int i = 0; i < NUMBER_OF_PEOPLE; ++i)
	{
		training_files[i]->close();
		testing_files[i]->close();

		delete training_files[i];
		delete testing_files[i];
	}

	cout << "Closed the files. " << endl;

	training_files.clear();
	testing_files.clear();

	cout << "Cleared..." << endl;
}

BagOfWordsRepresentation::BagOfWordsRepresentation(int num_clust, int ftr_dim) : NUMBER_OF_CLUSTERS(num_clust),
	FEATURE_DIMENSIONALITY(ftr_dim), NUMBER_OF_PEOPLE(0)
{
	loadClusters();
	normalizeClusters();

	motion_descriptor_size = 8;
	appearance_descriptor_size = 0;
	motion_is_binary = true;
	appearance_is_binary = true;
}

BagOfWordsRepresentation::BagOfWordsRepresentation(std::vector<std::string> &file_list, 
	int num_clust, int ftr_dim, int num_people, bool appearance_is_bin, 
	bool motion_is_bin) : NUMBER_OF_CLUSTERS(num_clust), 
	FEATURE_DIMENSIONALITY(ftr_dim), NUMBER_OF_PEOPLE(num_people), motion_is_binary(motion_is_bin), appearance_is_binary(appearance_is_bin)
{
	files = file_list;
	loadClusters();
	//normalizeClusters(); [maybe put this back] [TODO]

	// default values.
	motion_descriptor_size = 8; 
	appearance_descriptor_size = 0;
	motion_is_binary = true;
	appearance_is_binary = true;
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

BagOfWordsRepresentation::~BagOfWordsRepresentation()
{
	clusters->release();
	delete clusters;


}