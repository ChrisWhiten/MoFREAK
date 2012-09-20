#include "BagOfWordsRepresentation.h"

BagOfWordsFeature::BagOfWordsFeature() : bag_of_words(cv::Mat()), _ft(NULL), number_of_words(0), matcher_type(BF_L2)
{
	matcher = NULL;
}

BagOfWordsFeature::BagOfWordsFeature(const BagOfWordsFeature &b)
{}


void BagOfWordsRepresentation::normalizeClusters()
{
	const int MOTION_START = 16;//64;
	const int MOTION_END = 144;//192;

	for (unsigned int clust = 0; clust < clusters->rows; ++clust)
	{
		normalizeMotionOfFeature((*clusters)(cv::Range(clust, clust + 1), cv::Range(0, clusters->cols)));
	}
}

void BagOfWordsRepresentation::normalizeMotionOfFeature(cv::Mat &ftr)
{
	const int MOTION_START = 16;//64;
	const int MOTION_END = 144;//192;

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

void BagOfWordsRepresentation::findBestMatchFREAKAndOpticalFlow(cv::Mat &feature_vector, cv::Mat &clusters, int &best_cluster_index, float &best_cluster_score, ofstream &file)
{
	// constants
	const int FREAK_HAMMING_DIST_NORM = 512;
	const int FREAK_START_INDEX = 0;
	const int FREAK_END_INDEX = 16;//64;
	const int MOTION_START_INDEX = 16;//64;
	const int MOTION_END_INDEX = 144;//192;

	// clusters a pre-normalized.  normalize feature vector.
	normalizeMotionOfFeature(feature_vector);

	// base case: initialize with the score/index of the 0th cluster.
	best_cluster_index = 0;

	// start with the hamming distance over FREAK points.
	float FREAK_distance = 0.0;

	cv::Mat query_FREAK_descriptor = feature_vector(cv::Range(0, 1), cv::Range(FREAK_START_INDEX, FREAK_END_INDEX));
	cv::Mat cluster_FREAK_descriptor = clusters(cv::Range(0, 1), cv::Range(FREAK_START_INDEX, FREAK_END_INDEX));

	FREAK_distance = hammingDistance(query_FREAK_descriptor, cluster_FREAK_descriptor);
	FREAK_distance /= FREAK_HAMMING_DIST_NORM;
	file << FREAK_distance << ", ";

	cv::Mat query_motion_descriptor = feature_vector(cv::Range(0, 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));
	cv::Mat cluster_motion_descriptor = clusters(cv::Range(0, 1), cv::Range(MOTION_START_INDEX, MOTION_END_INDEX));

	float euclidean_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
	file << euclidean_distance << ", ";

	float final_dist = FREAK_distance + euclidean_distance;
	file << final_dist << std::endl;

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
		file << FREAK_distance << ", ";

		euclidean_distance = standardEuclideanDistance(query_motion_descriptor, cluster_motion_descriptor);
		file << euclidean_distance << ", ";

		final_dist = FREAK_distance + euclidean_distance;
		file << final_dist << std::endl;

		// compare to best.
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

cv::Mat BagOfWordsRepresentation::buildHistogram(QString &file)
{
	cv::Mat histogram(1, NUMBER_OF_CLUSTERS, CV_32FC1);
	for (unsigned col = 0; col < NUMBER_OF_CLUSTERS; ++col)
		histogram.at<float>(0, col) = 0;

	// open file.
	ifstream input_file(file.toStdString());
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
		findBestMatchFREAKAndOpticalFlow(feature_vector, *clusters, best_cluster_index, best_cluster_score, distances);
		//findBestMatchFREAKAndFrameDifference(feature_vector, *clusters, best_cluster_index, best_cluster_score);


		// + 1 to that codeword
		histogram.at<float>(0, best_cluster_index) = histogram.at<float>(0, best_cluster_index) + 1;
	}

	distances.close();
	ofstream unnormed_hist("unnormed.txt");
	for (unsigned col = 0; col < histogram.cols; ++col)
	{
		unnormed_hist << histogram.at<float>(0, col) << ", ";
	}
	unnormed_hist.close();

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

BagOfWordsRepresentation::BagOfWordsRepresentation(QStringList &qsl, int num_clust, int ftr_dim, int num_people) : NUMBER_OF_CLUSTERS(num_clust), 
	FEATURE_DIMENSIONALITY(ftr_dim), NUMBER_OF_PEOPLE(num_people)
{
	loadClusters();
	normalizeClusters();

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
	for (int i = 0; i < qsl.size(); ++i)// = qsl.begin(); it != qsl.end(); ++it)
	{
		
		QString temp = qsl[i];
		QStringList words = qsl[i].split("\\");
		QString file_name = words[words.length() - 1];

		int action, person, video_number;
		// get the action.
		if (qsl[i].contains("boxing"))
		{
			action = BOXING;
		}
		else if (qsl[i].contains("walking"))
		{
			action = WALKING;
		}
		else if (qsl[i].contains("jogging"))
		{
			action = JOGGING;
		}
		else if (qsl[i].contains("running"))
		{
			action = RUNNING;
		}
		else if (qsl[i].contains("handclapping"))
		{
			action = HANDCLAPPING;
		}
		else if (qsl[i].contains("handwaving"))
		{
			action = HANDWAVING;
		}
		else
		{
			action = HANDWAVING; // hopefully we never miss this?  Just giving a default value. 
		}

		
		// get the person.
		int first_underscore = file_name.indexOf("_");
		QString person_string = file_name.mid(first_underscore - 2, 2);
		person = person_string.toInt();

		// get the video number.
		int last_underscore = file_name.lastIndexOf("_");
		QString video_string = file_name.mid(last_underscore - 1, 1);
		video_number = video_string.toInt();

		// now extract each mosift point and assign it to the correct codeword.
		cv::Mat hist = buildHistogram(qsl[i]);

		// prepare line to be written to file.
		stringstream ss;
		string current_line;

		ss << (action + 1) << " ";
		for (unsigned col = 0; col < hist.cols; ++col)
		{
			ss << (col + 1) << ":" << hist.at<float>(0, col) << " ";
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

				/*
				(*(testing_files[p])) << action + 1 << " ";
				for (unsigned col = 0; col < hist.cols; ++col)
				{
					(*(testing_files[p])) << col + 1 << ":" << hist.at<float>(0, col) << " ";
				}
				(*(testing_files[p])) << std::endl;
				*/
			}

			// this shouldn't be left out. print to training file.
			else
			{
				training_file_lines[p].push_back(current_line);
				/*
				(*(training_files[p])) << action + 1 << " ";
				for (unsigned col = 0; col < hist.cols; ++col)
				{
					(*(training_files[p])) << col + 1 << ":" << hist.at<float>(0, col) << " ";
				}
				(*(training_files[p])) << std::endl;
				*/
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