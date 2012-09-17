#include "BagOfWordsRepresentation.h"

BagOfWordsFeature::BagOfWordsFeature() : bag_of_words(cv::Mat()), _ft(NULL), number_of_words(0), matcher_type(BF_L2)
{
	matcher = NULL;
}

BagOfWordsFeature::BagOfWordsFeature(const BagOfWordsFeature &b)
{}

void BagOfWordsRepresentation::doStuff()
{
	BagOfWordsFeature bow();
	// file name, matcher, 256 is the length of the mosift feature.
}

/*
float BagOfWordsRepresentation::standardEuclideanDistance(vector<float> a, vector<float> b) const
{
	float distance = 0.0;

	// compute distance between the two vectors.
	for (unsigned int i = 0; i < a.size(); ++i)
	{
		distance += ((a[i] - b[i]) * (a[i] - b[i]));
	}

	distance = sqrt(distance);
	return distance;
}*/

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

void BagOfWordsRepresentation::findBestMatch(cv::Mat &feature_vector, cv::Mat &clusters, int &best_cluster_index, float &best_cluster_score)
{
	// base case: initialize with the score/index of the 0th cluster.
	best_cluster_index = 0;

	// euclidean distance.
	float distance = 0.0;
	for (unsigned int i = 0; i < feature_vector.cols; ++i)
	{
		float a_i = feature_vector.at<float>(0, i);
		float b_i = clusters.at<float>(0, i);
		distance += ((a_i - b_i) * (a_i - b_i));
	}
	distance = sqrt(distance);
	best_cluster_score = distance;

	// now the remaining points.
	for (unsigned cluster = 1; cluster < clusters.rows; ++cluster)
	{
		distance = 0.0;
		for (unsigned int i = 0; i < feature_vector.cols; ++i)
		{
			float a_i = feature_vector.at<float>(0, i);
			float b_i = clusters.at<float>(cluster, i);
			distance += ((a_i - b_i) * (a_i - b_i));
		}
		distance = sqrt(distance);

		// compare to best.
		if (distance < best_cluster_score)
		{
			best_cluster_score = distance;
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

	while (std::getline(input_file, line))
	{
		// discard first 6 values.
		std::istringstream iss(line);
		double discard;

		for (unsigned i = 0; i < 6; ++i)
		{
			iss >> discard;
		}

		// read line and parse into 256-dim vector.
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
	}

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

	// open file streams to write data for SVM
	ofstream hist_file("hist.txt");
	ofstream label_file("label.txt");

	vector<ofstream *> training_files;
	vector<ofstream *> testing_files;

	for (unsigned int i = 0; i < NUMBER_OF_PEOPLE; ++i)
	{
		stringstream training_string;
		stringstream testing_string;

		training_string << "left_out_" << i + 1 << ".train";
		testing_string << "left_out_" << i + 1 << ".test";

		ofstream *training_file = new ofstream(training_string.str());
		ofstream *testing_file = new ofstream(testing_string.str());

		training_string.str("");
		training_string.clear();
		testing_string.str("");
		testing_string.clear();

		training_files.push_back(training_file);
		testing_files.push_back(testing_file);
	}

	// for each file, find the action + person number + video number
	// person01_boxing_d2_uncomp.txt....
	// So, split it on the underscore.
	// word_1[-2:] is the person.
	// word_2[:] should match one of the strings...
	// word_3[1:] is the video number
	for (auto it = qsl.begin(); it != qsl.end(); ++it)
	{
		
		int action, person, video_number;
		// get the action.
		if (it->contains("boxing"))
		{
			action = BOXING;
		}
		else if (it->contains("walking"))
		{
			action = WALKING;
		}
		else if (it->contains("jogging"))
		{
			action = JOGGING;
		}
		else if (it->contains("running"))
		{
			action = RUNNING;
		}
		else if (it->contains("handclapping"))
		{
			action = HANDCLAPPING;
		}
		else if (it->contains("handwaving"))
		{
			action = HANDWAVING;
		}
		else
		{
			action = HANDWAVING; // hopefully we never miss this?  Just giving a default value. 
		}

		// get the person.
		int first_underscore = it->indexOf("_");
		QString person_string = it->mid(first_underscore - 2, 2);
		person = person_string.toInt();

		// get the video number.
		int last_underscore = it->lastIndexOf("_");
		QString video_string = it->mid(last_underscore - 1, 1);
		video_number = video_string.toInt();

		// now extract each mosift point and assign it to the correct codeword.
		cv::Mat hist = buildHistogram(*it);

		// print output to correct files. libsvm-ready
		for (unsigned int i = 0; i < NUMBER_OF_PEOPLE; ++i)
		{
			if (i == person)
			{
				// print histogram to testing file.  leave this one out!
				(*(testing_files[i])) << action + 1 << " ";
				for (unsigned col = 0; col < hist.cols; ++col)
				{
					(*(testing_files[i])) << col + 1 << ":" << hist.at<float>(0, col);
				}
				(*(testing_files[i])) << std::endl;
			}

			// this shouldn't be left out. print to training file.
			else
			{
				(*(training_files[i])) << action + 1 << " ";
				for (unsigned col = 0; col < hist.cols; ++col)
				{
					(*(training_files[i])) << col + 1 << ":" << hist.at<float>(0, col);
				}
				(*(training_files[i])) << std::endl;
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

	// close the libsvm training and testing files.
	for (unsigned int i = 0; i < NUMBER_OF_PEOPLE; ++i)
	{
		training_files[i]->close();
		testing_files[i]->close();
	}
}