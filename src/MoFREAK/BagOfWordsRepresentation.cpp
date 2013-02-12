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

int BagOfWordsRepresentation::bruteForceMatch(cv::Mat &feature)
{
	int shortest_distance = INT_MAX;
	int shortest_index = -1;

	for (int i = 0; i < clusters_for_matching.size(); i++)
	{
		unsigned int dist = hammingDistance(feature, clusters_for_matching[i]);
		if (dist < shortest_distance)
		{
			shortest_distance = dist;
			shortest_index = i;
		}
	}
	return shortest_index;
}

unsigned int BagOfWordsRepresentation::hammingDistance(cv::Mat &a, cv::Mat &b)
{
	unsigned int distance = 0;
	for (int row = 0; row < a.rows; ++row)
	{
		for (int col = 0; col < a.cols; ++col)
		{
			distance += hammingDistance(a.at<unsigned char>(row, col), b.at<unsigned char>(row, col));
		}
	}

	return distance;
}

unsigned int BagOfWordsRepresentation::hammingDistance(unsigned char a, unsigned char b)
{
	unsigned int hamming_distance = 0;
	// start as 0000 0001
	unsigned int bit = 1;

	// get the xor of a and b, each 1 in the xor adds to the hamming distance...
	unsigned int xor_result = a ^ b;

	// now count the bits, using 'bit' as a mask and performing a bitwise AND
	for (bit = 1; bit != 0; bit <<= 1)
	{
		if ((xor_result & bit) != 0)
		{
			hamming_distance++;
		}
	}

	return hamming_distance;
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
		cv::Mat feature_vector(1, FEATURE_DIMENSIONALITY, CV_8U);
		float elem;
		
		for (unsigned int i = 0; i < FEATURE_DIMENSIONALITY; ++i)
		{
			iss >> elem;
			feature_vector.at<unsigned char>(0, i) = (unsigned char)elem;
		}

		// match that vector against centroids to assign to correct codeword.
		// brute force match each mofreak point against all clusters to find best match.
		//std::vector<cv::DMatch> matches;
		//bf_matcher->match(feature_vector, matches);
		int best_match = bruteForceMatch(feature_vector);

		// + 1 to that codeword
		//histogram.at<float>(0, matches[0].imgIdx) = histogram.at<float>(0, matches[0].imgIdx) + 1;
		histogram.at<float>(0, best_match) = histogram.at<float>(0, best_match) + 1;
		success = true;
		feature_vector.release();
	}

	input_file.close();

	if (!success)
		return histogram;

	// after doing that for all lines in file, normalize.
	float histogram_sum = 0;
	for (int col = 0; col < histogram.cols; ++col)
	{
		histogram_sum += histogram.at<float>(0, col);
	}

	for (int col = 0; col < histogram.cols; ++col)
	{
		histogram.at<float>(0, col) = histogram.at<float>(0, col)/histogram_sum;
	}
	
	return histogram;
}

void BagOfWordsRepresentation::loadClusters()
{
	clusters = new cv::Mat(NUMBER_OF_CLUSTERS, FEATURE_DIMENSIONALITY, CV_8U);

	string cluster_path = SVM_PATH + "/clusters.txt";

	ifstream cluster_file(cluster_path);
	if ((!cluster_file.is_open()) || cluster_file.bad())
	{
		cout << "could not open clusters file" << endl;
		exit(1);
	}

	string line;
	unsigned int row = 0;
	
	while (std::getline(cluster_file, line))
	{
		cv::Mat single_cluster(1, FEATURE_DIMENSIONALITY, CV_8U);
		float elem;
		for (unsigned int col = 0; col < FEATURE_DIMENSIONALITY; ++col)
		{
			cluster_file >> elem;
			clusters->at<unsigned char>(row, col) = (unsigned char)elem;
			single_cluster.at<unsigned char>(0, col) = (unsigned char)elem;
		}
		clusters_for_matching.push_back(single_cluster);
		++row;
	}
	cluster_file.close();

	// add clusters to bruteforce matcher.
	bf_matcher->add(clusters_for_matching);
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
				// brute force match each mofreak point against all clusters to find best match.
				std::vector<cv::DMatch> matches;
				bf_matcher->match(*it, matches);	

				// + 1 to that codeword
				new_histogram.at<float>(0, matches[0].imgIdx) = new_histogram.at<float>(0, matches[0].imgIdx) + 1;
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
				float normalizer = 0.0;
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
		float normalizer = 0.0;
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

// this function should be deprecated,
// since we are now using the automatic action-tagging, rather than
// hard-coding this stuff.. only exists for hmdb temporarily [DEPRECATED] [TODO]
int BagOfWordsRepresentation::actionStringToActionInt(string act)
{
	if (act == "brush_hair")
	{
		return BRUSH_HAIR;
	}

	else if (act == "cartwheel")
	{
		return CARTWHEEL;	
	}

	else if (act == "catch")
	{
		return CATCH;
	}

	else if (act == "chew")
	{
		return CHEW;
	}

	else if (act == "clap")
	{
		return CLAP;
	}

	else if (act == "climb")
	{
		return CLIMB;
	}

	else if (act == "climb_stairs")
	{
		return CLIMB_STAIRS;
	}

	else if (act == "draw_sword")
	{
		return DRAW_SWORD;
	}

	else if (act == "dribble")
	{
		return DRIBBLE;
	}

	else if (act == "drink")
	{
		return DRINK;
	}

	else if (act == "dive")
	{
		return DIVE;
	}

	else if (act == "eat")
	{
		return EAT;
	}

	else if (act == "fall_floor")
	{
		return FALL_FLOOR;
	}

	else if (act == "fencing")
	{
		return FENCING;
	}

	else if (act == "flic_flac")
	{
		return FLIC_FLAC;
	}

	else if (act == "golf")
	{
		return GOLF;
	}

	else if (act == "handstand")
	{
		return HANDSTAND;
	}

	else if (act == "hit")
	{
		return HIT;
	}

	else if (act == "hug")
	{
		return HUG;
	}

	else if (act == "jump")
	{
		return JUMP;
	}

	else if (act == "kick")
	{
		return KICK;
	}

	else if (act == "kick_ball")
	{
		return KICK_BALL;
	}

	else if (act == "kiss")
	{
		return KISS;
	}

	else if (act == "laugh")
	{
		return LAUGH;
	}

	else if (act == "pick")
	{
		return PICK;
	}

	else if (act == "pour")
	{
		return POUR;
	}

	else if (act == "pullup")
	{
		return PULLUP;
	}

	else if (act == "punch")
	{
		return PUNCH;
	}

	else if (act == "push")
	{
		return PUSH;
	}

	else if (act == "pushup")
	{
		return PUSHUP;
	}

	else if (act == "ride_bike")
	{
		return RIDE_BIKE;
	}

	else if (act == "ride_horse")
	{
		return RIDE_HORSE;
	}

	else if (act == "run")
	{
		return RUN;
	}

	else if (act == "shake_hands")
	{
		return SHAKE_HANDS;
	}

	else if (act == "shoot_ball")
	{
		return SHOOT_BALL;
	}

	else if (act == "shoot_bow")
	{
		return SHOOT_BOW;
	}

	else if (act == "shoot_gun")
	{
		return SHOOT_GUN;
	}

	else if (act == "sit")
	{
		return SIT;
	}

	else if (act == "situp")
	{
		return SITUP;
	}

	else if (act == "smile")
	{
		return SMILE;
	}

	else if (act == "smoke")
	{
		return SMOKE;
	}

	else if (act == "somersault")
	{
		return SOMERSAULT;
	}

	else if (act == "stand")
	{
		return STAND;
	}

	else if (act == "swing_baseball")
	{
		return SWING_BASEBALL;
	}

	else if (act == "sword")
	{
		return SWORD;
	}

	else if (act == "sword_exercise")
	{
		return SWORD_EXERCISE;
	}

	else if (act == "talk")
	{
		return TALK;
	}

	else if (act == "throw")
	{
		return THROW;
	}

	else if (act == "turn")
	{
		return TURN;
	}

	else if (act == "walk")
	{
		return WALK;
	}

	else if (act == "wave")
	{
		return WAVE;
	}

	else
	{
		cout << "****Didn't find action: " << act << endl;
		system("PAUSE");
		exit(1);
		return BRUSH_HAIR;
	}
}

void BagOfWordsRepresentation::computeHMDB51BagOfWords(string SVM_PATH, string MOFREAK_PATH, string METADATA_PATH)
{
	/*
	There are 3 splits to build.
	There are 3 files for every action,
	1 for each split, each ending with split(x).txt.
	Build a .test and .train file for each...
	*/

	ofstream train1(SVM_PATH + "/split1.train");
	ofstream train2(SVM_PATH + "/split2.train");
	ofstream train3(SVM_PATH + "/split3.train");
	ofstream test1(SVM_PATH + "/split1.test");
	ofstream test2(SVM_PATH + "/split2.test");
	ofstream test3(SVM_PATH + "/split3.test");

	vector<string> train1_lines;
	vector<string> train2_lines;
	vector<string> train3_lines;
	vector<string> test1_lines;
	vector<string> test2_lines;
	vector<string> test3_lines;

	/*
	go through each file in the metadata folder one by one...
	parse the filename to get the split number.
	read each line and parse lines by the last available space.
	parsed[0] + .mofreak is the features.
	parsed[1] is whether it goes in test, train, or nothing.
	*/

	directory_iterator end_iter;

	for (directory_iterator dir_iter(METADATA_PATH); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			vector<string> training_files;
			vector<string> testing_files;

			path current_file = dir_iter->path();
			string action_filename = current_file.filename().string();
			int pos = action_filename.find("_test_");
			string str_action = action_filename.substr(0, pos);
			cout << "Action: " << str_action << endl;
			int action = actionStringToActionInt(str_action);

			int split;
			pos = action_filename.find(".txt");
			istringstream(action_filename.substr(pos - 1, 1)) >> split;
			
			/*
			Parse the file to get the mofreak file and 
			whether it should go in test, train, or neither.
			*/
				
			cout << "current file: " << current_file.string() << "..." << endl;
			ifstream metadata_file(current_file.string());
			string line;

			while (std::getline(metadata_file, line))
			{
				int last_space_index = line.find_last_of(' ');
				string mofreak_file = line.substr(0, last_space_index - 2) + ".mofreak";
				string label = line.substr(line.length() - 2, 1);

				if (label == "1")
				{
					training_files.push_back(MOFREAK_PATH + "/" + str_action + "/" + mofreak_file);
				}
				else if (label == "2")
				{
					testing_files.push_back(MOFREAK_PATH + "/" + str_action + "/" + mofreak_file);
				}
			}
			metadata_file.close();

			/*
			Now we know which videos go where for this action.
			Go ahead and compute the BOW features.
			*/
#pragma omp parallel for
			for (int j = 0; j < training_files.size(); ++j)
			{
				bool success;
				cv::Mat hist = buildHistogram(training_files[j], success);
				
				if (!success)
					continue;

				// prepare line to be written to file.
				stringstream ss;
				string current_line;

				ss << (action + 1) << " ";
				for (int col = 0; col < hist.cols; ++col)
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

				switch(split)
				{
				case 1:
					train1_lines.push_back(current_line);
					break;
				case 2:
					train2_lines.push_back(current_line);
					break;
				case 3:
					train3_lines.push_back(current_line);
					break;
				default:
					cout << "Couldn't get right split! " << split << endl;
					break;
				}
			} // end training files 'for loop'

#pragma omp parallel for
			for (int j = 0; j < testing_files.size(); ++j)
			{
				bool success;
				cv::Mat hist = buildHistogram(testing_files[j], success);

				if (!success)
					continue;

				// prepare line to be written to file.
				stringstream ss;
				string current_line;

				ss << (action + 1) << " ";
				for (int col = 0; col < hist.cols; ++col)
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

				switch(split)
				{
				case 1:
					test1_lines.push_back(current_line);
					break;
				case 2:
					test2_lines.push_back(current_line);
					break;
				case 3:
					test3_lines.push_back(current_line);
					break;
				default:
					cout << "Couldn't get right split! " << split << endl;
					break;
				}
			} // end testing files 'for loop'
		}
	}

	// print all of the train/test files
	for (auto it = train1_lines.begin(); it != train1_lines.end(); ++it)
		train1 << *it << endl;
	for (auto it = train2_lines.begin(); it != train2_lines.end(); ++it)
		train2 << *it << endl;
	for (auto it = train3_lines.begin(); it != train3_lines.end(); ++it)
		train3 << *it << endl;

	for (auto it = test1_lines.begin(); it != test1_lines.end(); ++it)
		test1 << *it << endl;
	for (auto it = test2_lines.begin(); it != test2_lines.end(); ++it)
		test2 << *it << endl;
	for (auto it = test3_lines.begin(); it != test3_lines.end(); ++it)
		test3 << *it << endl;

	train1.close();
	train2.close();
	train3.close();
	test1.close();
	test2.close();
	test3.close();
}

void BagOfWordsRepresentation::extractMetadata(std::string filename, int &action, int &group, int &clip_number)
{
	if (false)//dataset == KTH)
	{
		// get the action.
		/*if (boost::contains(filename, "boxing"))
		{
			action = BOXING;
		}
		else if (boost::contains(filename, "walking"))
		{
			action = WALKING;
		}
		else if (boost::contains(filename, "jogging"))
		{
			action = JOGGING;
		}
		else if (boost::contains(filename, "running"))
		{
			action = RUNNING;
		}
		else if (boost::contains(filename, "handclapping"))
		{
			action = HANDCLAPPING;
		}
		else if (boost::contains(filename, "handwaving"))
		{
			action = HANDWAVING;
		}
		else
		{
			std::cout << "Didn't find action: " << filename << std::endl;
			exit(1);
		}*/

		std::vector<std::string> filename_parts = split(filename, '_');
		std::stringstream(filename_parts[0].substr(filename_parts[0].length() - 2, 2)) >> group;
		std::stringstream(filename_parts[2].substr(filename_parts[2].length() - 1, 1)) >> clip_number;
	}
	else if (dataset == UTI2)
	{
		// parse the filename.
		std::vector<std::string> filename_parts = split(filename, '_');

		// the "person" (video) is the number after the first underscore.
		std::string person_str = filename_parts[1];
		std::stringstream(filename_parts[1]) >> group;

		// the action is the number after the second underscore, before .avi.
		std::string vid_str = filename_parts[2];
		std::stringstream(filename_parts[2].substr(0, 1)) >> action;

		// video number.. not sure if useful for this dataset.
		std::stringstream(filename_parts[0]) >> clip_number;
	}
	else if (dataset == UCF101 || dataset == KTH)
	{
		std::vector<std::string> filename_parts = split(filename, '_');

		// extract action.
		std::string parsed_action = filename_parts[1];

		if (actions.find(parsed_action) == actions.end())
		{
			actions[parsed_action] = actions.size();
		}

		action = actions[parsed_action];

		// extract group
		if (dataset == KTH)
		{
			std::stringstream(filename_parts[0].substr(filename_parts[0].length() - 2, 2)) >> group;
		}
		else
		{
			std::string parsed_group = filename_parts[2].substr(1, 2);
			std::stringstream(parsed_group) >> group;
		}
		group--; // group indices start at 0, not 1, so decrement.

		// extract clip number.
		std::string parsed_clip = filename_parts[3].substr(2, 2);
		std::stringstream(parsed_clip) >> clip_number;
	}
}

void BagOfWordsRepresentation::intializeBOWMemory(string SVM_PATH)
{

	// initialize the output files and memory for BOW features for each grouping.
	for (int group = 0; group < NUMBER_OF_GROUPS; ++group)
	{
		stringstream training_filepath, testing_filepath;
		training_filepath << SVM_PATH << "/" << group + 1 << ".train";
		testing_filepath << SVM_PATH << "/" << group + 1 << ".test";

		ofstream *training_filestream = new ofstream(training_filepath.str());
		ofstream *testing_filestream = new ofstream(testing_filepath.str());

		training_files.push_back(training_filestream);
		testing_files.push_back(testing_filestream);

		training_filepath.str("");
		training_filepath.clear();
		testing_filepath.str("");
		testing_filepath.clear();

		std::vector<string> training_features;
		std::vector<string> testing_features;
		bow_training_crossvalidation_sets.push_back(training_features);
		bow_testing_crossvalidation_sets.push_back(testing_features);
	}
}

void BagOfWordsRepresentation::convertFileToBOWFeature(std::string file)
{
	boost::filesystem::path file_path(file);
	boost::filesystem::path file_name = file_path.filename();
	std::string file_name_str = file_name.generic_string();

	// extract the metadata from this file, such as the group and action performed.
	int action, group, video_number;
	extractMetadata(file_name_str, action, group, video_number);

	/*
	Now, extract each mofreak features and assign them to the correct codeword.
	buildHistogram returns a histogram representation (1 row, num_clust cols)
	of the bag-of-words feature.  If for any reason the process fails,
	the "success" boolean will be returned as false
	*/
	bool success;
	cv::Mat bow_feature;

	try
	{
		bow_feature = buildHistogram(file, success);
	}
	catch (cv::Exception &e)
	{
		cout << "Error: " << e.what() << endl;
		exit(1);
	}

	if (!success)
	{
		std::cout << "Bag-of-words feature construction was unsuccessful.  Investigate." << std::endl;
		exit(1);
	//	continue;
	}

	/*
	Prepare each histogram to be written as a line to multiple files.
	It gets assigned to each training file, except for the training
	file where the group id matches that leave-one-out iteration
	*/

	stringstream ss;
	ss << (action + 1) << " ";

	for (int col = 0; col < bow_feature.cols; ++col)
	{
		ss << (int)(col + 1) << ":" << (float)bow_feature.at<float>(0, col) << " ";
	}

	string current_line;
	current_line = ss.str();
	ss.str("");
	ss.clear();

	for (int g = 0; g < NUMBER_OF_GROUPS; ++g)
	{
		// for earlier datasets, we started groupings at 1 (first person, etc).
		// Transitioning to 0-indexing, but keeping this here as a reminder,
		// in case we run into any missed cases.

		//if ((g + 1) == group)
		if (g == group)
		{
			bow_testing_crossvalidation_sets[g].push_back(current_line);
		}
		else
		{
			bow_training_crossvalidation_sets[g].push_back(current_line);
		}
	}
}

void BagOfWordsRepresentation::writeBOWFeaturesToFiles()
{
	// ensure that we have the correct number of open files
	if (bow_training_crossvalidation_sets.size() != NUMBER_OF_GROUPS || bow_testing_crossvalidation_sets.size() != NUMBER_OF_GROUPS)
	{
		cout << "Incorrect number of training or testing file lines.  Check mapping from bow feature to test/train files" << endl;
		exit(1);
	}

	// for each group, write the training and testing cross-validation files.
	for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
	{
		cout << "number of training features: " << bow_training_crossvalidation_sets[i].size() << endl;
		for (unsigned line = 0; line < bow_training_crossvalidation_sets[i].size(); ++line)
		{
			try
			{
				*training_files[i] << bow_training_crossvalidation_sets[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
		
		cout << "number of testing features: " << bow_testing_crossvalidation_sets[i].size() << endl;
		for (unsigned line = 0; line < bow_testing_crossvalidation_sets[i].size(); ++line)
		{
			try
			{
				*testing_files[i] << bow_testing_crossvalidation_sets[i][line] << endl;
			}
			catch (exception &e)
			{
				cout << "Error: " << e.what() << endl;
				exit(1);
			}
		}
	}

	cout << "Finished writing to cross-validation files." << endl;

	// close the libsvm training and testing files.
	for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
	{
		training_files[i]->close();
		testing_files[i]->close();

		delete training_files[i];
		delete testing_files[i];
	}

	cout << "Closed all cross-validation files. " << endl;

	training_files.clear();
	testing_files.clear();
}

void BagOfWordsRepresentation::computeBagOfWords(string SVM_PATH, string MOFREAK_PATH, string METADATA_PATH)
{
	if (dataset == HMDB51)
	{
		computeHMDB51BagOfWords(SVM_PATH, MOFREAK_PATH, METADATA_PATH);
		return;
	}

	if (dataset == KTH || dataset == UCF101)
	{
		for (int i = 0; i < NUMBER_OF_GROUPS; ++i)
		{
			stringstream training_string;
			stringstream testing_string;

			training_string << SVM_PATH + "/left_out_" << i + 1 << ".train";
			testing_string << SVM_PATH + "/left_out_" << i + 1 << ".test";

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
			bow_training_crossvalidation_sets.push_back(training_lines);
			bow_testing_crossvalidation_sets.push_back(testing_lines);
		}
	}
	
	cout << "parallelizing bag-of-words matches across files." << endl;
#pragma omp parallel for
	for (int i = 0; i < files.size(); ++i)
	{
		convertFileToBOWFeature(files[i]);
	}
	cout << "done bag-of-words feature construction" << endl;

	/*
	Finally, after all of the BOW features have been computed,
	we write them to the corresponding files.
	This is outside of the parallelized loop,
	since writing to a file isn't thread-safe.
	*/
	writeBOWFeaturesToFiles();
}

BagOfWordsRepresentation::BagOfWordsRepresentation(int num_clust, int ftr_dim, std::string svm_path, int num_groups, int dset) : 
	NUMBER_OF_CLUSTERS(num_clust), FEATURE_DIMENSIONALITY(ftr_dim), SVM_PATH(svm_path), NUMBER_OF_GROUPS(num_groups), dataset(dset)
{
	bf_matcher = new cv::BFMatcher(cv::NORM_HAMMING);
	loadClusters();

	motion_descriptor_size = 8;
	appearance_descriptor_size = 8;
	motion_is_binary = true;
	appearance_is_binary = true;
}

BagOfWordsRepresentation::BagOfWordsRepresentation(std::vector<std::string> &file_list, 
	int num_clust, int ftr_dim, int num_groups, bool appearance_is_bin, 
	bool motion_is_bin, int dset, std::string svm_path) :

	NUMBER_OF_CLUSTERS(num_clust), FEATURE_DIMENSIONALITY(ftr_dim), 
	NUMBER_OF_GROUPS(num_groups), motion_is_binary(motion_is_bin), 
	appearance_is_binary(appearance_is_bin), dataset(dset),
	SVM_PATH(svm_path)
{
	files = file_list;
	bf_matcher = new cv::BFMatcher(cv::NORM_HAMMING);
	loadClusters();

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