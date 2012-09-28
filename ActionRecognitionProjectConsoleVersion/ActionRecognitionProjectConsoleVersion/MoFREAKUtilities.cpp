#include "MoFREAKUtilities.h"

// vanilla string split operation.  Direct copy-paste from stack overflow
// source: http://stackoverflow.com/questions/236129/splitting-a-string-in-c
std::vector<std::string> &MoFREAKUtilities::split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> MoFREAKUtilities::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}


// this is only implemented here, but not being used right now.
// the general idea is that we are going to replace optical flow by simple image difference.
// representing motion as a single unsigned integer.
unsigned int MoFREAKUtilities::extractMotionByImageDifference(cv::Mat &frame, cv::Mat &prev_frame, float x, float y)
{
	int f_rows = frame.rows;
	int p_rows = prev_frame.rows;

	int value_at_current_frame = frame.at<unsigned char>(y, x);
	int value_at_prev_frame = prev_frame.at<unsigned char>(y, x);

	return abs(value_at_current_frame - value_at_prev_frame);
}

unsigned int totalframediff(cv::Mat &frame, cv::Mat &prev)
{
	int frame_diff = 0;
	for (int row = 0; row < frame.rows; ++row)
	{
		for (int col = 0; col < frame.cols; ++col)
		{
			int value_at_current_frame = frame.at<unsigned char>(row, col);
			int value_at_prev_frame = prev.at<unsigned char>(row, col);

			frame_diff += abs(value_at_current_frame - value_at_prev_frame);
		}
	}

	return frame_diff;
}

// FREAK over the image difference image.
vector<unsigned int> MoFREAKUtilities::extractFREAK_ID(cv::Mat &frame, cv::Mat &prev_frame, float x, float y, float scale)
{
	// build the difference image.
	// to keep everything between 0-255 (as FREAK requires),
	// each value is divided by 2 and 128 is added to it.
	cv::Mat difference_image(frame.rows, frame.cols, CV_8U);
	for (int row = 0; row < frame.rows; ++row)
	{
		for (int col = 0; col < frame.cols; ++col)
		{
			int f_val = frame.at<unsigned char>(row, col);
			int p_val = prev_frame.at<unsigned char>(row, col);
			int diff_val = f_val - p_val;
			unsigned int shifted_val = (diff_val/2) + 128;
			difference_image.at<unsigned char>(row, col) = shifted_val;
		}
	}

	// difference image is computed.  Now extract the FREAK point.
	// [TODO] for efficiency, future iteration should only compute 
	// this difference image once per frame/prev_frame pair!
	vector<unsigned int> descriptor = extractFREAKFeature(difference_image, x, y, scale, true);
	return descriptor;
}

void MoFREAKUtilities::computeDifferenceImage(cv::Mat &current_frame, cv::Mat &prev_frame, cv::Mat &diff_img)
{
	for (int row = 0; row < current_frame.rows; ++row)
	{
		for (int col = 0; col < current_frame.cols; ++col)
		{
			int f_val = current_frame.at<unsigned char>(row, col);
			int p_val = prev_frame.at<unsigned char>(row, col);
			int diff_val = f_val - p_val;
			diff_img.at<unsigned char>(row, col) = abs(diff_val);
		}
	}
}

bool MoFREAKUtilities::sufficientMotion(cv::Mat &diff_integral_img, float &x, float &y, float &scale, int &motion)
{
	const int MOTION_THRESHOLD = 4096;
	// compute the sum of the values within this patch in the difference image.  It's that simple.
	int radius = ceil((scale)/2);

	// always + 1, since the integral image adds a row and col of 0s to the top-left.
	int tl_x = MAX(0, x - radius + 1);
	int tl_y = MAX(0, y - radius + 1);
	int br_x = MIN(diff_integral_img.cols, x + radius + 1);
	int br_y = MIN(diff_integral_img.rows, y + radius + 1);

	int br = diff_integral_img.at<int>(br_y, br_x);
	int tl = diff_integral_img.at<int>(tl_y, tl_x);
	int tr = diff_integral_img.at<int>(tl_y, br_x);
	int bl = diff_integral_img.at<int>(br_y, tl_x);
	motion = br + tl - tr - bl;

	return (motion > MOTION_THRESHOLD);
}

//void MoFREAKUtilities::computeMoFREAKFromFile(QString filename, bool clear_features_after_computation)
void MoFREAKUtilities::computeMoFREAKFromFile(std::string filename, bool clear_features_after_computation)
{
	std::string debug_filename = filename;
	debug_filename.append(".dbg");
	ofstream debug_stream(debug_filename);
	// ignore the first frame because we can't compute the frame difference with it.
	// Beyond that, go over all frames, extract FAST/SURF points, compute frame difference,
	// if frame difference is above some threshold, compute FREAK point and save.
	const int MOTION_THRESHOLD = 4;
	const int GAP_FOR_FRAME_DIFFERENCE = 5;

	cv::VideoCapture capture;
	capture.open(filename);

	if (!capture.isOpened())
	{
		debug_stream << "file wasn't opened! " << filename << endl;
	}

	cv::Mat current_frame;
	cv::Mat prev_frame;
	std::queue<cv::Mat> frame_queue;
	for (unsigned int i = 0; i < GAP_FOR_FRAME_DIFFERENCE; ++i)
	{
		capture >> prev_frame; // ignore first 'GAP_FOR_FRAME_DIFFERENCE' frames.  Read them in and carry on.
		frame_queue.push(prev_frame.clone());
	}
	prev_frame = frame_queue.front();
	frame_queue.pop();

	unsigned int frame_num = GAP_FOR_FRAME_DIFFERENCE - 1;

	while (true)
	{
		capture >> current_frame;
		if (current_frame.empty())	
		{
			break;
		}

		// compute the difference image for use in later computations.
		cv::Mat diff_img(current_frame.rows, current_frame.cols, CV_8U);
		cv::absdiff(current_frame, prev_frame, diff_img);
		//computeDifferenceImage(current_frame, prev_frame, diff_img);

		cv::Mat diff_integral_image(diff_img.rows + 1, diff_img.cols + 1, CV_32S);
		cv::integral(diff_img, diff_integral_image);

		vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;
		
		// detect all keypoints.
		cv::StarFeatureDetector *detector = new cv::StarFeatureDetector(15, 45, 50, 40);
		detector->detect(current_frame, keypoints);
		debug_stream << "detected " << keypoints.size() << " keypoints." << endl;

		// extract the FREAK descriptors efficiently over the whole frame
		// For now, we are just computing the motion FREAK!  It seems to be giving better results.
		cv::FREAK extractor;
		extractor.compute(diff_img, keypoints, descriptors);
		

		// for each detected keypoint
		unsigned char *pointer_to_descriptor_row = 0;
		unsigned int keypoint_row = 0;
		for (auto keypt = keypoints.begin(); keypt != keypoints.end(); ++keypt)
		{
			pointer_to_descriptor_row = descriptors.ptr<unsigned char>(keypoint_row);

			// only take points with sufficient motion.
			int motion = 0;
			if (sufficientMotion(diff_integral_image, keypt->pt.x, keypt->pt.y, keypt->size, motion))
			{
				debug_stream << "accepted motion of " << motion << " with scale " << keypt->size <<  endl;

				//vector<unsigned int> freak_descriptor = extractFREAKFeature(current_frame, keypt->pt.x, keypt->pt.y, keypt->size, false);
				//vector<unsigned int> motion_descriptor = extractFREAKFeature(diff_img, keypt->pt.x, keypt->pt.y, keypt->size, false);

				MoFREAKFeature ftr;
				ftr.frame_number = frame_num;
				ftr.scale = keypt->size;
				ftr.x = keypt->pt.x;
				ftr.y = keypt->pt.y;

				for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)
				{
					ftr.motion[i] = pointer_to_descriptor_row[i];
				}

				// gather metadata.
				int action, person, video_number;
				readMetadata(filename, action, video_number, person);

				ftr.action = action;
				ftr.video_number = video_number;
				ftr.person = person;

				// these parameters aren't useful right now.
				ftr.motion_x = 0;
				ftr.motion_y = 0;

				features.push_back(ftr);

				/*
				if ((freak_descriptor.size()) > 0 && (motion_descriptor.size() > 0))
				{
					MoFREAKFeature ftr;
					ftr.frame_number = frame_num;
					ftr.scale = keypt->size;
					ftr.x = keypt->pt.x;
					ftr.y = keypt->pt.y;

					for (unsigned i = 0; i < freak_descriptor.size(); ++i)
					{
						ftr.FREAK[i] = freak_descriptor[i];
					}

					for (unsigned i = 0; i < motion_descriptor.size(); ++i)
					{
						ftr.motion[i] = motion_descriptor[i];
					}

					// gather metadata.
					int action, person, video_number;
					readMetadata(filename, action, video_number, person);

					ftr.action = action;
					ftr.video_number = video_number;
					ftr.person = person;

					// these parameters aren't useful right now.
					ftr.motion_x = 0;
					ftr.motion_y = 0;

					features.push_back(ftr);
				}
				*/
			}
			/*else
			{
				debug_stream << "rejected motion of " << motion << " with scale " << keypt->size <<  endl;
			}*/
			keypoint_row++;
		} // at this point, gathered all the mofreak pts from the frame.
		debug_stream << "kept " << features.size() << " keypoints total." << endl;

		frame_queue.push(current_frame.clone());
		prev_frame = frame_queue.front();
		frame_queue.pop();
		frame_num++;
	}

	// in the end, print the mofreak file and reset the features for a new file.
	std::string mofreak_file = filename;
	mofreak_file.append(".mofreak");
	debug_stream << "writing this many features: " << features.size() << endl;
	debug_stream.close();
	writeMoFREAKFeaturesToFile(mofreak_file);

	if (clear_features_after_computation)
		features.clear();

}

vector<unsigned int> MoFREAKUtilities::extractFREAKFeature(cv::Mat &frame, float x, float y, float scale, bool extract_full_descriptor)
{
	const float SCALE_MULTIPLIER = 1;//6;
	cv::Mat descriptor;
	vector<cv::KeyPoint> ftrs;
	unsigned int DESCRIPTOR_SIZE;

	if (extract_full_descriptor)
		DESCRIPTOR_SIZE = 64;
	else
		DESCRIPTOR_SIZE = 16;

	// gather keypoint from image.
	cv::KeyPoint ftr;
	ftr.size = scale * SCALE_MULTIPLIER;
	ftr.angle = -1;
	ftr.pt = cv::Point2d(x, y);
	ftrs.push_back(ftr);
	
	// extract FREAK descriptor.
	cv::FREAK extractor;
	extractor.compute(frame, ftrs, descriptor);
	
	vector<unsigned int> ret_val;

	// when the feature is too close to the boundary, FREAK returns a null descriptor.  No good.
	if (descriptor.rows > 0)
	{
		for (unsigned i = 0; i < DESCRIPTOR_SIZE; ++i)//descriptor.cols; ++i)
		{
			ret_val.push_back(descriptor.at<unsigned char>(0, i));
		}
	}

	return ret_val;
}


void MoFREAKUtilities::addMoSIFTFeatures(int frame, vector<cv::KeyPoint> &pts, cv::VideoCapture &capture)
{
	cv::Mat prev_frame;
	cv::Mat current_frame;

	capture.set(CV_CAP_PROP_POS_FRAMES, frame - 5);
	capture >> prev_frame;
	prev_frame = prev_frame.clone();

	capture.set(CV_CAP_PROP_POS_FRAMES, frame);
	capture >> current_frame;

	// compute the difference image for use in later computations.
	cv::Mat diff_img(current_frame.rows, current_frame.cols, CV_8U);
	cv::absdiff(current_frame, prev_frame, diff_img);

	//extract
	cv::Mat descriptors;
	cv::FREAK extractor;
	extractor.compute(diff_img, pts, descriptors);
		

	// for each detected keypoint
	unsigned char *pointer_to_descriptor_row = 0;
	unsigned int keypoint_row = 0;
	for (auto keypt = pts.begin(); keypt != pts.end(); ++keypt)
	{
		pointer_to_descriptor_row = descriptors.ptr<unsigned char>(keypoint_row);

		// only take points with sufficient motion.
		MoFREAKFeature ftr;
		ftr.frame_number = frame;
		ftr.scale = keypt->size;
		ftr.x = keypt->pt.x;
		ftr.y = keypt->pt.y;

		for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)
		{
			ftr.motion[i] = pointer_to_descriptor_row[i];
		}

		// gather metadata.
		int action, person, video_number;
		//readMetadata(filename, action, video_number, person);
		action = 1;
		person = 1;
		video_number = 1; // these arne't useful for trecvid.

		ftr.action = action;
		ftr.video_number = video_number;
		ftr.person = person;

		// these parameters aren't useful right now.
		ftr.motion_x = 0;
		ftr.motion_y = 0;

		features.push_back(ftr);
		++keypoint_row;
	}

}
void MoFREAKUtilities::buildMoFREAKFeaturesFromMoSIFT(std::string mosift_file, string video_path, string mofreak_path)
{
	// remove old mofreak points.
	features.clear();
	int mofreak_file_id = 0;

	// gather mosift features.
	MoSIFTUtilities mosift;
	mosift.readMoSIFTFeatures(mosift_file);
	vector<MoSIFTFeature> mosift_features = mosift.getMoSIFTFeatures();
	cv::VideoCapture capture;

	capture.open(video_path);
	auto it = mosift_features.begin();
	vector<cv::KeyPoint> features_per_frame;
	int current_frame = it->frame_number;

	while (true)
	{
		if (it == mosift_features.end())
		{
			break;
		}

		if (it->frame_number == current_frame)
		{
			cv::KeyPoint keypt;
			keypt.pt = cv::Point2f(it->x, it->y);
			keypt.size = it->scale;
			features_per_frame.push_back(keypt);

			// move on to next mosift pt.
			++it;
		}
		else
		{
			// that's all the sift points for htis frame.  Do the computation.
			addMoSIFTFeatures(current_frame, features_per_frame, capture);
			current_frame = it->frame_number;
			features_per_frame.clear();

			// if running out of memory, write to file and continue.
			// > 5 mill, write.
			if (features.size() > 5000000)
			{
				string mofreak_out_file = mofreak_path;
				stringstream ss;
				ss << "." << mofreak_file_id;
				mofreak_out_file.append(ss.str());
				ss.str("");
				ss.clear();
				writeMoFREAKFeaturesToFile(mofreak_out_file);

				features.clear();
				mofreak_file_id++;
			}
		}
	}
	capture.release();
}

string MoFREAKUtilities::toBinaryString(unsigned int x)
{
	std::vector<bool> ret_vec;

	for (unsigned int z = x; z > 0; z /= 2)
	{
		bool r = ((z & 1) == 1);
		ret_vec.push_back(r);
	}

	int bits = std::numeric_limits<unsigned char>::digits;
	ret_vec.resize(bits);
	std::reverse(ret_vec.begin(), ret_vec.end());

	string ret_val = "";
	for (auto it = ret_vec.begin(); it != ret_vec.end(); ++it)
	{
		ret_val.push_back((*it) ? '1' : '0');
		ret_val.push_back(' ');
	}

	ret_val = ret_val.substr(0, ret_val.size() - 1);
	return ret_val;
}

void MoFREAKUtilities::writeMoFREAKFeaturesToFile(string output_file)
{
	ofstream f(output_file);

	for (auto it = features.begin(); it != features.end(); ++it)
	{
		// basics
		f << it->x << " " << it->y << " " << it->frame_number 
			<< " " << it->scale << " " << it->motion_x << " " << it->motion_y << " ";

		// FREAK
		/*for (int i = 0; i < NUMBER_OF_BYTES_FOR_APPEARANCE; ++i) // 64
		{
			int z = it->FREAK[i];
			f << it->FREAK[i] << " ";
			//f << toBinaryString(z) << " ";
		}*/

		// motion
		for (int i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)//64; ++i)//128; ++i)
		{
			int z = it->motion[i];
			f << z << " ";
		}
		f << "\n";
	}

	f.close();
}

// MoFREAK has 512 bits, so we can normalize this by summing the hamming distance
// over all bits in the MoFREAK vector and dividing the sum by 512..
unsigned int MoFREAKUtilities::hammingDistance(unsigned int a, unsigned int b)
{
	unsigned int hamming_distance = 0;
	// start as 0000 0001
	unsigned int bit = 1;

	// get the xor of a and b, each 1 in the xor adds to the hamming distance...
	unsigned int xor_result = a ^ b;

	// now count the bits, using 'bit' as a mask and performing a bitwise AND
	for (bit = 1; bit != 0; bit <<= 1)
	{
		hamming_distance += (xor_result & bit);
	}

	return hamming_distance;
}

double MoFREAKUtilities::motionNormalizedEuclideanDistance(vector<unsigned int> a, vector<unsigned int> b)
{
	double a_norm = 0.0, b_norm = 0.0, distance = 0.0;
	vector<double> normalized_a, normalized_b;

	// get euclidean norm for a.
	for (auto it = a.begin(); it != a.end(); ++it)
	{
		a_norm += (*it)*(*it);
	}
	a_norm = sqrt(a_norm);

	// get euclidean norm for b.
	for (auto it = b.begin(); it != b.end(); ++it)
	{
		b_norm += (*it)*(*it);
	}
	b_norm = sqrt(b_norm);

	// normalize vector a.
	for (auto it = a.begin(); it != a.end(); ++it)
	{
		normalized_a.push_back((*it)/a_norm);
	}

	// normalize vector b.
	for (auto it = b.begin(); it != b.end(); ++it)
	{
		normalized_b.push_back((*it)/b_norm);
	}

	// compute distance between the two normalized vectors.
	for (unsigned int i = 0; i < normalized_a.size(); ++i)
	{
		distance += ((normalized_a[i] - normalized_b[i]) * (normalized_a[i] - normalized_b[i]));
	}

	distance = sqrt(distance);
	return distance;
}

//void MoFREAKUtilities::readMetadata(QString filename, int &action, int &video_number, int &person)
void MoFREAKUtilities::readMetadata(std::string filename, int &action, int &video_number, int &person)
{
	//boost::filesystem::path file_path(filename.toStdString());
	boost::filesystem::path file_path(filename);
	boost::filesystem::path file_name = file_path.filename();
	std::string file_name_str = file_name.generic_string();
	
		//QStringList words = filename.split("\\");
		//QString file_name = words[words.length() - 1];

		// get the action.
		//if (file_name.contains("boxing"))
		if (boost::contains(file_name_str, "boxing"))
		{
			action = BOXING;
		}
		//else if (file_name.contains("walking"))
		else if (boost::contains(file_name_str, "walking"))
		{
			action = WALKING;
		}
		//else if (file_name.contains("jogging"))
		else if (boost::contains(file_name_str, "jogging"))
		{
			action = JOGGING;
		}
		//else if (file_name.contains("running"))
		else if (boost::contains(file_name_str, "running"))
		{
			action = RUNNING;
		}
		//else if (file_name.contains("handclapping"))
		else if (boost::contains(file_name_str, "handclapping"))
		{
			action = HANDCLAPPING;
		}
		//else if (file_name.contains("handwaving"))
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
		//person = atoi((filename_parts[0].substr(filename_parts[0].length() - 2, 2)).c_str());

		// the video number is the last character of the 3rd section of the filename.
		std::stringstream(filename_parts[2].substr(filename_parts[2].length() - 1, 1)) >> video_number;
		//video_number = atoi((filename_parts[2].substr(filename_parts[2].length() - 1, 1)).c_str());

		/*
		int first_underscore = file_name.indexOf("_");
		QString person_string = file_name.mid(first_underscore - 2, 2);
		person = person_string.toInt();

		// get the video number.
		int last_underscore = file_name.lastIndexOf("_");
		QString video_string = file_name.mid(last_underscore - 1, 1);
		video_number = video_string.toInt();
		*/
}

//void MoFREAKUtilities::readMoFREAKFeatures(QString filename)
void MoFREAKUtilities::readMoFREAKFeatures(std::string filename)
{
	int action, video_number, person;
	readMetadata(filename, action, video_number, person);

	ifstream stream;
	//stream.open(filename.toStdString());
	stream.open(filename);
	
	while (!stream.eof())
	{
		if (features.size() == 1049869)
		{
			int xy = 0;
			xy++;
		}
		// single feature
		MoFREAKFeature ftr;
		stream >> ftr.x >> ftr.y >> ftr.frame_number >> ftr.scale >> ftr.motion_x >> ftr.motion_y;
	
		// FREAK
		
		for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)//64; ++i)
		{
			unsigned int a;
			stream >> a;
			ftr.FREAK[i] = a;
		}
		

		// motion
		for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)//64; ++i)//128; ++i)
		{
			unsigned int a;
			stream >> a;
			ftr.motion[i] = a;
		}

		// metadata
		ftr.action = action;
		ftr.video_number = video_number;
		ftr.person = person;

		// add new feature to collection.
		features.push_back(ftr);
	}
	stream.close();
}


vector<MoFREAKFeature> MoFREAKUtilities::getMoFREAKFeatures()
{
	return features;
}

void MoFREAKUtilities::setAllFeaturesToLabel(int label)
{
	for (unsigned i = 0; i < features.size(); ++i)
	{
		features[i].action = label;
	}
}