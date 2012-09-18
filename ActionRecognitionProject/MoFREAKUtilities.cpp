#include "MoFREAKUtilities.h"

// this is only implemented here, but not being used right now.
// the general idea is that we are going to replace optical flow by simple image difference.
// representing motion as a single unsigned integer.
unsigned int MoFREAKUtilities::extractMotionByImageDifference(cv::Mat &frame, cv::Mat &prev_frame, float x, float y)
{
	int value_at_current_frame = frame.at<unsigned char>(x, y);
	int value_at_prev_frame = prev_frame.at<unsigned char>(x, y);

	return abs(value_at_current_frame - value_at_prev_frame);
}

vector<unsigned int> MoFREAKUtilities::extractFREAKFeature(cv::Mat &frame, float x, float y, float scale)
{
	const float SCALE_MULTIPLIER = 6;
	cv::Mat descriptor;
	vector<cv::KeyPoint> ftrs;

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
		for (unsigned i = 0; i < descriptor.cols; ++i)
		{
			ret_val.push_back(descriptor.at<unsigned char>(0, i));
		}
	}

	return ret_val;
}

void MoFREAKUtilities::buildMoFREAKFeaturesFromMoSIFT(QString mosift_file, string video_path)
{
	// remove old mofreak points.
	features.clear();

	// gather mosift features.
	MoSIFTUtilities mosift;
	mosift.readMoSIFTFeatures(mosift_file);
	vector<MoSIFTFeature> mosift_features = mosift.getMoSIFTFeatures();
	cv::VideoCapture capture;

	capture.open(video_path);

	for (auto it = mosift_features.begin(); it != mosift_features.end(); ++it)
	{
		// navigate into the corresponding video sequence and extract the FREAK feature.
		capture.set(CV_CAP_PROP_POS_FRAMES, it->frame_number);
		cv::Mat frame;
		capture >> frame;
		vector<unsigned int> freak_ftr = extractFREAKFeature(frame, it->x, it->y, it->scale);

		// compute motion as image difference.
		unsigned int image_difference = 0;
		if (it->frame_number != 1)
		{
			// set the capture to the previous frame.
			capture.set(CV_CAP_PROP_POS_FRAMES, it->frame_number - 1);
			cv::Mat prev_frame;
			capture >> prev_frame;

			image_difference = extractMotionByImageDifference(frame, prev_frame, it->x, it->y);
		}

		if (freak_ftr.size() > 0)
		{
			// build MoFREAK feature and add it to our feature list.
			MoFREAKFeature ftr;
			ftr.x = it->x;
			ftr.y = it->y;
			ftr.scale = it->scale;
			ftr.motion_x = it->motion_x;
			ftr.motion_y = it->motion_y;
			ftr.frame_number = it->frame_number;

			for (unsigned i = 0; i < 128; ++i)
				ftr.motion[i] = it->motion[i];

			for (unsigned i = 0; i < 64; ++i)
				ftr.FREAK[i] = freak_ftr[i];

			features.push_back(ftr);
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
		for (int i = 0; i < 64; ++i)
		{
			int z = it->FREAK[i];
			f << it->FREAK[i] << " ";
			//f << toBinaryString(z) << " ";
		}

		// motion
		for (int i = 0; i < 128; ++i)
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

void MoFREAKUtilities::readMetadata(QString filename, int &action, int &video_number, int &person)
{
	//int action, person, video_number;
		// get the action.
		if (filename.contains("boxing"))
		{
			action = BOXING;
		}
		else if (filename.contains("walking"))
		{
			action = WALKING;
		}
		else if (filename.contains("jogging"))
		{
			action = JOGGING;
		}
		else if (filename.contains("running"))
		{
			action = RUNNING;
		}
		else if (filename.contains("handclapping"))
		{
			action = HANDCLAPPING;
		}
		else if (filename.contains("handwaving"))
		{
			action = HANDWAVING;
		}
		else
		{
			action = HANDWAVING; // hopefully we never miss this?  Just giving a default value. 
		}

		// get the person.
		int first_underscore = filename.indexOf("_");
		QString person_string = filename.mid(first_underscore - 2, 2);
		person = person_string.toInt();

		// get the video number.
		int last_underscore = filename.lastIndexOf("_");
		QString video_string = filename.mid(last_underscore - 1, 1);
		video_number = video_string.toInt();
}

void MoFREAKUtilities::readMoFREAKFeatures(QString filename)
{
	int action, video_number, person;
	readMetadata(filename, action, video_number, person);

	ifstream stream;
	stream.open(filename.toStdString());
	
	while (!stream.eof())
	{
		// single feature
		MoFREAKFeature ftr;
		stream >> ftr.x >> ftr.y >> ftr.frame_number >> ftr.scale >> ftr.motion_x >> ftr.motion_y;
	
		// FREAK
		for (unsigned i = 0; i < 64; ++i)
		{
			unsigned a;
			stream >> a;
			ftr.FREAK[i] = a;
		}

		// motion
		for (unsigned i = 0; i < 128; ++i)
		{
			unsigned a;
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