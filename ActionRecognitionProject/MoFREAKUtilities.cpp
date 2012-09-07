#include "MoFREAKUtilities.h"

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

void MoFREAKUtilities::buildMoFREAKFeaturesFromMoSIFT(string mosift_file, string video_path)
{
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
			f << toBinaryString(z) << " ";
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