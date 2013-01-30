#include "MoFREAKUtilities.h"
#include <brisk\brisk.h>


MoFREAKUtilities::MoFREAKUtilities(int dset)
{
	dataset = dset;
}

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

unsigned int MoFREAKUtilities::countOnes(unsigned int byte)
{
	unsigned int num_ones = 0;

	for (unsigned int i = 0; i < 8; ++i)
	{
		if ((byte & (1 << i)) != 0)
		{
			++num_ones;
		}
	}
	return num_ones;
}

// Computes the motion interchange pattern between the current and previous frame.
// Assumes both matrices are 19 x 19, and we will check the 8 motion patch locations in the previous frame
// returns a binary descriptor representing the MIP responses for the patch at the SURF descriptor.
// x, y correspond to the location in the 19x19 roi that we are centering a patch around.
/*
unsigned int MoFREAKUtilities::motionInterchangePattern(cv::Mat &current_frame, cv::Mat &prev_frame, int x, int y)
{
	const int THETA = 10368;//5184;//2592;//1296; // recommended by MIP paper
	// extract patch on current frame.
	cv::Rect roi(x - 1, y - 1, 3, 3);
	cv::Mat patch_t(current_frame, roi);

	// extract patches from previous frame.
	vector<cv::Mat> previous_patches;
	// (-4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x - 4) - 1, y - 1, 3, 3)));
	// (-3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y + 3) - 1, 3, 3)));
	// (0, 4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y + 4) - 1, 3, 3)));
	// (3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y + 3) - 1, 3, 3)));
	// (4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x + 4) - 1, y - 1, 3, 3)));
	// (3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y - 3) - 1, 3, 3)));
	// (0, -4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y - 4) - 1, 3, 3)));
	// (-3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y - 3) - 1, 3, 3)));

	// now do SSD between current patch and all of those patches.
	// opencv might have an optimized ssd, i didn't find it though.
	unsigned int bit = 1;
	unsigned int descriptor = 0;
	for (auto it = previous_patches.begin(); it != previous_patches.end(); ++it)
	{
		int ssd = 0;
		for (int row = 0; row < 3; ++row)
		{
			uchar *p = patch_t.data;
			uchar *p2 = it->data;
			for (int col = 0; col < 3; ++col)
			{
				ssd += (int)pow((float)((*p) - (*p2)), 2);
				p++;
				p2++;
			}
		}

		if (ssd > THETA) // try switching to <...
		{
			descriptor |= bit;
		}
		bit <<= 1;
	}

	return descriptor;
}
*/
unsigned int MoFREAKUtilities::motionInterchangePattern(cv::Mat &current_frame, cv::Mat &prev_frame, int x, int y)
{
	//return 0;
	bool GAUSSIAN_CHECK = true;
	const int THETA = 64;//10368;//5184;//2592;//1296; // recommended by MIP paper
	unsigned int bit = 1;
	unsigned int descriptor = 0;
	cv::Mat *blurred_current, *blurred_prev;

	if (GAUSSIAN_CHECK)
	{
		blurred_current = new cv::Mat();
		blurred_prev = new cv::Mat();

		// blur with a 5x5 Gaussian
		//std::cout << "Size: " << current_frame.rows << ", " << current_frame.cols << endl;//current_frame.size() << std::endl;
		cv::GaussianBlur(current_frame, *blurred_current, cv::Size(5, 5), 0);
		cv::GaussianBlur(prev_frame, *blurred_prev, cv::Size(5, 5), 0);

		
		// (-4, 0)
		try
		{
			//cout << abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x - 4, y)) << ", " << THETA << endl;
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x - 4, y)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "Out of bounds maybe? -4, 0" << endl;
		}
		bit <<= 1;
		// (-3, 3)
		try
		{
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x - 3, y + 3)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "Out of bounds? -3, 3" << endl;
		}
		bit <<= 1;
		
		// (0, 4)
		try
		{
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x, y + 4)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "Out of bounds? 0, 4" << endl;
		}
		bit <<= 1;
		
		// (3, 3)
		try
		{
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x + 3, y + 3)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "out of bounds? 3, 3" << endl;
		}
		bit <<= 1;
		
		// (4, 0)
		try
		{
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x + 4, y)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "out of bounds? 4, 0" << endl;
		}
		bit <<= 1;
		
		// (3, -3)
		try
		{
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x + 3, y - 3)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "out of bounds? 3, -3" << endl;
		}
		bit <<= 1;
		
		// (0, -4)
		try
		{
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x, y - 4)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "out of bounds? 0 -4" << endl;
		}
		bit <<= 1;
		
		// (-3, -3)
		try
		{
			if (abs(blurred_current->at<unsigned char>(x, y) - blurred_prev->at<unsigned char>(x - 3, y - 3)) > THETA)
			{
				descriptor |= bit;
			}
		}
		catch (...)
		{
			cout << "out of bounds? -3 -3" << endl;
		}
		bit <<= 1;

		delete blurred_current;
		delete blurred_prev;
		
		return descriptor;
	}
	// extract patch on current frame.
	cv::Rect roi(x - 1, y - 1, 3, 3);
	cv::Mat patch_t(current_frame, roi);

	// extract patches from previous frame.
	vector<cv::Mat> previous_patches;
	// (-4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x - 4) - 1, y - 1, 3, 3)));
	// (-3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y + 3) - 1, 3, 3)));
	// (0, 4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y + 4) - 1, 3, 3)));
	// (3, 3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y + 3) - 1, 3, 3)));
	// (4, 0)
	previous_patches.push_back(prev_frame(cv::Rect((x + 4) - 1, y - 1, 3, 3)));
	// (3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x + 3) - 1, (y - 3) - 1, 3, 3)));
	// (0, -4)
	previous_patches.push_back(prev_frame(cv::Rect(x - 1, (y - 4) - 1, 3, 3)));
	// (-3, -3)
	previous_patches.push_back(prev_frame(cv::Rect((x - 3) - 1, (y - 3) - 1, 3, 3)));

	// now do SSD between current patch and all of those patches.
	// opencv might have an optimized ssd, i didn't find it though.
	for (auto it = previous_patches.begin(); it != previous_patches.end(); ++it)
	{
		int ssd = 0;
		for (int row = 0; row < 3; ++row)
		{
			uchar *p = patch_t.data;
			uchar *p2 = it->data;
			for (int col = 0; col < 3; ++col)
			{
				ssd += (int)pow((float)((*p) - (*p2)), 2);
				p++;
				p2++;
			}
		}

		if (ssd > THETA) // try switching to <...
		{
			descriptor |= bit;
		}
		bit <<= 1;
	}

	return descriptor;
}

void MoFREAKUtilities::extractMotionByMotionInterchangePatterns(cv::Mat &current_frame, cv::Mat &prev_frame,
	vector<unsigned int> &motion_descriptor, 
	float scale, int x, int y)
{
	// get region of interest from frames at keypt + scale
	int tl_x = x - (int)scale/2;
	int tl_y = y - (int)scale/2;
	cv::Rect roi(tl_x, tl_y, ceil(scale), ceil(scale));
	cv::Mat current_roi = current_frame(roi);
	cv::Mat prev_roi = prev_frame(roi);

	// resize to 19x19
	cv::Mat frame_t(19, 19, CV_32F);
	cv::Mat frame_t_minus_1(19, 19, CV_32F);

	cv::resize(current_roi, frame_t, frame_t.size());
	cv::resize(prev_roi, frame_t_minus_1, frame_t_minus_1.size());

	// we will compute the descriptor around these 9 points...
	vector<cv::Point2d> patch_centers;
	patch_centers.push_back(cv::Point2d(5, 5));
	patch_centers.push_back(cv::Point2d(5, 9));
	patch_centers.push_back(cv::Point2d(5, 13));
	patch_centers.push_back(cv::Point2d(9, 5));
	//patch_centers.push_back(cv::Point2d(9, 9));
	patch_centers.push_back(cv::Point2d(9, 13));
	patch_centers.push_back(cv::Point2d(13, 5));
	patch_centers.push_back(cv::Point2d(13, 9));
	patch_centers.push_back(cv::Point2d(13, 13));

	// over each of these patch centers, we compute a 1-byte motion descriptor.
	for (auto it = patch_centers.begin(); it != patch_centers.end(); ++it)
	{
		//cout << "patch centered at " << it->x << ", " << it->y << endl;
		unsigned int descriptor = motionInterchangePattern(frame_t, frame_t_minus_1, it->x, it->y);
		motion_descriptor.push_back(descriptor);
	}
}

// to decide if there is sufficient motion, we compute the MIP on the center location (9, 9)
// If there are a sufficient number of ones, then we have sufficient motion.
bool MoFREAKUtilities::sufficientMotion(cv::Mat &current_frame, cv::Mat prev_frame, float x, float y, float scale)
{
	// get region of interest from frames at keypt + scale
	int tl_x = x - (int)scale/2;
	int tl_y = y - (int)scale/2;
	cv::Rect roi(tl_x, tl_y, ceil(scale), ceil(scale));
	cv::Mat current_roi = current_frame(roi);
	cv::Mat prev_roi = prev_frame(roi);

	// resize to 19x19
	cv::Mat frame_t(19, 19, CV_32F);
	cv::Mat frame_t_minus_1(19, 19, CV_32F);

	cv::resize(current_roi, frame_t, frame_t.size());
	cv::resize(prev_roi, frame_t_minus_1, frame_t_minus_1.size());

	unsigned int descriptor = motionInterchangePattern(frame_t, frame_t_minus_1, 9, 9);

	unsigned int num_ones = countOnes(descriptor);
	
	//return (num_ones > 0);//3);
	return true;
}

bool MoFREAKUtilities::sufficientMotion(cv::Mat &diff_integral_img, float &x, float &y, float &scale, int &motion)
{
	// compute the sum of the values within this patch in the difference image.  It's that simple.
	int radius = ceil((scale));///2);
	const int MOTION_THRESHOLD = 4 * radius * 5;

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

void MoFREAKUtilities::computeMoFREAKFromFile(std::string video_filename, std::string mofreak_filename, bool clear_features_after_computation)
{
	std::string debug_filename = video_filename;
	// ignore the first frames because we can't compute the frame difference with them.
	const int GAP_FOR_FRAME_DIFFERENCE = 5;

	cv::VideoCapture capture;
	capture.open(video_filename);

	if (!capture.isOpened())
	{
		cout << "Could not open file: " << video_filename << endl;
	}

	cv::Mat current_frame;
	cv::Mat prev_frame;

	std::queue<cv::Mat> frame_queue;
	for (unsigned int i = 0; i < GAP_FOR_FRAME_DIFFERENCE; ++i)
	{
		capture >> prev_frame; // ignore first 'GAP_FOR_FRAME_DIFFERENCE' frames.  Read them in and carry on.
		cv::cvtColor(prev_frame, prev_frame, CV_BGR2GRAY);
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
		cv::cvtColor(current_frame ,current_frame, CV_BGR2GRAY);

		// compute the difference image for use in later computations.
		cv::Mat diff_img(current_frame.rows, current_frame.cols, CV_8U);
		cv::absdiff(current_frame, prev_frame, diff_img);

		vector<cv::KeyPoint> keypoints, diff_keypoints;
		cv::Mat descriptors;
		
		// detect all keypoints.
		cv::BriskFeatureDetector *detector = new cv::BriskFeatureDetector(30);
		cv::BriskFeatureDetector *diff_detector = new cv::BriskFeatureDetector(30);
		//detector->detect(current_frame, keypoints);
		diff_detector->detect(diff_img, keypoints);

		// extract the FREAK descriptors efficiently over the whole frame
		// For now, we are just computing the motion FREAK!  It seems to be giving better results.
		cv::FREAK extractor;
		extractor.compute(diff_img, keypoints, descriptors);
		//cout << "--------------------------------" << keypoints.size() << " detected features" << endl;
		

		// for each detected keypoint
		vector<cv::KeyPoint> current_frame_keypts;
		unsigned char *pointer_to_descriptor_row = 0;
		unsigned int keypoint_row = 0;
		for (auto keypt = keypoints.begin(); keypt != keypoints.end(); ++keypt)
		{
			pointer_to_descriptor_row = descriptors.ptr<unsigned char>(keypoint_row);

			// only take points with sufficient motion.
			int motion = 0;
			
			if (sufficientMotion(current_frame, prev_frame, keypt->pt.x, keypt->pt.y, keypt->size))
			{
				//cout << "feature: motion bytes: " << NUMBER_OF_BYTES_FOR_MOTION << endl;
				//cout << "feature: app bytes: " << NUMBER_OF_BYTES_FOR_APPEARANCE << endl;
				MoFREAKFeature ftr(NUMBER_OF_BYTES_FOR_MOTION, NUMBER_OF_BYTES_FOR_APPEARANCE);
				ftr.frame_number = frame_num;
				ftr.scale = keypt->size;
				ftr.x = keypt->pt.x;
				ftr.y = keypt->pt.y;

				for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)
				{
					ftr.motion[i] = pointer_to_descriptor_row[i];
				}

				// MIP
				vector<unsigned int> motion_desc;
				extractMotionByMotionInterchangePatterns(current_frame, prev_frame, motion_desc, keypt->size, keypt->pt.x, keypt->pt.y);

				for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)
				{
					ftr.appearance[i] = motion_desc[i];
				}

				// gather metadata.
				int action, person, video_number;
				readMetadata(video_filename, action, video_number, person);

				ftr.action = action;
				ftr.video_number = video_number;
				ftr.person = person;

				// these parameters aren't useful right now.
				ftr.motion_x = 0;
				ftr.motion_y = 0;

				features.push_back(ftr);
				current_frame_keypts.push_back(*keypt);
			}
			keypoint_row++;
		} // at this point, gathered all the mofreak pts from the frame.

		frame_queue.push(current_frame.clone());
		prev_frame = frame_queue.front();
		frame_queue.pop();
		frame_num++;
	}

	// in the end, print the mofreak file and reset the features for a new file.
	cout << "Writing this mofreak file: " << mofreak_filename << endl;
	writeMoFREAKFeaturesToFile(mofreak_filename);

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
		MoFREAKFeature ftr(NUMBER_OF_BYTES_FOR_MOTION, NUMBER_OF_BYTES_FOR_APPEARANCE);
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
	mosift.openMoSIFTStream(mosift_file);
	cv::VideoCapture capture;

	cout << "Opening video path... " << endl;
	capture.open(video_path);

	if (!capture.isOpened())
	{
		cout << "ERROR: problem reading video file. " << endl;
	}

	MoSIFTFeature *feature = new MoSIFTFeature();
	mosift.readNextMoSIFTFeatures(feature);
	vector<cv::KeyPoint> features_per_frame;
	int current_frame = feature->frame_number;

	while (true)
	{
		if (feature->frame_number == current_frame)
		{
			cv::KeyPoint keypt;
			keypt.pt = cv::Point2f(feature->x, feature->y);
			keypt.size = feature->scale;
			features_per_frame.push_back(keypt);

			// move on to next mosift pt.
			if (!mosift.readNextMoSIFTFeatures(feature)){
				break;
			}
		}
		else
		{
			// that's all the sift points for this frame.  Do the computation.
			addMoSIFTFeatures(current_frame, features_per_frame, capture);
			current_frame = feature->frame_number;
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
				cout << "Over 5 million features, writing to secondary file. " << endl;
				writeMoFREAKFeaturesToFile(mofreak_out_file);

				features.clear();
				mofreak_file_id++;
			}
		}
	}
	delete feature;
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
	ret_vec.clear();

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
		for (int i = 0; i < NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)
		{
			//int z = it->appearance[i];
			f << it->appearance[i] << " ";
			//f << toBinaryString(z) << " ";
		}

		// motion
		for (int i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)
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

void MoFREAKUtilities::setCurrentAction(string folder_name)
{
	if (dataset == HMDB51)
	{
		if (folder_name == "brush_hair")
		{
			current_action = BRUSH_HAIR;
		}

		else if (folder_name == "cartwheel")
		{
			current_action = CARTWHEEL;	
		}

		else if (folder_name == "catch")
		{
			current_action = CATCH;
		}

		else if (folder_name == "chew")
		{
			current_action = CHEW;
		}

		else if (folder_name == "clap")
		{
			current_action = CLAP;
		}

		else if (folder_name == "climb")
		{
			current_action = CLIMB;
		}

		else if (folder_name == "climb_stairs")
		{
			current_action = CLIMB_STAIRS;
		}

		else if (folder_name == "draw_sword")
		{
			current_action = DRAW_SWORD;
		}

		else if (folder_name == "dribble")
		{
			current_action = DRIBBLE;
		}

		else if (folder_name == "drink")
		{
			current_action = DRINK;
		}

		else if (folder_name == "dive")
		{
			current_action = DIVE;
		}

		else if (folder_name == "eat")
		{
			current_action = EAT;
		}

		else if (folder_name == "fall_floor")
		{
			current_action = FALL_FLOOR;
		}

		else if (folder_name == "fencing")
		{
			current_action = FENCING;
		}

		else if (folder_name == "flic_flac")
		{
			current_action = FLIC_FLAC;
		}

		else if (folder_name == "golf")
		{
			current_action = GOLF;
		}

		else if (folder_name == "handstand")
		{
			current_action = HANDSTAND;
		}

		else if (folder_name == "hit")
		{
			current_action = HIT;
		}

		else if (folder_name == "hug")
		{
			current_action = HUG;
		}

		else if (folder_name == "jump")
		{
			current_action = JUMP;
		}

		else if (folder_name == "kick")
		{
			current_action = KICK;
		}

		else if (folder_name == "kick_ball")
		{
			current_action = KICK_BALL;
		}

		else if (folder_name == "kiss")
		{
			current_action = KISS;
		}

		else if (folder_name == "laugh")
		{
			current_action = LAUGH;
		}

		else if (folder_name == "pick")
		{
			current_action = PICK;
		}

		else if (folder_name == "pour")
		{
			current_action = POUR;
		}

		else if (folder_name == "pullup")
		{
			current_action = PULLUP;
		}

		else if (folder_name == "punch")
		{
			current_action = PUNCH;
		}

		else if (folder_name == "push")
		{
			current_action = PUSH;
		}

		else if (folder_name == "pushup")
		{
			current_action = PUSHUP;
		}

		else if (folder_name == "ride_bike")
		{
			current_action = RIDE_BIKE;
		}

		else if (folder_name == "ride_horse")
		{
			current_action = RIDE_HORSE;
		}

		else if (folder_name == "run")
		{
			current_action = RUN;
		}

		else if (folder_name == "shake_hands")
		{
			current_action = SHAKE_HANDS;
		}

		else if (folder_name == "shoot_ball")
		{
			current_action = SHOOT_BALL;
		}

		else if (folder_name == "shoot_bow")
		{
			current_action = SHOOT_BOW;
		}

		else if (folder_name == "shoot_gun")
		{
			current_action = SHOOT_GUN;
		}

		else if (folder_name == "sit")
		{
			current_action = SIT;
		}

		else if (folder_name == "situp")
		{
			current_action = SITUP;
		}

		else if (folder_name == "smile")
		{
			current_action = SMILE;
		}

		else if (folder_name == "smoke")
		{
			current_action = SMOKE;
		}

		else if (folder_name == "somersault")
		{
			current_action = SOMERSAULT;
		}

		else if (folder_name == "stand")
		{
			current_action = STAND;
		}

		else if (folder_name == "swing_baseball")
		{
			current_action = SWING_BASEBALL;
		}

		else if (folder_name == "sword")
		{
			current_action = SWORD;
		}

		else if (folder_name == "sword_exercise")
		{
			current_action = SWORD_EXERCISE;
		}

		else if (folder_name == "talk")
		{
			current_action = TALK;
		}

		else if (folder_name == "throw")
		{
			current_action = THROW;
		}

		else if (folder_name == "turn")
		{
			current_action = TURN;
		}

		else if (folder_name == "walk")
		{
			current_action = WALK;
		}

		else if (folder_name == "wave")
		{
			current_action = WAVE;
		}

		else
		{
			current_action = BRUSH_HAIR;
			cout << "****Didn't find action" << endl;
			system("PAUSE");
			exit(1);
		}
	}
	else if (dataset == UCF101)
	{
		// This is the right way to do this.  Verify that it works,
		// Then replace the other examples.
		if (actions.find(folder_name) == actions.end())
		{
			actions[folder_name] = actions.size();
		}

		current_action = actions[folder_name];
	}
}

void MoFREAKUtilities::readMetadata(std::string filename, int &action, int &video_number, int &person)
{
	boost::filesystem::path file_path(filename);
	boost::filesystem::path file_name = file_path.filename();
	std::string file_name_str = file_name.generic_string();

	if (false)//dataset == KTH)
	{
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
	}

	else if (dataset == HMDB51)
	{
		video_number = 0;
		person = 0;
		action = current_action;
	}

	else if (dataset == UTI2)
	{
		// parse the filename.
		std::vector<std::string> filename_parts = split(file_name_str, '_');

		// the "person" (video) is the number after the first underscore.
		std::string person_str = filename_parts[1];
		std::stringstream(filename_parts[1]) >> person;

		// the action is the number after the second underscore, before .avi.
		std::string vid_str = filename_parts[2];
		std::stringstream(filename_parts[2].substr(0, 1)) >> action;

		// video number.. not sure if useful for this dataset.
		std::stringstream(filename_parts[0]) >> video_number;
	}
}

void MoFREAKUtilities::readMoFREAKFeatures(std::string filename, int num_to_sample)
{
	std::deque<MoFREAKFeature> new_features;

	int action, video_number, person;
	readMetadata(filename, action, video_number, person);
	std::cout << "read metadata." << std::endl;

	ifstream stream;
	stream.open(filename);
	int iter = 0;
	while (!stream.eof())
	{
		// single feature
		MoFREAKFeature ftr(NUMBER_OF_BYTES_FOR_MOTION, NUMBER_OF_BYTES_FOR_APPEARANCE);
		stream >> ftr.x >> ftr.y >> ftr.frame_number >> ftr.scale >> ftr.motion_x >> ftr.motion_y;
		//std::cout << ftr.x << ftr.y << ftr.frame_number << std::endl;
	
		// if there was some extra data and it's not an actual feature point, leave.
		if (stream.eof())
		{
			break;
		}
		// otherwise, we read appearanace data
		for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_APPEARANCE; ++i)
		{
			unsigned int a;
			stream >> a;
			ftr.appearance[i] = a;
		}
		
		// motion data
		for (unsigned i = 0; i < NUMBER_OF_BYTES_FOR_MOTION; ++i)
		{
			unsigned int a;
			stream >> a;
			ftr.motion[i] = a;
		}

		// TEMP.  THIS IS FOR AN EXPERIMENT.  DELETE LATER. [TODO]
		/*for (unsigned i = 0; i < 64 - NUMBER_OF_BYTES_FOR_MOTION; ++i)
		{
			unsigned int a;
			stream >> a;
		}*/


		// metadata
		ftr.action = action;
		ftr.video_number = video_number;
		ftr.person = person;

		// add new feature to collection.
		new_features.push_back(ftr);
		++iter;
	}
	stream.close();

	// if we are randomly sampling the new features, do that sampling process now.
	if (num_to_sample && (new_features.size() > num_to_sample))
	{
		std::random_shuffle(new_features.begin(), new_features.end());

		for (int i = 0; i < num_to_sample; ++i)
		{
			features.push_back(new_features.back());
			new_features.pop_back();
		}
	}
	else
	{
		while (!new_features.empty())
		{
			features.push_back(new_features.back());
			new_features.pop_back();
		}
	}
}

std::deque<MoFREAKFeature> MoFREAKUtilities::getMoFREAKFeatures()
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

void MoFREAKUtilities::clearFeatures()
{
	features.clear();
}

/*
MoFREAKUtilities::MoFREAKUtilities(int motion_bytes)
{
	NUMBER_OF_BYTES_FOR_MOTION = motion_bytes;
	NUMBER_OF_BYTES_FOR_APPEARANCE = appearance_bytes;
}
*/

/*
MoFREAKUtilities::~MoFREAKUtilities()
{
	recent_frame.release();
	features.clear();
}
*/