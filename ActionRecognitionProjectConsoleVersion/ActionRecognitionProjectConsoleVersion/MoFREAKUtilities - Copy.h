#ifndef MOFREAKUTILITIES_H
#define MOFREAKUTILITIES_H

#include <stdio.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <opencv2\core\core.hpp>
#include <opencv2\video\video.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <queue>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "MoSIFTUtilities.h"
using namespace std;

#define MOTION_BYTES 8
#define APPEARANCE_BYTES 0

struct MoFREAKFeature
{
	MoFREAKFeature() { using_image_difference = false; } // just to set the default value, for compatibility.

	bool using_image_difference;
	float x;
	float y;
	float scale;
	float motion_x;
	float motion_y;

	int frame_number;
	unsigned int FREAK[1];
	//unsigned int FREAK[APPEARANCE_BYTES];
	unsigned int motion[MOTION_BYTES];

	int action;
	int video_number;
	int person;
};

class MoFREAKUtilities
{
public:
	void readMoFREAKFeatures(std::string filename);
	vector<MoFREAKFeature> getMoFREAKFeatures();
	void buildMoFREAKFeaturesFromMoSIFT(std::string mosift_file, string video_path, string mofreak_path);
	void writeMoFREAKFeaturesToFile(string output_file);
	void computeMoFREAKFromFile(std::string filename, bool clear_features_after_computation);
	void setAllFeaturesToLabel(int label);

private:
	string toBinaryString(unsigned int x);
	vector<unsigned int> extractFREAKFeature(cv::Mat &frame, float x, float y, float scale, bool extract_full_descriptor = false);
	vector<unsigned int> extractFREAK_ID(cv::Mat &frame, cv::Mat &prev_frame, float x, float y, float scale);
	unsigned int extractMotionByImageDifference(cv::Mat &frame, cv::Mat &prev_frame, float x, float y);
	void computeDifferenceImage(cv::Mat &current_frame, cv::Mat &prev_frame, cv::Mat &diff_img);
	bool sufficientMotion(cv::Mat &diff_img, float &x, float &y, float &scale, int &motion);
	unsigned int hammingDistance(unsigned int a, unsigned int b);
	double motionNormalizedEuclideanDistance(vector<unsigned int> a, vector<unsigned int> b);
	void readMetadata(std::string filename, int &action, int &video_number, int &person);
	void addMoSIFTFeatures(int frame, vector<cv::KeyPoint> &pts, cv::VideoCapture &capture);

	std::vector<std::string> split(const std::string &s, char delim);
	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

	vector<MoFREAKFeature> features;
	static const int NUMBER_OF_BYTES_FOR_MOTION = MOTION_BYTES;
	static const int NUMBER_OF_BYTES_FOR_APPEARANCE = APPEARANCE_BYTES;
	cv::Mat recent_frame;
};
#endif