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
#include <unordered_map>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include "MoSIFTUtilities.h"
using namespace std;

#define MOTION_BYTES 8
#define APPEARANCE_BYTES 8

struct MoFREAKFeature
{
	MoFREAKFeature(int motion_bytes, int appearance_bytes) 
	{ 
		for (int i = 0; i < motion_bytes; ++i)
		{
			motion.push_back(0);
		}

		for (int i = 0; i < appearance_bytes; ++i)
		{
			appearance.push_back(0);
		}
		using_image_difference = false; 
	}

	bool using_image_difference;
	float x;
	float y;
	float scale;
	float motion_x;
	float motion_y;

	int frame_number;
	std::vector<unsigned int> motion;
	std::vector<unsigned int> appearance;

	int action;
	int video_number;
	int person;
};

class MoFREAKUtilities
{
public:
	MoFREAKUtilities(int dset);
	void readMoFREAKFeatures(std::string filename, int num_to_sample = 0);
	std::deque<MoFREAKFeature> getMoFREAKFeatures();
	void clearFeatures();

	void buildMoFREAKFeaturesFromMoSIFT(std::string mosift_file, string video_path, string mofreak_path);
	void writeMoFREAKFeaturesToFile(string output_file);
	void computeMoFREAKFromFile(std::string video_filename, std::string mofreak_filename, bool clear_features_after_computation);
	void setAllFeaturesToLabel(int label);

	void setCurrentAction(string folder_name);
	int current_action; // for hmdb51..
	std::unordered_map<std::string, int> actions;

	static const int NUMBER_OF_BYTES_FOR_APPEARANCE = APPEARANCE_BYTES;
	static const int NUMBER_OF_BYTES_FOR_MOTION = MOTION_BYTES;

private:
	
	vector<unsigned int> extractFREAKFeature(cv::Mat &frame, float x, float y, float scale, bool extract_full_descriptor = false);
	vector<unsigned int> extractFREAK_ID(cv::Mat &frame, cv::Mat &prev_frame, float x, float y, float scale);
	unsigned int extractMotionByImageDifference(cv::Mat &frame, cv::Mat &prev_frame, float x, float y);
	void computeDifferenceImage(cv::Mat &current_frame, cv::Mat &prev_frame, cv::Mat &diff_img);
	bool sufficientMotion(cv::Mat &diff_img, float &x, float &y, float &scale, int &motion);
	bool sufficientMotion(cv::Mat &current_frame, cv::Mat prev_frame, float x, float y, float scale);
	void extractMotionByMotionInterchangePatterns(cv::Mat &current_frame, cv::Mat &prev_frame, 
		vector<unsigned int> &motion_descriptor, 
		float scale, int x, int y);

	string toBinaryString(unsigned int x);
	unsigned int hammingDistance(unsigned int a, unsigned int b);
	double motionNormalizedEuclideanDistance(vector<unsigned int> a, vector<unsigned int> b);
	unsigned int motionInterchangePattern(cv::Mat &current_frame, cv::Mat &prev_frame, int x, int y);
	unsigned int countOnes(unsigned int byte);

	void readMetadata(std::string filename, int &action, int &video_number, int &person);
	void addMoSIFTFeatures(int frame, vector<cv::KeyPoint> &pts, cv::VideoCapture &capture);

	std::vector<std::string> split(const std::string &s, char delim);
	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

	std::deque<MoFREAKFeature> features; // using deque because I'm running into memory issues..
	cv::Mat recent_frame;

	int dataset;
	enum datasets {KTH, TRECVID, HOLLYWOOD, UTI1, UTI2, HMDB51, UCF101};
};
#endif