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
#include <qstring.h>
#include <qstringlist.h>
#include <opencv2/nonfree/features2d.hpp>
#include <queue>

#include "MoSIFTUtilities.h"
using namespace std;

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
	unsigned int FREAK[64];
	unsigned char motion[128]; // is there a difference between unsigned int and unsigned char?  to investigate..
	unsigned int img_diff;

	int action;
	int video_number;
	int person;
};

class MoFREAKUtilities
{
public:
	void readMoFREAKFeatures(string filename);
	vector<MoFREAKFeature> getMoFREAKFeatures();
	void buildMoFREAKFeaturesFromMoSIFT(QString mosift_file, string video_path);
	void writeMoFREAKFeaturesToFile(string output_file,  bool img_diff = false);
	void readMoFREAKFeatures(QString filename);
	void computeMoFREAKFromFile(QString filename, bool clear_features_after_computation);

private:
	string toBinaryString(unsigned int x);
	vector<unsigned int> extractFREAKFeature(cv::Mat &frame, float x, float y, float scale);
	unsigned int extractMotionByImageDifference(cv::Mat &frame, cv::Mat &prev_frame, float x, float y); // I suppose we don't need scale for this.
	unsigned int hammingDistance(unsigned int a, unsigned int b);
	double motionNormalizedEuclideanDistance(vector<unsigned int> a, vector<unsigned int> b);
	void readMetadata(QString filename, int &action, int &video_number, int &person);

	vector<MoFREAKFeature> features;
};
#endif