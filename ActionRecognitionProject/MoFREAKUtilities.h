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

#include "MoSIFTUtilities.h"
using namespace std;

struct MoFREAKFeature
{
	float x;
	float y;
	float scale;
	float motion_x;
	float motion_y;

	int frame_number;
	unsigned int FREAK[64]; // this is going to change to an array of bits.  temporary.
	unsigned char motion[128];
};

class MoFREAKUtilities
{
public:
	void readMoFREAKFeatures(string filename);
	vector<MoFREAKFeature> getMoFREAKFeatures();
	void buildMoFREAKFeaturesFromMoSIFT(string mosift_file, string video_path);
	void writeMoFREAKFeaturesToFile(string output_file);

private:
	string toBinaryString(unsigned int x);
	vector<unsigned int> extractFREAKFeature(cv::Mat &frame, float x, float y, float scale);
	unsigned int extractMotionByImageDifference(cv::Mat &frame, float x, float y); // I suppose we don't need scale for this.

	vector<MoFREAKFeature> features;
};
#endif