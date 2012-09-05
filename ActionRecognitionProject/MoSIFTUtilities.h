#ifndef MOSIFTUTILITIES_H
#define MOSIFTUTILITIES_H

#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

using namespace std;

struct MoSIFTFeature
{
	float x;
	float y;
	float scale;
	float motion_x;
	float motion_y;

	int frame_number;
	unsigned char SIFT[128];
	unsigned char motion[128];
};

class MoSIFTUtilities
{
public:
	void readMoSIFTFeatures(string filename);
	vector<MoSIFTFeature> getMoSIFTFeatures();

private:
	vector<MoSIFTFeature> features;
};
#endif