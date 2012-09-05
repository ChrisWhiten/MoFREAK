#ifndef MOFREAKUTILITIES_H
#define MOFREAKUTILITIES_H

#include <stdio.h>
#include <vector>

#include "MoSIFTUtilities.h"

struct MoFREAKFeature
{
	float x;
	float y;
	float scale;
	float motion_x;
	float motion_y;

	int frame_number;
	unsigned char FREAK[128]; // this is going to change to an array of bits.  temporary.
	unsigned char motion[128];
};

class MoFREAKUtilities
{
public:
	void readMoFREAKFeatures(string filename);
	vector<MoFREAKFeature> getMoFREAKFeatures();
	void buildMoFREAKFeaturesFromMoSIFT(vector<MoSIFTFeature> mosift_ftrs);

private:
	vector<MoFREAKFeature> features;
};
#endif