#include "MoSIFTUtilities.h"

void MoSIFTUtilities::readMoSIFTFeatures(string filename)
{
	ifstream stream;
	stream.open(filename);
	
	while (!stream.eof())
	{
		// single feature
		MoSIFTFeature ftr;
		stream >> ftr.x >> ftr.y >> ftr.frame_number >> ftr.scale >> ftr.motion_x >> ftr.motion_y;
	
		// sift
		for (unsigned i = 0; i < 128; ++i)
		{
			unsigned a;
			stream >> a;
			ftr.SIFT[i] = a;
		}

		// motion
		for (unsigned i = 0; i < 128; ++i)
		{
			unsigned a;
			stream >> a;
			ftr.motion[i] = a;
		}
		// add new feature to collection.
		features.push_back(ftr);
	}
	stream.close();
}

vector<MoSIFTFeature> MoSIFTUtilities::getMoSIFTFeatures()
{
	return features;
}