#ifndef MOSIFTUTILITIES_H
#define MOSIFTUTILITIES_H

#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <qstring.h>

using namespace std;

enum action {BOXING, HANDCLAPPING, HANDWAVING, JOGGING, RUNNING, WALKING};
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

	int action;
	int video_number;
	int person;
};

class MoSIFTUtilities
{
public:
	void readMoSIFTFeatures(QString filename);
	vector<MoSIFTFeature> getMoSIFTFeatures();

private:
	void readMetadata(QString filename, int &action, int &video_number, int &person);

	vector<MoSIFTFeature> features;
};
#endif