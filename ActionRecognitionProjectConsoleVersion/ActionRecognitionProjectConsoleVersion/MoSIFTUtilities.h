#ifndef MOSIFTUTILITIES_H
#define MOSIFTUTILITIES_H

#include <stdio.h>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

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
	void readMoSIFTFeatures(std::string filename);
	vector<MoSIFTFeature> getMoSIFTFeatures();

private:
	void readMetadata(std::string filename, int &action, int &video_number, int &person);
	std::vector<std::string> split(const std::string &s, char delim);
	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

	vector<MoSIFTFeature> features;
};
#endif