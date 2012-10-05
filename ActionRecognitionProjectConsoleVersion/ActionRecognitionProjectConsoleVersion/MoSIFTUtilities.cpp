#include "MoSIFTUtilities.h"

#include <fstream>
#include <sstream>

// vanilla string split operation.  Direct copy-paste from stack overflow
// source: http://stackoverflow.com/questions/236129/splitting-a-string-in-c
std::vector<std::string> &MoSIFTUtilities::split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}


std::vector<std::string> MoSIFTUtilities::split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    return split(s, delim, elems);
}

void MoSIFTUtilities::readMetadata(std::string filename, int &action, int &video_number, int &person)
{
	// get the action.
	boost::filesystem::path file_path(filename);
	boost::filesystem::path file_name = file_path.filename();
	std::string file_name_str = file_name.generic_string();
	

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
	//person = atoi((filename_parts[0].substr(filename_parts[0].length() - 2, 2)).c_str());

	// the video number is the last character of the 3rd section of the filename.
	std::stringstream(filename_parts[2].substr(filename_parts[2].length() - 1, 1)) >> video_number;
}

void MoSIFTUtilities::openMoSIFTStream(std::string filename)
{
	//readMetadata(filename, action, video_number, person);

	moSIFTFeaturesStream.open(filename);

	if (!moSIFTFeaturesStream.is_open())
	{
		cout << "MoSIFT file didn't open: " << filename << endl;
	}
}

bool MoSIFTUtilities::readNextMoSIFTFeatures(MoSIFTFeature* ftr)
{
	int action, video_number, person;
	action = 1;
	video_number = 1;
	person = 1; /// irrelevant for trecvid.

	string line;
	getline(moSIFTFeaturesStream, line);
	istringstream iss(line);
	// single feature
	iss >> ftr->x >> ftr->y >> ftr->frame_number >> ftr->scale >> ftr->motion_x >> ftr->motion_y;
	
	// sift
	for (unsigned i = 0; i < 128; ++i)
	{
		unsigned a;
		iss >> a;
		ftr->SIFT[i] = a;
	}

	// motion
	for (unsigned i = 0; i < 128; ++i)
	{
		unsigned a;
		iss >> a;
		ftr->motion[i] = a;
	}

	// metadata
	ftr->action = action;
	ftr->video_number = video_number;
	ftr->person = person;

	if (moSIFTFeaturesStream.eof()){
		moSIFTFeaturesStream.close();
		return false;
	}
	return true;
}