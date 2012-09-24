#include "MoSIFTUtilities.h"

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
	//int action, person, video_number;
		// get the action.
		boost::filesystem::path file_path(filename);
		boost::filesystem::path file_name = file_path.filename();
		std::string file_name_str = file_name.generic_string();
	
		//QStringList words = filename.split("\\");
		//QString file_name = words[words.length() - 1];

		// get the action.
		//if (file_name.contains("boxing"))
		if (boost::contains(file_name_str, "boxing"))
		{
			action = BOXING;
		}
		//else if (file_name.contains("walking"))
		else if (boost::contains(file_name_str, "walking"))
		{
			action = WALKING;
		}
		//else if (file_name.contains("jogging"))
		else if (boost::contains(file_name_str, "jogging"))
		{
			action = JOGGING;
		}
		//else if (file_name.contains("running"))
		else if (boost::contains(file_name_str, "running"))
		{
			action = RUNNING;
		}
		//else if (file_name.contains("handclapping"))
		else if (boost::contains(file_name_str, "handclapping"))
		{
			action = HANDCLAPPING;
		}
		//else if (file_name.contains("handwaving"))
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

void MoSIFTUtilities::readMoSIFTFeatures(std::string filename)
{
	int action, video_number, person;
	readMetadata(filename, action, video_number, person);

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

		// metadata
		ftr.action = action;
		ftr.video_number = video_number;
		ftr.person = person;

		// add new feature to collection.
		features.push_back(ftr);
	}
	stream.close();
}

vector<MoSIFTFeature> MoSIFTUtilities::getMoSIFTFeatures()
{
	return features;
}