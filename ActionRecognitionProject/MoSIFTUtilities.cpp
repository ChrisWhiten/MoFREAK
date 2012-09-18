#include "MoSIFTUtilities.h"

void MoSIFTUtilities::readMetadata(QString filename, int &action, int &video_number, int &person)
{
	//int action, person, video_number;
		// get the action.
		if (filename.contains("boxing"))
		{
			action = BOXING;
		}
		else if (filename.contains("walking"))
		{
			action = WALKING;
		}
		else if (filename.contains("jogging"))
		{
			action = JOGGING;
		}
		else if (filename.contains("running"))
		{
			action = RUNNING;
		}
		else if (filename.contains("handclapping"))
		{
			action = HANDCLAPPING;
		}
		else if (filename.contains("handwaving"))
		{
			action = HANDWAVING;
		}
		else
		{
			action = HANDWAVING; // hopefully we never miss this?  Just giving a default value. 
		}

		// get the person.
		int first_underscore = filename.indexOf("_");
		QString person_string = filename.mid(first_underscore - 2, 2);
		person = person_string.toInt();

		// get the video number.
		int last_underscore = filename.lastIndexOf("_");
		QString video_string = filename.mid(last_underscore - 1, 1);
		video_number = video_string.toInt();
}

void MoSIFTUtilities::readMoSIFTFeatures(QString filename)
{
	int action, video_number, person;
	readMetadata(filename, action, video_number, person);

	ifstream stream;
	stream.open(filename.toStdString());
	
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