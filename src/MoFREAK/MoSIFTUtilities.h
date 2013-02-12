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

enum KTH_action {BOXING, HANDCLAPPING, HANDWAVING, JOGGING, RUNNING, WALKING};
enum HMDB_action {BRUSH_HAIR, CARTWHEEL, CATCH, CHEW, CLAP, CLIMB, CLIMB_STAIRS,
		DIVE, DRAW_SWORD, DRIBBLE, DRINK, EAT, FALL_FLOOR, FENCING, FLIC_FLAC, GOLF,
		HANDSTAND, HIT, HUG, JUMP, KICK, KICK_BALL, KISS, LAUGH, PICK, POUR, PULLUP,
		PUNCH, PUSH, PUSHUP, RIDE_BIKE, RIDE_HORSE, RUN, SHAKE_HANDS, SHOOT_BALL,
		SHOOT_BOW, SHOOT_GUN, SIT, SITUP, SMILE, SMOKE, SOMERSAULT, STAND,
		SWING_BASEBALL, SWORD, SWORD_EXERCISE, TALK, THROW, TURN, WALK, WAVE};
enum WEIZMANN_action {BEND, JACK, JUMP_W, PJUMP, RUN_W, SIDE, SKIP, WALK_W, WAVE1, WAVE2};

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
	void openMoSIFTStream(std::string filename);
	bool readNextMoSIFTFeatures(MoSIFTFeature* ftr);

private:
	ifstream moSIFTFeaturesStream;

	void readMetadata(std::string filename, int &action, int &video_number, int &person);
	std::vector<std::string> split(const std::string &s, char delim);
	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

};
#endif