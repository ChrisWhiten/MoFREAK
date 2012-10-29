#ifndef BAGOFWORDSREPRESENTATION_H
#define BAGOFWORDSREPRESENTATION_H

#include <opencv2\core\core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <fstream>
#include <list>

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::filesystem;
using namespace std;

class BagOfWordsRepresentation
{
public:
	BagOfWordsRepresentation(std::vector<std::string> &file_list, int num_clust, int ftr_dim, int num_people,
		bool appearance_is_bin, bool motion_is_bin, int dset);

	~BagOfWordsRepresentation();
	BagOfWordsRepresentation(int num_clust, int ftr_dim);
	void computeBagOfWords(string SVM_PATH, string MOFREAK_PATH, string METADATA_PATH);
	void setMotionDescriptor(unsigned int size, bool binary = false);
	void setAppearanceDescriptor(unsigned int size, bool binary = false);
	void computeSlidingBagOfWords(std::string &input_file, int alpha, int label, ofstream &out);

private:
	void computeHMDB51BagOfWords(string SVM_PATH, string MOFREAK_PATH, string METADATA_PATH);
	void loadClusters();
	cv::Mat buildHistogram(std::string &file, bool &success);
	float standardEuclideanDistance(cv::Mat &a, cv::Mat &b) const;
	void findBestMatch(cv::Mat &feature_vector, cv::Mat &clusters, int &best_cluster_index, float &best_cluster_score);
	unsigned int hammingDistance(cv::Mat &a, cv::Mat &b);
	int actionStringToActionInt(string act);

	void normalizeClusters();
	void normalizeMotionOfFeature(cv::Mat &ftr);
	void normalizeAppearanceOfFeature(cv::Mat &ftr);

	std::vector<std::string> split(const std::string &s, char delim);
	std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

	std::vector<std::string> files;
	const int NUMBER_OF_CLUSTERS;
	const int FEATURE_DIMENSIONALITY;
	const int NUMBER_OF_PEOPLE;
	cv::Mat *clusters;
	enum KTH_action {BOXING, HANDCLAPPING, HANDWAVING, JOGGING, RUNNING, WALKING};
	enum HMDB_action {BRUSH_HAIR, CARTWHEEL, CATCH, CHEW, CLAP, CLIMB, CLIMB_STAIRS,
		DIVE, DRAW_SWORD, DRIBBLE, DRINK, EAT, FALL_FLOOR, FENCING, FLIC_FLAC, GOLF,
		HANDSTAND, HIT, HUG, JUMP, KICK, KICK_BALL, KISS, LAUGH, PICK, POUR, PULLUP,
		PUNCH, PUSH, PUSHUP, RIDE_BIKE, RIDE_HORSE, RUN, SHAKE_HANDS, SHOOT_BALL,
		SHOOT_BOW, SHOOT_GUN, SIT, SITUP, SMILE, SMOKE, SOMERSAULT, STAND,
		SWING_BASEBALL, SWORD, SWORD_EXERCISE, TALK, THROW, TURN, WALK, WAVE};

	unsigned int motion_descriptor_size;
	unsigned int appearance_descriptor_size;
	bool motion_is_binary;
	bool appearance_is_binary;

	int dataset;
	enum datasets {KTH, TRECVID, HOLLYWOOD, UTI1, UTI2, HMDB51};
};
#endif