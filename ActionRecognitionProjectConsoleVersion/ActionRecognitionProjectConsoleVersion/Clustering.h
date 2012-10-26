#ifndef CLUSTERING_H
#define CLUSTERING_H

#include <opencv2/core/core.hpp>
#include <time.h>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "MoSIFTUtilities.h"
#include "MoFREAKUtilities.h"

using namespace std;

class Clustering
{
public:
	Clustering(int dim, int num_clust, int num_pts, int num_classes, vector<int> poss_classes);
	~Clustering();
	void buildDataFromMoSIFT(vector<MoSIFTFeature> &mosift_ftrs, bool sample_pts);
	void buildDataFromMoFREAK(vector<MoFREAKFeature> &mofreak_ftrs, bool sample_pts, bool img_diff = false, bool fix_class = false, int fixed_class = 1);
	void clusterWithKMeans();
	void randomClusters();
	void writeClusters();

	void setMotionDescriptor(unsigned int size, bool binary);
	void setAppearanceDescriptor(unsigned int size, bool binary);
private:
	void shuffleCVMat(cv::Mat &mx);

	const int DIMENSIONALITY;
	const int NUMBER_OF_CLUSTERS;
	const int NUMBER_OF_POINTS_TO_SAMPLE;
	const int NUMBER_OF_CLASSES;

	cv::Mat *centers;
	cv::Mat *data_pts;
	cv::Mat labels;
	vector<int> possible_classes;

	 // default values. MoSIFT.
	int motion_descriptor_size; 
	int appearance_descriptor_size;
	bool motion_is_binary;
	bool appearance_is_binary;
};

#endif