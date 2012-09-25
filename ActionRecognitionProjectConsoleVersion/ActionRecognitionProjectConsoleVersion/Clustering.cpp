#include "Clustering.h"



// taken from feng's work.
 void Clustering::shuffleCVMat(cv::Mat &mx)
 {
	 srand( (unsigned int)time(NULL) );
	 int rowNo = mx.rows;
	 int row0 = rowNo - 1;

	 //suuffle the array with Fisher-Yates shuffling
	 while (row0 > 0)
	 {
		int row1 = rand() % row0;
		cv::Mat m1 = mx.row(row1);
		cv::Mat mt = m1.clone();
		mx.row(row0).copyTo(m1);
		mt.copyTo(mx.row(row0));
		row0--;
	 }
 }

 Clustering::Clustering(int dim, int num_clust, int num_pts, int num_classes) : DIMENSIONALITY(dim), NUMBER_OF_CLUSTERS(num_clust), 
	 NUMBER_OF_POINTS_TO_SAMPLE(num_pts), NUMBER_OF_CLASSES(num_classes)
 {
	 // default values. MoSIFT.
	motion_descriptor_size = 128; 
	appearance_descriptor_size = 128;
	motion_is_binary = false;
	appearance_is_binary = false;
 }

 void Clustering::buildDataFromMoFREAK(vector<MoFREAKFeature> &mofreak_ftrs, bool sample_pts, bool img_diff)
 {
	 // allocate data matrix.
	 // + 3 for metadata
	 int num_ftrs = mofreak_ftrs.size();
	 data_pts = new cv::Mat(num_ftrs, DIMENSIONALITY + 3, CV_32FC1);

	 // convert mofreak pts into rows in the matrix
	 for (unsigned int row = 0; row < num_ftrs; ++row)
	 {
		MoFREAKFeature ftr = mofreak_ftrs[row];

		// first, metadata.
		data_pts->at<float>(row, 0) = ftr.action;
		data_pts->at<float>(row, 1) = ftr.person;
		data_pts->at<float>(row, 2) = ftr.video_number;

		// appearance.
		for (unsigned col = 0; col < appearance_descriptor_size; ++col)//64; ++col)
		{
			data_pts->at<float>(row, col + 3) = (float)ftr.FREAK[col];
		}

		// motion.
		for (unsigned col = 0; col < motion_descriptor_size; ++col)
		{
			data_pts->at<float>(row, col + appearance_descriptor_size + 3) = (float)ftr.motion[col];//67) = (float)ftr.motion[col]; // 67 = 3 + 64.
		}
		
	 }

	// sample points for computational efficiency.
	if (sample_pts)
	{
		// shuffle 3 times.
		shuffleCVMat(*data_pts);
		shuffleCVMat(*data_pts);
		shuffleCVMat(*data_pts);

		// remove excessive points.
		if (data_pts->rows > NUMBER_OF_POINTS_TO_SAMPLE)
		{
			data_pts->pop_back(data_pts->rows - NUMBER_OF_POINTS_TO_SAMPLE);
		}
	}
 }

 void Clustering::buildDataFromMoSIFT(vector<MoSIFTFeature> &mosift_ftrs, bool sample_pts)
 {
	 // allocate data matrix.
	 // + 3 for metadata.
	 int num_ftrs = mosift_ftrs.size();
	 data_pts = new cv::Mat(num_ftrs, DIMENSIONALITY + 3, CV_32FC1);

	 // convert mosift pts into rows in the matrix.
	for (unsigned int row = 0; row < num_ftrs; ++row)
	{
		MoSIFTFeature ftr = mosift_ftrs[row];

		// first, metadata.
		data_pts->at<float>(row, 0) = ftr.action;
		data_pts->at<float>(row, 1) = ftr.person;
		data_pts->at<float>(row, 2) = ftr.video_number;

		for (unsigned col = 0; col < 128; ++col)
		{
			data_pts->at<float>(row, col + 3) = (float)ftr.SIFT[col];
			data_pts->at<float>(row, col + 131) = (float)ftr.motion[col]; // 131 = 128 + 3
		}
	}

	// sample points for computational efficiency.
	if (sample_pts)
	{
		// shuffle 3 times.
		shuffleCVMat(*data_pts);
		shuffleCVMat(*data_pts);
		shuffleCVMat(*data_pts);

		// remove excessive points.
		if (data_pts->rows > NUMBER_OF_POINTS_TO_SAMPLE)
		{
			data_pts->pop_back(data_pts->rows - NUMBER_OF_POINTS_TO_SAMPLE);
		}
	}
 }

 // these methods always assume we have more data points than clusters.
 // checks aren't hard to implement if that's not the case but i'm in a rush here!
 // this is how bad code gets written.
 void Clustering::randomClusters()
 {
	centers = new cv::Mat(NUMBER_OF_CLUSTERS, DIMENSIONALITY, data_pts->type());

	// shuffle and take the top NUMBER_OF_CLUSTERS points as the clusters.
	shuffleCVMat(*data_pts);
	shuffleCVMat(*data_pts);
	shuffleCVMat(*data_pts);

	// sample an even number of points from each class to keep the classes balanced.
	const int CLUSTERS_PER_CLASS = NUMBER_OF_CLUSTERS/NUMBER_OF_CLASSES;

	for (unsigned int c = 0; c < NUMBER_OF_CLASSES; ++c)
	{
		unsigned int sampled_from_this_class = 0;

		for (unsigned row = 0; row < data_pts->rows; ++row)
		{
			if ((unsigned int)data_pts->at<float>(row, 0) == c)
			{
				// sample this point.
				// 3 metadata parameters at the start...?
				for (unsigned col = 3; col < data_pts->cols; ++col)
				{
					centers->at<float>((c*CLUSTERS_PER_CLASS) + sampled_from_this_class, col - 3) = data_pts->at<float>(row, col);
				}

				sampled_from_this_class++;
				if (sampled_from_this_class == CLUSTERS_PER_CLASS)
					break;
			}
		}
	}

 }

 void Clustering::clusterWithKMeans()
 {
	 // remove meta-data from data points for clustering.
	 cv::Mat clusterable_data(data_pts->rows, data_pts->cols - 3, data_pts->type());

	 for (unsigned row = 0; row < data_pts->rows; ++row)
	 {
		 for (unsigned col = 0; col < data_pts->cols - 3; ++col)
		 {
			 clusterable_data.at<float>(row, col) = data_pts->at<float>(row, col + 3);
		 }
	 }

	 // now, cluster.
	 centers = new cv::Mat(NUMBER_OF_CLUSTERS, 1, clusterable_data.type());
	 kmeans(clusterable_data, NUMBER_OF_CLUSTERS, labels,  cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100000, 0.001),1,cv::KMEANS_PP_CENTERS, *centers);
 }

 void Clustering::writeClusters()
 {
	 ofstream output_file("clusters.txt");

	for (unsigned i = 0; i < NUMBER_OF_CLUSTERS; ++i)
	{
		for (unsigned j = 0; j < centers->cols; ++j)
		{
			output_file << centers->at<float>(i, j) << " ";
		}

		output_file << std::endl;
	}
	output_file.close();
 }

 void Clustering::setMotionDescriptor(unsigned int size, bool binary)
{
	motion_is_binary = binary;
	motion_descriptor_size = size;
}

void Clustering::setAppearanceDescriptor(unsigned int size, bool binary)
{
	appearance_is_binary = binary;
	appearance_descriptor_size = size;
}