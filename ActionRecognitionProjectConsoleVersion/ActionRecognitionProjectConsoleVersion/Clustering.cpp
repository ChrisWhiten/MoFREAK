#include "Clustering.h"
#include <exception>


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

		m1.release();
		mt.release();
	 }
 }

 Clustering::~Clustering()
 {
	 data_pts->release();
	 delete data_pts;

	 centers->release();
	 delete centers;

	 labels.release();
 }

 Clustering::Clustering(int dim, int num_clust, int num_pts, int num_classes, vector<int> poss_classes, string svm_path) : 
	DIMENSIONALITY(dim), NUMBER_OF_CLUSTERS(num_clust), NUMBER_OF_POINTS_TO_SAMPLE(num_pts), NUMBER_OF_CLASSES(num_classes),
		SVM_PATH(svm_path)
 {
	possible_classes = poss_classes;
	center_row = 0;
	data_pts = new cv::Mat();
	centers = new cv::Mat(NUMBER_OF_CLUSTERS, DIMENSIONALITY, CV_8U);

	// some default values that will be overwritten..
	// If debugging program and both descriptors are size 0,
	// then we have not set the descriptor attributes with setMotionDescriptor and setAppearanceDescriptor.
	motion_descriptor_size = 0; 
	appearance_descriptor_size = 0;
	motion_is_binary = true;
	appearance_is_binary = true;
 }

 void Clustering::buildDataFromMoFREAK(std::deque<MoFREAKFeature> &mofreak_ftrs, bool sample_pts, bool img_diff, bool fix_class, int fixed_class)
 {
	 if (data_pts)
	 {
		 delete data_pts;
	 }

	 // allocate data matrix.
	 // + 3 for metadata
	 int num_ftrs = mofreak_ftrs.size();
	 cout << "allocating data_pts in builddatafrommofreaK: " << num_ftrs << endl;
	 data_pts = new cv::Mat(num_ftrs, DIMENSIONALITY + 3, CV_8U);
	 cout << "allocated" << endl;

	 // convert mofreak pts into rows in the matrix
	 for (unsigned int row = 0; row < num_ftrs; ++row)
	 {
		MoFREAKFeature ftr = mofreak_ftrs[row];

		// first, metadata.
		if (fix_class)
		{
			data_pts->at<unsigned char>(row, 0) = fixed_class;
		}
		else
		{
			data_pts->at<unsigned char>(row, 0) = (unsigned char)ftr.action;
		}
		data_pts->at<unsigned char>(row, 1) = (unsigned char)ftr.person;
		data_pts->at<unsigned char>(row, 2) = (unsigned char)ftr.video_number;

		// appearance.
		for (unsigned col = 0; col < appearance_descriptor_size; ++col)
		{
			data_pts->at<unsigned char>(row, col + 3) = (unsigned char)ftr.appearance[col];
		}

		// motion.
		for (unsigned col = 0; col < motion_descriptor_size; ++col)
		{
			data_pts->at<unsigned char>(row, col + appearance_descriptor_size + 3) = (unsigned char)ftr.motion[col];
		}
	 }

	 cout << "converted into matrix of pts" << endl;

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

	// we've stored the data in a clusterable format.  Now clear out the intermediate data.
	mofreak_ftrs.clear();
 }

 // [DEPRECATED] [TODO]
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
 void Clustering::randomClusters(bool only_one_class)
 {
	 std::cout << "Random clustering" << std::endl;

	// shuffle and take the top NUMBER_OF_CLUSTERS points as the clusters.
	shuffleCVMat(*data_pts);
	shuffleCVMat(*data_pts);
	shuffleCVMat(*data_pts);

	// sample an even number of points from each class to keep the classes balanced.
	const int CLUSTERS_PER_CLASS = NUMBER_OF_CLUSTERS/NUMBER_OF_CLASSES;

	cout << "Sampling " << CLUSTERS_PER_CLASS << endl;
	cout << "Current center row:" << center_row << endl;
	for (auto c = possible_classes.begin(); c != possible_classes.end(); ++c)
	{
		unsigned int sampled_from_this_class = 0;

		for (unsigned row = 0; row < data_pts->rows; ++row)
		{
			if (((unsigned int)data_pts->at<unsigned char>(row, 0) == *c) || only_one_class)
			{
				// sampling this point.
				for (unsigned col = 3; col < data_pts->cols; ++col)
				{
					try
					{
						centers->at<unsigned char>(center_row, col - 3) = data_pts->at<unsigned char>(row, col);
					}
					catch (exception &e)
					{
						cout << "Error: " << e.what() << endl;
						system("PAUSE");
						exit(1);
					}
				}

				center_row++;
				sampled_from_this_class++;
				if (sampled_from_this_class == CLUSTERS_PER_CLASS)
					break;
			}
		}

		if (only_one_class)
			break;
	}

 }

 void Clustering::clusterWithKMeans()
 {
	 // remove meta-data from data points for clustering.
	 cv::Mat clusterable_data(data_pts->rows, data_pts->cols - 3, data_pts->type());

	 for (int row = 0; row < data_pts->rows; ++row)
	 {
		 for (int col = 0; col < data_pts->cols - 3; ++col)
		 {
			 clusterable_data.at<float>(row, col) = data_pts->at<float>(row, col + 3);
		 }
	 }

	 // now, cluster.
	 centers = new cv::Mat(NUMBER_OF_CLUSTERS, 1, clusterable_data.type());
	 kmeans(clusterable_data, NUMBER_OF_CLUSTERS, labels,  cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100000, 0.001),1,cv::KMEANS_PP_CENTERS, *centers);
 }

 void Clustering::writeClusters(bool append)
 {
	 cout << "writing" << endl;
	 std::ofstream output_file;

	 if (append)
	 {
		 output_file.open(SVM_PATH + "/clusters.txt", std::ios_base::app);
	 }
	 else
	 {
		output_file.open(SVM_PATH + "/clusters.txt");
	 }

	for (int i = 0; i < NUMBER_OF_CLUSTERS; ++i)
	{
		for (unsigned j = 0; j < centers->cols; ++j)
		{
			output_file << (float)centers->at<unsigned char>(i, j) << " ";
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