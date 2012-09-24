#include <iostream>
#include <time.h>
#include <opencv2/nonfree/nonfree.hpp>
#include <vector>
#include <string>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;
using namespace boost::filesystem;

bool  sufficientMotion(cv::Mat &diff_integral_img, float &x, float &y, float scale, int &motion)
{
	const int MOTION_THRESHOLD = 1028;
	// compute the sum of the values within this patch in the difference image.  It's that simple.
	int radius = ceil((scale)/2);

	// always + 1, since the integral image adds a row and col of 0s to the top-left.
	int tl_x = MAX(0, x - radius + 1);
	int tl_y = MAX(0, y - radius + 1);
	int br_x = MIN(diff_integral_img.cols, x + radius + 1);
	int br_y = MIN(diff_integral_img.rows, y + radius + 1);

	int br = diff_integral_img.at<int>(br_y, br_x);
	int tl = diff_integral_img.at<int>(tl_y, tl_x);
	int tr = diff_integral_img.at<int>(tl_y, br_x);
	int bl = diff_integral_img.at<int>(br_y, tl_x);
	motion = br + tl - tr - bl;

	return  (motion > MOTION_THRESHOLD);
}

void computeMoFREAK(string video_file)
{
	clock_t start, finish;
	cv::VideoCapture capture;
	capture.open(video_file);

	video_file.append(".mofreak");
	ofstream out(video_file);

	cv::Mat frame, frame2;

	start = clock();
	int frame_num = 5;
	while (true)
	{
		capture.set(CV_CAP_PROP_POS_FRAMES, frame_num);
		capture >> frame;
		frame = frame.clone();

		capture.set(CV_CAP_PROP_POS_FRAMES, frame_num - 5);
		capture >> frame2;

		if (frame.data == NULL)
		{
			break;
		}

		cv::Mat diff_img(frame.rows, frame.cols, CV_8U);
		cv::absdiff(frame, frame2, diff_img);

		cv::Mat diff_int_img(diff_img.rows + 1, diff_img.cols + 1, CV_32S);
		cv::integral(diff_img, diff_int_img);

		//cv::StarFeatureDetector *detector_ = new cv::StarFeatureDetector(15, 45, 50, 40);
		cv::SiftFeatureDetector *detector_ = new cv::SiftFeatureDetector();
		
		vector<cv::KeyPoint> keypoints;
		vector<cv::KeyPoint> ftrs;
		cv::Mat motion_descriptors, appearance_descriptors;

		detector_->detect(frame, keypoints);


		// extract FREAK descriptor.
		cv::FREAK extractor;
		extractor.compute(diff_img, keypoints, motion_descriptors);
		extractor.compute(frame, keypoints, appearance_descriptors);

		unsigned keypoint_row = 0;
		unsigned char *pointer_to_motion_descriptor_row, *pointer_to_appearance_descriptor_row = 0;
		for (auto keypt = keypoints.begin(); keypt != keypoints.end(); ++keypt)
		{
			pointer_to_motion_descriptor_row = motion_descriptors.ptr<unsigned char>(keypoint_row);
			pointer_to_appearance_descriptor_row = appearance_descriptors.ptr<unsigned char>(keypoint_row);

			int motion = 0;
			if (sufficientMotion(diff_int_img, keypt->pt.x, keypt->pt.y, keypt->size * 6, motion))
			{
				int i = 0;
				out << keypt->pt.x << " " << keypt->pt.y << " " << frame_num <<
					" " << keypt->size << " 0 0 ";

				// appearance
				for (unsigned i = 0; i < 16; ++i)
				{
					out << (int)pointer_to_appearance_descriptor_row[i] << " ";
				}

				// motion
				for (unsigned i = 0; i < 16; ++i)
				{
					out << (int)pointer_to_motion_descriptor_row[i] << " ";
				}
				out << "\n";
			}
			keypoint_row++;
		}


		frame_num++;
	}
	out.close();
	capture.release();
	finish = clock();
	cout << "looped in  " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;

}

void main()
{
	// find all files in this folder that end with .avi.
	string root_folder = "C:/data/kth/all_in_one/videos/";
	vector<string> files;

	path file_path(root_folder);
	directory_iterator end_iter;

	for (directory_iterator dir_iter(root_folder); dir_iter != end_iter; ++dir_iter)
	{
		if (is_regular_file(dir_iter->status()))
		{
			path current_file = dir_iter->path();
			string filename = current_file.filename().generic_string();
			if (filename.substr(filename.length() - 3, 3) == "avi")
			{
				files.push_back(current_file.string());
				//cout << filename << endl;
			}
		}
	}

	
	for (auto it = files.begin(); it != files.end(); ++it)
	{
		computeMoFREAK(*it);
	}
	
	
	/*string video_file = "C://data//kth//all_in_one//videos/person17_jogging_d1_uncomp.avi";
	computeMoFREAK(video_file);*/
	

	cout << "ALL DONE!" << endl;
	int x = 0;
	cin >> x;
}