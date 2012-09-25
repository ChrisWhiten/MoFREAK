// This exists because Qt is making life miserable.
// Qt is slowing down this process to taking 30 hours for 3 hours of video.
// That's nowhere near what it should be, so I'm offseting it to a console program.
// Tada.



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
#include <limits>

using namespace std;
using namespace boost::filesystem;

bool CELL_TO_EAR = false;
bool EMBRACE = false;
bool OBJECT_PUT = false;
bool PEOPLE_MEET = false;
bool PEOPLE_SPLIT_UP = false;
bool PERSON_RUNS = false;
bool POINTING = false;
bool TAKE_PICTURE = false;
bool ELEVATOR_NO_ENTRY = false;

bool sufficientMotion(cv::Mat &diff_integral_img, float &x, float &y, float scale, int &motion)
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

void computeMoFREAK(cv::VideoCapture &capture, int start_frame, int end_frame, ofstream &out)
{
	clock_t start, finish;
	cv::Mat frame, frame2;

	start = clock();
	int frame_num = start_frame;
	while (true)
	{
		capture.set(CV_CAP_PROP_POS_FRAMES, frame_num);
		capture >> frame;
		frame = frame.clone();

		capture.set(CV_CAP_PROP_POS_FRAMES, frame_num - 5);
		capture >> frame2;

		if ((frame.data == NULL) || frame_num > end_frame)
		{
			break;
		}

		cv::Mat diff_img(frame.rows, frame.cols, CV_8U);
		cv::absdiff(frame, frame2, diff_img);

		cv::Mat diff_int_img(diff_img.rows + 1, diff_img.cols + 1, CV_32S);
		cv::integral(diff_img, diff_int_img);

		//cv::StarFeatureDetector *detector_ = new cv::StarFeatureDetector(15, 45, 50, 40);
		cv::SiftFeatureDetector *detector_ = new cv::SiftFeatureDetector(2000, 5);
		
		vector<cv::KeyPoint> keypoints;
		vector<cv::KeyPoint> ftrs;
		cv::Mat motion_descriptors, appearance_descriptors;

		detector_->detect(frame, keypoints);

		// temp. trying this out to see if there's any improvement....
		/*for (unsigned int i = 0; i < keypoints.size(); ++i)
		{
			keypoints[i].size = sqrt((keypoints[i].size * keypoints[i].size) - (0.5 * 0.5));
		}*/

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
	cout << "computed mofreak in  " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;

}


void extractFeaturesAtSpecificIntervals(cv::VideoCapture &capture, string &interval_file, ofstream &out)
{
	// each interval is read from one line of the file, in format: start_frame end_frame
	// just seek to those intervals and compute MoFREAK over that interval.
	// Write it to the out file
	ifstream in(interval_file);
	string dbg_filename = interval_file;
	dbg_filename.append(".dbg");
	ofstream dbg_file(dbg_filename);

	unsigned int current_line = 0;
	while (!in.eof())
	{
		int start_frame, end_frame;
		in >> start_frame >> end_frame;

		dbg_file << "Starting line " << current_line << endl;
		computeMoFREAK(capture, start_frame, end_frame, out);
		current_line++;
	}
	dbg_file.close();
}

// pass, as input to this function,
// a folder containing one video and 
// a txt file for each action.

// also (for simplicity's sake),
// just pass the path to the video file directly.
void processTRECVID(string root_folder, string video_file)
{
	clock_t start, finish, total_start, total_finish;
	total_start = clock();

	cv::VideoCapture capture;
	capture.open(video_file);

	string mofreak_file_name = video_file;
	mofreak_file_name.append(".mofreak");
	ofstream out(mofreak_file_name);

	// figure out which events to process...
	if (CELL_TO_EAR)
	{
		start = clock();
		string cell_to_ear_name = root_folder;
		cell_to_ear_name.append("/CellToEar.txt");
		extractFeaturesAtSpecificIntervals(capture, cell_to_ear_name, out);

		finish = clock();
		cout << "processed cell to ear in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (EMBRACE)
	{
		start = clock();
		string embrace_name = root_folder;
		embrace_name.append("/Embrace.txt");

		finish = clock();
		cout << "processed embrace in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (OBJECT_PUT)
	{
		start = clock();
		string put_name = root_folder;
		put_name.append("/ObjectPut.txt");

		finish = clock();
		cout << "processed object put in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (PEOPLE_MEET)
	{
		start = clock();
		string meet_name = root_folder;
		meet_name.append("/PeopleMeet.txt");

		finish = clock();
		cout << "processed people meet in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (PEOPLE_SPLIT_UP)
	{
		start = clock();
		string split_name = root_folder;
		split_name.append("/PeopleSplitUp.txt");

		finish = clock();
		cout << "processed people split up in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (PERSON_RUNS)
	{
		start = clock();
		string runs_name = root_folder;
		runs_name.append("/PersonRuns.txt");

		finish = clock();
		cout << "processed person runs in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (POINTING)
	{
		start = clock();
		string point_name = root_folder;
		point_name.append("/Pointing.txt");

		finish = clock();
		cout << "processed pointing in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (TAKE_PICTURE)
	{
		start = clock();
		string pic_name = root_folder;
		pic_name.append("/TakePicture.txt");

		finish = clock();
		cout << "processed take picture in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}
	if (ELEVATOR_NO_ENTRY)
	{
		start = clock();
		string elevator_name = root_folder;
		elevator_name.append("/ElevatorNoEntry.txt");

		finish = clock();
		std::cout << "processed elevator no entry in " << (finish - start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
	}


	total_finish = clock();
	std::cout << "processed video in " << (total_finish - total_start)/(double)CLOCKS_PER_SEC << " seconds." << endl;
}

// pass, as input to this function,
// the path to one folder containing all video files.
void processKTH(string root_folder)
{
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
		string mofreak_file = *it;
		mofreak_file.append(".mofreak");
		ofstream out(mofreak_file);

		cv::VideoCapture capture;
		capture.open(*it);
		// 5 since we go 5 frames back.  We can not pull out the interfreak any earlier.
		computeMoFREAK(capture, 5, std::numeric_limits<int>::max(), out);
	}
}

void main()
{
	// find all files in this folder that end with .avi.
	
	string root_folder = "C:/data/kth/all_in_one/videos/";
	processKTH(root_folder);
	
	
	/*string video_file = "C://data//kth//all_in_one//videos/person17_jogging_d1_uncomp.avi";
	computeMoFREAK(video_file);*/
	

	cout << "All done." << endl;
	int x = 0;
	cin >> x;
}