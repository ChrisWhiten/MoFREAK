#include "ActionRecognitionProject.h"

ActionRecognitionProject::ActionRecognitionProject(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	connect(ui.actionLoad, SIGNAL(triggered()), this, SLOT(loadFiles()));
	connect(ui.play_pause_button, SIGNAL(clicked()), this, SLOT(playOrPause()));
	connect(ui.actionLoadMoSIFT, SIGNAL(triggered()), this, SLOT(loadMoSIFTFile()));
	connect(ui.actionLoadEverything, SIGNAL(triggered()), this, SLOT(loadEverything()));
	connect(ui.convertMoSIFTToMoFREAK, SIGNAL(clicked()), this, SLOT(convertMoSIFTToMoFREAK()));
	connect(ui.train_svm_button, SIGNAL(clicked()), this, SLOT(trainSVM()));
	connect(ui.test_svm_button, SIGNAL(clicked()), this, SLOT(testSVM()));
	connect(ui.actionLoadTrainingFile, SIGNAL(triggered()), this, SLOT(loadSVMTrainingFile()));
	connect(ui.leave_one_out_button, SIGNAL(clicked()), this, SLOT(evaluateSVMWithLeaveOneOut()));
	connect(ui.actionLoadMoSIFTForClustering, SIGNAL(triggered()), this, SLOT(loadMoSIFTFilesForClustering()));
	connect(ui.cluster_push_button, SIGNAL(clicked()), this, SLOT(clusterMoSIFTPoints()));

	// Update the UI
	timer = new QTimer(this);
	timer->setInterval(1000/Constants::FPS);
	connect(timer, SIGNAL(timeout()), this, SLOT(nextFrame()));

	frame_number = 0;
	state = STOPPED;
}

ActionRecognitionProject::~ActionRecognitionProject()
{

}

void ActionRecognitionProject::pause()
{
	timer->stop();
}

void ActionRecognitionProject::play()
{
	timer->start();
}

// Slot associated with the user wanting to select files to load.
void ActionRecognitionProject::loadFiles()
{
    files = QFileDialog::getOpenFileNames(this, tr("Directory"), directory.path());
	if (files.length() > 1)
	{
		reading_sequence_of_images = true;
	}
	else
	{
		// initialize video to be read.
		capture = new cv::VideoCapture(files[0].toStdString());
		reading_sequence_of_images = false;
	}
}

// taken from feng's work.
 void ActionRecognitionProject::shuffleCVMat(cv::Mat &mx)
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

void ActionRecognitionProject::clusterMoSIFTPoints()
{
	// organize pts into a cv::Mat.
	const int FEATURE_DIMENSIONALITY = 256;
	const int NUM_CLUSTERS = 2;
	const int POINTS_TO_SAMPLE = 12000;
	cv::Mat data_pts(mosift_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	ui.frame_label->setText("Formatting features...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	for (unsigned int row = 0; row < mosift_ftrs.size(); ++row)
	{
		MoSIFTFeature ftr = mosift_ftrs[row];
		for (unsigned col = 0; col < 128; ++col)
		{
			data_pts.at<float>(row, col) = (float)ftr.SIFT[col];
			data_pts.at<float>(row, col + 128) = (float)ftr.motion[col];
		}
	}

	// shuffle 3 times.
	shuffleCVMat(data_pts);
	shuffleCVMat(data_pts);
	shuffleCVMat(data_pts);

	// remove excessive points.
	if (data_pts.rows > POINTS_TO_SAMPLE)
	{
		data_pts.pop_back(data_pts.rows - POINTS_TO_SAMPLE);
	}

	ui.frame_label->setText("Clustering...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	// call k-means.
	cv::Mat labels;//(data_pts.rows, 1, CV_32SC1);
	cv::Mat centers(NUM_CLUSTERS, 1, data_pts.type());//data_pts.cols, data_pts.type());
	//cv::kmeans(data_pts, NUM_CLUSTERS, labels, cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER,100000, 0.001), 1, cv::KMEANS_PP_CENTERS,  centers);
	kmeans(data_pts, NUM_CLUSTERS, labels,  cvTermCriteria( CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 100000, 0.001),1,cv::KMEANS_PP_CENTERS, centers);

	// print clusters to file
	ui.frame_label->setText("Writing clusters to file...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	ofstream output_file("clusters.txt");

	for (unsigned i = 0; i < NUM_CLUSTERS; ++i)
	{
		for (unsigned j = 0; j < centers.cols; ++j)
		{
			output_file << centers.at<float>(i, j) << " ";
		}

		output_file << std::endl;
	}
	output_file.close();
	ui.frame_label->setText("Clusters written...");
	ui.frame_label->adjustSize();
}

void ActionRecognitionProject::evaluateSVMWithLeaveOneOut()
{
	ofstream output_file("C:/data/kth/ehsan/whatthe.txt");

	double summed_accuracy = 0.0;
	for (unsigned i = 0; i < svm_training_files.size(); ++i)
	{
		SVMInterface svm_guy;
		// tell GUI where we're at in the l-o-o process
		stringstream ss;
		ss << "Leaving out person " << (i + 1);

		QString left_out_text = QString::fromStdString(ss.str());
		ss.str("");
		ss.clear();

		ui.frame_label->setText(left_out_text);
		ui.frame_label->adjustSize();

		// update gui.
		qApp->processEvents();

		// build model.
		svm_guy.trainModel(svm_training_files[i].toStdString());

		// get accuracy.
		double accuracy = svm_guy.testModel(svm_testing_files[i].toStdString());
		summed_accuracy += accuracy;

		// debugging...print to testing file.
		output_file << svm_training_files[i].toStdString() <<", " << svm_testing_files[i].toStdString() << ", " << accuracy << std::endl;
	}	

	output_file.close();

	// output average accuracy.
	double average_accuracy = summed_accuracy/(double)svm_training_files.size();

	stringstream ss;
	ss << "Averaged accuracy: " << average_accuracy << "%";

	QString avg_acc_text = QString::fromStdString(ss.str());
	ss.str("");
	ss.clear();

	ui.frame_label->setText(avg_acc_text);
	ui.frame_label->adjustSize();
}

void ActionRecognitionProject::trainSVM()
{
	// How can we force these labels to update before running the following lines?
	// The GUI kind of hangs and "training..." never appears.
	ui.frame_label->setText("Training...");
	ui.frame_label->adjustSize();

	svm_interface.trainModel(svm_training_files[0].toStdString());

	ui.frame_label->setText("Finished training.");
	ui.frame_label->adjustSize();
}

void ActionRecognitionProject::testSVM()
{
	ui.frame_label->setText("Testing...");
	ui.frame_label->adjustSize();

	double accuracy = svm_interface.testModel(svm_testing_files[0].toStdString());

	stringstream ss;
	ss << "Accuracy: " << accuracy << "%";

	QString accuracy_text = QString::fromStdString(ss.str());
	ss.str("");
	ss.clear();

	ui.frame_label->setText(accuracy_text);
	ui.frame_label->adjustSize();
}

// given the mosift features file, assuming the corresponding video files exist, remove sift and replace with freak descriptor.
void ActionRecognitionProject::convertMoSIFTToMoFREAK()
{
	int x = 0;
	for (auto it = files.begin(); it != files.end(); ++it)
	{
		// parse the path to extract the corresponding AVI file.
		QDir txt_file(*it);
		QString file_name = txt_file.dirName();
		
		txt_file.cdUp();
		QString video_path = txt_file.canonicalPath(); // this is the folder.
		video_path.append("/videos/");
		video_path.append(file_name);
		video_path.chop(3);
		video_path.append("avi");

		// ship it away for modification
		mofreak.buildMoFREAKFeaturesFromMoSIFT(it->toStdString(), video_path.toStdString());
		it->append(".mofreak.txt");
		mofreak.writeMoFREAKFeaturesToFile(it->toStdString());
	}
}

void ActionRecognitionProject::loadSVMTrainingFile()
{
	svm_training_files = QFileDialog::getOpenFileNames(this, tr("Directory"), directory.path());;

	// training files end with .train.  The assumption is that a matching .test file exists
	for (auto it = svm_training_files.begin(); it != svm_training_files.end(); ++it)
	{
		QString testing_file = (*it);
		testing_file.chop(5);
		testing_file.append("test");

		svm_testing_files.push_back(testing_file);
	}
}

void ActionRecognitionProject::loadMoSIFTFilesForClustering()
{
	files = QFileDialog::getOpenFileNames(this, tr("Directory"), directory.path());

	ui.frame_label->setText("Loading MoSIFT Features...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	for (auto it = files.begin(); it != files.end(); ++it)
	{
		mosift.readMoSIFTFeatures(it->toStdString());
	}

	mosift_ftrs = mosift.getMoSIFTFeatures();
	
	ui.frame_label->setText("MoSIFT Features Loaded.");
	ui.frame_label->adjustSize();
}
void ActionRecognitionProject::loadMoSIFTFile()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Directory"), directory.path());
	
	mosift.readMoSIFTFeatures(filename.toStdString());
	mosift_ftrs = mosift.getMoSIFTFeatures();

	/*
	for (unsigned i = 0; i < mosift_ftrs.size(); ++i)
	{
		int x = mosift_ftrs[i].frame_number;
	}*/
}

void ActionRecognitionProject::loadEverything()
{
	files = QFileDialog::getOpenFileNames(this, tr("Directory"), directory.path());
}

void ActionRecognitionProject::playOrPause()
{
	if (state == STOPPED)
	{
		ui.play_pause_button->setText("Pause");
		state = PLAYING;
		play();
	}
	else if (state == PLAYING)
	{
		ui.play_pause_button->setText("Play");
		state = PAUSED;
		pause();
	}
	else
	{
		ui.play_pause_button->setText("Pause");
		state = PLAYING;
		play();
	}
}

cv::Mat3b ActionRecognitionProject::getFrame()
{
	cv::Mat3b frame;
	if (reading_sequence_of_images)
	{
		if (frame_number >= files.size())
		{
			return frame;
		}

		QString filename = files[frame_number];
		frame = cv::imread(filename.toStdString());
	}
	else
	{
		*capture >> frame;	
	}
	return frame;
}

void ActionRecognitionProject::nextFrame()
{
	// load frame.
	cv::Mat3b frame = getFrame();
	if (!frame.data)
	{
		return;
	}

	cv::Mat gray(frame);
	cv::cvtColor(frame, gray, CV_BGR2GRAY);

	cv::Mat3b output_frame(frame.rows, frame.cols);
	frame.copyTo(output_frame);

	processFrame(gray, output_frame);
	updateGUI(frame, output_frame);
	frame_number++;
}

void ActionRecognitionProject::processFrame(cv::Mat &input, cv::Mat &output)
{
	// maybe display us some mosift here.
}

void ActionRecognitionProject::updateGUI(cv::Mat3b &raw_frame, cv::Mat3b &output_frame)
{
	// arrange raw and tracked frames to be viewable in our window.
	if (Constants::RESIZE_OUTPUT)
	{
		int frame_width = ui.centralWidget->width()/3;
		int frame_height = ui.centralWidget->height()/2;
		cv::resize(raw_frame, raw_frame, cv::Size(frame_width, frame_height), 0, 0);
		cv::resize(output_frame, output_frame, cv::Size(frame_width, frame_height), 0, 0);
	}

	// arrange raw image.
	ui.input_sequence->setGeometry(0, 0, raw_frame.cols, raw_frame.rows);

	QImage qimage_frame = OpenCVToQtInterfacing::Mat2QImage(raw_frame);
	ui.input_sequence->setPixmap(QPixmap::fromImage(qimage_frame));

	int input_label_x = (raw_frame.cols/2) - (ui.input_label->width()/2);
	ui.input_label->setGeometry(input_label_x, raw_frame.rows + 10, ui.input_label->width(), ui.input_label->height());

	// arrange output image.
	ui.output_sequence->setGeometry(output_frame.cols + 10, 0, output_frame.cols, output_frame.rows);

	QImage qimage_output_frame = OpenCVToQtInterfacing::Mat2QImage(output_frame);
	ui.output_sequence->setPixmap(QPixmap::fromImage(qimage_output_frame));

	int output_label_x = ((3 * output_frame.cols)/2 + 10) - (ui.output_label->width()/2);
	ui.output_label->setGeometry(output_label_x, output_frame.rows + 10, ui.output_label->width(), ui.output_label->height());

	// update frame number on GUI.
	stringstream ss;
	ss << "Frame " << frame_number;

	QString frametext = QString::fromStdString(ss.str());
	ss.str("");
	ss.clear();
	ui.frame_label->setText(frametext);

	repaint();
}