#include "ActionRecognitionProject.h"

ActionRecognitionProject::ActionRecognitionProject(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	// file menu.
	connect(ui.actionLoad, SIGNAL(triggered()), this, SLOT(loadFiles()));
	connect(ui.actionLoadMoSIFT, SIGNAL(triggered()), this, SLOT(loadMoSIFTFile()));
	connect(ui.actionLoadEverything, SIGNAL(triggered()), this, SLOT(loadEverything()));
	connect(ui.actionLoadTrainingFile, SIGNAL(triggered()), this, SLOT(loadSVMTrainingFile()));
	connect(ui.actionLoadMoSIFTForClustering, SIGNAL(triggered()), this, SLOT(loadMoSIFTFilesForClustering()));
	connect(ui.actionLoadMoFREAKForClustering, SIGNAL(triggered()), this, SLOT(loadMoFREAKFilesForClustering()));
	connect(ui.actionLoadToComputeMoFREAK, SIGNAL(triggered()), this, SLOT(loadVideosForMoFREAK()));
	

	// buttons on the GUI.
	connect(ui.play_pause_button, SIGNAL(clicked()), this, SLOT(playOrPause()));
	connect(ui.convertMoSIFTToMoFREAK, SIGNAL(clicked()), this, SLOT(convertMoSIFTToMoFREAK()));
	connect(ui.train_svm_button, SIGNAL(clicked()), this, SLOT(trainSVM()));
	connect(ui.test_svm_button, SIGNAL(clicked()), this, SLOT(testSVM()));
	connect(ui.leave_one_out_button, SIGNAL(clicked()), this, SLOT(evaluateSVMWithLeaveOneOut()));
	connect(ui.cluster_push_button, SIGNAL(clicked()), this, SLOT(clusterMoSIFTPoints()));
	connect(ui.cluster_mofreak_push_button, SIGNAL(clicked()), this, SLOT(clusterMoFREAKPoints()));
	connect(ui.push_button_compute_mofreak_from_videos, SIGNAL(clicked()), this, SLOT(buildMoFREAKFromVideos()));
	
	

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

void ActionRecognitionProject::clusterMoFREAKPoints()
{
	// organize pts into a cv::Mat.
	const int FEATURE_DIMENSIONALITY = 32;//192;//65;//192;
	const int NUM_CLUSTERS = 600;
	const int POINTS_TO_SAMPLE = 12000;
	const int NUM_CLASSES = 6;
	const int NUMBER_OF_PEOPLE = 25;
	cv::Mat data_pts(mofreak_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, POINTS_TO_SAMPLE, NUM_CLASSES);
	clustering.setAppearanceDescriptor(16, true);
	clustering.setMotionDescriptor(16, true);

	ui.frame_label->setText("Formatting features...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	clustering.buildDataFromMoFREAK(mofreak_ftrs, false, false);

	ui.frame_label->setText("Clustering...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	//clustering.clusterWithKMeans();
	clustering.randomClusters();

	// print clusters to file
	ui.frame_label->setText("Writing clusters to file...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	clustering.writeClusters();

	ui.frame_label->setText("Computing BOW Representation...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	BagOfWordsRepresentation bow_rep(files, NUM_CLUSTERS, FEATURE_DIMENSIONALITY, NUMBER_OF_PEOPLE, true, true);
	bow_rep.setAppearanceDescriptor(16, true);
	bow_rep.setMotionDescriptor(16, true);
	bow_rep.computeBagOfWords();

	ui.frame_label->setText("BOW Representation Computed.");
	ui.frame_label->adjustSize();
}

void ActionRecognitionProject::clusterMoSIFTPoints()
{
	// organize pts into a cv::Mat.
	const int FEATURE_DIMENSIONALITY = 256;
	const int NUM_CLUSTERS = 600;
	const int POINTS_TO_SAMPLE = 12000;
	const int NUM_CLASSES = 6;
	const int NUMBER_OF_PEOPLE = 25;
	cv::Mat data_pts(mosift_ftrs.size(), FEATURE_DIMENSIONALITY, CV_32FC1);

	Clustering clustering(FEATURE_DIMENSIONALITY, NUM_CLUSTERS, POINTS_TO_SAMPLE, NUM_CLASSES);
	clustering.setAppearanceDescriptor(128, false);
	clustering.setMotionDescriptor(128, false);

	ui.frame_label->setText("Formatting features...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	clustering.buildDataFromMoSIFT(mosift_ftrs, false);

	ui.frame_label->setText("Clustering...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	//clustering.clusterWithKMeans();
	clustering.randomClusters();

	// print clusters to file
	ui.frame_label->setText("Writing clusters to file...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	clustering.writeClusters();

	ui.frame_label->setText("Computing BOW Representation...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	BagOfWordsRepresentation bow_rep(files, NUM_CLUSTERS, FEATURE_DIMENSIONALITY, NUMBER_OF_PEOPLE, false, false);
	bow_rep.setAppearanceDescriptor(128, false);
	bow_rep.setMotionDescriptor(128, false);
	bow_rep.computeBagOfWords();

	ui.frame_label->setText("BOW Representation Computed.");
	ui.frame_label->adjustSize();
}

void ActionRecognitionProject::evaluateSVMWithLeaveOneOut()
{
	ofstream output_file("C:/data/kth/chris/whatthe.txt");

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
		string test_filename = svm_testing_files[i].toStdString();
		double accuracy = svm_guy.testModel(test_filename);
		summed_accuracy += accuracy;

		// debugging...print to testing file.
		output_file << svm_training_files[i].toStdString() <<", " << svm_testing_files[i].toStdString() << ", " << accuracy << std::endl;
	}	

	output_file.close();

	// output average accuracy.
	double denominator = (double)svm_training_files.size();
	double average_accuracy = summed_accuracy/denominator;

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

void ActionRecognitionProject::loadVideosForMoFREAK()
{
	files = QFileDialog::getOpenFileNames(this, tr("Directory"), directory.path());
}

void ActionRecognitionProject::buildMoFREAKFromVideos()
{
	
	for (auto it = files.begin(); it != files.end(); ++it)
	{
		// tell the user where we are...
		stringstream ss;
		ss << "Computing MoFREAK on " << it->toStdString();

		QString status_text = QString::fromStdString(ss.str());
		ss.str("");
		ss.clear();

		ui.frame_label->setText(status_text);
		ui.frame_label->adjustSize();
		qApp->processEvents();

		// and compute.
		mofreak.computeMoFREAKFromFile(*it, true);
	}

	ui.frame_label->setText("Done");
	ui.frame_label->adjustSize();
	qApp->processEvents();
}

// given the mosift features file, assuming the corresponding video files exist, remove sift and replace with freak descriptor.
void ActionRecognitionProject::convertMoSIFTToMoFREAK()
{
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
		mofreak.buildMoFREAKFeaturesFromMoSIFT(*it, video_path.toStdString());
		it->append(".mofreak.txt");
		mofreak.writeMoFREAKFeaturesToFile(it->toStdString(), false);
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

void ActionRecognitionProject::loadMoFREAKFilesForClustering()
{
	files = QFileDialog::getOpenFileNames(this, tr("Directory"), directory.path());

	ui.frame_label->setText("Loading MoFREAK Features...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	for (auto it = files.begin(); it != files.end(); ++it)
	{
		mofreak.readMoFREAKFeatures(*it, USING_IMG_DIFF);
	}

	mofreak_ftrs = mofreak.getMoFREAKFeatures();
	
	ui.frame_label->setText("MoFREAK Features Loaded.");
	ui.frame_label->adjustSize();
}

void ActionRecognitionProject::loadMoSIFTFilesForClustering()
{
	files = QFileDialog::getOpenFileNames(this, tr("Directory"), directory.path());

	ui.frame_label->setText("Loading MoSIFT Features...");
	ui.frame_label->adjustSize();
	qApp->processEvents();

	for (auto it = files.begin(); it != files.end(); ++it)
	{
		mosift.readMoSIFTFeatures(*it);
	}

	mosift_ftrs = mosift.getMoSIFTFeatures();
	
	ui.frame_label->setText("MoSIFT Features Loaded.");
	ui.frame_label->adjustSize();
}
void ActionRecognitionProject::loadMoSIFTFile()
{
	QString filename = QFileDialog::getOpenFileName(this, tr("Directory"), directory.path());
	
	mosift.readMoSIFTFeatures(filename);
	mosift_ftrs = mosift.getMoSIFTFeatures();
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
	// maybe display some mosift points overlaid on top of the frame here.
	// clearly not used.
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