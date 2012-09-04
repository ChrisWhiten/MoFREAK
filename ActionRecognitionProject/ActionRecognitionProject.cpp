#include "ActionRecognitionProject.h"

ActionRecognitionProject::ActionRecognitionProject(QWidget *parent, Qt::WFlags flags)
	: QMainWindow(parent, flags)
{
	ui.setupUi(this);

	connect(ui.actionLoad, SIGNAL(triggered()), this, SLOT(loadFiles()));
	connect(ui.play_pause_button, SIGNAL(clicked()), this, SLOT(playOrPause()));

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

	// process frame here.
	//trackFrame(gray, output_frame);
	updateGUI(frame, output_frame);
	frame_number++;
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