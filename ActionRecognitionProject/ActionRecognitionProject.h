#ifndef ACTIONRECOGNITIONPROJECT_H
#define ACTIONRECOGNITIONPROJECT_H

#include <QtGui/QMainWindow>
#include <qtimer.h>

// for file/directory manipulations
#include <qdir.h>
#include <qfiledialog.h>

// for image processing and representation
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// basic C++ operations.
#include <iostream>
#include <string>
#include <sstream>

#include "ui_ActionRecognitionProject.h"
#include "Constants.h"
#include "OpenCVToQtInterfacing.h"

using namespace std;

class ActionRecognitionProject : public QMainWindow
{
	Q_OBJECT

public:
	ActionRecognitionProject(QWidget *parent = 0, Qt::WFlags flags = 0);
	~ActionRecognitionProject();

private slots:
	void loadFiles();
	void playOrPause();
	void nextFrame();

private:
	int frame_number;
	Ui::ActionRecognitionProjectClass ui;
	QDir directory;
	QTimer *timer;
	int state;

	// false if reading from an actual video file
	bool reading_sequence_of_images;

	QStringList files;
	cv::VideoCapture *capture;

	cv::Mat3b getFrame();
	void updateGUI(cv::Mat3b &raw_frame, cv::Mat3b &tracked_frame);
	void pause();
	void play();


	enum states {PAUSED, PLAYING, STOPPED};
};

#endif // ACTIONRECOGNITIONPROJECT_H
