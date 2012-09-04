#include "ActionRecognitionProject.h"
#include <QtGui/QApplication>

int main(int argc, char *argv[])
{
	QApplication a(argc, argv);
	ActionRecognitionProject w;
	w.show();
	return a.exec();
}
