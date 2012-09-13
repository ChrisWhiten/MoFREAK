/********************************************************************************
** Form generated from reading UI file 'ActionRecognitionProject.ui'
**
** Created: Thu Sep 13 12:01:24 2012
**      by: Qt User Interface Compiler version 4.8.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_ACTIONRECOGNITIONPROJECT_H
#define UI_ACTIONRECOGNITIONPROJECT_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QMainWindow>
#include <QtGui/QMenu>
#include <QtGui/QMenuBar>
#include <QtGui/QPushButton>
#include <QtGui/QStatusBar>
#include <QtGui/QToolBar>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_ActionRecognitionProjectClass
{
public:
    QAction *actionLoad;
    QAction *actionLoadMoSIFT;
    QAction *actionLoadEverything;
    QAction *actionLoadTrainingFile;
    QAction *actionLoadTestingFile;
    QWidget *centralWidget;
    QPushButton *play_pause_button;
    QPushButton *stop_button;
    QLabel *input_sequence;
    QLabel *output_sequence;
    QLabel *input_label;
    QLabel *output_label;
    QLabel *frame_label;
    QPushButton *convertMoSIFTToMoFREAK;
    QPushButton *train_svm_button;
    QPushButton *test_svm_button;
    QPushButton *leave_one_out_button;
    QMenuBar *menuBar;
    QMenu *menuFile;
    QToolBar *mainToolBar;
    QStatusBar *statusBar;

    void setupUi(QMainWindow *ActionRecognitionProjectClass)
    {
        if (ActionRecognitionProjectClass->objectName().isEmpty())
            ActionRecognitionProjectClass->setObjectName(QString::fromUtf8("ActionRecognitionProjectClass"));
        ActionRecognitionProjectClass->resize(600, 400);
        actionLoad = new QAction(ActionRecognitionProjectClass);
        actionLoad->setObjectName(QString::fromUtf8("actionLoad"));
        actionLoadMoSIFT = new QAction(ActionRecognitionProjectClass);
        actionLoadMoSIFT->setObjectName(QString::fromUtf8("actionLoadMoSIFT"));
        actionLoadEverything = new QAction(ActionRecognitionProjectClass);
        actionLoadEverything->setObjectName(QString::fromUtf8("actionLoadEverything"));
        actionLoadTrainingFile = new QAction(ActionRecognitionProjectClass);
        actionLoadTrainingFile->setObjectName(QString::fromUtf8("actionLoadTrainingFile"));
        actionLoadTestingFile = new QAction(ActionRecognitionProjectClass);
        actionLoadTestingFile->setObjectName(QString::fromUtf8("actionLoadTestingFile"));
        centralWidget = new QWidget(ActionRecognitionProjectClass);
        centralWidget->setObjectName(QString::fromUtf8("centralWidget"));
        play_pause_button = new QPushButton(centralWidget);
        play_pause_button->setObjectName(QString::fromUtf8("play_pause_button"));
        play_pause_button->setGeometry(QRect(30, 270, 75, 23));
        stop_button = new QPushButton(centralWidget);
        stop_button->setObjectName(QString::fromUtf8("stop_button"));
        stop_button->setGeometry(QRect(130, 270, 75, 23));
        input_sequence = new QLabel(centralWidget);
        input_sequence->setObjectName(QString::fromUtf8("input_sequence"));
        input_sequence->setGeometry(QRect(60, 60, 46, 13));
        output_sequence = new QLabel(centralWidget);
        output_sequence->setObjectName(QString::fromUtf8("output_sequence"));
        output_sequence->setGeometry(QRect(390, 60, 46, 13));
        input_label = new QLabel(centralWidget);
        input_label->setObjectName(QString::fromUtf8("input_label"));
        input_label->setGeometry(QRect(70, 250, 46, 13));
        output_label = new QLabel(centralWidget);
        output_label->setObjectName(QString::fromUtf8("output_label"));
        output_label->setGeometry(QRect(400, 250, 46, 13));
        frame_label = new QLabel(centralWidget);
        frame_label->setObjectName(QString::fromUtf8("frame_label"));
        frame_label->setGeometry(QRect(490, 320, 46, 13));
        convertMoSIFTToMoFREAK = new QPushButton(centralWidget);
        convertMoSIFTToMoFREAK->setObjectName(QString::fromUtf8("convertMoSIFTToMoFREAK"));
        convertMoSIFTToMoFREAK->setGeometry(QRect(390, 270, 151, 23));
        train_svm_button = new QPushButton(centralWidget);
        train_svm_button->setObjectName(QString::fromUtf8("train_svm_button"));
        train_svm_button->setGeometry(QRect(20, 310, 75, 23));
        test_svm_button = new QPushButton(centralWidget);
        test_svm_button->setObjectName(QString::fromUtf8("test_svm_button"));
        test_svm_button->setGeometry(QRect(120, 310, 75, 23));
        leave_one_out_button = new QPushButton(centralWidget);
        leave_one_out_button->setObjectName(QString::fromUtf8("leave_one_out_button"));
        leave_one_out_button->setGeometry(QRect(220, 310, 161, 23));
        ActionRecognitionProjectClass->setCentralWidget(centralWidget);
        menuBar = new QMenuBar(ActionRecognitionProjectClass);
        menuBar->setObjectName(QString::fromUtf8("menuBar"));
        menuBar->setGeometry(QRect(0, 0, 600, 21));
        menuFile = new QMenu(menuBar);
        menuFile->setObjectName(QString::fromUtf8("menuFile"));
        ActionRecognitionProjectClass->setMenuBar(menuBar);
        mainToolBar = new QToolBar(ActionRecognitionProjectClass);
        mainToolBar->setObjectName(QString::fromUtf8("mainToolBar"));
        ActionRecognitionProjectClass->addToolBar(Qt::TopToolBarArea, mainToolBar);
        statusBar = new QStatusBar(ActionRecognitionProjectClass);
        statusBar->setObjectName(QString::fromUtf8("statusBar"));
        ActionRecognitionProjectClass->setStatusBar(statusBar);

        menuBar->addAction(menuFile->menuAction());
        menuFile->addAction(actionLoad);
        menuFile->addAction(actionLoadMoSIFT);
        menuFile->addAction(actionLoadEverything);
        menuFile->addSeparator();
        menuFile->addAction(actionLoadTrainingFile);

        retranslateUi(ActionRecognitionProjectClass);

        QMetaObject::connectSlotsByName(ActionRecognitionProjectClass);
    } // setupUi

    void retranslateUi(QMainWindow *ActionRecognitionProjectClass)
    {
        ActionRecognitionProjectClass->setWindowTitle(QApplication::translate("ActionRecognitionProjectClass", "ActionRecognitionProject", 0, QApplication::UnicodeUTF8));
        actionLoad->setText(QApplication::translate("ActionRecognitionProjectClass", "Load Video", 0, QApplication::UnicodeUTF8));
        actionLoadMoSIFT->setText(QApplication::translate("ActionRecognitionProjectClass", "Load MoSIFT Features", 0, QApplication::UnicodeUTF8));
        actionLoadEverything->setText(QApplication::translate("ActionRecognitionProjectClass", "Load Group of MoSIFT Features and AVI Files", 0, QApplication::UnicodeUTF8));
        actionLoadTrainingFile->setText(QApplication::translate("ActionRecognitionProjectClass", "Load SVM Training File", 0, QApplication::UnicodeUTF8));
        actionLoadTestingFile->setText(QApplication::translate("ActionRecognitionProjectClass", "Load SVM Testing File", 0, QApplication::UnicodeUTF8));
        play_pause_button->setText(QApplication::translate("ActionRecognitionProjectClass", "Play", 0, QApplication::UnicodeUTF8));
        stop_button->setText(QApplication::translate("ActionRecognitionProjectClass", "Restart", 0, QApplication::UnicodeUTF8));
        input_sequence->setText(QString());
        output_sequence->setText(QString());
        input_label->setText(QApplication::translate("ActionRecognitionProjectClass", "Input", 0, QApplication::UnicodeUTF8));
        output_label->setText(QApplication::translate("ActionRecognitionProjectClass", "Output", 0, QApplication::UnicodeUTF8));
        frame_label->setText(QString());
        convertMoSIFTToMoFREAK->setText(QApplication::translate("ActionRecognitionProjectClass", "Convert MoSIFT to MoFREAK", 0, QApplication::UnicodeUTF8));
        train_svm_button->setText(QApplication::translate("ActionRecognitionProjectClass", "Train SVM", 0, QApplication::UnicodeUTF8));
        test_svm_button->setText(QApplication::translate("ActionRecognitionProjectClass", "Test SVM", 0, QApplication::UnicodeUTF8));
        leave_one_out_button->setText(QApplication::translate("ActionRecognitionProjectClass", "Test SVM with Leave-One-Out", 0, QApplication::UnicodeUTF8));
        menuFile->setTitle(QApplication::translate("ActionRecognitionProjectClass", "File", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class ActionRecognitionProjectClass: public Ui_ActionRecognitionProjectClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_ACTIONRECOGNITIONPROJECT_H
