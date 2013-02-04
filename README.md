MoFREAK
========================

Action recognition for surveillance scenarios with local binary feature descriptors.
Root project is not currently being maintained.  Instead, work is being done in ActionRecognitionProjectConsoleVersion (for efficiency, losing the GUI)


Dependencies:
------------------
- Boost 1.51.0 (http://www.boostpro.com/download/)
- Qt 4.8 (http://qt.digia.com/product/) [gui version only]
- OpenCV 2.4.2 (http://opencv.org)


Notes:
------------------
- Qt and Boost have opposing views of wchar_t variables.  In VS2010, go to Project Properties-> C/C++-> Language and turn on "Treat WChar_t As Built in Type"

- Within main.cpp, there are a few things to set before using this software.
- At the top, there is a bool called TRECVID.  Set it to true if performing tests on the TRECVID dataset, false for KTH.
- The function setParameters() sets the folder locations that can be modified.  My apologies that this is not more modifiable outside of the source.  Future work :)
- Finally, to set the task, there is a 'state' variable in the main() function.  MOFREAK_TO_DETECTION computes MoFREAK descriptors, gets bag of words features,
tests each feature against the SVM, and finds detections by local maxima in the SVM response space.
