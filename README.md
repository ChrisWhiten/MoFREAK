MoFREAK
========================

Action recognition for surveillance scenarios with local binary feature descriptors.  Work done by Chris Whiten for the VIVA Research Lab at University of Ottawa.  Work was completed for TRECVID 2012, as well as further research in the action recognition domain.  

Questions can be forwarded to Chris Whiten at chris.whiten@gmail.com


Dependencies:
------------------
- Boost 1.51.0 (http://www.boostpro.com/download/)
- OpenCV 2.4.2 (http://opencv.org)


Notes:
------------------
- Within main.cpp, there are a few things to set before using this software.
- At the top, there is a bool called TRECVID.  Set it to true if performing tests on the TRECVID dataset, false for anything else.  
- This source code has been heavily modified since TRECVID submission was last tested.  Some functionality may be broken, but the general idea persists within the source code.
- The function setParameters() sets the folder locations that can be modified.  My apologies that this is not more modifiable outside of the source.  Future work 
- Finally, to set the task, there is a 'state' variable in the main() function.  MOFREAK_TO_DETECTION computes MoFREAK descriptors, gets bag of words features,
tests each feature against the SVM, and finds detections by local maxima in the SVM response space.  DETECT_MOFREAK simply computes the MoFREAK descriptors for the entire dataset.
