MoFREAK
========================

Action recognition for surveillance scenarios with local binary feature descriptors.  Work done by Chris Whiten for the VIVA Research Lab at University of Ottawa.  Work was completed for TRECVID 2012, as well as further research in the action recognition domain.  

Questions can be forwarded to Chris Whiten at chris.whiten@gmail.com


Dependencies:
------------------
- Boost 1.51.0 (http://www.boostpro.com/download/)
- OpenCV 2.4.2 (http://opencv.org)
- Built with Visual Studio 2010

Usage:
-----------------
The constructs for performing action recognition already exist for some datasets.  Within main.cpp, there is a setParameters() function that outlines the file structure required for each dataset.  Dataset can be selected at the top of main.cpp with the "dataset" variable, selected from the "datasets" enum.

To exclusively compute the MoFREAK features across a dataset, set the "state" variable at the top of main.cpp to "DETECT_MOFREAK".  This will process each video file, creating a .mofreak file containing its descriptors.

To compute MoFREAK features across the dataset and perform the entire recognition pipeline, set the "state" variable to "DETECTION_TO_CLASSIFICATION".  This will compute the MoFREAK files, cluster the features and compute a bag-of-words representation.  Finally, it will take the bag-of-words representations and classify them with an SVM.

Feature Format:
----------------
Within a .mofreak file, each row consists of a single feature.  That feature is organized as follows:
- [x location] [y location] [frame number] [scale] [throw-away] [throw-away] [8 bytes of appearance data] [8 bytes of motion data]

The x and y location, as well as the scale, are floating point numbers.  The frame number is an integer.  The final 16 bytes of descriptor data are unsigned integers (1 byte per integer).  The throw-away values are floating point values that should always be 0.  They are simply artifacts from previous iterations of the descriptor, and should be ignored.