#ifndef OPENCVTOQTINTERFACING_H
#define OPENCVTOQTINTERFACING_H

#include <opencv2/core/core.hpp>
#include "qimage.h"

namespace OpenCVToQtInterfacing
{
	// These functions below convert a cv::Mat to a QImage.
	// Adapted from http://stackoverflow.com/questions/5026965/how-to-convert-an-opencv-cvmat-to-qimage
	// The grayscale ones seem to be inverted.  Look into this later.

	inline QImage Mat2QImage(const cv::Mat3b &src) 
	{
		QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
		for (int row = 0; row < src.rows; ++row) 
		{
			const cv::Vec3b *srcrow = src[row];
			QRgb *destrow = (QRgb*)dest.scanLine(row);
			for (int col = 0; col < src.cols; ++col) 
			{
				destrow[col] = qRgba(srcrow[col][2], srcrow[col][1], srcrow[col][0], 255);
			}
		}
		return dest;
	}

	inline QImage Mat2QImage(const cv::Mat_<double> &src)
	{
		double scale = 255.0;
		QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
		for (int y = 0; y < src.rows; ++y) 
		{
			const double *srcrow = src[y];
			QRgb *destrow = (QRgb*)dest.scanLine(y);
			for (int x = 0; x < src.cols; ++x) 
			{
				unsigned int color = srcrow[x] * scale;
				destrow[x] = qRgba(color, color, color, 255);
			}
		}
		return dest;
	}

	inline QImage Mat2QImage(const cv::Mat_<unsigned char> &src)
	{
		double scale = 255.0;
		QImage dest(src.cols, src.rows, QImage::Format_ARGB32);
		for (int y = 0; y < src.rows; ++y) 
		{
			const unsigned char *srcrow = src[y];
			QRgb *destrow = (QRgb*)dest.scanLine(y);
			for (int x = 0; x < src.cols; ++x) 
			{
				unsigned int color = srcrow[x] * scale;
				destrow[x] = qRgba(color, color, color, 255);
			}
		}
		return dest;
	}
};

#endif