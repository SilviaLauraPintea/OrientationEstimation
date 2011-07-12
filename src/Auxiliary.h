/* Auxiliary.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef AUXILIARY_H_
#define AUXILIARY_H_
#include "eigenbackground/src/Helpers.hh"
class Auxiliary:public Helpers{
	public:
		/** Converts a pointer to an IplImage to an OpenCV Mat.
		 */
		static cv::Mat ipl2mat(IplImage* ipl_image);

		/** Converts an OpenCV Mat to a pointer to an IplImage.
		 */
		static IplImage* mat2ipl(const cv::Mat &image);

		/** Convert the values from a cv::Mat of floats to be between 0 and 1.
		 */
		static void normalizeMat(cv::Mat &matrix);

		/** Changes the values of the matrix to be between [-1,1].
		 */
		static void range1Mat(cv::Mat &matrix);

		/** Write a 2D-matrix to a text file (first row is the dimension of the matrix).
		 */
		static void mat2TxtFile(cv::Mat &matrix,char* fileName,bool append = false);

		/** Reads a 2D-matrix from a text file (first row is the dimension of the matrix).
		 */
		static void txtFile2Mat(cv::Mat &matrix,char* fileName);

		/** Write a 2D-matrix to a binary file (first the dimension of the matrix).
		 */
		static void mat2BinFile(cv::Mat &matrix,char* fileName,bool append = false);

		/** Reads a 2D-matrix from a binary file (first the dimension of the matrix).
		 */
		static void binFile2mat(cv::Mat &matrix,char* fileName);
		/** Convert int to string.
		 */
		static std::string int2string(int i);
		/** Changes a given angle in RADIANS to be positive and between [0,2*M_PI).
		 */
		static void angle0to360(float &angle);
		/** Changes a given angle in RADIANS to be positive and between [-M_PI,M_PI).
		 */
		static void angle180to180(float &angle);
		/** Get perpendicular to a line given by 2 points A,B in point C.
		 */
		static void perpendicularLine(const cv::Point2f &A,const cv::Point2f &B,\
			const cv::Point2f &C,float &m,float &b);
		/** Checks to see if a point is on the same side of a line like another given point.
		 */
		static bool sameSubplane(const cv::Point2f &test,const cv::Point2f &point,\
			float m,float b);
		/** Just displaying an image a bit larger to visualize it better.
		 */
		static void showZoomedImage(const cv::Mat &image,const std::string &title);
		/** A function that transforms the data such that it has zero mean and unit
		 * variance: img = (img-mean(img(:)))/std(img(:)).
		 */
		static void mean0Variance1(cv::Mat &mat);
		/** Used to sort a vector of points -- compares points on the X coordinate.
		 */
		static bool isSmallerPointX(const cv::Point2f &p1,const cv::Point2f &p2);
		/** Compares 2 keypoints based on their response.
		 */
		static bool isLargerKey(const cv::KeyPoint &k1,const cv::KeyPoint &k2);
		/** Compares 2 the lengths of 2 openCV contours (vectors of vectors of cv::Point).
		 */
		static bool isLongerContours(const std::vector<cv::Point> &c1,\
			const std::vector<cv::Point> &c2);
		/** Store the PCA model locally so you can load it next time when you need it.
		 */
		static void savePCA(const std::tr1::shared_ptr<cv::PCA> pcaPtr,\
			const std::string &file);
		/** Load the PCA model locally so you can load it next time when you need it.
		 */
		static std::tr1::shared_ptr<cv::PCA> loadPCA(const std::string &file);
		/** Deallocates a PCA pointed by a pointer.
		 */
		static void getRidOfPCA(cv::PCA *pca);
		/** Mean and stddev for matrices.
		 */
		static void mean0Variance1(cv::Mat &mat,cv::Mat &mean, cv::Mat &var);
	private:
		DISALLOW_COPY_AND_ASSIGN(Auxiliary);
};
#endif /* AUXILIARY_H_ */
