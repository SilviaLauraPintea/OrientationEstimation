/* Auxiliary.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef AUXILIARY_H_
#define AUXILIARY_H_
#include <opencv2/opencv.hpp>

/** Converts a pointer to an IplImage to an OpenCV Mat.
 */
void ipl2mat(IplImage* ipl_image,cv::Mat &mat_image);

/** Converts an OpenCV Mat to a pointer to an IplImage.
 */
void mat2ipl(const cv::Mat &image,IplImage* ipl_image);

/** Convert the values from a cv::Mat of doubles to be between 0 and 1.
 */
void normalizeMat(cv::Mat &matrix);

/** Changes the values of the matrix to be between [-1,1].
 */
void range1Mat(cv::Mat &matrix);

/** Write a 2D-matrix to a text file (first row is the dimension of the matrix).
 */
void mat2TxtFile(cv::Mat &matrix,char* fileName,bool append = false);

/** Reads a 2D-matrix from a text file (first row is the dimension of the matrix).
 */
void txtFile2Mat(cv::Mat &matrix,char* fileName);

/** Write a 2D-matrix to a binary file (first the dimension of the matrix).
 */
void mat2BinFile(cv::Mat &matrix,char* fileName,bool append = false);

/** Reads a 2D-matrix from a binary file (first the dimension of the matrix).
 */
void binFile2mat(cv::Mat &matrix,char* fileName);
/** Convert int to string.
 */
std::string int2string(int i);
/** Changes a given angle in RADIANS to be positive and between [0,2*M_PI).
 */
void angle0to360(float &angle);
/** Changes a given angle in RADIANS to be positive and between [-M_PI,M_PI).
 */
void angle180to180(float &angle);
/** Get perpendicular to a line given by 2 points A,B in point C.
 */
void perpendicularLine(const cv::Point2f &A,const cv::Point2f &B,\
const cv::Point2f &C,float &m,float &b);
/** Checks to see if a point is on the same side of a line like another given point.
 */
bool sameSubplane(const cv::Point2f &test,const cv::Point2f &point,float m,float b);
/** Just displaying an image a bit larger to visualize it better.
 */
void showZoomedImage(const cv::Mat &image,std::string &title);
/** A function that transforms the data such that it has zero mean and unit
 * variance: img = (img-mean(img(:)))/std(img(:)).
 */
void mean0Variance1(cv::Mat &mat);
#endif /* AUXILIARY_H_ */
