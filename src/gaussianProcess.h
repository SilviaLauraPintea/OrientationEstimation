/* gaussianProcess.h
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 */
#ifndef GAUSSIANPROCESS_H_
#define GAUSSIANPROCESS_H_
#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <exception>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "cholesky.h"
/** Class implementing the Gaussian Process Regression.
 */
class gaussianProcess {
	public:
		/** All available distributions for the functions.
		 */
		enum DISTRIBUTION {BETA, GAUSS, GAUSS2D, GAUSSnD, LOGGAUSSnD};
		//======================================================================
		gaussianProcess();
		virtual ~gaussianProcess();

		/** Generates a selected distribution of the functions given the parameters (the
		 * mean: mu, the covariance: cov, the data x).
		 */
		void distribution(cv::Mat x,gaussianProcess::DISTRIBUTION distrib,\
			cv::Mat mu = cv::Mat(),cv::Mat cov = cv::Mat(),float a=0,float b=0,float s=0);
		//======================================================================
	protected:
		cholesky chlsky;

};
#endif /* GAUSSIANPROCESS_H_ */
