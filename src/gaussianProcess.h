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
#include <deque>
#include <opencv2/opencv.hpp>
#include "cholesky.h"

/** Class implementing the Gaussian Process Regression.
 */
class gaussianProcess {
	public:
		/** Define a pointer to the kernel function
		 */
		typedef double (gaussianProcess::*kernelFunction)(cv::Mat, cv::Mat, double);

		/** A structure used to define predictions.
		 */
		struct prediction {
		  std::deque<double> mean;
		  std::deque<double> variance;
		  ~prediction(){
			  if(!this->mean.empty()){
				  this->mean.clear();
			  }
			  if(!this->variance.empty()){
				  this->variance.clear();
			  }
		  }
		};

		/** All available distributions for the functions.
		 */
		enum DISTRIBUTION {BETA, GAUSS, GAUSS2D, GAUSSnD, LOGGAUSSnD};
		//======================================================================
		gaussianProcess(){
			this->_norm_fast = false;
			this->_norm_max  = RAND_MAX/2.0;
			this->_norm_next = 0.0;
			this->rand_x     = 0;
			this->rand_y     = 1;
			this->N          = 0;
			this->kFunction  = &gaussianProcess::sqexp;
		};
		virtual ~gaussianProcess(){
			if(!this->alpha.empty()){
				this->alpha.release();
			}
			if(!this->data.empty()){
				this->data.release();
			}
		};

		/** Generates a selected distribution of the functions given the parameters (the
		 * mean: mu, the covariance: cov, the data x).
		 */
		double distribution(cv::Mat x,gaussianProcess::DISTRIBUTION distrib,\
			cv::Mat mu = cv::Mat(),cv::Mat cov = cv::Mat(),double a=0,double b=0,\
			double s=0);

		/** Trains the Gaussian process.
		 */
		void train(cv::Mat X, cv::Mat y, double (gaussianProcess::*fFunction)\
			(cv::Mat, cv::Mat, double), double sigmasq, double length=1.0);

		/** Returns the prediction for the test data, x (only one test data point).
		 */
		void predict(cv::Mat x, gaussianProcess::prediction &predi,\
			double length=1.0);

		/** Samples an N-dimensional Gaussian.
		 */
		void sampleGaussND(cv::Mat mu, cv::Mat cov, cv::Mat &smpl);

		/** Returns a random number from the normal distribution.
		 */
		double rand_normal();

		/** Samples the process that generates the inputs.
		 */
		void sample(cv::Mat inputs, cv::Mat &smpl);

		/** Samples the Gaussian Process Prior.
		 */
		void sampleGPPrior(double (gaussianProcess::*fFunction)(cv::Mat,\
		cv::Mat, double), cv::Mat inputs, cv::Mat &smpl);

		// Squared exponential kernel function.
		double sqexp(cv::Mat x1, cv::Mat x2, double l=1.0);

		// Matern05 kernel function.
		double matern05(cv::Mat x1, cv::Mat x2, double l=1.0);

		// Exponential Covariance kernel function.
		double expCovar(cv::Mat x1, cv::Mat x2, double l=1.0);

		// Matern15 kernel function.
		double matern15(cv::Mat x1, cv::Mat x2, double l=1.0);

		// Matern25 kernel function.
		double matern25(cv::Mat x1, cv::Mat x2, double l=1.0);
		//======================================================================
	protected:
		/** @var chlsky
		 * An instance of the class \c cholesky.
		 */
		cholesky chlsky;

		/** @var alpha
		 * A variable to chace the values of \i alpha from the algorithm.
		 */
		cv::Mat alpha;

		/** @var data
		 * Data matrix used for training.
		 */
		cv::Mat data;

		/** @var N
		 * Number of training data points (data.rows).
		 */
		unsigned N;

		/** @var kFunction
		 * Pointer to the kernel function to be used.
		 */
		kernelFunction kFunction;

		bool _norm_fast;
		double _norm_next,_norm_max;
		int rand_x, rand_y;
};
#endif /* GAUSSIANPROCESS_H_ */
