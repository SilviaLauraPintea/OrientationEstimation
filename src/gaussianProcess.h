/* gaussianProcess.h
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef GAUSSIANPROCESS_H_
#define GAUSSIANPROCESS_H_
#include "cholesky.h"

/** Class implementing the Gaussian Process Regression.
 */
class gaussianProcess {
	public:
		/** Define a pointer to the kernel function
		 */
		typedef float (gaussianProcess::*kernelFunction)(const cv::Mat&,\
			const cv::Mat&,float);

		/** A structure used to define predictions.
		 */
		struct prediction {
			public:
			  std::deque<float> mean;
			  std::deque<float> variance;
			  prediction(){};
			  virtual ~prediction(){
				  if(!this->mean.empty()){
					  this->mean.clear();
				  }
				  if(!this->variance.empty()){
					  this->variance.clear();
				  }
			  }
			prediction(const prediction &pred){
				this->mean = pred.mean;
				this->variance = pred.variance;
			}
			prediction& operator=(const prediction &pred){
				if(this == &pred) return *this;
				this->mean = pred.mean;
				this->variance = pred.variance;
				return *this;
			}
		};

		/** All available distributions for the functions.
		 */
		enum DISTRIBUTION {BETA,GAUSS,GAUSS2D,GAUSSnD,LOGGAUSSnD};
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
		gaussianProcess(const gaussianProcess &rhs){
			if(!this->data.empty()){
				this->data.release();
			}
			this->chlsky     = rhs.chlsky;
			this->alpha      = rhs.alpha;
			rhs.data.copyTo(this->data);
			this->N          = rhs.N;
			this->kFunction  = rhs.kFunction;
			this->_norm_fast = rhs._norm_fast;
			this->_norm_next = rhs._norm_next;
			this->_norm_max  = rhs._norm_max;
			this->rand_x     = rhs.rand_x;
			this->rand_y     = rhs.rand_y;
		}
		gaussianProcess& operator=(const gaussianProcess &rhs){
			if(this == &rhs){return *this;}
				if(!this->data.empty()){
					this->data.release();
				}
				this->chlsky     = rhs.chlsky;
				this->alpha      = rhs.alpha;
				rhs.data.copyTo(this->data);
				this->N          = rhs.N;
				this->kFunction  = rhs.kFunction;
				this->_norm_fast = rhs._norm_fast;
				this->_norm_next = rhs._norm_next;
				this->_norm_max  = rhs._norm_max;
				this->rand_x     = rhs.rand_x;
				this->rand_y     = rhs.rand_y;
				return *this;
		}

		/** Generates a selected distribution of the functions given the parameters (the
		 * mean: mu,the covariance: cov,the data x).
		 */
		float distribution(const cv::Mat &x,const gaussianProcess::DISTRIBUTION \
			&distrib,const cv::Mat &mu,const cv::Mat &cov,float a=0,float b=0,\
			float s=0);

		/** Trains the Gaussian process.
		 */
		void train(const cv::Mat &X,const cv::Mat &y,float (gaussianProcess::*fFunction)\
			(const cv::Mat&,const cv::Mat&,float),float sigmasq,float length);

		/** Returns the prediction for the test data,x (only one test data point).
		 */
		void predict(const cv::Mat &x,gaussianProcess::prediction &predi,\
			float length);

		/** Samples an N-dimensional Gaussian.
		 */
		void sampleGaussND(const cv::Mat &mu,const cv::Mat &cov,cv::Mat &smpl);

		/** Returns a random number from the normal distribution.
		 */
		float rand_normal();

		/** Samples the process that generates the inputs.
		 */
		void sample(const cv::Mat &inputs,cv::Mat &smpl);

		/** Samples the Gaussian Process Prior.
		 */
		void sampleGPPrior(float (gaussianProcess::*fFunction)\
		(const cv::Mat&,const cv::Mat&,float),const cv::Mat &inputs,cv::Mat &smpl);

		// Squared exponential kernel function.
		float sqexp(const cv::Mat &x1,const cv::Mat &x2,float l=1.0);

		// Matern05 kernel function.
		float matern05(const cv::Mat &x1,const cv::Mat &x2,float l=1.0);

		// Exponential Covariance kernel function.
		float expCovar(const cv::Mat &x1,const cv::Mat &x2,float l=1.0);

		// Matern15 kernel function.
		float matern15(const cv::Mat &x1,const cv::Mat &x2,float l=1.0);

		// Matern25 kernel function.
		float matern25(const cv::Mat &x1,const cv::Mat &x2,float l=1.0);
		/** Initializes or re-initializes a Gaussian Process.
		 */
		void init(gaussianProcess::kernelFunction theKFunction =\
			&gaussianProcess::sqexp);
		//======================================================================
	public:
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
		float _norm_next,_norm_max;
		int rand_x,rand_y;
};
#endif /* GAUSSIANPROCESS_H_ */
