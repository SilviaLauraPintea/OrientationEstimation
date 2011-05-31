/* GaussianProcess.h
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef GAUSSIANPROCESS_H_
#define GAUSSIANPROCESS_H_
#include <tr1/memory>
#include "Cholesky.h"

/** Class implementing the Gaussian Process Regression.
 */
class GaussianProcess {
	public:
		/** Define a pointer to the kernel function
		 */
		typedef _float (GaussianProcess::*kernelFunction)(const cv::Mat&,\
			const cv::Mat&,_float);
		/** A structure used to define predictions.
		 */
		struct prediction {
			public:
			  std::deque<float> mean_;
			  std::deque<float> variance_;
			  prediction(){};
			  virtual ~prediction(){
				  if(!this->mean_.empty()){
					  this->mean_.clear();
				  }
				  if(!this->variance_.empty()){
					  this->variance_.clear();
				  }
			  }
			prediction(const prediction &pred){
				this->mean_ = pred.mean_;
				this->variance_ = pred.variance_;
			}
			prediction& operator=(const prediction &pred){
				if(this == &pred) return *this;
				this->mean_ = pred.mean_;
				this->variance_ = pred.variance_;
				return *this;
			}
		};
		/** All available distributions for the functions.
		 */
		enum DISTRIBUTION {BETA,GAUSS,GAUSS2D,GAUSSnD,LOGGAUSSnD};
		//======================================================================
		GaussianProcess(){
			this->_norm_fast_ = false;
			this->_norm_max_  = RAND_MAX/2.0;
			this->_norm_next_ = 0.0;
			this->rand_x_     = 0;
			this->rand_y_     = 1;
			this->N_          = 0;
			this->kFunction_  = &GaussianProcess::sqexp;
			this->chlsky_     = std::tr1::shared_ptr<Cholesky>(new Cholesky());
		};
		virtual ~GaussianProcess(){
			if(!this->alpha_.empty()){
				this->alpha_.release();
			}
			if(!this->data_.empty()){
				this->data_.release();
			}
			if(this->chlsky_){
				this->chlsky_.reset();
			}
		};
		GaussianProcess(const GaussianProcess &rhs){
			if(!this->data_.empty()){
				this->data_.release();
			}
			this->chlsky_     = rhs.chlsky_;
			this->alpha_      = rhs.alpha_;
			rhs.data_.copyTo(this->data_);
			this->N_          = rhs.N_;
			this->kFunction_  = rhs.kFunction_;
			this->_norm_fast_ = rhs._norm_fast_;
			this->_norm_next_ = rhs._norm_next_;
			this->_norm_max_  = rhs._norm_max_;
			this->rand_x_     = rhs.rand_x_;
			this->rand_y_     = rhs.rand_y_;
		}
		GaussianProcess& operator=(const GaussianProcess &rhs){
			if(this == &rhs){return *this;}
				if(!this->data_.empty()){
					this->data_.release();
				}
				this->chlsky_     = rhs.chlsky_;
				this->alpha_      = rhs.alpha_;
				rhs.data_.copyTo(this->data_);
				this->N_          = rhs.N_;
				this->kFunction_  = rhs.kFunction_;
				this->_norm_fast_ = rhs._norm_fast_;
				this->_norm_next_ = rhs._norm_next_;
				this->_norm_max_  = rhs._norm_max_;
				this->rand_x_     = rhs.rand_x_;
				this->rand_y_     = rhs.rand_y_;
				return *this;
		}
		/** Generates a selected distribution of the functions given the parameters (the
		 * mean: mu,the covariance: cov,the data x).
		 */
		_float distribution(const cv::Mat &x,const GaussianProcess::DISTRIBUTION \
			&distrib,const cv::Mat &mu,const cv::Mat &cov,_float a=0,_float b=0,\
			_float s=0);
		/** Trains the Gaussian process.
		 */
		void train(cv::Mat &X,cv::Mat &y,_float (GaussianProcess::*fFunction)\
			(const cv::Mat&,const cv::Mat&,_float),_float sigmasq,_float length);
		/** Returns the prediction for the test data,x (only one test data point).
		 */
		void predict(cv::Mat &x,GaussianProcess::prediction &predi,\
			_float length);
		/** Samples an N-dimensional Gaussian.
		 */
		void sampleGaussND(const cv::Mat &mu,const cv::Mat &cov,cv::Mat &smpl);
		/** Returns a random number from the normal distribution.
		 */
		_float rand_normal();
		/** Samples the process that generates the inputs.
		 */
		void sample(const cv::Mat &inputs,cv::Mat &smpl);
		/** Samples the Gaussian Process Prior.
		 */
		void sampleGPPrior(_float (GaussianProcess::*fFunction)\
		(const cv::Mat&,const cv::Mat&,_float),const cv::Mat &inputs,cv::Mat &smpl);
		// Squared exponential kernel function.
		_float sqexp(const cv::Mat &x1,const cv::Mat &x2,_float l=1.0);
		// Matern05 kernel function.
		_float matern05(const cv::Mat &x1,const cv::Mat &x2,_float l=1.0);
		// Exponential Covariance kernel function.
		_float expCovar(const cv::Mat &x1,const cv::Mat &x2,_float l=1.0);
		// Matern15 kernel function.
		_float matern15(const cv::Mat &x1,const cv::Mat &x2,_float l=1.0);
		// Matern25 kernel function.
		_float matern25(const cv::Mat &x1,const cv::Mat &x2,_float l=1.0);
		/** Initializes or re-initializes a Gaussian Process.
		 */
		void init(GaussianProcess::kernelFunction theKFunction =\
			&GaussianProcess::sqexp);
		/** Useful to compute the distance between 2 edges.
		 */
		_float matchShapes(const cv::Mat &x1,const cv::Mat &x2,_float l);
		/** Checks to see if the Gaussian process was trained.
		 */
		bool empty();
		//======================================================================
	private:
		/** @var chlsky_
		 * An instance of the class \c Cholesky.
		 */
		std::tr1::shared_ptr<Cholesky> chlsky_;
		/** @var alpha_
		 * A variable to chace the values of \i alpha from the algorithm.
		 */
		cv::Mat alpha_;
		/** @var data_
		 * Data matrix used for training.
		 */
		cv::Mat data_;
		/** @var N_
		 * Number of training data points (data.rows).
		 */
		unsigned N_;
		/** @var kFunction_
		 * Pointer to the kernel function to be used.
		 */
		kernelFunction kFunction_;
		bool _norm_fast_;
		_float _norm_next_,_norm_max_;
		int rand_x_,rand_y_;
};
#endif /* GAUSSIANPROCESS_H_ */
