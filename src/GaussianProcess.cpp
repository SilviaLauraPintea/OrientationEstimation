/* GaussianProcess.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include <iostream>
#include <fstream>
#include <cmath>
#include <exception>
#include <deque>
#include <opencv2/opencv.hpp>
#include "Cholesky.h"
#include "GaussianProcess.h"
#include "Auxiliary.h"
//==============================================================================
/** Initializes or re-initializes a Gaussian Process.
 */
void GaussianProcess::init(GaussianProcess::kernelFunction theKFunction){
	this->chlsky_->init();
	this->_norm_fast_ = false;
	this->_norm_max_  = RAND_MAX/2.0;
	this->_norm_next_ = 0.0;
	this->rand_x_     = 0;
	this->rand_y_     = 1;
	this->N_          = 0;
	this->kFunction_  = theKFunction;
	if(!this->alpha_.empty()){
		this->alpha_.release();
	}
	if(!this->data_.empty()){
		this->data_.release();
	}
}
//==============================================================================
/** Generates a selected distribution of the functions given the parameters (the
 * mean: mu,the covariance: cov,the data_ x).
 */
_float GaussianProcess::distribution(const cv::Mat &x,\
const GaussianProcess::DISTRIBUTION &distrib,const cv::Mat &mu,const cv::Mat &cov,\
_float a,_float b,_float s){
	assert(cov.type()==_CV_32FC1);
	assert(mu.type()==_CV_32FC1);

	_float det2,result;
	cv::Mat diff;
	cv::Mat inv;

	switch(distrib){
		case (GaussianProcess::BETA):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess BETA distribution: size(x) = (1,1)!"<<std::endl;
				goto err;
			}
			result = (gamma(a+b)*(std::pow(x.at<_float>(0,0),(a-1.0)))*\
						(std::pow(1.0-x.at<_float>(0,0),(b-1.0))))/(gamma(a)+gamma(b));
			break;
		case (GaussianProcess::GAUSS):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(x) = (1,1)!"<<std::endl;
				goto err;
			}
			if(mu.cols!=1 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(mu) = (1,1)!(mean)"<<std::endl;
				goto err;
			}
			result = std::exp(-std::pow((x.at<_float>(0,0)-mu.at<_float>(0,0)),2)/\
						(2.0*std::pow(s,2)))/(std::sqrt(2.0*M_PI)*s);
			break;
		case (GaussianProcess::GAUSS2D):
			if(x.cols!=2 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(x)=(1,2)!(x.x,x.y)"<<std::endl;
				goto err;
			}
			if(mu.cols!=2 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(mu)=(1,2)!(mu.x,mu.y)"<<std::endl;
				goto err;
			}
			if(cov.cols!=2 || cov.rows!=2){
				std::cerr<<"GaussianProcess GAUSS2D distribution: size(cov)=(2,2)!(covariance)"<<std::endl;
				goto err;
			}
			det2   = (cov.at<_float>(0,0)*cov.at<_float>(1,1) -\
							cov.at<_float>(0,1)*cov.at<_float>(1,0));
			diff   = x-mu;
			result = 1.0/(2.0*M_PI*std::sqrt(det2))*std::exp(-0.5*\
						diff.dot(diff*cov.inv()));
			break;
		case (GaussianProcess::GAUSSnD):
			if(x.cols<2 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSSnD distribution: size(x)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSSnD distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			// IF NO DECOMPOSITION WAS DONE,DO THAT
			if(!this->chlsky_->checkDecomposition()){
				this->chlsky_->decomposeCov(cov);
			}
			this->chlsky_->inverse(inv);
			diff   = x-mu;
			result = 1.0/(std::pow((2.0*M_PI),(x.cols/2.0))*\
						exp(0.5*this->chlsky_->logDet()))*exp(-0.5*diff.dot(diff*inv));
			break;
		case (GaussianProcess::LOGGAUSSnD):
			if(x.cols<2 || x.rows!=1){
				std::cerr<<"GaussianProcess LOGGAUSS2D distribution: size(x)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess LOGGAUSS2D distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			// IF NO DECOMPOSITION WAS DONE,DO THAT
			if(!this->chlsky_->checkDecomposition()){
				this->chlsky_->decomposeCov(cov);
			}
			this->chlsky_->inverse(inv);
			diff   = x-mu;
			result = -0.5*(this->chlsky_->n()*std::log(2.0*M_PI)+this->chlsky_->logDet()+\
						diff.dot(diff*inv));
			break;
	}
	inv.release();
	diff.release();
	return result;
	err:
		diff.release();
		inv.release();
		exit(1);
}
//==============================================================================
/** Trains the Gaussian process.
 */
void GaussianProcess::train(cv::Mat &X,cv::Mat &y,\
_float (GaussianProcess::*fFunction)(const cv::Mat&,const cv::Mat&,_float),\
_float sigmasq,_float length){
	if(y.rows != X.rows){
		std::cerr<<"In Gaussian Process - train: X and y need to be defined for the"<<\
			" same number of points"<<std::endl;
		return;
	}
	// MAKE SURE THE MATRIXES ARE IN THE RIGHT FORMAT TO WORK WITH THEM
	X.convertTo(X,_CV_32FC1);
	y.convertTo(y,_CV_32FC1);

	this->chlsky_->init();
	this->kFunction_ = fFunction;
	this->N_         = X.rows;// NUMBER OF TRAINING DATA POINTS!
	if(!this->data_.empty()){
		this->data_.release();
	}
	X.copyTo(this->data_);
	this->data_.convertTo(this->data_,_CV_32FC1);

	// BUILD THE KERNEL MARIX K: K(i,j) = k(x[i],x[j])
	cv::Mat K = cv::Mat::zeros(cv::Size(this->N_,this->N_),_CV_32FC1);
	for(int indy=0;indy<this->N_;++indy){
		for(int indx=0;indx<this->N_;++indx){
			K.at<_float>(indy,indx) = (this->*kFunction_)\
				(X.row(indy),X.row(indx),length);
		}
	}

	// ADD sigma^2 TO THE KERNEL MATRIX,K
	for(int indy=0;indy<this->N_;++indy){
		K.at<_float>(indy,indy) += sigmasq;
	}

    // BUILD THE CHOLESKY DECOMPOSITON IF IT WAS NOT BUILT YET
	if(!this->chlsky_->checkDecomposition()){
		if(!this->chlsky_->decomposeCov(K)){
			std::cerr<<"Cholesky decomposition failed"<<std::endl;
			exit(1);
		}
	}
	this->chlsky_->solve(y,this->alpha_);
	this->alpha_.convertTo(this->alpha_,_CV_32FC1);

	std::cout<<"N: ("<<this->N_<<")"<<std::endl;
	std::cout<<"size of alpha: ("<<this->alpha_.cols<<","<<this->alpha_.rows<<")"<<std::endl;
	std::cout<<"size of data: ("<<this->data_.cols<<","<<this->data_.rows<<")"<<std::endl;
	K.release();
}
//==============================================================================
/** Returns the prediction for the test data,x (only one test data point).
 */
void GaussianProcess::predict(cv::Mat &x,GaussianProcess::prediction &predi,\
_float length){
	// MAKE SURE THE INPUT MATRIX HAS THE REQUESTED TYPE
	x.convertTo(x,_CV_32FC1);

	cv::Mat kstar(cv::Size(1,this->N_),_CV_32FC1);
	for(int indy=0;indy<this->N_;++indy){
		kstar.at<_float>(indy,0) = (this->*kFunction_)(this->data_.row(indy),x,length);
	}
	for(int i=0;i<this->alpha_.cols;++i){
		predi.mean_.push_back(static_cast<float>(kstar.dot(this->alpha_.col(i))));
	}
	cv::Mat v;
	this->chlsky_->solveL(kstar,v);
	v.convertTo(v,_CV_32FC1);
	predi.variance_.push_back(static_cast<float>((this->*kFunction_)(x,x,length)-v.dot(v)));
	kstar.release();
	v.release();
}
//==============================================================================
/** Samples the process that generates the inputs.
 */
void GaussianProcess::sample(const cv::Mat &inputs,cv::Mat &smpl){
	cv::Mat Kxstarx(this->N_,inputs.cols,_CV_32FC1);
	cv::Mat Kxstarxstar(inputs.cols,inputs.cols,_CV_32FC1);

	for(int indy=0;indy<inputs.cols;++indy){
		for(int indx=0;indx<this->N_;++indx){
			Kxstarx.at<_float>(indy,indx) = (this->*kFunction_)(inputs.row(indy),\
											this->data_.row(indx),1.0);
		}
		for(int indx=0;indx<inputs.cols;++indx){
			Kxstarxstar.at<_float>(indy,indx) = (this->*kFunction_)(inputs.row(indy),\
												inputs.row(indx),1.0);
		}
	}

	cv::Mat Kxxstar = Kxstarx.t();
	cv::Mat mu      = Kxstarx * this->alpha_;
	cv::Mat inv;
	this->chlsky_->inverse(inv);
	cv::Mat cov     = Kxstarxstar-(Kxstarx*inv)*Kxxstar;

	for(int indy=0;indy<cov.cols;++indy){
		cov.at<_float>(indy,indy) += 1.0e-6;
	}
	this->sampleGaussND(mu,cov,smpl);

	Kxstarx.release();
	Kxstarxstar.release();
	Kxxstar.release();
	mu.release();
	cov.release();
	inv.release();
}
//==============================================================================
/** Samples an N-dimensional Gaussian.
 */
void GaussianProcess::sampleGaussND(const cv::Mat &mu,const cv::Mat &cov,cv::Mat &smpl){
	assert(cov.type()==_CV_32FC1);
	assert(mu.type()==_CV_32FC1);
	if(!this->chlsky_->checkDecomposition()){
		this->chlsky_->decomposeCov(cov);
	}

	for(int indy=0;indy<mu.cols;++indy){
		smpl.at<_float>(indy,0) = this->rand_normal();
	}

	smpl = mu + (this->chlsky_->covar() * smpl);
}
//==============================================================================
/** Returns a random number from the normal distribution.
 */
_float GaussianProcess::rand_normal(){
	if(this->_norm_fast_){
		this->_norm_fast_ = false;
		return (this->_norm_next_);
	}
	this->_norm_fast_ = true;

	while(true){
		_float u = (std::rand() - _norm_max_ + 0.5)/_norm_max_;
		_float v = (std::rand() - _norm_max_ + 0.5)/_norm_max_;

		_float x,w =u*u+v*v;
		if (w >= 1) continue;
		x = std::sqrt(-2.0 * std::log(w)/w);
		this->_norm_next_ = u*x;
		return (v*x);
     }
     return 12345.789;
}
//==============================================================================
/** Samples the Gaussian Process Prior.
 */
void GaussianProcess::sampleGPPrior(_float (GaussianProcess::*fFunction)\
(const cv::Mat&,const cv::Mat&,_float),const cv::Mat &inputs,cv::Mat &smpl){
	this->kFunction_ = fFunction;
	cv::Mat mu;
	cv::Mat cov;

	for(int indy=0;indy<inputs.cols;++indy){
		mu.at<_float>(indy,0) = 0.0;
		for(int indx=0;indx<inputs.cols;++indx){
			cov.at<_float>(indy,indx) = (this->*kFunction_)(inputs.row(indy),\
										inputs.row(indx),1.0);
		}
	}

	for(int indy=0;indy<inputs.cols;++indy){
		cov.at<_float>(indy,indy) += 1.0e-6;
	}
	this->sampleGaussND(mu,cov,smpl);
	mu.release();
	cov.release();
}
//==============================================================================
// Squared exponential kernel function.
_float GaussianProcess::sqexp(const cv::Mat &x1,const cv::Mat &x2,_float l){
	assert(x1.type()==_CV_32FC1);
	assert(x2.type()==_CV_32FC1);
	cv::Mat diff = x1-x2;
	diff.convertTo(diff,_CV_32FC1);
	cv::Mat tmpDiff = diff.colRange(0,diff.cols-2);
	_float result1  = std::sqrt(tmpDiff.dot(tmpDiff))/(2.0*l);
	_float result2  = std::sqrt(diff.at<_float>(0,diff.cols-2)*diff.at<_float>\
		(0,diff.cols-2))/(2.0*l);
	_float result  = std::exp(-1.0*(result1+result2));
	diff.release();
	tmpDiff.release();
	return result;
}
//==============================================================================
// Matern05 kernel function.
_float GaussianProcess::matern05(const cv::Mat &x1,const cv::Mat &x2,_float l){
	assert(x1.type()==_CV_32FC1);
	assert(x2.type()==_CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,_CV_32FC1);
	_float result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Exponential Covariance kernel function.
_float GaussianProcess::expCovar(const cv::Mat &x1,const cv::Mat &x2,_float l){
	assert(x1.type()==_CV_32FC1);
	assert(x2.type()==_CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,_CV_32FC1);
	_float result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Matern15 kernel function.
_float GaussianProcess::matern15(const cv::Mat &x1,const cv::Mat &x2,_float l){
	assert(x1.type()==_CV_32FC1);
	assert(x2.type()==_CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,_CV_32FC1);
	_float result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(3.0)*result/l) * std::exp(-1.0 * std::sqrt(3.0)*result/l);
}
//==============================================================================
// Matern25 kernel function.
_float GaussianProcess::matern25(const cv::Mat &x1,const cv::Mat &x2,_float l){
	assert(x1.type()==_CV_32FC1);
	assert(x2.type()==_CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,_CV_32FC1);
	_float result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(5.0)*result/l + (5.0*result*result)/(3.0*l*l))*\
		std::exp(-1.0 * std::sqrt(5.0)*result/l);
}
//==============================================================================
/** Useful to compute the distance between 2 edges.
 */
_float GaussianProcess::matchShapes(const cv::Mat &x1,const cv::Mat &x2,_float l){
	assert(x1.type()==_CV_32FC1);
	assert(x2.type()==_CV_32FC1);
	cv::Mat tmpX1,tmpX2;

	// IT NEEDS TO HAVE 2 CHANNELS => NEEDS TO HAVE COLS%2=0
	if(x1.cols%2==1){
		tmpX1        = cv::Mat::zeros(cv::Size(x1.cols+1,x1.rows),x1.type());
		cv::Mat dumm = tmpX1.colRange(0,x1.cols);
		x1.copyTo(dumm);
		dumm.release();

		// ALSO FOR X2 IT SHOULD BE TRUE
		tmpX2 = cv::Mat::zeros(cv::Size(x2.cols+1,x2.rows),x2.type());
		dumm  = tmpX2.colRange(0,x2.cols);
		x2.copyTo(dumm);
		dumm.release();
	}else{
		x1.copyTo(tmpX1);
		x2.copyTo(tmpX2);
	}
	tmpX1.convertTo(tmpX1,_CV_32FC1);
	tmpX2.convertTo(tmpX2,_CV_32FC1);
	tmpX1 = tmpX1.reshape(2,0);
	tmpX2 = tmpX2.reshape(2,0);
	_float result = cv::matchShapes(tmpX1,tmpX2,1,0.0);
	tmpX1.release();
	tmpX2.release();


	if(!result){
		result = 1e-3;
		return 10*std::exp(-result/l);
	}
	return std::exp(-result/l);
}
//==============================================================================
/** Checks to see if the Gaussian process was trained.
 */
bool GaussianProcess::empty(){
	return (!this->N_ || this->data_.empty() || this->alpha_.empty() ||\
		!this->chlsky_->checkDecomposition());
}
//==============================================================================
//==============================================================================
/*
int main(){
	cv::Mat test(10,5,_CV_32FC1);
	cv::Mat train(100,5,_CV_32FC1);
	cv::Mat targets = cv::Mat::zeros(100,1,_CV_32FC1);
	cv::Mat ttargets = cv::Mat::zeros(10,1,_CV_32FC1);
	train = cv::Mat::zeros(100,5,_CV_32FC1);

	for(unsigned i=0;i<100;++i){
		cv::Mat stupid = train.row(i);
		cv::add(stupid,cv::Scalar(i),stupid);
		targets.at<_float>(i,0) = i;
		if(i<10){
			cv::Mat stupid2 = test.row(i);
			cv::add(stupid2,cv::Scalar(i*2.5),stupid2);
			ttargets.at<_float>(i,0) = i*2.5;
		}
	}

	std::cout<<"test: "<<test<<std::endl;
	std::cout<<"ttargets: "<<ttargets<<std::endl;

	GaussianProcess gp;
	gp.train(train,targets,&GaussianProcess::sqexp,0.1);

	cv::Mat result;
	for(unsigned i=0;i<test.rows;++i){
		GaussianProcess::prediction predi;
		gp.predict(test.row(i),predi);
		std::cout<<"label: "<<ttargets.at<_float>(i,0)<<"\t"<<\
			predi.mean[0]<<" variance:"<<predi.variance[0]<<std::endl;
	}
}
*/
