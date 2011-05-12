/* gaussianProcess.cpp
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
#include "cholesky.h"
#include "gaussianProcess.h"
//==============================================================================
/** Initializes or re-initializes a Gaussian Process.
 */
void gaussianProcess::init(gaussianProcess::kernelFunction theKFunction){
	this->chlsky.init();
	this->_norm_fast = false;
	this->_norm_max  = RAND_MAX/2.0;
	this->_norm_next = 0.0;
	this->rand_x     = 0;
	this->rand_y     = 1;
	this->N          = 0;
	this->kFunction  = theKFunction;
	if(!this->alpha.empty()){
		this->alpha.release();
	}
	if(!this->data.empty()){
		this->data.release();
	}
}
//==============================================================================
/** Generates a selected distribution of the functions given the parameters (the
 * mean: mu,the covariance: cov,the data x).
 */
float gaussianProcess::distribution(cv::Mat x,gaussianProcess::DISTRIBUTION distrib,
cv::Mat mu,cv::Mat cov,float a,float b,float s){
	float det2,result;
	cv::Mat diff;
	cv::Mat inv;

	switch(distrib){
		case (gaussianProcess::BETA):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess BETA distribution: size(x) = (1,1)!"<<std::endl;
				goto err;
			}
			result = (gamma(a+b)*(std::pow(x.at<float>(0,0),(a-1.0)))*\
						(std::pow(1.0-x.at<float>(0,0),(b-1.0))))/(gamma(a)+gamma(b));
			break;
		case (gaussianProcess::GAUSS):
			if(x.cols!=1 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(x) = (1,1)!"<<std::endl;
				goto err;
			}
			if(mu.cols!=1 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSS distribution: size(mu) = (1,1)!(mean)"<<std::endl;
				goto err;
			}
			result = std::exp(-std::pow((x.at<float>(0,0)-mu.at<float>(0,0)),2)/\
						(2.0*std::pow(s,2)))/(std::sqrt(2.0*M_PI)*s);
			break;
		case (gaussianProcess::GAUSS2D):
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
			det2   = (cov.at<float>(0,0)*cov.at<float>(1,1) -\
							cov.at<float>(0,1)*cov.at<float>(1,0));
			diff   = x-mu;
			result = 1.0/(2.0*M_PI*std::sqrt(det2))*std::exp(-0.5*\
						diff.dot(diff*cov.inv()));
			break;
		case (gaussianProcess::GAUSSnD):
			if(x.cols<2 || x.rows!=1){
				std::cerr<<"GaussianProcess GAUSSnD distribution: size(x)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess GAUSSnD distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			// IF NO DECOMPOSITION WAS DONE,DO THAT
			if(!this->chlsky.checkDecomposition()){
				this->chlsky.decomposeCov(cov);
			}
			this->chlsky.inverse(inv);
			diff   = x-mu;
			result = 1.0/(std::pow((2.0*M_PI),(x.cols/2.0))*\
						exp(0.5*this->chlsky.logDet()))*exp(-0.5*diff.dot(diff*inv));
			break;
		case (gaussianProcess::LOGGAUSSnD):
			if(x.cols<2 || x.rows!=1){
				std::cerr<<"GaussianProcess LOGGAUSS2D distribution: size(x)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			if(mu.cols<2 || mu.rows!=1){
				std::cerr<<"GaussianProcess LOGGAUSS2D distribution: size(mu)=(1,n)!(a row)"<<std::endl;
				goto err;
			}
			// IF NO DECOMPOSITION WAS DONE,DO THAT
			if(!this->chlsky.checkDecomposition()){
				this->chlsky.decomposeCov(cov);
			}
			this->chlsky.inverse(inv);
			diff   = x-mu;
			result = -0.5*(this->chlsky.n*std::log(2.0*M_PI)+this->chlsky.logDet()+\
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
void gaussianProcess::train(cv::Mat X,cv::Mat y,float (gaussianProcess::*fFunction)\
(cv::Mat,cv::Mat,float),float sigmasq,float length){
	if(y.rows != X.rows){
		std::cerr<<"In Gaussian Process - train: X and y need to be defined for the"<<\
			" same number of points"<<std::endl;
		return;
	}
	X.convertTo(X,CV_32FC1);
	y.convertTo(y,CV_32FC1);
	this->chlsky.init();
	this->kFunction = fFunction;
	this->N         = X.rows;// NUMBER OF TRAINING DATA POINTS!
	if(!this->data.empty()){
		this->data.release();
	}
	X.copyTo(this->data);
	this->data.convertTo(this->data,CV_32FC1);

	// BUILD THE KERNEL MARIX K: K(i,j) = k(x[i],x[j])
	cv::Mat K = cv::Mat::zeros(cv::Size(this->N,this->N),CV_32FC1);
	for(int indy=0;indy<this->N;++indy){
		for(int indx=0;indx<this->N;++indx){
			K.at<float>(indy,indx) = (this->*kFunction)(X.row(indy),\
										X.row(indx),length);
		}
	}

	// ADD sigma^2 TO THE KERNEL MATRIX,K
	for(int indy=0;indy<this->N;++indy){
		K.at<float>(indy,indy) += sigmasq;
	}

    // BUILD THE CHOLESKY DECOMPOSITON IF IT WAS NOT BUILT YET
	if(!this->chlsky.checkDecomposition()){
		K.convertTo(K,CV_32FC1);
		if(!this->chlsky.decomposeCov(K)){
			std::cerr<<"Cholesky decomposition failed"<<std::endl;
			exit(1);
		}
	}
	this->chlsky.solve(y,this->alpha);
	this->alpha.convertTo(this->alpha,CV_32FC1);

	std::cout<<"N: ("<<this->N<<")"<<std::endl;
	std::cout<<"size of alpha: ("<<this->alpha.cols<<","<<this->alpha.rows<<")"<<std::endl;
	std::cout<<"size of data: ("<<this->data.cols<<","<<this->data.rows<<")"<<std::endl;
	K.release();
}
//==============================================================================
/** Returns the prediction for the test data,x (only one test data point).
 */
void gaussianProcess::predict(cv::Mat x,gaussianProcess::prediction &predi,\
float length){
	x.convertTo(x,CV_32FC1);
	cv::Mat kstar(cv::Size(1,this->N),CV_32FC1);
	for(int indy=0;indy<this->N;++indy){
		kstar.at<float>(indy,0) = (this->*kFunction)(this->data.row(indy),x,length);
	}

	for(int i=0;i<this->alpha.cols;++i){
		predi.mean.push_back(kstar.dot(this->alpha.col(i)));
	}
	cv::Mat v;
	this->chlsky.solveL(kstar,v);
	v.convertTo(v,CV_32FC1);
	predi.variance.push_back((this->*kFunction)(x,x,length) - v.dot(v));
	kstar.release();
	v.release();
}
//==============================================================================
/** Samples the process that generates the inputs.
 */
void gaussianProcess::sample(cv::Mat inputs,cv::Mat &smpl){
	cv::Mat Kxstarx(this->N,inputs.cols,CV_32FC1);
	cv::Mat Kxstarxstar(inputs.cols,inputs.cols,CV_32FC1);

	for(int indy=0;indy<inputs.cols;++indy){
		for(int indx=0;indx<this->N;++indx){
			Kxstarx.at<float>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
											this->data.row(indx),1.0);
		}
		for(int indx=0;indx<inputs.cols;++indx){
			Kxstarxstar.at<float>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
												inputs.row(indx),1.0);
		}
	}

	cv::Mat Kxxstar = Kxstarx.t();
	cv::Mat mu      = Kxstarx * this->alpha;
	cv::Mat inv;
	this->chlsky.inverse(inv);
	cv::Mat cov     = Kxstarxstar-(Kxstarx*inv)*Kxxstar;

	for(int indy=0;indy<cov.cols;++indy){
		cov.at<float>(indy,indy) += 1.0e-6;
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
void gaussianProcess::sampleGaussND(cv::Mat mu,cv::Mat cov,cv::Mat &smpl){
	if(!this->chlsky.checkDecomposition()){
		this->chlsky.decomposeCov(cov);
	}

	for(int indy=0;indy<mu.cols;++indy){
		smpl.at<float>(indy,0) = this->rand_normal();
	}

	smpl = mu + (this->chlsky.covar * smpl);
}
//==============================================================================
/** Returns a random number from the normal distribution.
 */
float gaussianProcess::rand_normal(){
	if(this->_norm_fast){
		this->_norm_fast = false;
		return (this->_norm_next);
	}
	this->_norm_fast = true;

	while(true){
		float u = (std::rand() - _norm_max + 0.5)/_norm_max;
		float v = (std::rand() - _norm_max + 0.5) / _norm_max;

		float x,w =u*u+v*v;
		if (w >= 1) continue;
		x = std::sqrt(-2.0 * std::log(w)/w);
		this->_norm_next = u*x;
		return (v*x);
     }
     return 12345.789;
}
//==============================================================================
/** Samples the Gaussian Process Prior.
 */
void gaussianProcess::sampleGPPrior(float (gaussianProcess::*fFunction)(cv::Mat,\
cv::Mat,float),cv::Mat inputs,cv::Mat &smpl){
	this->kFunction = fFunction;
	cv::Mat mu;
	cv::Mat cov;

	for(int indy=0;indy<inputs.cols;++indy){
		mu.at<float>(indy,0) = 0.0;
		for(int indx=0;indx<inputs.cols;++indx){
			cov.at<float>(indy,indx) = (this->*kFunction)(inputs.row(indy),\
										inputs.row(indx),1.0);
		}
	}

	for(int indy=0;indy<inputs.cols;++indy){
		cov.at<float>(indy,indy) += 1.0e-6;
	}
	this->sampleGaussND(mu,cov,smpl);
	mu.release();
	cov.release();
}
//==============================================================================
// Squared exponential kernel function.
float gaussianProcess::sqexp(cv::Mat x1,cv::Mat x2,float l){
	x1.convertTo(x1,CV_32FC1);
	x2.convertTo(x2,CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,CV_32FC1);
	float result = std::exp(-1.0 * diff.dot(diff)/(2.0*l));
	diff.release();
	return result;
}
//==============================================================================
// Matern05 kernel function.
float gaussianProcess::matern05(cv::Mat x1,cv::Mat x2,float l){
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,CV_32FC1);
	float result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Exponential Covariance kernel function.
float gaussianProcess::expCovar(cv::Mat x1,cv::Mat x2,float l){
	x1.convertTo(x1,CV_32FC1);
	x2.convertTo(x1,CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,CV_32FC1);
	float result = std::sqrt(diff.dot(diff));
	diff.release();
	return std::exp(-1.0 * result/l);
}
//==============================================================================
// Matern15 kernel function.
float gaussianProcess::matern15(cv::Mat x1,cv::Mat x2,float l){
	x1.convertTo(x1,CV_32FC1);
	x2.convertTo(x1,CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,CV_32FC1);
	float result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(3.0)*result/l) * std::exp(-1.0 * std::sqrt(3.0)*result/l);
}
//==============================================================================
// Matern25 kernel function.
float gaussianProcess::matern25(cv::Mat x1,cv::Mat x2,float l){
	x1.convertTo(x1,CV_32FC1);
	x2.convertTo(x1,CV_32FC1);
	cv::Mat diff  = x1-x2;
	diff.convertTo(diff,CV_32FC1);
	float result = std::sqrt(diff.dot(diff));
	diff.release();
	return (1.0 + std::sqrt(5.0)*result/l + (5.0*result*result)/(3.0*l*l))*\
		std::exp(-1.0 * std::sqrt(5.0)*result/l);
}
//==============================================================================
// Chamfer 2D distance metric.
/*
float gaussianProcess::chamfer(cv::Mat x1,cv::Mat x2,float l){
	for(int x=0;x<x1.cols;++x){
		float m1 = x
	}
}
*/
//==============================================================================

/*
int main(){
	cv::Mat test(10,5,CV_32FC1);
	cv::Mat train(100,5,CV_32FC1);
	cv::Mat targets = cv::Mat::zeros(100,1,CV_32FC1);
	cv::Mat ttargets = cv::Mat::zeros(10,1,CV_32FC1);
	train = cv::Mat::zeros(100,5,CV_32FC1);

	for(unsigned i=0;i<100;++i){
		cv::Mat stupid = train.row(i);
		cv::add(stupid,cv::Scalar(i),stupid);
		targets.at<float>(i,0) = i;
		if(i<10){
			cv::Mat stupid2 = test.row(i);
			cv::add(stupid2,cv::Scalar(i*2.5),stupid2);
			ttargets.at<float>(i,0) = i*2.5;
		}
	}

	std::cout<<"test: "<<test<<std::endl;
	std::cout<<"ttargets: "<<ttargets<<std::endl;

	gaussianProcess gp;
	gp.train(train,targets,&gaussianProcess::sqexp,0.1);

	cv::Mat result;
	for(unsigned i=0;i<test.rows;++i){
		gaussianProcess::prediction predi;
		gp.predict(test.row(i),predi);
		std::cout<<"label: "<<ttargets.at<float>(i,0)<<"\t"<<\
			predi.mean[0]<<" variance:"<<predi.variance[0]<<std::endl;
	}
}
*/
