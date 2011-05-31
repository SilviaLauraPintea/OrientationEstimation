/* Cholesky.cpp
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#include <iostream>
#include <cmath>
#include <err.h>
#include <exception>
#include "Cholesky.h"
//==============================================================================
/** Checks to see if the decomposition was already done (returns true if it is
 * done).
 */
bool Cholesky::checkDecomposition(){
	return (!this->covar_.empty());
}
//==============================================================================
/** (Re)Initializes the class variables so the same instance of the class can be
 * used for multiple decompositions.
 */
void Cholesky::init(){
	if(!this->covar_.empty()){
		this->covar_.release();
		this->n_ = 0;
	}
}
//==============================================================================
/** Decomposes the (covariance) matrix A into A = LL*.
 */
int Cholesky::decomposeCov(const cv::Mat &a){
	if(a.cols!=a.rows){
		std::cerr<<"For Cholesky decomposeCov: the input matrix needs to "<<\
			"be square"<<std::endl;
		exit(1);
	}
	assert(a.type()==_CV_32FC1);
	if(!this->covar_.empty()){
		this->covar_.release();
	}
	this->n_ = static_cast<unsigned>(a.rows);
	a.copyTo(this->covar_);
    for(int indy=0;indy<this->n_;++indy){
    	for(int indx=0;indx<this->n_;++indx){
        	_float sum = this->covar_.at<_float>(indy,indx);
            for(int k=indy-1;k>=0;--k){
            	sum -= this->covar_.at<_float>(indy,k)*this->covar_.at<_float>(indx,k);
            }
            if(indx==indy){
            	if(sum <= 0.0){
            		std::cerr<<"Decomposition failed,not positive defined";
            		std::abort();
            	}
            	this->covar_.at<_float>(indy,indx) = std::sqrt(sum);
            }else{
            	this->covar_.at<_float>(indx,indy) = sum/this->covar_.at<_float>(indy,indy);
            }
        }
    }

    // GET THE THE UPPER TRIANGLE
	for(int indy=0;indy<this->n_;++indy){
		for(int indx=0;indx!=indy;++indx){
			this->covar_.at<_float>(indx,indy) = 0.0;
		}
	}
	this->covar_.convertTo(this->covar_,_CV_32FC1);
    return 1;
}
//==============================================================================
/** Solves the general linear system: Ax = b and returns x.
 */
void Cholesky::solve(const cv::Mat &b,cv::Mat &x){
	assert(b.type()==_CV_32FC1);
	if(b.rows != this->n_){
		std::cerr<<"In Cholesky solve: in Ax=b,b has the wrong size"<<std::endl;
		exit(1);
	}
	x = cv::Mat::zeros(cv::Size(b.cols,this->n_),_CV_32FC1);
	for(int indx=0;indx<b.cols; ++indx){ // NOT REALLY NEEDED (JUST 1 COL)
		for(int indy=0;indy<this->n_;++indy){
			_float sum = b.at<_float>(indy,indx);
			for(int k=indy-1;k>=0;--k){
				sum -= this->covar_.at<_float>(indy,k)*x.at<_float>(k,indx);
			}
			x.at<_float>(indy,indx) = sum/this->covar_.at<_float>(indy,indy);
		}
	}

	for(int indx=0;indx<x.cols;++indx){ // NOT NEEDED (JUST 1 COL)
		for(int indy=this->n_-1;indy>=0;--indy){
			_float sum = x.at<_float>(indy,indx);
			for(int k=indy+1;k<this->n_;++k){
				sum -= this->covar_.at<_float>(k,indy)*x.at<_float>(k,indx);
			}
			x.at<_float>(indy,indx) = sum/this->covar_.at<_float>(indy,indy);
		}
	}
	x.convertTo(x,_CV_32FC1);
}
//==============================================================================
/** Solve the simplified equation Ly = b,and return y (where A=LL*).
 */
void Cholesky::solveL(const cv::Mat &b,cv::Mat &y){
	assert(b.type()==_CV_32FC1);
	if(b.rows != this->n_){
		std::cerr<<"In Cholesky solveL: in Ly=b,b has the wrong size"<<std::endl;
		exit(1);
	}
	y = cv::Mat::zeros(cv::Size(1,this->n_),_CV_32FC1);
	for(int indy=0;indy<this->n_;++indy){
		_float sum = b.at<_float>(indy,0);
		for(int indx=0;indx<indy;++indx){
			sum -= this->covar_.at<_float>(indy,indx) * y.at<_float>(indx,0);
		}
		y.at<_float>(indy,0) = sum/this->covar_.at<_float>(indy,indy);
	}
	y.convertTo(y,_CV_32FC1);
}
//==============================================================================
/** Solve the simplified equation L'y = b,and return y (where A=LL*).
 */
void Cholesky::solveLTranspose(const cv::Mat &b,cv::Mat &y){
	assert(b.type()==_CV_32FC1);
	if(b.rows != this->n_){
		std::cerr<<"In Cholesky solveLTranspose: in L'y=b,b has the wrong size"<<std::endl;
		exit(1);
	}
	y = cv::Mat::zeros(cv::Size(1,this->n_),_CV_32FC1);
	for(int indy=this->n_-1;indy>=0;--indy){
		_float sum = b.at<_float>(indy,0);
		for(int indx=indy+1;indx<this->n_;++indx){
			sum -= this->covar_.at<_float>(indx,indy) * y.at<_float>(indx,0);
		}
		y.at<_float>(indy,0) = sum/this->covar_.at<_float>(indy,indy);
	}
	y.convertTo(y,_CV_32FC1);
}
//==============================================================================
/** Returns the inverse of the covariance: A^{-1}.
 */
void Cholesky::inverse(cv::Mat &ainv){
	ainv = cv::Mat::zeros(cv::Size(this->n_,this->n_),_CV_32FC1);
	for(int indy=0;indy<this->n_;++indy){
		for(int indx=0;indx<this->n_;++indx){
			_float sum = (indx==indy?1.0:0.0);
			for(int k=indy-1;k>=indx;--k){
				sum -= this->covar_.at<_float>(indy,k) * ainv.at<_float>(indx,k);
			}
			ainv.at<_float>(indx,indy) = sum/this->covar_.at<_float>(indy,indy);
		}
	}
	for(int indy=this->n_-1;indy>=0;--indy){
		for(int indx=0;indx<=indy;++indx){
			_float sum = (indy<=indx ? 0.0 : ainv.at<_float>(indx,indy));
			for(int k=indy+1;k<this->n_;++k){
				sum -= this->covar_.at<_float>(k,indy)*ainv.at<_float>(indx,k);
			}
			ainv.at<_float>(indy,indx) = sum/this->covar_.at<_float>(indy,indy);
			ainv.at<_float>(indx,indy) = sum/this->covar_.at<_float>(indy,indy);
		}
	}
	ainv.convertTo(ainv,_CV_32FC1);
}
//==============================================================================
/** Returns the log of the determiner of the (covariance) matrix,A.
 */
_float Cholesky::logDet(){
	_float sum=0;
	for(int indy=0;indy<this->n_;++indy){
		sum += std::log(this->covar_.at<_float>(indy,indy));
	}
	return 2.0*sum;
}
//==============================================================================
unsigned Cholesky::n(){return this->n_;}
//==============================================================================
cv::Mat Cholesky::covar(){return this->covar_;}
//==============================================================================
//==============================================================================


