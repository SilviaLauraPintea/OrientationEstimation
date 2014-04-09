/* Cholesky.h
 * Author: Silvia-Laura Pintea
 * Copyright (c) 2010-2011 Silvia-Laura Pintea. All rights reserved.
 * Feel free to use this code,but please retain the above copyright notice.
 */
#ifndef CHOLESKY_H_
#define CHOLESKY_H_
#include "eigenbackground/src/Helpers.hh"

/** The \c Cholesky decomposition is used to solve Ax = b;if A is symmetric and
 * positive definite => we can decompose A = LL* and instead of solving Ax = b,
 * solve Ly = b for y,and the solve L*x = y for x.
 */
class Cholesky {
	//==========================================================================
	public:
		Cholesky(){
			this->n_ = 0;
		};
		virtual ~Cholesky(){
			if(!this->covar_.empty()){
				this->covar_.release();
			}
		}
		Cholesky(const Cholesky &c){
			this->n_ = c.n_;
			if(!this->covar_.empty()){
				this->covar_.release();
			}
			c.covar_.copyTo(this->covar_);
		}
		Cholesky& operator=(const Cholesky &c){
			if(this == &c){return *this;}
			this->n_ = c.n_;
			if(!this->covar_.empty()){
				this->covar_.release();
			}
			c.covar_.copyTo(this->covar_);
			return *this;
		}
		/** (Re)Initializes the class variables so the same instance of the class
		 * can be used for multiple decompositions.
		 */
		void init();
		/** Checks to see if the decomposition was already done (returns true
		 * if it is done).
		 */
		bool checkDecomposition();
		/** Decomposes the (covariance) matrix A into A = LL*.
		 */
		int decomposeCov(const cv::Mat &a);
		/** Solves the general linear system: Ax = b and returns x.
		 */
		void solve(const cv::Mat &b,cv::Mat &x);
		/** Solve the simplified equation Ly = b,and return y (where A=LL*).
		 */
		void solveL(const cv::Mat &b,cv::Mat &y);
		/** Solve the simplified equation L'y = b,and return y (where A=LL*).
		 */
		void solveLTranspose(const cv::Mat &b,cv::Mat &y);
		/** Returns the inverse of the covariance: A^{-1}.
		 */
		void inverse(cv::Mat &ainv);
		/** Returns the log of the determiner of the (covariance) matrix,A.
		 */
		_float logDet();
		unsigned n();
		cv::Mat covar();
	//==========================================================================
	private:
		/** @var n_
		 * The number of elements of the covariance matrix.
		 */
		unsigned n_;
		/** @var covar_
		 * The covariance matrix.
		 */
		cv::Mat covar_;
};
#endif /* CHOLESKY_H_ */
