/* cholesky.h
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
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
class cholesky {
	//==========================================================================
	public:
		cholesky(){
			this->n = 0;
		};
		virtual ~cholesky(){
			if(!this->covar.empty()){
				this->covar.release();
			}
		}
		cholesky(const cholesky &c){
			this->n = c.n;
			if(!this->covar.empty()){
				this->covar.release();
			}
			c.covar.copyTo(this->covar);
		}
		cholesky& operator=(const cholesky &c){
			if(this == &c){return *this;}
			this->n = c.n;
			if(!this->covar.empty()){
				this->covar.release();
			}
			c.covar.copyTo(this->covar);
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
		float logDet();
	//==========================================================================
	public:
		unsigned n;
		cv::Mat covar;
};
#endif /* CHOLESKY_H_ */
