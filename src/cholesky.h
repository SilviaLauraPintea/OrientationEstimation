/* cholesky.h
 * Original code: Dr Gwenn Englebienne
 * Modified by: Silvia-Laura Pintea
 */
#ifndef CHOLESKY_H_
#define CHOLESKY_H_
#include <iostream>
#include <string>
#include <cmath>
#include <err.h>
#include <exception>
#include <opencv2/opencv.hpp>

/** The \c Cholesky decomposition is used to solve Ax = b; if A is symmetric and
 * positive definite => we can decompose A = LL* and instead of solving Ax = b,
 * solve Ly = b for y, and the solve L*x = y for x.
 */
class cholesky {
	//==========================================================================
	public:
		cholesky(){
			this->n = 0;
		};
		virtual ~cholesky(){ this->covar.release(); };

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
		int decomposeCov(cv::Mat a);

		/** Solves the general linear system: Ax = b and returns x.
		 */
		void solve(cv::Mat b, cv::Mat &x);

		/** Solve the simplified equation Ly = b, and return y (where A=LL*).
		 */
		void solveL(cv::Mat b, cv::Mat &y);

		/** Solve the simplified equation L'y = b, and return y (where A=LL*).
		 */
		void solveLTranspose(cv::Mat b, cv::Mat &y);

		/** Returns the inverse of the covariance: A^{-1}.
		 */
		void inverse(cv::Mat &ainv);

		/** Returns the log of the determiner of the (covariance) matrix, A.
		 */
		double logDet();
	//==========================================================================
	public:
		unsigned n;
		cv::Mat_<double> covar;
};
#endif /* CHOLESKY_H_ */
