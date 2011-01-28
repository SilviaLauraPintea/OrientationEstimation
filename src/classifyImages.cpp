/* classifyImages.cpp
 * Author: Silvia-Laura Pintea
 */
#include "classifyImages.h"
//==============================================================================
classifyImages::classifyImages(int argc, char **argv){
	if(argc != 8){
		cerr<<"Usage: classifier <bgTrain|bgModel> <calib> <prior> "<<\
			"<trainFolder> <annotationsTrain> <testFolder> <annotationsTest>"<<endl;
		exit(1);
	}else{
		for(unsigned i=0; i<argc; i++){
			if(i==4){
				this->trainFolder = (std::string)argv[i];
			}else if(i==5){
				this->annotationsTrain = (std::string)argv[i];
				argv[i] = "";
			}else if(i==6){
				this->testFolder = (std::string)argv[i];
				argv[i] = "";
			}else if(i==7){
				this->annotationsTest = (std::string)argv[i];
				argv[i] = "";
			}
		}
		argc -= 3;
		this->features = new featureDetector(argc,argv,false);
	}
}
//==============================================================================
classifyImages::~classifyImages(){
	this->trainData.release();
	this->testData.release();
	this->trainTargets.release();
	this->testTargets.release();
}
//==============================================================================
/** Creates the training data (according to the options), the labels and
 * trains the a \c GaussianProcess on the data.
 */
void classifyImages::trainGP(std::vector<std::string> options){
	this->features->init(this->trainFolder);
	this->features->run();
	this->trainData = this->features->data.clone();

	this->trainTargets = cv::Mat::zeros(cv::Size(1, this->trainData.rows),\
					this->trainData.type());

	std::vector<annotationsHandle::FULL_ANNOTATIONS> targetAnno;
	annotationsHandle::loadAnnotations(const_cast<char*>\
		(this->annotationsTrain.c_str()), targetAnno);
	for(std::size_t i=0; i<targetAnno.size(); i++){
		for(std::size_t j=0; j<targetAnno[i].annos.size(); j++){
			unsigned aSize = i*targetAnno[i].annos.size()+j;
			this->trainTargets.at<double>(aSize,1) = static_cast<double>\
				(targetAnno[i].annos[j].poses[annotationsHandle::ORIENTATION]);
			cout<<"targets "<<this->trainTargets.at<double>(aSize,1)<<endl;
		}
	}
	this->trainData.convertTo(this->trainData, cv::DataType<double>::type,1.0/255.0);
	this->trainTargets.convertTo(this->trainTargets, cv::DataType<double>::type);
	this->gp.train(this->trainData,this->trainTargets,&gaussianProcess::sqexp,1.0);
}
//==============================================================================
/** Creates the test data and applies \c GaussianProcess prediction on the test
 * data.
 */
void classifyImages::predictGP(std::vector<std::string> options){
	this->features->init(this->testFolder);
	this->features->run();
	this->testData = this->features->data.clone();

	std::vector<annotationsHandle::FULL_ANNOTATIONS> targetAnno;
	this->testTargets = cv::Mat::zeros(cv::Size(1, this->testData.rows),\
						this->testData.type());

	annotationsHandle::loadAnnotations(const_cast<char*>\
		(this->annotationsTest.c_str()),targetAnno);
	for(std::size_t i=0; i<targetAnno.size(); i++){
		for(std::size_t j=0; j<targetAnno[i].annos.size(); j++){
			unsigned aSize = i*targetAnno[i].annos.size()+j;
			this->testTargets.at<double>(aSize,1) = static_cast<double>\
				(targetAnno[i].annos[j].poses[annotationsHandle::ORIENTATION]);
			cout<<"test targets "<<this->testTargets.at<double>(aSize,1)<<endl;
		}
	}
	this->testData.convertTo(this->testData, cv::DataType<double>::type,1.0/255.0);
	this->testTargets.convertTo(this->testTargets, cv::DataType<double>::type);
	gaussianProcess::prediction predi;

	//std::cout<<this->testData.cols<<" "<<this->testData.rows<<endl;
	//std::cout<<this->testTargets.cols<<" "<<this->testTargets.rows<<endl;

	for(unsigned i=0; i<this->testData.rows; i++){
		this->gp.predict(this->testData.row(i), predi);
		std::cout<<"label: "<<this->testTargets.at<double>(i,1)<<\
			" mean:"<<predi.mean<<" variance:"<<predi.variance<<std::endl;
	}
}
//==============================================================================
int main(int argc, char **argv){
	classifyImages classi(argc, argv);
	classi.trainGP();
	classi.predictGP();

/**
 * int main(){
	cv::Mat test(10, 100, CV_64FC1);
	cv::Mat train(100, 100, CV_64FC1);
	cv::Mat targets = cv::Mat::zeros(100,1,CV_64FC1);

	train = cv::Mat::zeros(100,100,CV_64FC1);
	cv::imshow("pretrain",train);

	for(unsigned i=0; i<100; i++){
		cv::Mat stupid = train(cv::Range(i,i+1),cv::Range::all());
		cv::add(stupid, cv::Scalar(i/100.0), stupid);

		//-------------------------------
		for(unsigned j=0; j<100; j++){
			std::cout<<train.at<double>(i,j)<<std::endl;
		}
		//-------------------------------

		if(i<10){
			cv::Mat stupid2 = test(cv::Range(i,i+1),cv::Range::all());
			cv::add(stupid2, cv::Scalar((100.0-(i*10))/100.0), stupid2);
		}
		targets.at<double>(i,1) = i;
	}
	cv::imshow("train",train);
	cv::imshow("test",test);
	cv::waitKey(0);
	//-----------------------------------

	gaussianProcess gp;
	gp.train(train, targets, &gaussianProcess::expCovar, 1.0);
	gaussianProcess::prediction predi;

	cv::Mat result;
	for(unsigned i=0; i<test.rows; i++){
		cv::Mat dummy = test(cv::Range(i,i+1),cv::Range::all());
		gp.predict(dummy, predi);
		std::cout<<"label: "<<(100.0-(i*10))<<" mean:"<<predi.mean<<" variance:"<<\
			predi.variance<<std::endl;
	}
}
 *
 */
}
