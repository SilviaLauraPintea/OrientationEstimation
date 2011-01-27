/* classifyImages.cpp
 * Author: Silvia-Laura Pintea
 */
#include "classifyImages.h"
//==============================================================================
classifyImages::classifyImages(int argc, char **argv){
	cerr<<">>>>>>>>>>>>>>>>"<<argc<<endl;
	if(argc != 6){
		cerr<<"Usage: classifier <trainFolder> <testFolder> <bgTrain|bgModel> "<<\
			"<calib> <prior>"<<endl;
		exit(1);
	}else{
		char** argvTest = argv;
		for(unsigned i=0; i<argc; i++){
			cout<<(std::string)argv[i]<<endl;

			if(i==1){
				this->trainFolder = (std::string)argv[i];
				argvTest[i] = "";
			}else if(i==2){
				this->testFolder = (std::string)argv[i];
				argv[i] = "";
			}
		}
		argc--;

		for(unsigned i=0; i<argc; i++){
			cout<<(std::string)argv[i]<<endl;
			cout<<(std::string)argvTest[i]<<endl;
		}

		this->testFeatures = new featureDetector(argc,argv,false);
		this->trainFeatures = new featureDetector(argc,argvTest,false);
	}
}
//==============================================================================
classifyImages::~classifyImages(){
	this->trainData.release();
	this->testData.release();
}
//==============================================================================
/** Creates the training data/test data.
 */
void classifyImages::createData(std::vector<std::string> options){
	if(options[0] == "test"){
		this->testFeatures->run();
	}else{
		this->trainFeatures->run();
	}
}
//==============================================================================
int main(int argc, char **argv){
	classifyImages classfier(int argc, char **argv);
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
