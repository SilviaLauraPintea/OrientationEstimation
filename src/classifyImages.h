#ifndef CLASSIFYIMAGES_H_
#define CLASSIFYIMAGES_H_

class classifyImages {
	public:
		featureDetector features;
		//======================================================================
		classifyImages();
		virtual ~classifyImages();

		/** Neural NEtworks classification.
		 */
		void classifyNN();
};

#endif /* CLASSIFYIMAGES_H_ */
