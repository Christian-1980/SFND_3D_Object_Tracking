#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;

        // SIFT needs gradients
        if (descriptorType.compare("DES_HOG") == 0)
        {
            normType = cv::NORM_L2;
        }

        // with all other binary descriptors
        else
        {
            normType = cv::NORM_HAMMING;
        }

        matcher = cv::BFMatcher::create(normType, true);
    }

    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // SIFT needs gradients
        if (descriptorType.compare("DES_HOG") == 0)
        {
            matcher = cv::FlannBasedMatcher::create();
        }
        // with all other binary descriptorTypes
        else
        {
            const cv::Ptr<cv::flann::IndexParams>& indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
            // std::cout << "descSource size: " << descSource.size() << " x " << descSource.cols << std::endl;
            // std::cout << "descRef size: " << descRef.size() << " x " << descRef.cols << std::endl;

        }
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    
    else if (selectorType.compare("SEL_KNN") == 0)
    { 
        // k nearest neighbors (k=2)
        int k = 2;

        // vector to store the 2 neighbours
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        // Filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it : knn_matches) {
            // Check if the  2 matches are near to each other % if so psuh to resulting vector
            if ( 2 == it.size() && (it[0].distance < minDescDistRatio * it[1].distance) ) {
                matches.push_back(it[0]);
            }
        }
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::SIFT::create();
    }
    else
    {
        // Specified descriptorType is unsupported
        throw invalid_argument(descriptorType + " is not a valid descriptorType");
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    // cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2;     // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)
    
    // Non-maximum suppression (NMS) settings
    double maxOverlap = 0.0;  // Maximum overlap between two features in %

    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Apply non-maximum suppression (NMS)
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int)dst_norm.at<float>(j, i);

            // Apply the minimum threshold for Harris cornerness response
            if (response < minResponse) continue;

            // Otherwise create a tentative new keypoint
            cv::KeyPoint newKeyPoint;
            newKeyPoint.pt = cv::Point2f(i, j);
            newKeyPoint.size = 2 * apertureSize;
            newKeyPoint.response = response;

            // Perform non-maximum suppression (NMS) in local neighbourhood around the new keypoint
            bool bOverlap = false;
            // Loop over all existing keypoints
            for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                // Test if overlap exceeds the maximum percentage allowable
                if (kptOverlap > maxOverlap) {
                    bOverlap = true;
                    // If overlapping, test if new response is the local maximum
                    if (newKeyPoint.response > (*it).response) {
                        *it = newKeyPoint;  // Replace the old keypoint
                        break;  // Exit for loop
                    }
                }
            }

            // If above response threshold and not overlapping any other keypoint
            if (!bOverlap) {
                keypoints.push_back(newKeyPoint);  // Add to keypoints list
            }
        }
    }

    if (bVis)
    {
        // visualize results
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
    
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    if (detectorType.compare("FAST") == 0) {
        
        // Standard parameters for FAST detector
        int threshold = 10; // Lower values detect more corners
        bool nonmaxSuppression = true;
        //int type = cv::FastFeatureDetector::TYPE_9_16; // Other options: TYPE_7_12, TYPE_5_8
        
        auto detector = cv::FastFeatureDetector::create(threshold, nonmaxSuppression, cv::FastFeatureDetector::TYPE_9_16);
        detector->detect(img, keypoints);

    } else if (detectorType.compare("BRISK") == 0)  {

        // Standard parameters for BRISK detector
        int threshold = 30; // Detection threshold (lower = more keypoints)
        int octaves = 3;     // Detection octaves (scale space)
        float patternScale = 1.0f; // Apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        auto detector = cv::BRISK::create(threshold, octaves, patternScale);
        detector->detect(img, keypoints);

    } else if (detectorType.compare("ORB") == 0)  {

        // Standard parameters for ORB detector
        int nfeatures = 500; // Number of desired features
        float scaleFactor = 1.2f; // Pyramid decimation ratio
        int nlevels = 8; // Number of pyramid levels
        int edgeThreshold = 31; // Size of the border where features are not detected
        int firstLevel = 0; // Which pyramid level to start from
        int wta_k = 2; // The number of points used to produce each element of the oriented BRIEF descriptor
        cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE; // HARRIS_SCORE or FAST_SCORE
        int patchSize = 31; // Size of the patch used by the oriented BRIEF descriptor
        int fastThreshold = 20; // FAST threshold

        auto detector = cv::ORB::create(nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, wta_k, scoreType, patchSize, fastThreshold);
        detector->detect(img, keypoints);

    } else if (detectorType.compare("AKAZE") == 0) {
        
        // Standard parameters for AKAZE detector
        cv::AKAZE::DescriptorType descriptorType = cv::AKAZE::DESCRIPTOR_MLDB; // Type of the extracted descriptor
        int descriptorSize = 0; // Size of the descriptor in bits. 0 -> full size
        int descriptorChannels = 3; // Number of channels in the descriptor (1, 2, 3)
        float threshold = 0.001f; // Detector response threshold to accept point
        int nOctaves = 4; // Maximum octave evolution of the image
        int nOctaveLayers = 4; // Default number of sublevels per scale level
        cv::KAZE::DiffusivityType diffusivity = cv::KAZE::DIFF_PM_G2; // Diffusivity used by the nonlinear diffusion filter

        auto detector = cv::AKAZE::create(descriptorType, descriptorSize, descriptorChannels, threshold, nOctaves, nOctaveLayers, diffusivity);
        detector->detect(img, keypoints);

    } else if (detectorType.compare("SIFT") == 0){
        
        // Standard parameters for SIFT detector
        int nfeatures = 0; // The number of best features to retain. The features are sorted by their response (score).
        int nOctaveLayers = 3; // The number of layers in each octave. 3 is the value used in D. Lowe paper.
        double contrastThreshold = 0.04; // The contrast threshold used to filter out weak features in semi-uniform (low-contrast) regions.
        double edgeThreshold = 10; // The edge threshold used to filter out bad features from edges.
        double sigma = 1.6; // The sigma of the Gaussian applied to the input image at the octave #0.

        auto detector = cv::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
        detector->detect(img, keypoints);
    
    } else {
        // Specified detectorType is unsupported
        throw invalid_argument(detectorType + " is not a valid detectorType");
    }

    if (bVis) {
        cv::Mat img_keypoints;
        cv::drawKeypoints(img, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow("Keypoints (" + detectorType + ")", img_keypoints);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }
}