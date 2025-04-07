/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    // #####################################
    // ### VARIABLES AND DATA STRUCTURES ###
    // #####################################

    // Detector Choice:
    string detectorType = "FAST";      // -> HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
    
    // Descriptor Choice:
    string descriptorType = "BRIEF";    // -> BRIEF, ORB, FREAK, AKAZE, SIFT

    // FLAGS
    bool flag_all_combinations = false; // to process all above Detector/Descriptor combinations

    // Matching Choice:
    string matcherType = "MAT_BF";        // -> MAT_BF, MAT_FLANN
    string matchSelectorType = "SEL_NN";       // -> SEL_NN, SEL_KNN
    

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; 
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";
    
    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";
    
    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector

    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;

    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;

    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;   

    // Visualize matching results
    bool bVis = false;            // visualize results

    // reduce search to proceeding vehicle box
    bool bFocusOnVehicle = true; // focus only on the proceeding vehicle
    cv::Rect vehicleRect(535, 180, 180, 150); // fix pixle locations

    // ############
    // ### CODE ###
    // ############

    vector<string> selectedDetectorType;
    vector<string> selectedDescriptorType;

    if (flag_all_combinations)
    {
        selectedDetectorType = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"}; //"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"
        selectedDescriptorType = {"BRIEF", "BRISK", "FREAK", "ORB", "AKAZE", "SIFT"}; // "BRIEF", "BRISK", "FREAK", "ORB", "AKAZE", "SIFT"
    } else {
        selectedDetectorType = {detectorType};
        selectedDescriptorType = {descriptorType};
    }

    /* MAIN LOOP OVER ALL Detectors */

    for (const std::string& dectType : selectedDetectorType) 
        {
            /* MAIN LOOP OVER ALL Descriptors */
        for (const std::string& descType : selectedDescriptorType)
            {
                // Catch invalid combinations 
                if ((descType.compare("AKAZE") == 0 && dectType.compare("AKAZE") != 0)  ||
                    (descType.compare("ORB") == 0 && dectType.compare("SIFT") == 0 ))
                    {
                        // AKAZE descriptor extractor works only with key-points detected with KAZE/AKAZE detectors
                        // ORB descriptor extractor does not work with the SIFT detetor
                        continue;
                    }
                
                // data buffer
                int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
                vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    
                /* MAIN LOOP OVER ALL IMAGES */

                for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex++)
                    {
                        /* LOAD IMAGE INTO BUFFER */
                        // cout << "Start processing all requested combinations for image " << imgIndex << endl;

                        // assemble filenames for current index
                        ostringstream imgNumber;
                        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;
                    
                        // load image from file 
                        cv::Mat img = cv::imread(imgFullFilename);

                        //// STUDENT ASSIGNMENT
                        //// TASK MP.1 -> replace the following code with ring buffer called dataBuffer of size dataBufferSize
                        //ImageRingBuffer dataBuffer(dataBufferSize);

                        // push image into data frame buffer
                        DataFrame frame;
                        frame.cameraImg = img;
                        dataBuffer.push_back(frame);

                        // limit data frame buffer size by removing oldest frame
                        if (dataBuffer.size() > dataBufferSize) {
                            dataBuffer.erase(dataBuffer.begin());
                        }

                        //// EOF STUDENT ASSIGNMENT
                        // //cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;
                        
                        /* DETECT & CLASSIFY OBJECTS */

                        float confThreshold = 0.2;
                        float nmsThreshold = 0.4;        
                        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                                    yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

                        //cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;

                        /* CROP LIDAR POINTS */

                        // load 3D Lidar points from file
                        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
                        std::vector<LidarPoint> lidarPoints;
                        loadLidarFromFile(lidarPoints, lidarFullFilename);

                        // remove Lidar points based on distance properties
                        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
                        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
                    
                        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

                        //cout << "#3 : CROP LIDAR POINTS done" << endl;

                        /* CLUSTER LIDAR POINT CLOUD */

                        // associate Lidar points with camera-based ROI
                        float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
                        clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

                        // Visualize 3D objects
                        bVis = false;
                        if(bVis)
                        {
                            show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
                        }
                        bVis = false;

                        //cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;

                        /* DETECT IMAGE KEYPOINTS */
                        // cout << "Start processing dectetors: " << dectType << "." << endl;
                        // convert current image to grayscale
                        cv::Mat imgGray;
                        cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);
                        
                        // Measure time for detectors
                        double detectorTime = (double)cv::getTickCount();
                        
                        // extract 2D keypoints from current image
                        vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                        
                        //// STUDENT ASSIGNMENT
                        //// TASK MP.2 -> add the following keypoint detectors in file matching2D.cpp and enable string-based selection based on detectorType
                        if (dectType.compare("SHITOMASI") == 0)
                        {
                            detKeypointsShiTomasi(keypoints, imgGray, false);
                        }
                        else if (dectType.compare("HARRIS") == 0) {
                            detKeypointsHarris(keypoints, imgGray, false);
                        }
                        else 
                        {
                            detKeypointsModern(keypoints, imgGray, dectType, false);
                        }
                        
                        detectorTime = ((double)cv::getTickCount() - detectorTime) / cv::getTickFrequency();
                        // cout << "Detector processing time: " << detectorTime << " []" << endl;

                        //// EOF STUDENT ASSIGNMENT
                        //// STUDENT ASSIGNMENT
                        //// TASK MP.3 -> only keep keypoints on the preceding vehicle
                        // only keep keypoints on the preceding vehicle
                       
                        if (bFocusOnVehicle)
                        {
                           // temp vector to write out the keypoints of interest
                           vector<cv::KeyPoint> framedKeypoints;
                           for (auto kp : keypoints) {
                               if (vehicleRect.contains(kp.pt)) framedKeypoints.push_back(kp);
                           }
                           // reframed keypoints
                           keypoints = framedKeypoints;
                        }
                        //// EOF STUDENT ASSIGNMENT

                        // optional : limit number of keypoints (helpful for debugging and learning)
                        bool bLimitKpts = false;
                        
                        if (bLimitKpts)
                        {
                            int maxKeypoints = 50;
                            if (dectType.compare("SHITOMASI") == 0)
                            { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                            }
                            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                            cout << " NOTE: Keypoints have been limited!" << endl;
                        }
                        
                        // push keypoints and descriptor for current frame to end of data buffer
                        (dataBuffer.end() - 1)->keypoints = keypoints;

                        /* EXTRACT KEYPOINT DESCRIPTORS */
                        
                        //// STUDENT ASSIGNMENT
                        
                        double descTime = (double)cv::getTickCount();
                        
                        //// TASK MP.4 -> add the following descriptors in file matching2D.cpp and enable string-based selection based on descriptorType
                        cv::Mat descriptors;
                        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, descType);
                        
                        descTime = ((double)cv::getTickCount() - descTime) / cv::getTickFrequency();
                        //// EOF STUDENT ASSIGNMENT
                        
                        // push descriptors for current frame to end of data buffer
                        (dataBuffer.end() - 1)->descriptors = descriptors;

                        // //cout << "#3 : EXTRACT DESCRIPTORS done." << endl;
                        if (dataBuffer.size() > 1) // wait until at least two images have been processed
                        {
                            /* MATCH KEYPOINT DESCRIPTORS */
                            // cout << "Starting matching process with " << matchDescriptorType << "and" << matchSelectorType << " ." << endl;
                            
                            double matchTime = (double)cv::getTickCount();
                            
                            // Catch SIFT needs gradient based and all others are going with binary
                            string matchDescriptorType;
                            
                            if (descType.compare("SIFT")==0)
                            {
                                matchDescriptorType = "DES_HOG"; // -> DES_BINARY, DES_HOG
                            } else {
                                matchDescriptorType = "DES_BINARY";
                            }
                                
                            vector<cv::DMatch> matches;

                            //// STUDENT ASSIGNMENT
                            //// TASK MP.5 -> add FLANN matching in file matching2D.cpp
                            //// TASK MP.6 -> add KNN match selection and perform descriptor distance ratio filtering with t=0.8 in file matching2D.cpp
                            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                            (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                            matches, matchDescriptorType, matcherType, matchSelectorType);
                            
                            matchTime = ((double)cv::getTickCount() - matchTime) / cv::getTickFrequency();
                            
                            // cout << "IMAGE_" << imgIndex << ",";
                            // cout << dectType << ",";
                            // cout << descType << ",";
                            // cout << keypoints.size() << ",";
                            // cout << matches.size() << ",";
                            // cout << 1000 * detectorTime / 1.0 << ",";
                            // cout << 1000 * descTime / 1.0 << ",";
                            // cout << 1000 * matchTime / 1.0 << ",";
                            // cout << (1000 * detectorTime / 1.0) + (1000 * descTime / 1.0) + (1000 * matchTime / 1.0)<< endl;
                            //// EOF STUDENT ASSIGNMENT

                            // store matches in current data frame
                            (dataBuffer.end() - 1)->kptMatches = matches;
                            
                            //cout << "#4 : MATCH KEYPOINT DESCRIPTORS done." << endl;

                            // cout << "Finalized processing following combinatio: " << dectType << "/ " << selectedDescriptorType[i] << "/ " << matchDescriptorType << "/ " << matchSelectorType << endl;
                            // visualize matches between current and previous image
                            if (bVis)
                            {
                                // here there need to be setting the "unset GTK_PATH" before it works under my local setup!
                                cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                                cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,            
                                                (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                                matches, matchImg,
                                                cv::Scalar::all(-1), cv::Scalar::all(-1),
                                                vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
                                string windowName = "Matching keypoints between two camera images";
                                cv::namedWindow(windowName, cv::WINDOW_AUTOSIZE);
                                cv::imshow(windowName, matchImg);
                                cout << "Press key to continue to next image" << endl;
                                cv::waitKey(0); // wait for key to be pressed
                            }
                            bVis=false;

                            /* TRACK 3D OBJECT BOUNDING BOXES */

                            //// STUDENT ASSIGNMENT
                            //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
                            map<int, int> bbBestMatches;
                            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
                            //// EOF STUDENT ASSIGNMENT

                            // store matches in current data frame
                            (dataBuffer.end()-1)->bbMatches = bbBestMatches;

                            //cout << "#5 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


                            /* COMPUTE TTC ON OBJECT IN FRONT */

                            // loop over all BB match pairs
                            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                            {
                                // find bounding boxes associates with current match
                                BoundingBox *prevBB, *currBB;
                                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                                {
                                    if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                                    {
                                        currBB = &(*it2);
                                    }
                                }

                                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                                {
                                    if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                                    {
                                        prevBB = &(*it2);
                                    }
                                }

                                // compute TTC for current match
                                if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                                {
                                    //// STUDENT ASSIGNMENT
                                    //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                                    double ttcLidar; 
                                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                                    //// EOF STUDENT ASSIGNMENT

                                    //// STUDENT ASSIGNMENT
                                    //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                                    //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                                    double ttcCamera;
                                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                                    //// EOF STUDENT ASSIGNMENT

                                    double diffTTC = std::abs(ttcLidar - ttcCamera);

                                    cout << "IMAGE_" << imgIndex << ",";
                                    cout << dectType << ",";
                                    cout << descType << ",";
                                    cout << ttcLidar<< ",";
                                    cout << ttcCamera<< ",";
                                    cout << diffTTC << endl;


                                    bVis = false;
                                    if (bVis)
                                    {
                                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                                        
                                        char str[200];
                                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", ttcLidar, ttcCamera);
                                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                                        string windowName = "Final Results : TTC";
                                        cv::namedWindow(windowName, 4);
                                        cv::imshow(windowName, visImg);
                                        cout << "Press key to continue to next frame" << endl;
                                        cv::waitKey(0);
                                    }
                                    bVis = false;

                                } // eof TTC computation
                            } // eof loop over all BB matches 

                            bVis = false;
                                    
                        }
                    }
            }
    } // eof loop over all images

    return 0;
}