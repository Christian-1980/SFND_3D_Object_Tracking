
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera like in excercise
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers all bounding boxes incl. points
        
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrinked box to reduce noisy data
            cv::Rect shrinkedBox;

            shrinkedBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            shrinkedBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            shrinkedBox.width = (*it2).roi.width * (1 - shrinkFactor);
            shrinkedBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (shrinkedBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        }

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    }
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    double sum_distance = 0;

    // matrix to store the matches in the Dmatrix
    std::vector<cv::DMatch> matches;

    // Find the mean distance between all matched keypoints
    for(auto it = kptMatches.begin(); it != kptMatches.end(); ++it) {
        // internal keypoints
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);

        if(boundingBox.roi.contains(kpCurr.pt)) {
            // store it to matches
            matches.push_back(*it);
            sum_distance += cv::norm(kpCurr.pt - kpPrev.pt);
        }
    }

    // Find the threshold distance
    double distMean = sum_distance / matches.size();
    double threshold = distMean * 0.7;

    // Find the matches that are within the threshold distance to filter out outliers
    for(auto it = matches.begin(); it != matches.end(); ++it) {
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);

        if(cv::norm(kpCurr.pt - kpPrev.pt) < threshold) {
            // associates as a member of boundigBox
            boundingBox.kptMatches.push_back(*it);
        }
    }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint>& kptsPrev, std::vector<cv::KeyPoint>& kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, double& TTC) 
{        
        // vectore to store the ratio of distances for eval
        std::vector<double> ratio_distances;

        double dist_min = 100.0;
        
        for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1) {

            const cv::KeyPoint &prev_keypoint1 = kptsPrev[it1 -> queryIdx];
            const cv::KeyPoint &curr_keypoint1 = kptsCurr[it1 -> trainIdx];
            
            for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2) {

                const cv::KeyPoint &prev_keypoint2 = kptsPrev[it2 -> queryIdx];
                const cv::KeyPoint &curr_keypoint2 = kptsCurr[it2 -> trainIdx];
                
                double prev_distance = cv::norm(prev_keypoint1.pt - prev_keypoint2.pt);
                double curr_distance = cv::norm(curr_keypoint1.pt - curr_keypoint2.pt);

                if (curr_distance > std::numeric_limits<double>::epsilon() && prev_distance >= dist_min) {
                    ratio_distances.push_back(curr_distance / prev_distance);
                }
            }
        }
            
        if (ratio_distances.empty()) {
            TTC = NAN;
            std::cout << "TTC Camera: NAN (medians are equal)" << std::endl;
            return;
        }
        
        // Trick: sort the vector then find the median by middle index
        std::sort(ratio_distances.begin(), ratio_distances.end());
        long median_index = floor(ratio_distances.size() / 2);

        //std::cout << "Sorted Vector: " << ratio_distances.size() << std::endl;
        double median_distance_ratio;
            
        if (ratio_distances.size() % 2 == 0) {
            median_distance_ratio = (ratio_distances[median_index - 1] + ratio_distances[median_index]) / 2.0;
        } else {
            median_distance_ratio = ratio_distances[median_index];
        }
        // std::cout << "Median Ratio: "<< median_distance_ratio << std::endl;
            
        TTC = (-1.0 / frameRate) / (1 - median_distance_ratio);
        //std::cout << "TTC Camera: " << TTC << " s" << std::endl;
}
            

void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev, std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC) 
{
    
    double lane_width = 2.5; // European width
    
    // vector to store the distances x
    std::vector<double> x_values_prev, x_values_curr;

    // A lambda function filter_and_extract_x is used to encapsulate 
    // the common logic of filtering and extracting x-values
    auto filter_and_extract_x = [&](const std::vector<LidarPoint>& points, std::vector<double>& x_values) {
        for (const auto& point : points) {
            if (std::abs(point.y) <= lane_width / 2.0) {
                x_values.push_back(point.x);
            }
        }
        std::sort(x_values.begin(), x_values.end());
    };

    filter_and_extract_x(lidarPointsPrev, x_values_prev);
    filter_and_extract_x(lidarPointsCurr, x_values_curr);

    if (x_values_prev.empty() || x_values_curr.empty()) {
        TTC = NAN;
        std::cout << "TTC Lidar: NAN (empty point cloud)" << std::endl;
        return;
    }

    double x_median_prev = x_values_prev[x_values_prev.size() / 2];
    double x_median_curr = x_values_curr[x_values_curr.size() / 2];

    if (x_median_prev == x_median_curr) {
        TTC = NAN;
        std::cout << "TTC Lidar: NAN (medians are equal)" << std::endl;
        return;
    }

    TTC = x_median_curr * (1.0 / frameRate) / (x_median_prev - x_median_curr);
    //std::cout << "TTC Lidar: " << TTC << " s" << std::endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // A function to match bounding boxes between 2 frames wrt to the keypoint that are in those

    // Store the counts for the matches for each combination in a matrix
    cv::Mat matching_matrix = cv::Mat::zeros(prevFrame.boundingBoxes.size(), currFrame.boundingBoxes.size(), CV_32S);

    // Now loop over the frames and check the matches plus put the data; queryIdx is the previous, tranIdx the current indices of the 
    // keypoints from the DMtach Matrix
    for (const auto &match : matches){
        const cv::KeyPoint &prev_keypoint = prevFrame.keypoints[match.queryIdx];
        const cv::KeyPoint &curr_keypoint = currFrame.keypoints[match.trainIdx];

        for (const auto &prev_bbox : prevFrame.boundingBoxes){
            if (prev_bbox.roi.contains(prev_keypoint.pt)){
                for (const auto &curr_bbox : currFrame.boundingBoxes){
                    if (curr_bbox.roi.contains(curr_keypoint.pt)){
                        matching_matrix.at<int>(prev_bbox.boxID, curr_bbox.boxID) += 1;
                    }
                }
            }
        }
    }
    
    // loop over the filled matrix, rows and find the max count in second loop over cols
    for (int i = 0; i < matching_matrix.rows; ++i) {
        int max_count = 0;
        int max_id = -1;

        for (int j = 0; j < matching_matrix.cols; ++j) {
            int current_value = matching_matrix.at<int>(i, j);
            if (current_value > max_count && current_value > 0) {
                max_count = current_value;
                max_id = j;
            }
        }

        if (max_id != -1) {
            bbBestMatches[i] = max_id;
        }
    }
}
