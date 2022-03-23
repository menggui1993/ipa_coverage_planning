#include "ipa_room_exploration/room_seg.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

/**********************************************************************************************************************/
// MACRO definition

// smooth the map outline
#define SMOOTH_CONTOUR

// smooth inside
#define SMOOTH_INSIDE

/**********************************************************************************************************************/
// constant definition

// maximum size(m^2) of obstacle regions to be filtered
static float SMALL_OBSTACLE_THRESHOLD = 0.25;

// distance from robot to the wall
static int WALL_DISTANCE = 4;

// width between obstacle for robot to pass
static float PASS_WIDTH = 0.45;

// maximum erode step
static int MAX_ERODE_STEP = 20;

// maximum/minimum eroded room area
static float MAX_AREA = 10.0;
static float MIN_AREA = 0.3;

// ideal region area
static float REGION_AREA = 4.0;

// minimum area for a region to be merged
static float REGION_MERGE_THRESHOLD = 2.0;

// houghline threshold in rotation adjust
static int LINE_THRESHOLD = 80;

// when eroded region matches room condition. erode more steps to check if the region can be further split. larger value will lead to more rooms
static int SPLIT_FURTHER_STEP = 2;

// distance threshold, approximation accuracy in Douglas-Peucker algorithm
static float APPROX_EPSILON = 1.5;

// number of iterations for smooth preprocess in simplify curve
static int SMOOTH_ITERATION = 5;

// window size for calculating cornerity
static int CORNERITY_WINDOW_SIZE = 10;

/**********************************************************************************************************************/
// function declaration

void occMap2Mat(int8_t* data, int w, int h, cv::Mat& mat);

float rotateAdjust(cv::Mat& srcImg, cv::Mat& rotImg);

void collideInContours(cv::Mat& image, cv::Mat& kernel, std::vector<cv::Point>& contour);

void wavefrontRegionGrowing(cv::Mat& image, cv::Rect roi, std::vector<int>& radius);

void fillEmpty(cv::Mat& image, int roomCount);

void mergeRegions(cv::Mat& image, int& regionCount, float minArea, float resolution);

void smoothContour(std::vector<cv::Point>& contour, float epsilon, std::vector<cv::Point>& simpContour);

void DouglasPeucker(std::vector<cv::Point>& contour, std::vector<float>& cornerity, int startIdx, int endIdx, float epsilon, std::vector<cv::Point>& simpContour);

float lineAngle(int x1, int y1, int x2, int y2);
/**********************************************************************************************************************/
// function definition

void segmentHouse(cv::Mat& src, float resolution, cv::Mat& seg, float& rotation, int& roomCount)
{
    // rotate and resize image to better position for segmentation
    // cv::Mat rotImg;
    // rotation = rotateAdjust(src, rotImg);
    // cv::imshow("rot", rotImg);
    cv::Mat rotImg = src.clone();
    rotation = 0.;

    // smooth outer contour
    cv::Mat invImg;
    cv::threshold(rotImg, invImg, 250, 255, cv::THRESH_BINARY_INV);
    std::vector<std::vector<cv::Point>> tcontours;
    std::vector <cv::Vec4i> hierarchy;
    cv::findContours(invImg, tcontours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_NONE);

# ifdef SMOOTH_CONTOUR
    int level1 = -1;
    for (int i = 0; i < tcontours.size(); i++)
    {
        if (hierarchy[i][3] == -1)
        {
            level1 = i;
        }
    }
    int max_id = -1;
    int max_len = 0;
    for (int i = 0; i < tcontours.size(); i++)
    {
        if (hierarchy[i][3] == level1)
        {
            if (tcontours[i].size() > max_len)
            {
                max_id = i;
                max_len = tcontours[i].size();
            }
        }
    }
    std::vector<cv::Point> simpContour;
    smoothContour(tcontours[max_id], APPROX_EPSILON, simpContour);
    // cv::approxPolyDP(tcontours[i], simpContour, APPROX_EPSILON, true);

    // std::vector<cv::Point> corners;
    // corners.push_back(simpContour[0]);
    // int head = 0;
    // while (head < simpContour.size()-1)
    // {
    //     int tail = head+1;
    //     cv::Point v1 = simpContour[tail] - simpContour[head];

    //     while (tail+1 < simpContour.size())
    //     {
    //         cv::Point v2 = simpContour[tail+1] - simpContour[tail];
    //         float angle = lineAngle(v1.x, v1.y, v2.x, v2.y);
    //         if (angle < 5)
    //         {
    //             tail++;
    //             v1 = simpContour[tail] - simpContour[head];
    //         }
    //         else
    //         {
    //             break;
    //         }
    //     }
    //     corners.push_back(simpContour[tail]);
    //     head = tail;
    // }

    
    tcontours[max_id] = simpContour;
    cv::Mat smoothImg(rotImg.size(), CV_8UC1, cv::Scalar(200));
    cv::drawContours(smoothImg, tcontours, max_id, cv::Scalar(255), cv::FILLED);
    cv::drawContours(smoothImg, tcontours, max_id, cv::Scalar(0), 1);
    for (int i = 0; i < tcontours.size(); i++)
    {
        if (hierarchy[i][3] == max_id)
        {
#ifdef SMOOTH_INSIDE
            if (tcontours[i].size() > 25)
            {
                smoothContour(tcontours[i], APPROX_EPSILON, simpContour);
                tcontours[i] = simpContour;
            }
#endif
            cv::drawContours(smoothImg, tcontours, i, cv::Scalar(200), cv::FILLED);
            cv::drawContours(smoothImg, tcontours, i, cv::Scalar(0), 1);
        }
    }

    // cv::Mat tmp1;
    // cv::cvtColor(smoothImg, tmp1, cv::COLOR_GRAY2BGR);
    // cv::Mat tmp2 = tmp1.clone();
    // for (int i = 0; i < tcontours[max_id].size(); i++)
    // {
    //     cv::circle(tmp1, tcontours[max_id][i], 1, cv::Scalar(0,0,255), 1);
    // }
    // for (int i = 0; i < corners.size(); i++)
    // {
    //     cv::circle(tmp2, corners[i], 1, cv::Scalar(0,0,255), 1);
    // }
    // cv::imshow("s1", tmp1);
    // cv::imshow("s2", tmp2);

    // cv::imshow("smooth", smoothImg);
    // cv::imwrite("smooth.png", smoothImg);
    rotImg = smoothImg.clone();
#endif

    // generate binary image
    cv::Mat binImg;
    cv::threshold(rotImg, binImg, 250, 255, cv::THRESH_BINARY);
    cv::Rect roiRect = cv::boundingRect(binImg);

    // remove small obstacles
    cv::Mat workImg = binImg.clone();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(workImg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); i++)
    {
        float area = cv::contourArea(contours[i]) * resolution * resolution;
        if (area < SMALL_OBSTACLE_THRESHOLD)
        {
            cv::drawContours(workImg, contours, i, cv::Scalar(255), cv::FILLED);
        }
    }

    // remove unreachable region
	std::vector<cv::Point> pcont;
    int blocks = ceil(PASS_WIDTH / resolution);
    cv::Mat collideKernel;
    cv::Mat dilateKernel;
    if (blocks % 2 == 1)
    {
        collideKernel = cv::Mat::zeros(blocks, blocks, CV_8UC1);
        int center = (blocks - 1) / 2;
        collideKernel.at<uint8_t>(center, center) = 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::dilate(collideKernel, collideKernel, kernel, cv::Point(-1, -1), center);
        dilateKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(blocks, blocks));
    }
    else
    {
        collideKernel = cv::Mat::zeros(blocks+1, blocks+1, CV_8UC1);
        int center = blocks / 2;
        collideKernel.at<uint8_t>(center, center) = 1;
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        cv::dilate(collideKernel, collideKernel, kernel, cv::Point(-1, -1), center-1);
        kernel.at<uint8_t>(1, 2) = 0;
        kernel.at<uint8_t>(0, 1) = 0;
        cv::dilate(collideKernel, collideKernel, kernel);
        dilateKernel = cv::Mat::zeros(blocks+1, blocks+1, CV_8UC1);
        cv::Mat sub = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(blocks, blocks));
        sub.copyTo(dilateKernel(cv::Rect(1,0,blocks,blocks)));
    }
    // std::cout << collideKernel << "\n";
    // std::cout << dilateKernel << "\n";
	collideInContours(workImg, collideKernel, pcont);
	contours.clear();
	contours.push_back(pcont);
	cv::Mat connected = cv::Mat::zeros(workImg.size(), CV_8UC1);
	cv::drawContours(connected, contours, 0, cv::Scalar(255), cv::FILLED);
	cv::dilate(connected, connected, dilateKernel);
	workImg = workImg & connected;


    // keep eroding the image to find possible rooms
	std::vector<cv::Mat> roomMasks; // vector to save room masks
	std::vector<int> spreadRadius;  // vector to save erode steps for each room
	for (int counter = 0; counter < MAX_ERODE_STEP; counter++)
	{
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
        cv::erode(workImg, workImg, kernel);
		// std::cout << "Step" << counter << "\n";

		std::vector <std::vector<cv::Point>> contours;
		std::vector <cv::Vec4i> hierarchy;
		cv::findContours(workImg, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
		if (contours.size() != 0)
		{
			//check every contour if it fullfills the criteria of a room
			for (int idx = 0; idx < contours.size(); idx++)
			{
				if (hierarchy[idx][3] == -1)
				{
					double roomArea = resolution * resolution * cv::contourArea(contours[idx]);
					// subtract the area from the hole contours inside the found contour
					for (int hole = 0; hole < contours.size(); hole++)
					{
						if (hierarchy[hole][3] == idx)
						{
							roomArea -= resolution * resolution * cv::contourArea(contours[hole]);
						}
					}
					// std::cout << roomArea << "\n";
                    // if area fit requirements, or eroded too many steps
					if (roomArea > MIN_AREA && (roomArea < MAX_AREA || counter == MAX_ERODE_STEP-1))
					{
                        cv::Mat roomMask = cv::Mat::zeros(workImg.size(), CV_8UC1);
						cv::drawContours(roomMask, contours, idx, cv::Scalar(255), cv::FILLED);
                        // eliminate holes
						for (int hole = 0; hole < contours.size(); hole++)
						{
							if (hierarchy[hole][3] == idx)
							{
								cv::drawContours(roomMask, contours, hole, cv::Scalar(0), cv::FILLED);
							}
						}

						// erode 2 more times to check if the room can be further segmented
						bool split = false;
						cv::Mat tmp = roomMask.clone();
						for (int c = 0; c < SPLIT_FURTHER_STEP; c++)
						{
							cv::erode(tmp, tmp, cv::Mat());
							std::vector<std::vector<cv::Point>> tmpContours;
							std::vector<cv::Vec4i> tmpHierarchy;
							cv::findContours(tmp, tmpContours, tmpHierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
                            std::vector<int> valid;
							for (int f = 0; f < tmpContours.size(); f++)
							{
								if (tmpHierarchy[f][3] == -1)
								{
									double roomArea = resolution * resolution * cv::contourArea(tmpContours[f]);
									for (int hole = 0; hole < tmpContours.size(); hole++)
									{
										if (tmpHierarchy[hole][3] == f)
										{
											roomArea -= resolution * resolution * cv::contourArea(tmpContours[hole]);
										}
									}
									if (roomArea >= MIN_AREA)
									{
                                        valid.push_back(f);
									}
								}
							}
                            if (valid.size() > 1)
							{
                                for (int k = 0; k < valid.size(); k++)
                                {
                                    cv::Mat mask = cv::Mat::zeros(workImg.size(), CV_8UC1);
                                    cv::drawContours(mask, tmpContours, valid[k], cv::Scalar(255), cv::FILLED);
                                    for (int hole = 0; hole < tmpContours.size(); hole++)
                                    {
                                        if (tmpHierarchy[hole][3] == valid[k])
                                        {
                                            cv::drawContours(mask, tmpContours, hole, cv::Scalar(0), cv::FILLED);
                                        }
                                    }
                                    spreadRadius.push_back(counter + c);
                                    roomMasks.push_back(mask);
                                }
								split = true;
								break;
							}
						}

						if (!split)
						{
							spreadRadius.push_back(counter);
							roomMasks.push_back(roomMask);
						}

                        // blackout detected region on working image
                        workImg.setTo(0, roomMask);
					}
				}
			}
			// cv::imshow("erode", workImg);
			// cv::waitKey(0);
		}
		else
		{
			break;
		}
	}

    roomCount = roomMasks.size();
    // std::cout << roomCount << " rooms\n";
    if (roomCount > 64)
    {
        std::cout << "Too many rooms\n";
    }

    // assign segment result to single image
    cv::Mat segImg = cv::Mat(workImg.size(), CV_8UC1, cv::Scalar(255));
    for (uint8_t i = 0; i < roomMasks.size(); i++)
    {
        segImg.setTo(i<<2, roomMasks[i]);
    }
    cv::Mat uncMask;
    cv::inRange(connected, 0, 0, uncMask);
    segImg.setTo(3, uncMask);
    cv::Mat blkMask;
    cv::inRange(rotImg, 0, 50, blkMask);
    segImg.setTo(1, blkMask);
    cv::Mat unkMask;
    cv::inRange(rotImg, 50, 250, unkMask);
    segImg.setTo(3, unkMask);

    // expand room region
    wavefrontRegionGrowing(segImg, roiRect, spreadRadius);
    fillEmpty(segImg, roomCount);

    seg = segImg.clone();
}

void drawSegmentation(cv::Mat& src, int roomCount, cv::Mat& outImg)
{
    int segH = src.rows;
    int segW = src.cols;
    outImg = cv::Mat(segH, segW, CV_8UC3);

    std::vector<cv::Vec3b> colors;
    for (int i = 0; i < roomCount; i++)
    {
        cv::Vec3b color((rand() % 250) + 1, (rand() % 250) + 1, (rand() % 250) + 1);
        colors.push_back(color);
    }

    for (int r = 0; r < segH; r++)
    {
        for (int c = 0; c < segW; c++)
        {
            // block region
            if (src.at<uint8_t>(r, c) % 4 == 1)
            {
                outImg.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
            }
            // unknown region
            else if (src.at<uint8_t>(r, c) % 4 == 3)
            {
                outImg.at<cv::Vec3b>(r, c) = cv::Vec3b(150, 150, 150);
                // outImg.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);
            }
            // // unassigned region
            // else if (seg[r*segW+c] == 255)
            // {
            //     outImg.at<cv::Vec3b>(r, c) = cv::Vec3b(255, 255, 255);
            // }
            // room region
            else if (src.at<uint8_t>(r, c) % 4 == 0)
            {
                int id = src.at<uint8_t>(r, c) >> 2;
                outImg.at<cv::Vec3b>(r, c) = colors[id];
            }
        }
    }
}

void drawPath(cv::Mat& img, std::vector<Trace>& traces)
{
    std::vector<std::vector<cv::Point>> contours;
    for (int j = 0; j < traces.size(); j++)
    {
        std::vector<cv::Point> contour;
        for (int k = 0; k < traces[j].size(); k++)
        {
            contour.push_back(cv::Point(traces[j][k].x, traces[j][k].y));
        }
        contours.push_back(contour);
    }
    cv::drawContours(img, contours, -1, cv::Scalar(0,0,255), 1);
}


void occMap2Mat(int8_t* data, int w, int h, cv::Mat& mat)
{
    mat = cv::Mat::zeros(h, w, CV_8UC1);
    for (int y = 0; y < h; y++)
    {
        for (int x = 0; x < w; x++)
        {
            // unknown region
            if (data[y*w+x] == -1)
            {
                mat.at<uint8_t>(h-y-1, x) = 200;
            }
            // clear region
            else if (data[y*w+x] == 0)
            {
                mat.at<uint8_t>(h-y-1, x) = 255;
            }
            // otherwise block region
        }
    }
}

float rotateAdjust(cv::Mat& srcImg, cv::Mat& rotImg)
{
	cv::Mat lineImg;
	cv::threshold(srcImg, lineImg, 50, 255, cv::THRESH_BINARY_INV);
	// cv::imshow("binaryline", line_img);
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(lineImg, lines, 1, CV_PI/180, LINE_THRESHOLD);
	// cv::Mat res;
	// cv::cvtColor(srcImage, res, cv::COLOR_GRAY2BGR);
	std::vector<float> angle_bins;
	std::vector<int> angle_counts;
	for( size_t i = 0; i < lines.size(); i++ )
    {
        float rho = lines[i][0];
		float theta = lines[i][1];
		// cv::Point pt1, pt2;
		// double a = cos(theta);
		// double b = sin(theta);
		// double x0 = a*rho;
		// double y0 = b*rho;
		// pt1.x = cvRound(x0 + 1000*(-b));
		// pt1.y = cvRound(y0 + 1000*(a));
		// pt2.x = cvRound(x0 - 1000*(-b));
		// pt2.y = cvRound(y0 - 1000*(a));
		// cv::line(res, pt1, pt2, cv::Scalar(0,0,255), 1);
		theta = theta * 180 / CV_PI;
		if (theta > 175)
		{
			theta = theta - 180;
		}
		// std::cout << theta << "\n";

		bool added = false;
		for (int j = 0; j < angle_bins.size(); j++)
		{
			if (fabs(theta - angle_bins[j]) < 10)
			{
				angle_bins[j] = (angle_bins[j] * angle_counts[j] + theta) / (angle_counts[j] + 1);
				angle_counts[j] ++;
				added = true;
				break;
			}
		}
		if (!added)
		{
			angle_bins.push_back(theta);
			angle_counts.push_back(1);
		}
    }
	// cv::imshow("line", res);
	// cv::waitKey(0);

	float angle = 360;
	for (int i = 0; i < angle_bins.size(); i++)
	{
        // std::cout << angle_bins[i] << ", " << angle_counts[i] << "\n";
		if (angle_bins[i] < angle && angle_counts[i] >= 5)
		{
			angle = angle_bins[i];
		}
	}
	if (angle < 180)
	{
		if (angle > 90)
		{
			angle -= 90;
		}
        if (angle > 45)
        {
            angle -= 90;
        }
		cv::Point2f center(srcImg.cols/2.0, srcImg.rows/2.0);
		cv::Mat rot_mat = cv::getRotationMatrix2D(center, angle, 1.0);
		double abs_cos = fabs(rot_mat.at<double>(0, 0));
		double abs_sin = fabs(rot_mat.at<double>(0, 1));
		int bound_w = int(srcImg.rows * abs_sin + srcImg.cols * abs_cos);
		int bound_h = int(srcImg.rows * abs_cos + srcImg.cols * abs_sin);
		rot_mat.at<double>(0, 2) += bound_w / 2.0 - center.x;
		rot_mat.at<double>(1, 2) += bound_h / 2.0 - center.y;
		// std::cout << rot_mat << "\n";

		cv::warpAffine(srcImg, rotImg, rot_mat, cv::Size(bound_w, bound_h), cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(200));
		return angle;
	}
	else
	{
		rotImg = srcImg.clone();
		return 0.0;
    }
}

void collideInContours(cv::Mat& image, cv::Mat& kernel, std::vector<cv::Point>& contour)
{
    cv::Mat eroded;
    cv::erode(image, eroded, kernel);
    // find the longest contour
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(eroded, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
	int maxLen = 0;
	for (int i = 0; i < contours.size(); i++)
	{
		int curLen = cv::arcLength(contours[i], true);
		if (curLen > maxLen)
		{
			maxLen = curLen;
			contour = contours[i];
		}
	}
}

void wavefrontRegionGrowing(cv::Mat& image, cv::Rect roi, std::vector<int>& radius)
{
	cv::Mat spreadingMap = image.clone();
	int counter = 0;    // spreading counter
    // Stop spreading until no pixel changes
	bool finished = false;
	while (finished == false)
	{
		finished = true;
		for (int row = roi.y; row < roi.y+roi.height; ++row)
		{
			for (int column = roi.x; column < roi.x+roi.width; ++column)
			{
				if (spreadingMap.at<uint8_t>(row, column) == 255)		// unassigned pixels
				{
					//check 3x3 area around white pixel for fillcolour, if filled Pixel around fill white pixel with that colour
					bool set_value = false;
					for (int row_counter = -1; row_counter <= 1 && set_value == false; ++row_counter)
					{
						for (int column_counter = -1; column_counter <= 1 && set_value == false; ++column_counter)
						{
                            // if (row+row_counter < 0 || row+row_counter >= image.rows || column+column_counter < 0 || column+column_counter >= image.cols)
                            // {
                            //     continue;
                            // }
							int value = image.at<uint8_t>(row + row_counter, column + column_counter);
							if (value % 4 == 0 && counter <= radius[value>>2]+1)
							{
								spreadingMap.at<uint8_t>(row, column) = value;
								set_value = true;
								finished = false;
							}
						}
					}
				}
			}
		}
		image = spreadingMap.clone();
		counter ++;
	}
}

void fillEmpty(cv::Mat& image, int roomCount)
{
    // make binary image
	cv::Mat binary;
	cv::threshold(image, binary, 254, 255, cv::THRESH_BINARY);
	// find white spaces
    std::vector<std::vector<cv::Point>> whiteSpaces;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(binary, whiteSpaces, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

    // create room mask
	std::vector<cv::Mat> rooms;
	for (int i = 0; i < roomCount; i++)
	{
		cv::Mat room;
		cv::inRange(image, i<<2, i<<2, room);
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
		cv::dilate(room, room, kernel);
		rooms.push_back(room);
	}

    // go through all white spaces
	for (int i = 0; i < whiteSpaces.size(); i++)
	{
		if (hierarchy[i][3] != -1)
		{
			continue;
		}
        // create white space mask
		cv::Mat whiteSpace = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
		cv::drawContours(whiteSpace, whiteSpaces, i, cv::Scalar(255), cv::FILLED);
		for (int hole = 0; hole < whiteSpaces.size(); hole++)
		{
			if (hierarchy[hole][3] == i)
			{
				cv::drawContours(whiteSpace, whiteSpaces, hole, cv::Scalar(0), cv::FILLED);
			}
		}
		
        // find the most connected room region, assign the white space to that room
		int maxInter = 0;
        int maxId = 0;
		for (int j = 0; j < rooms.size(); j++)
		{
			cv::Mat interRegion;
			cv::bitwise_and(whiteSpace, rooms[j], interRegion);
			int inter = cv::countNonZero(interRegion);
			if (inter > maxInter)
			{
				maxInter = inter;
				maxId = j;
			}
		}
		if (maxInter > 0)
		{
			image.setTo(maxId<<2, whiteSpace);
		}
	}
}

void mergeRegions(cv::Mat& image, int& regionCount, float minArea, float resolution)
{
	std::vector<cv::Mat> regions;
	for (int i = 0; i < regionCount; i++)
	{
		cv::Mat region;
		cv::inRange(image, i<<2, i<<2, region);
		regions.push_back(region);
	}

	std::vector<int> removeIdx;
	for (int i = 0; i < regionCount; i++)
	{
		cv::Mat curMask = regions[i].clone();
		float area = cv::countNonZero(curMask) * resolution * resolution;
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
		cv::dilate(curMask, curMask, kernel);
		int maxInter = 0;
		int maxIdx = -1;
		for (int j = 0; j < regionCount; j++)
		{
			if (j == i || std::find(removeIdx.begin(), removeIdx.end(), j) != removeIdx.end())
			{
				continue;
			}
			cv::Mat interRegion;
			cv::bitwise_and(curMask, regions[j], interRegion);
			int inter = cv::countNonZero(interRegion);
			if (inter > maxInter)
			{
				maxInter = inter;
				maxIdx = j;
			}
		}
		// std::cout << area << "," << maxIdx << "\n";
		if (area < minArea && maxIdx != -1)
		{
			image.setTo(maxIdx<<2, regions[i]);
			cv::bitwise_or(regions[maxIdx], regions[i], regions[maxIdx]);
			removeIdx.push_back(i);
		}
	}

	// reassign id after merging
	int tail = regionCount - 1;
	while (std::find(removeIdx.begin(), removeIdx.end(), tail) != removeIdx.end())
	{
		tail --;
	}
	for (int i = 0; i < removeIdx.size();  i++)
	{
		if (removeIdx[i] > tail)
		{
			break;
		}
		image.setTo(removeIdx[i]<<2, regions[tail]);
		do
		{
			tail --;
		} while (std::find(removeIdx.begin(), removeIdx.end(), tail) != removeIdx.end());
	}

	regionCount -= removeIdx.size();
}

// simplify the contour
// reference to An Improved Douglas-Peucker Algorithm for Fast Curve Approximation(https://ieeexplore.ieee.org/document/5647972/)
// some details are different from the paper
void smoothContour(std::vector<cv::Point>& contour, float epsilon, std::vector<cv::Point>& simpContour)
{
    simpContour.clear();

    std::vector<cv::Point2f> fPts;
    int ptCount = contour.size();
    for (int i = 0; i < ptCount; i++)
    {
        fPts.push_back(cv::Point2f(contour[i].x, contour[i].y));
    }

    // smooth curve several iterations
    // calculate average of previous point, current point and next point
    for (int i = 0; i < SMOOTH_ITERATION; i++)
    {
        std::vector<cv::Point2f> ptsCopy(fPts);
        // since contour is closed, we need to deal with first and last points that will cross the end of vector
        ptsCopy.insert(ptsCopy.begin(), ptsCopy[ptCount-1]);
        ptsCopy.push_back(ptsCopy[0]);
        for (int j = 1; j <= ptCount; j++)
        {
            cv::Point2f prev = ptsCopy[j-1];
            cv::Point2f next = ptsCopy[j+1];
            fPts[j].x = (prev.x + ptsCopy[j].x + next.x) / 3.0;
            fPts[j].y = (prev.y + ptsCopy[j].y + next.y) / 3.0;
        }
    }

    // calculate cornerity indices (larger value, more likely to be corner)
    // cornerity defined in Boundary based corner  detection and localization using new 'cornerity' index: a robust approach(https://ieeexplore.ieee.org/document/1301477)
    // also record two points with largest cornerity
    int max1Idx = -1;
    float max1Val = 0;
    int max2Idx = -1;
    float max2Val = 0;
    std::vector<float> cornerity(ptCount);
    for (int i = 0; i < ptCount; i++)
    {
        // calculate average point inside the window
        cv::Point2f avgPt(0, 0);
        for (int w = -CORNERITY_WINDOW_SIZE; w <= CORNERITY_WINDOW_SIZE; w++)
        {
            int idx = i + w;
            if (idx < 0)
            {
                idx = idx + ptCount;
            }
            else if (idx >= ptCount)
            {
                idx = idx - ptCount;
            }
            avgPt.x += fPts[idx].x;
            avgPt.y += fPts[idx].y;
        }
        avgPt.x /= (2*CORNERITY_WINDOW_SIZE+1);
        avgPt.y /= (2*CORNERITY_WINDOW_SIZE+1);
        // cornerity is the distance between the point and the average point
        cornerity[i] = (fPts[i].x - avgPt.x) * (fPts[i].x - avgPt.x) + (fPts[i].y - avgPt.y) * (fPts[i].y - avgPt.y);

        if (cornerity[i] > max1Val)
        {
            max2Val = max1Val;
            max2Idx = max1Idx;
            max1Val = cornerity[i];
            max1Idx = i;
        }
        else if (cornerity[i] > max2Val)
        {
            max2Val = cornerity[i];
            max2Idx = i;
        }
    }

    // // find local maximum cornerity indices
    // std::vector<int> local_maxima_idx;
    // int window_size = 3;
    // for (int i = 0; i < ptCount; i++)
    // {
    //     bool local_maxima = true;
    //     for (int w = -window_size; w <= window_size; w++)
    //     {
    //         if (w == 0) continue;
    //         int idx = i + w;
    //         if (idx < 0)
    //         {
    //             idx = ptCount + idx;
    //         }
    //         else if (idx >= ptCount)
    //         {
    //             idx = idx - ptCount;
    //         }
    //         if (cornerity[i] < cornerity[idx])
    //         {
    //             local_maxima = false;
    //             break;
    //         }
    //     }
    //     if (local_maxima && cornerity[i] > 0.1)
    //     {
    //         local_maxima_idx.push_back(i);
    //     }
    // }


    // start Douglas-Peucker with the two maximum cornerity points
    simpContour.push_back(contour[max1Idx]);
    DouglasPeucker(contour, cornerity, max1Idx, max2Idx, epsilon, simpContour);
    simpContour.push_back(contour[max2Idx]);
    DouglasPeucker(contour, cornerity, max2Idx, max1Idx, epsilon, simpContour);
    simpContour.push_back(contour[max1Idx]);
    
    // std::cout << contour.size() << "," << simpContour.size() << "\n";
}

// perpendicular distance from point (xk,yk) to line with end (xi,yi) and (xj,yj)
// D = |(yk-yi)(xj-xi) - (xk-xi)(yj-yi)| / sqrt((xj-xi)^2 + (yj-yi)^2)
float perpendicularDistance(cv::Point lineStart, cv::Point lineEnd, cv::Point point)
{
    float d1 = fabs((point.y-lineStart.y) * (lineEnd.x-lineStart.x) - (point.x-lineStart.x) * (lineEnd.y-lineStart.y));
    float d2 = sqrt((lineEnd.x-lineStart.x)*(lineEnd.x-lineStart.x) + (lineEnd.y-lineStart.y)*(lineEnd.y-lineStart.y));
    return d1 / d2;
}

// improved Douglas-Peucker algorithm. 
// check all points between start and end. if maximum perpendicular distance to the line is smaller than epsilon, end;
// otherwise, find the point with largest cornerity and split the line into two segments, recursion on the two segments
void DouglasPeucker(std::vector<cv::Point>& contour, std::vector<float>& cornerity, int startIdx, int endIdx, float epsilon, std::vector<cv::Point>& simpContour)
{
    // std::cout << "Call DouglasPeucker " << startIdx << "," << endIdx << "\n";
    if (startIdx < endIdx)
    {
        for (int i = startIdx+1; i < endIdx; i++)
        {
            if (perpendicularDistance(contour[startIdx], contour[endIdx], contour[i]) > epsilon)
            {
                auto maxIter = std::max_element(cornerity.begin()+startIdx+1, cornerity.begin()+endIdx);
                int maxIdx = maxIter - cornerity.begin();

                DouglasPeucker(contour, cornerity, startIdx, maxIdx, epsilon, simpContour);
                simpContour.push_back(contour[maxIdx]);
                DouglasPeucker(contour, cornerity, maxIdx, endIdx, epsilon, simpContour);
                break;
            }
        }
    }
    else
    {
        for (int i = startIdx+1; i < endIdx+contour.size(); i++)
        {
            int ii = i;
            if (ii >= contour.size())
            {
                ii -= contour.size();
            }
            if (perpendicularDistance(contour[startIdx], contour[endIdx], contour[ii]) > epsilon)
            {
                auto rightMaxIter = std::max_element(cornerity.begin()+startIdx+1, cornerity.end());
                auto leftMaxIter = std::max_element(cornerity.begin(), cornerity.begin()+endIdx);
                int maxIdx;
                if (*rightMaxIter > *leftMaxIter || endIdx == 0)
                {
                    maxIdx = rightMaxIter - cornerity.begin();
                }
                else
                {
                    maxIdx = leftMaxIter - cornerity.begin();
                }

                DouglasPeucker(contour, cornerity, startIdx, maxIdx, epsilon, simpContour);
                simpContour.push_back(contour[maxIdx]);
                DouglasPeucker(contour, cornerity, maxIdx, endIdx, epsilon, simpContour);
                break;
            }
        }
    }
}

float lineAngle(int x1, int y1, int x2, int y2)
{
    float cosa = (x1 * x2 + y1 * y2) / (sqrt(x1*x1+y1*y1) * sqrt(x2*x2+y2*y2));
    float angle = acos(cosa) * 180.0 / CV_PI;
    return angle;
}