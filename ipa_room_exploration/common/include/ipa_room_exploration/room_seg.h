#ifndef _ROOM_SEG_H_
#define _ROOM_SEG_H_

#include <stdint.h>
#include <vector>
#include <opencv2/core.hpp>

/**
 * @brief 2D int点
 * 
 */
struct PointInt
{
    PointInt():x(0),y(0){}
    PointInt(int x, int y):x(x),y(y){}
    int x;
    int y;
};

/**
 * @brief 2D float点
 * 
 */
struct PointFloat
{
    PointFloat():x(0),y(0){}
    PointFloat(float x, float y):x(x),y(y){}
    float x;
    float y;
};

/** 轨迹。由点序列组成 */
typedef std::vector<PointInt> Trace;

/**
 * @brief 将地图分割为一个个房间。输出结果保存为地图，每个点一个字节，高6位标识房间ID（0-63），低2位定义如下00：可通行区域，01：障碍物，11：未知区域。
 *        会对房间做一个角度校正预处理，因此输入输出地图大小不同。
 * 
 * @param[in] src 输入地图。-1表示灰色未知区域，0表示白色可通行区域，100表示黑色障碍区域。
 * @param[in] srcW 输入地图宽度
 * @param[in] srcH 输入地图高度
 * @param[in] resolution 地图分辨率
 * @param[out] pSeg 输出分割结果地图（需要在函数外释放内存）
 * @param[out] segW 输出地图宽度
 * @param[out] segH 输出地图高度
 * @param[out] rotation 图像旋转角度（角度制，逆时针为正）
 * @param[out] roomCount 分割得到房间数量
 */
void segmentHouse(cv::Mat& src, float resolution, cv::Mat& seg, float& rotation, int& roomCount);

/**
 * @brief 根据分割结果，绘制可视化分割图像 
 * 
 * @param[in] seg 输入分割结果地图
 * @param[in] segW 输入地图宽度
 * @param[in] segH 输入地图高度
 * @param[in] roomCount 房间数量
 * @param[out] outImg 输出图像
 */
void drawSegmentation(cv::Mat& src, int roomCount, cv::Mat& outImg);

/**
 * @brief 将轨迹路径画到图像上
 * 
 * @param[in,out] img 输入输出图像 
 * @param[in] traces 轨迹路径数组
 */
void drawPath(cv::Mat& img, std::vector<Trace>& traces);

#endif