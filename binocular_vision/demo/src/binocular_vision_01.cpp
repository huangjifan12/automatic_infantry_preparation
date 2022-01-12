// #include <iostream>
// #include <opencv2/core/core.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/highgui/highgui.hpp>

// using namespace cv;
// using namespace std;

// int main ( int argc, char** argv )
// {
//     if ( argc != 3 )
//     {
//         cout<<"usage: feature_extraction img1 img2"<<endl;
//         return 1;
//     }
//     //-- 读取图像
//     Mat img_1 = imread ( argv[1], 0 );
//     Mat img_2 = imread ( argv[2], 0 );

//     //-- 初始化
//     std::vector<KeyPoint> keypoints_1, keypoints_2;
//     Mat descriptors_1, descriptors_2;
//     Ptr<FeatureDetector> detector = ORB::create();
//     Ptr<DescriptorExtractor> descriptor = ORB::create();
//     // Ptr<FeatureDetector> detector =
//     FeatureDetector::create(detector_name);
//     // Ptr<DescriptorExtractor> descriptor =
//     DescriptorExtractor::create(descriptor_name); Ptr<DescriptorMatcher>
//     matcher  = DescriptorMatcher::create ( "BruteForce-Hamming" );

//     //-- 第一步:检测 Oriented FAST 角点位置
//     detector->detect ( img_1,keypoints_1 );
//     detector->detect ( img_2,keypoints_2 );

//     //-- 第二步:根据角点位置计算 BRIEF 描述子
//     descriptor->compute ( img_1, keypoints_1, descriptors_1 );
//     descriptor->compute ( img_2, keypoints_2, descriptors_2 );

//     Mat outimg1;
//     drawKeypoints( img_1, keypoints_1, outimg1, Scalar::all(-1),
//     DrawMatchesFlags::DEFAULT ); imshow("ORB特征点",outimg1);

//     //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
//     vector<DMatch> matches;
//     //BFMatcher matcher ( NORM_HAMMING );
//     matcher->match ( descriptors_1, descriptors_2, matches );

//     //-- 第四步:匹配点对筛选
//     double min_dist=10000, max_dist=0;

//     //找出所有匹配之间的最小距离和最大距离,
//     即是最相似的和最不相似的两组点之间的距离 for ( int i = 0; i <
//     descriptors_1.rows; i++ )
//     {
//         double dist = matches[i].distance;
//         if ( dist < min_dist ) min_dist = dist;
//         if ( dist > max_dist ) max_dist = dist;
//     }

//     // 仅供娱乐的写法
//     min_dist = min_element( matches.begin(), matches.end(), [](const DMatch&
//     m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;
//     max_dist = max_element( matches.begin(), matches.end(), [](const DMatch&
//     m1, const DMatch& m2) {return m1.distance<m2.distance;} )->distance;

//     printf ( "-- Max dist : %f \n", max_dist );
//     printf ( "-- Min dist : %f \n", min_dist );

//     //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
//     std::vector< DMatch > good_matches;
//     for ( int i = 0; i < descriptors_1.rows; i++ )
//     {
//         if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
//         {
//             good_matches.push_back ( matches[i] );
//         }
//     }

//     //-- 第五步:绘制匹配结果
//     Mat img_match;
//     Mat img_goodmatch;
//     drawMatches ( img_1, keypoints_1, img_2, keypoints_2, matches, img_match
//     ); drawMatches ( img_1, keypoints_1, img_2, keypoints_2, good_matches,
//     img_goodmatch ); imshow ( "所有匹配点对", img_match ); imshow (
//     "优化后匹配点对", img_goodmatch ); waitKey(0);

//     return 0;
// }
/////////////////////////////////////////////////////
// #include <iostream>
// #include <opencv2/core/core.hpp>
// #include <opencv2/features2d/features2d.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/calib3d/calib3d.hpp>
// // #include "extra.h" // used in opencv2
// using namespace std;
// using namespace cv;

// void find_feature_matches (
//     const Mat& img_1, const Mat& img_2,
//     std::vector<KeyPoint>& keypoints_1,
//     std::vector<KeyPoint>& keypoints_2,
//     std::vector< DMatch >& matches );

// void pose_estimation_2d2d (
//     const std::vector<KeyPoint>& keypoints_1,
//     const std::vector<KeyPoint>& keypoints_2,
//     const std::vector< DMatch >& matches,
//     Mat& R, Mat& t );

// void triangulation (
//     const vector<KeyPoint>& keypoint_1,
//     const vector<KeyPoint>& keypoint_2,
//     const std::vector< DMatch >& matches,
//     const Mat& R, const Mat& t,
//     vector<Point3d>& points
// );

// // 像素坐标转相机归一化坐标
// Point2f pixel2cam( const Point2d& p, const Mat& K );

// int main ( int argc, char** argv )
// {
//     if ( argc != 3 )
//     {
//         cout<<"usage: triangulation img1 img2"<<endl;
//         return 1;
//     }
//     //-- 读取图像
//     Mat img_1 = imread ( argv[1], 1 );
//     Mat img_2 = imread ( argv[2], 1 );

//     vector<KeyPoint> keypoints_1, keypoints_2;
//     vector<DMatch> matches;
//     find_feature_matches ( img_1, img_2, keypoints_1, keypoints_2, matches );
//     cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

//     //-- 估计两张图像间运动
//     Mat R,t;
//     pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );

//     //-- 三角化
//     vector<Point3d> points;
//     triangulation( keypoints_1, keypoints_2, matches, R, t, points );

//     //-- 验证三角化点与特征点的重投影关系
//     Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0,
//     1 ); for ( int i=0; i<matches.size(); i++ )
//     {
//         Point2d pt1_cam = pixel2cam( keypoints_1[ matches[i].queryIdx ].pt, K
//         ); Point2d pt1_cam_3d(
//             points[i].x/points[i].z,
//             points[i].y/points[i].z
//         );

//         cout<<"point in the first camera frame: "<<pt1_cam<<endl;
//         cout<<"point projected from 3D "<<pt1_cam_3d<<",
//         d="<<points[i].z<<endl;

//         // 第二个图
//         Point2f pt2_cam = pixel2cam( keypoints_2[ matches[i].trainIdx ].pt, K
//         ); Mat pt2_trans = R*( Mat_<double>(3,1) << points[i].x, points[i].y,
//         points[i].z ) + t; pt2_trans /= pt2_trans.at<double>(2,0);
//         cout<<"point in the second camera frame: "<<pt2_cam<<endl;
//         cout<<"point reprojected from second frame: "<<pt2_trans.t()<<endl;
//         cout<<endl;
//     }

//     return 0;
// }

// void find_feature_matches ( const Mat& img_1, const Mat& img_2,
//                             std::vector<KeyPoint>& keypoints_1,
//                             std::vector<KeyPoint>& keypoints_2,
//                             std::vector< DMatch >& matches )
// {
//     //-- 初始化
//     Mat descriptors_1, descriptors_2;
//     // used in OpenCV3
//     Ptr<FeatureDetector> detector = ORB::create();
//     Ptr<DescriptorExtractor> descriptor = ORB::create();
//     // use this if you are in OpenCV2
//     // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
//     // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create (
//     "ORB" ); Ptr<DescriptorMatcher> matcher  =
//     DescriptorMatcher::create("BruteForce-Hamming");
//     //-- 第一步:检测 Oriented FAST 角点位置
//     detector->detect ( img_1,keypoints_1 );
//     detector->detect ( img_2,keypoints_2 );

//     //-- 第二步:根据角点位置计算 BRIEF 描述子
//     descriptor->compute ( img_1, keypoints_1, descriptors_1 );
//     descriptor->compute ( img_2, keypoints_2, descriptors_2 );

//     //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
//     vector<DMatch> match;
//    // BFMatcher matcher ( NORM_HAMMING );
//     matcher->match ( descriptors_1, descriptors_2, match );

//     //-- 第四步:匹配点对筛选
//     double min_dist=10000, max_dist=0;

//     //找出所有匹配之间的最小距离和最大距离,
//     即是最相似的和最不相似的两组点之间的距离 for ( int i = 0; i <
//     descriptors_1.rows; i++ )
//     {
//         double dist = match[i].distance;
//         if ( dist < min_dist ) min_dist = dist;
//         if ( dist > max_dist ) max_dist = dist;
//     }

//     printf ( "-- Max dist : %f \n", max_dist );
//     printf ( "-- Min dist : %f \n", min_dist );

//     //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
//     for ( int i = 0; i < descriptors_1.rows; i++ )
//     {
//         if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
//         {
//             matches.push_back ( match[i] );
//         }
//     }
// }

// void pose_estimation_2d2d (
//     const std::vector<KeyPoint>& keypoints_1,
//     const std::vector<KeyPoint>& keypoints_2,
//     const std::vector< DMatch >& matches,
//     Mat& R, Mat& t )
// {
//     // 相机内参,TUM Freiburg2
//     Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0,
//     1 );

//     //-- 把匹配点转换为vector<Point2f>的形式
//     vector<Point2f> points1;
//     vector<Point2f> points2;

//     for ( int i = 0; i < ( int ) matches.size(); i++ )
//     {
//         points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
//         points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
//     }

//     //-- 计算基础矩阵
//     Mat fundamental_matrix;
//     fundamental_matrix = findFundamentalMat ( points1, points2, FM_8POINT );
//     cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

//     //-- 计算本质矩阵
//     Point2d principal_point ( 325.1, 249.7 );
//     //相机主点, TUM dataset标定值 int focal_length = 521;
//     //相机焦距, TUM dataset标定值 Mat essential_matrix; essential_matrix =
//     findEssentialMat ( points1, points2, focal_length, principal_point );
//     cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

//     //-- 计算单应矩阵
//     Mat homography_matrix;
//     homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
//     cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

//     //-- 从本质矩阵中恢复旋转和平移信息.
//     recoverPose ( essential_matrix, points1, points2, R, t, focal_length,
//     principal_point ); cout<<"R is "<<endl<<R<<endl; cout<<"t is
//     "<<endl<<t<<endl;
// }

// void triangulation (
//     const vector< KeyPoint >& keypoint_1,
//     const vector< KeyPoint >& keypoint_2,
//     const std::vector< DMatch >& matches,
//     const Mat& R, const Mat& t,
//     vector< Point3d >& points )
// {
//     Mat T1 = (Mat_<float> (3,4) <<
//         1,0,0,0,
//         0,1,0,0,
//         0,0,1,0);
//     Mat T2 = (Mat_<float> (3,4) <<
//         R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2),
//         t.at<double>(0,0), R.at<double>(1,0), R.at<double>(1,1),
//         R.at<double>(1,2), t.at<double>(1,0), R.at<double>(2,0),
//         R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
//     );

//     Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0,
//     1 ); vector<Point2f> pts_1, pts_2; for ( DMatch m:matches )
//     {
//         // 将像素坐标转换至相机坐标
//         pts_1.push_back ( pixel2cam( keypoint_1[m.queryIdx].pt, K) );
//         pts_2.push_back ( pixel2cam( keypoint_2[m.trainIdx].pt, K) );
//     }

//     Mat pts_4d;
//     cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );

//     // 转换成非齐次坐标
//     for ( int i=0; i<pts_4d.cols; i++ )
//     {
//         Mat x = pts_4d.col(i);
//         x /= x.at<float>(3,0); // 归一化
//         Point3d p (
//             x.at<float>(0,0),
//             x.at<float>(1,0),
//             x.at<float>(2,0)
//         );
//         points.push_back( p );
//     }
// }

// Point2f pixel2cam ( const Point2d& p, const Mat& K )
// {
//     return Point2f
//     (
//         ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0),
//         ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1)
//     );
// }
//////////////////////////////////////////////////////////////
// #include <iostream>
// #include <fstream>
// #include <map>
// #include <vector>
// #include <cmath>

// #include <opencv2/highgui.hpp>
// #include <opencv2/calib3d.hpp>
// #include <opencv2/imgproc.hpp>

// using namespace std;

// //****这里设置三种颜色来让程序运行的更醒目****
// #define WHITE "\033[37m"
// #define BOLDRED "\033[1m\033[31m"
// #define BOLDGREEN "\033[1m\033[32m"

// //****相机内参****
// struct CAMERA_INTRINSIC_PARAMETERS
// {
//     double cx,cy,fx,fy,baseline,scale;
// };

// struct FILE_FORM
// {
//     cv::Mat left,right;//存放左右视图数据
//     string dispname,depthname,colorname;//存放三种输出数据的文件名
// };

// //****读取配置文件****
// class ParameterReader
// {
// public:
//     ParameterReader(string filename = "./parameters.txt")//配置文件目录
//     {
//         ifstream fin(filename.c_str());
//         if(!fin)
//         {
//             cerr<<BOLDRED"can't find parameters file!"<<endl;
//             return;
//         }
//         while(!fin.eof())
//         {
//             string str;
//             getline(fin,str);
//             if(str[0] == '#')//遇到’#‘视为注释
//             {
//                 continue;
//             }

//             int pos = str.find("=");//遇到’=‘将其左右值分别赋给key和alue
//             if (pos == -1)
//             {
//                 continue;
//             }
//             string key = str.substr(0,pos);
//             string value = str.substr(pos+1,str.length());
//             data[key] = value;

//             if (!fin.good())
//             {
//                 break;
//             }
//         }
//     }
//     string getData(string key)//获取配置文件参数值
//     {
//         map<string,string>::iterator iter = data.find(key);
//         if (iter == data.end())
//         {
//             cerr<<BOLDRED"can't find:"<<key<<" parameters!"<<endl;
//             return string("NOT_FOUND");
//         }
//         return iter->second;
//     }
// public:
//     map<string,string> data;
// };

// FILE_FORM readForm(int index,ParameterReader
// pd);//存入当前序列左右视图数据和三种输出结果文件名 void stereoSGBM(cv::Mat
// lpng,cv::Mat rpng,string filename,cv::Mat &disp);//SGBM方法获取视差图 void
// stereoBM(cv::Mat lpng,cv::Mat rpng,string filename,cv::Mat
// &disp);//BM方法获取视差图 void disp2Depth(cv::Mat disp,cv::Mat
// &depth,CAMERA_INTRINSIC_PARAMETERS camera);//由视察图计算深度图

// int main(int argc, char const *argv[])
// {
// 	ParameterReader pd;//读取配置文件
// 	CAMERA_INTRINSIC_PARAMETERS camera;//相机内参结构赋值
// 	camera.fx = atof(pd.getData("camera.fx").c_str());
//     camera.fy = atof(pd.getData("camera.fy").c_str());
//     camera.cx = atof(pd.getData("camera.cx").c_str());
//     camera.cy = atof(pd.getData("camera.cy").c_str());
//     camera.baseline = atof(pd.getData("camera.baseline").c_str());
//     camera.scale = atof(pd.getData("camera.scale").c_str());

//     int startIndex = atoi(pd.getData("start_index").c_str());//起始序列
//     int endIndex = atoi(pd.getData("end_index").c_str());//截止序列

//     bool is_color = pd.getData("is_color") ==
//     string("yes");//判断是否要输出彩色深度图
//     cout<<BOLDRED"......START......"<<endl;
//     for (int currIndex = startIndex;currIndex <
//     endIndex;currIndex++)//从起始序列循环至截止序列
//     {
//     	cout<<BOLDGREEN"Reading file： "<<currIndex<<endl;
//     	FILE_FORM form =
//     readForm(currIndex,pd);//获取当前序列的左右视图以及输出结果文件名
//      	cv::Mat disp,depth,color;

//         //****判断使用何种算法计算视差图****
//     	if (pd.getData("algorithm") == string("SGBM"))
//     	{
//     		stereoSGBM(form.left,form.right,form.dispname,disp);
//     	}
//     	else if (pd.getData("algorithm") == string("BM"))
//     	{
//     		stereoBM(form.left,form.right,form.dispname,disp);
//     	}
//     	else
//     	{
//     		cout<<BOLDRED"Algorithm is wrong!"<<endl;
//     		return 0;
//     	}

// 	    disp2Depth(disp,depth,camera);//输出深度图
// 	    cv::imwrite(form.depthname,depth);
// 	    cout<<WHITE"Depth saved!"<<endl;

//         //****判断是否输出彩色深度图****
// 	    if (is_color)
// 	    {
// 	    	cv::applyColorMap(depth,color,cv::COLORMAP_JET);//转彩色图
// 	    	cv::imwrite(form.colorname,color);
// 	    	cout<<WHITE"Color saved!"<<endl;
// 	    }
//     }
//     cout<<BOLDRED"......END......"<<endl;

// 	return 0;
// }

// FILE_FORM readForm(int index,ParameterReader pd)
// {
//     FILE_FORM f;
//     string lpngDir = pd.getData("left_dir");//获取左视图输入目录名
//     string rpngDir = pd.getData("right_dir");//获取右视图输入目录名
//     string dispDir = pd.getData("disp_dir");//获取视差图输出目录名
//     string depthDir = pd.getData("depth_dir");//获取深度图输出目录名
//     string colorDir = pd.getData("color_dir");//获取彩色深度图输出目录名
//     string rgbExt = pd.getData("rgb_extension");//获取图片数据格式后缀名

//     //输出当前文件序号（使用的TUM数据集，其双目视图命名从000000至004540，详情参看博文末尾ps）
//     string numzero;
//     if ( index >= 0 && index <= 9 )
//     {
//         numzero = "00000";
//     }
//     else if ( index >= 10 && index <= 99 )
//     {
//         numzero = "0000";
//     }
//     else if ( index >= 100 && index <= 999 )
//     {
//         numzero = "000";
//     }
//     else if ( index >= 1000 && index <= 9999 )
//     {
//         numzero = "00";
//     }
//     else if ( index >= 10000 && index <= 99999 )
//     {
//         numzero = "0";
//     }

//     //获取左视图文件名
//     stringstream ss;
//     ss<<lpngDir<<numzero<<index<<rgbExt;
//     string filename;
//     ss>>filename;
//     f.left = cv::imread(filename,0);//这里要获取单通道数据

//     //获取右视图文件名
//     ss.clear();
//     filename.clear();
//     ss<<rpngDir<<numzero<<index<<rgbExt;
//     ss>>filename;
//     f.right = cv::imread(filename,0);//这里要获取单通道数据

//     //获取深度图输出文件名
//     ss.clear();
//     filename.clear();
//     ss<<depthDir<<index<<rgbExt;
//     ss>>filename;
//     f.depthname = filename;

//     //获取视差图输出文件名
//     ss.clear();
//     filename.clear();
//     ss<<dispDir<<index<<rgbExt;
//     ss>>filename;
//     f.dispname = filename;

//     //获取彩色深度图输出文件名
//     ss.clear();
//     filename.clear();
//     ss<<colorDir<<index<<rgbExt;
//     ss>>filename;
//     f.colorname = filename;

//     return f;
// }

// void stereoSGBM(cv::Mat lpng,cv::Mat rpng,string filename,cv::Mat &disp)
// {
//     disp.create(lpng.rows,lpng.cols,CV_16S);
//     cv::Mat disp1 = cv::Mat(lpng.rows,lpng.cols,CV_8UC1);
//     cv::Size imgSize = lpng.size();
//     cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();

//     int nmDisparities = ((imgSize.width / 8) + 15) & -16;//视差搜索范围
//     int pngChannels = lpng.channels();//获取左视图通道数
//     int winSize = 5;

//     sgbm->setPreFilterCap(31);//预处理滤波器截断值
//     sgbm->setBlockSize(winSize);//SAD窗口大小
//     sgbm->setP1(8*pngChannels*winSize*winSize);//控制视差平滑度第一参数
//     sgbm->setP2(32*pngChannels*winSize*winSize);//控制视差平滑度第二参数
//     sgbm->setMinDisparity(0);//最小视差
//     sgbm->setNumDisparities(nmDisparities);//视差搜索范围
//     sgbm->setUniquenessRatio(5);//视差唯一性百分比
//     sgbm->setSpeckleWindowSize(100);//检查视差连通区域变化度的窗口大小
//     sgbm->setSpeckleRange(32);//视差变化阈值
//     sgbm->setDisp12MaxDiff(1);//左右视差图最大容许差异
//     sgbm->setMode(cv::StereoSGBM::MODE_SGBM);//采用全尺寸双通道动态编程算法
//     sgbm->compute(lpng,rpng,disp);

//     disp.convertTo(disp1,CV_8U,255 / (nmDisparities*16.));//转8位

//     cv::imwrite(filename,disp1);
//     cout<<WHITE"Disp saved!"<<endl;
// }

// void stereoBM(cv::Mat lpng,cv::Mat rpng,string filename,cv::Mat &disp)
// {
//     disp.create(lpng.rows,lpng.cols,CV_16S);
//     cv::Mat disp1 = cv::Mat(lpng.rows,lpng.cols,CV_8UC1);
//     cv::Size imgSize = lpng.size();
//     cv::Rect roi1,roi2;
//     cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16,9);

//     int nmDisparities = ((imgSize.width / 8) + 15) & -16;//视差搜索范围
//     bm->setPreFilterType(1);//预处理滤波器类型
//     bm->setPreFilterSize(9);//预处理滤波器窗口大小
//     bm->setPreFilterCap(31);//预处理滤波器截断值
//     bm->setBlockSize(9);//SAD窗口大小
//     bm->setMinDisparity(0);//最小视差
//     bm->setNumDisparities(nmDisparities);//视差搜索范围
//     bm->setTextureThreshold(10);//低纹理区域的判断阈值
//     bm->setUniquenessRatio(5);//视差唯一性百分比
//     bm->setSpeckleWindowSize(100);//检查视差连通区域变化度窗口大小
//     bm->setSpeckleRange(32);//视差变化阈值
//     bm->setROI1(roi1);
//     bm->setROI2(roi2);
//     bm->setDisp12MaxDiff(1);//左右视差图最大容许差异
//     bm->compute(lpng,rpng,disp);

//     disp.convertTo(disp1,CV_8U,255 / (nmDisparities*16.));

//     cv::imwrite(filename,disp1);
//     cout<<WHITE"Disp saved!"<<endl;
// }

// void disp2Depth(cv::Mat disp,cv::Mat &depth,CAMERA_INTRINSIC_PARAMETERS
// camera)
// {
//         depth.create(disp.rows,disp.cols,CV_8UC1);
//         cv::Mat depth1 = cv::Mat(disp.rows,disp.cols,CV_16S);
//         for (int i = 0;i < disp.rows;i++)
//         {
//             for (int j = 0;j < disp.cols;j++)
//             {
//                 if (!disp.ptr<ushort>(i)[j])//防止除0中断
//                     continue;
//                 depth1.ptr<ushort>(i)[j] = camera.scale * camera.fx *
//                 camera.baseline / disp.ptr<ushort>(i)[j];
//             }
//         }
//         depth1.convertTo(depth,CV_8U,1./256);//转8位
// }

// //原文链接：https://blog.csdn.net/ADDfish/article/details/1104335981

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int imageWidth = 1920;
const int imageHeight = 1024;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;
Mat Rl, Rr, Pl, Pr, Q;
Mat xyz;

Point origin;
Rect selection;
bool selectObject = false;

Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);

Mat cameraMatrixL = (Mat_<double>(3, 3) << 4334.09568,
                     0,
                     959.50000,
                     0,
                     4334.09568,
                     511.50000,
                     0,
                     0,
                     1.0);

Mat distCoeffL = (Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

Mat cameraMatrixR = (Mat_<double>(3, 3) << 4489.55770,
                     0,
                     801.86552,
                     0,
                     4507.13412,
                     530.72579,
                     0,
                     0,
                     1.0);

Mat distCoeffR = (Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);

Mat T = (Mat_<double>(3, 1) << -518.97666, 01.20629, 9.14632);  // T平移向量
Mat rec = (Mat_<double>(3, 1) << 0.04345, -0.05236, -0.01810);  // rec旋转向量
Mat R;  // R 旋转矩阵

static void saveXYZ(const char* filename, const Mat& mat) {
    const double max_z = 16.0e4;
    FILE* fp = fopen(filename, "wt");
    printf("%d %d \n", mat.rows, mat.cols);
    for (int y = 0; y < mat.rows; y++) {
        for (int x = 0; x < mat.cols; x++) {
            Vec3f point = mat.at<Vec3f>(y, x);
            if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)
                continue;
                fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);        
        }
    }
    fclose(fp);
}

/*给深度图上色*/
void GenerateFalseMap(cv::Mat &src, cv::Mat &disp)
{
    // color map  
    float max_val = 255.0f;
    float map[8][4] = { { 0,0,0,114 },{ 0,0,1,185 },{ 1,0,0,114 },{ 1,0,1,174 },
    { 0,1,0,114 },{ 0,1,1,185 },{ 1,1,0,114 },{ 1,1,1,0 } };
    float sum = 0;
    for (int i = 0; i<8; i++)
        sum += map[i][3];

    float weights[8]; // relative   weights  
    float cumsum[8];  // cumulative weights  
    cumsum[0] = 0;
    for (int i = 0; i<7; i++) {
        weights[i] = sum / map[i][3];
        cumsum[i + 1] = cumsum[i] + map[i][3] / sum;
    }

    int height_ = src.rows;
    int width_ = src.cols;
    // for all pixels do  
    for (int v = 0; v<height_; v++) {
        for (int u = 0; u<width_; u++) {

            // get normalized value  
            float val = std::min(std::max(src.data[v*width_ + u] / max_val, 0.0f), 1.0f);

            // find bin  
            int i;
            for (i = 0; i<7; i++)
                if (val<cumsum[i + 1])
                    break;

            // compute red/green/blue values  
            float   w = 1.0 - (val - cumsum[i])*weights[i];
            uchar r = (uchar)((w*map[i][0] + (1.0 - w)*map[i + 1][0]) * 255.0);
            uchar g = (uchar)((w*map[i][1] + (1.0 - w)*map[i + 1][1]) * 255.0);
            uchar b = (uchar)((w*map[i][2] + (1.0 - w)*map[i + 1][2]) * 255.0);
            //rgb内存连续存放  
            disp.data[v*width_ * 3 + 3 * u + 0] = b;
            disp.data[v*width_ * 3 + 3 * u + 1] = g;
            disp.data[v*width_ * 3 + 3 * u + 2] = r;
        }
    }
}

      /*****立体匹配*****/
void stereo_match(int, void*)
{
    sgbm->setPreFilterCap(63);
    int sgbmWinSize =  5;//根据实际情况自己设定
    int NumDisparities = 416;//根据实际情况自己设定
    int UniquenessRatio = 6;//根据实际情况自己设定
    sgbm->setBlockSize(sgbmWinSize);
    int cn = rectifyImageL.channels();

    sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(NumDisparities);
    sgbm->setUniquenessRatio(UniquenessRatio);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(10);
    sgbm->setDisp12MaxDiff(1);
    sgbm->setMode(StereoSGBM::MODE_SGBM);
    Mat disp, dispf, disp8;
    sgbm->compute(rectifyImageL, rectifyImageR, disp);
    //去黑边
    Mat img1p, img2p;
    copyMakeBorder(rectifyImageL, img1p, 0, 0, NumDisparities, 0, BORDER_REPLICATE);
    copyMakeBorder(rectifyImageR, img2p, 0, 0, NumDisparities, 0, BORDER_REPLICATE);
    dispf = disp.colRange(NumDisparities, img2p.cols- NumDisparities);

    dispf.convertTo(disp8, CV_8U, 255 / (NumDisparities *16.));
    reprojectImageTo3D(dispf, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    imshow("disparity", disp8);
    Mat color(dispf.size(), CV_8UC3);
    GenerateFalseMap(disp8, color);//转成彩图
    imshow("disparity", color);
    saveXYZ("xyz.xls", xyz);
}

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        cout << origin << "in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
            break;
    }
}

/*****主函数*****/
int main()
{
    /*  立体校正    */
    Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pl, imageSize, CV_16SC2, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_16SC2, mapRx, mapRy);

    /*  读取图片    */
    // rgbImageL = imread("left_cor.bmp", 1);//CV_LOAD_IMAGE_COLOR
    // rgbImageR = imread("right_cor.bmp", -1);
    rgbImageL = imread("../image/1.png", 1);
    rgbImageR = imread("../image/2.png", 1);

    /*  经过remap之后，左右相机的图像已经共面并且行对准了 */
    remap(rgbImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);//INTER_LINEAR
    remap(rgbImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

    /*  把校正结果显示出来*/

    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 700. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

                                        //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
    resize(rectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
    resize(rectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);

    /*  立体匹配    */
    namedWindow("disparity", WINDOW_NORMAL);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);//disparity
    stereo_match(0, 0);

    waitKey(0);
    return 0;
}
//原文链接：https://blog.csdn.net/weixin_39449570/article/details/79033314