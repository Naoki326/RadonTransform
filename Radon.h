//fft.h
#ifndef NK_RADON
#define NK_RADON

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include <cmath>

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#include "fftw32/include/fftw3.h"
#include "fft.h"
#include "fft2.h"


using std::vector;
using std::shared_ptr;
using namespace cv;

namespace MyAlg
{
	const double M_PI = 3.141592653;

	class RadonLines
	{
	public:
		//输入参数分别是查找的直线的角度的起始值和终止值
		//范围是(-180, 180]
		RadonLines(double stAng = -179, double edAng = 180, double inc = 1);

		~RadonLines()
		{
			if (ifft1d_ptr_.unique() == true)
				ifft1d_ptr_.reset();
			if (fft2d_ptr_.unique() == true)
				fft2d_ptr_.reset();
		}

		//重新设置接收的输入图的大小
		//主要是因为fftw需要预设大小
		void ChangeSize(int width, int height, double stAng = -179, double edAng = 180, double inc = 1);

		//设置补零
		void PadData(const Mat& image, Mat& dst);

		//输入数据
		void FeedData(const Mat& image);

		//峰值搜索函数
		void RadonPeaks(int numpeaks, double threshold, int nhoodx = 0, int nhoody = 0);

		//取直线
		void GetResult(int &n, shared_ptr<double> rho, shared_ptr<double> the);

	private:
		int width_;
		int height_;

		int padedw_;
		int padedh_;

		Mat trans_x_;
		Mat trans_y_;

		//极坐标下角度轴的像素长度
		int trans_len_;
		double stAng_;
		double edAng_;
		double inc_;

		shared_ptr<iFFT1D<true> > ifft1d_ptr_;
		shared_ptr<FFT2D<true> > fft2d_ptr_;

		//存放Radon变换结果
		Mat result_;
		//存放搜索
		vector<double> rho_, the_;

	};
	
}

#endif