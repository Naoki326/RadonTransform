//fft.h
#ifndef NK_RADON
#define NK_RADON

#include <iostream>
#include <memory>
#include <algorithm>
#include <vector>
#include "fftw3.h"
#include "fft.h"
#include "fft2.h"
#include <cmath>
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/core.hpp"

#pragma comment(lib, "libfftw3f-3.lib")
#pragma comment(lib, "libfftw3-3.lib")
#pragma comment(lib, "libfftw3l-3.lib")


using std::vector;
using std::shared_ptr;
using namespace cv;

namespace MyAlg
{
	class RadonLines
	{
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

	public:
		RadonLines(double stAng, double edAng, double inc)
		{
			assert(stAng > -180 && edAng <= 180);

			stAng_ = stAng;
			edAng_ = edAng;
			inc_ = inc;
			
			//尽量将图片统一resize到2的整数次幂，可以加速fft计算
			padedw_ = 2048;
			padedh_ = 2048;

			fft2d_ptr_.reset(new FFT2D<true>(padedw_, padedh_));
			ifft1d_ptr_.reset(new iFFT1D<true>(padedw_));

			trans_len_ = ceil((edAng - stAng) / inc);
			trans_x_ = Mat(padedw_, trans_len_, CV_32FC1);
			trans_y_ = Mat(padedw_, trans_len_, CV_32FC1);

			for (int i = 0; i < padedw_; i++)
			{
				for (int j = 0; j < trans_len_; j++)
				{
					double theta = j * inc + stAng;
					if (theta < 0) theta = 360 + theta;
					theta *= M_PI / 180;
					trans_x_.at<float>(i, j) = (i - padedw_ / 2.0) * cos(-theta) + padedw_ / 2;
					trans_y_.at<float>(i, j) = (i - padedw_ / 2.0) * sin(-theta) + padedh_ / 2;
				}
			}

			result_ = Mat(padedw_, trans_len_, CV_64FC1);

		}
		~RadonLines()
		{

		}

		void ChangeSize(int width, int height, double stAng, double edAng, double inc)
		{
			assert(stAng > -180 && edAng <= 180);

			if (width_ == width && height_ == height && stAng_ == stAng && edAng_ == edAng && inc_ == inc)
				return;

			if (width_ != width || height_ != height)
			{
				width_ = width;
				height_ = height;

				double diagonal = sqrt(2) * max(width, height);
				int padw = int(ceil(diagonal - width_));
				int padh = int(ceil(diagonal - height_));

				if (padedw_ != padw + width_ || padedh_ != padh + height_)
				{
					padedw_ = padw + width_;
					padedh_ = padh + height_;

					fft2d_ptr_.reset(new FFT2D<true>(padedw_, padedh_));
					ifft1d_ptr_.reset(new iFFT1D<true>(padedw_));
				}
			}

			if (stAng_ != stAng || edAng_ != edAng || inc_ != inc)
			{
				stAng_ = stAng;
				edAng_ = edAng;
				inc_ = inc;

				trans_len_ = ceil((edAng - stAng) / inc);
			}

			trans_x_ = Mat(padedw_, trans_len_, CV_32FC1);
			trans_y_ = Mat(padedw_, trans_len_, CV_32FC1);

			for (int i = 0; i < padedw_; i++)
			{
				for (int j = 0; j < trans_len_; j++)
				{
					double theta = j * inc + stAng;
					if (theta < 0) theta = 360 + theta;
					theta *= M_PI / 180;
					trans_x_.at<float>(i, j) = (i - padedw_ / 2.0) * cos(-theta) + padedw_ / 2;
					trans_y_.at<float>(i, j) = (i - padedw_ / 2.0) * sin(-theta) + padedh_ / 2;
				}
			}

			result_ = Mat(padedw_, trans_len_, CV_64FC1);

		}

		void PadData(const Mat& image, Mat& dst)
		{
			assert(image.rows == height_ && image.cols == width_);

			//Radon变换，先作pad
			double padw = padedw_ - width_;
			double padh = padedh_ - height_;
			copyMakeBorder(image, dst, floor(padh / 2), ceil(padh / 2), floor(padw / 2), ceil(padw / 2), BORDER_CONSTANT, Scalar(0));

		}

		void FeedData(const Mat& image)
		{
			assert(image.rows == height_ && image.cols == width_);
			
			//Radon变换，先作pad
			double padw = padedw_ - width_;
			double padh = padedh_ - height_;
			Mat pad_image;
			copyMakeBorder(image, pad_image, floor(padh / 2), ceil(padh / 2), floor(padw / 2), ceil(padw / 2), BORDER_CONSTANT, Scalar(0));
			
			//利用傅里叶切片定理实现Radon变换
			//执行二维傅里叶变换
			shared_ptr<double> p_origin(new double[padedw_ * padedh_], default_delete<double[]>());//输入
			Mat m_fft2_real(padedw_, padedh_, CV_64FC1);//输出
			Mat m_fft2_imag(padedw_, padedh_, CV_64FC1);

			for (int i = 0; i < padedw_ * padedh_; i++)
			{
				p_origin.get()[i] = static_cast<double>(pad_image.data[i]);
			}

			fft2d_ptr_.get()->FeedData(p_origin);
			fft2d_ptr_.get()->iFFTShift2D(shift_flag::In);
			fft2d_ptr_.get()->Execute();
			fft2d_ptr_.get()->FFTShift2D(shift_flag::Out);
			fft2d_ptr_.get()->GetResultRC(m_fft2_real, m_fft2_imag);
			//极坐标转换
			Mat m_polar_real(padedw_, trans_len_, CV_64FC1);//输出
			Mat m_polar_imag(padedw_, trans_len_, CV_64FC1);
			remap(m_fft2_real, m_polar_real, trans_x_, trans_y_, INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0));
			remap(m_fft2_imag, m_polar_imag, trans_x_, trans_y_, INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0));
			//执行一维傅里叶逆变换
			shared_ptr<double> p_polar_real(new double[padedw_], default_delete<double[]>());//输入
			shared_ptr<double> p_polar_imag(new double[padedw_], default_delete<double[]>());
			shared_ptr<double> it_amp(new double[padedw_], default_delete<double[]>());//输出
			for (int i = 0; i < trans_len_; i++)
			{
				//不确定是否可以使用memcpy
				for (int j = 0; j < padedw_; j++)
				{
					p_polar_real.get()[j] = m_polar_real.at<double>(j, i);
					p_polar_imag.get()[j] = m_polar_imag.at<double>(j, i);
				}
				ifft1d_ptr_.get()->FeedData(p_polar_real, p_polar_imag);
				ifft1d_ptr_.get()->Execute();
				ifft1d_ptr_.get()->iFFTShift(shift_flag::Out);
				ifft1d_ptr_.get()->GetResultA(it_amp);
				for (int j = 0; j < padedw_; j++)
				{
					result_.at<double>(j, i) = it_amp.get()[j];
				}
			}

		}

		//峰值搜索函数
		void RadonPeaks(int numpeaks, double threshold, int nhoodx = 0, int nhoody = 0)
		{
			rho_.clear();
			the_.clear();
			Mat image_ = result_.clone();
			int h = image_.rows;
			int w = image_.cols;
			double maxVal = 0;
			minMaxIdx(image_, 0, &maxVal);
			threshold *= maxVal;
			if (nhoodx == 0)
			{
				nhoodx = h / 100 * 2 + 1;
			}
			else
			{
				nhoodx = nhoodx;
			}
			if (nhoody == 0)
			{
				nhoody = 40;
			}
			else
			{
				nhoody = nhoody;
			}
			while (true)
			{
				double maxv = 0.0;
				int maxp[2] = { -1, -1 };
				minMaxIdx(image_, 0, &maxv, 0, maxp);
				if (maxv < threshold)
					return;


				//当前最大值及其邻域所在的矩形，后续将其置零
				int p1_x = maxp[1] - (nhoody - 1) / 2;
				int p1_y = maxp[0] - (nhoodx - 1) / 2;

				int p2_x = maxp[1] + (nhoody - 1) / 2 + 1;
				int p2_y = maxp[0] + (nhoodx - 1) / 2 + 1;

				//若当前位置与之前查到的峰值位置有重叠，则不记录当前数据
				if (
					(p1_x > 0) && (p1_y > 0) && (p2_x < w - 1) && (p2_y < h - 1)
					&& abs(image_.at<double>(p1_y, p1_x) + 1) > 0.1
					&& abs(image_.at<double>(p2_y, p2_x) + 1) > 0.1
					&& abs(image_.at<double>(p1_y, p2_x) + 1) > 0.1
					&& abs(image_.at<double>(p2_y, p1_x) + 1) > 0.1
					)
				{
					double p = maxp[0];
					double q = maxp[1];
					p = p - h / 2;
					q = q * inc_ + stAng_;

					rho_.push_back(p);
					the_.push_back(q);
				}
				if (rho_.size() >= numpeaks)
					return;


				//置-1
				if (p1_x < 0) p1_x = 0;
				if (p1_y < 0) p1_y = 0;
				if (p2_x > w - 1) p2_x = w - 1;
				if (p2_y > h - 1) p2_y = h - 1;
				image_(Rect(Point(p1_x, p1_y), Point(p2_x + 1, p2_y + 1))) = -1;
			}
		}

		//取直线
		void GetResult(int &n, shared_ptr<double> rho, shared_ptr<double> the)
		{
			for (int i = 0; i < rho_.size() && i < n; i++)
			{
				rho.get()[i] = rho_.at(i);
				the.get()[i] = the_.at(i);
			}
			n = n < rho_.size() ? n : rho_.size();
		}


	};
	
}

#endif