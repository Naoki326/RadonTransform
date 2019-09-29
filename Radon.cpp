#include "Radon.h"

namespace MyAlg
{

	RadonLines::RadonLines(double stAng, double edAng, double inc)
	{
		assert(stAng > -180 && edAng <= 180);

		stAng_ = stAng;
		edAng_ = edAng;
		inc_ = inc;

		//������ͼƬͳһresize��2���������ݣ����Լ���fft����
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

	void RadonLines::ChangeSize(int width, int height, double stAng, double edAng, double inc)
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

	void RadonLines::PadData(const Mat& image, Mat& dst)
	{
		assert(image.rows == height_ && image.cols == width_);

		//Radon�任������pad
		double padw = padedw_ - width_;
		double padh = padedh_ - height_;
		copyMakeBorder(image, dst, floor(padh / 2), ceil(padh / 2), floor(padw / 2), ceil(padw / 2), BORDER_CONSTANT, Scalar(0));
	}

	void RadonLines::FeedData(const Mat& image)
	{
		assert(image.rows == height_ && image.cols == width_);

		//Radon�任������pad
		double padw = padedw_ - width_;
		double padh = padedh_ - height_;
		Mat pad_image;
		copyMakeBorder(image, pad_image, floor(padh / 2), ceil(padh / 2), floor(padw / 2), ceil(padw / 2), BORDER_CONSTANT, Scalar(0));

		//���ø���Ҷ��Ƭ����ʵ��Radon�任
		//ִ�ж�ά����Ҷ�任
		shared_ptr<double> p_origin(new double[padedw_ * padedh_], std::default_delete<double[]>());//����
		Mat m_fft2_real(padedw_, padedh_, CV_64FC1);//���
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
		//������ת��
		Mat m_polar_real(padedw_, trans_len_, CV_64FC1);//���
		Mat m_polar_imag(padedw_, trans_len_, CV_64FC1);
		remap(m_fft2_real, m_polar_real, trans_x_, trans_y_, INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0));
		remap(m_fft2_imag, m_polar_imag, trans_x_, trans_y_, INTER_LANCZOS4, BORDER_CONSTANT, Scalar(0));
		//ִ��һά����Ҷ��任
		shared_ptr<double> p_polar_real(new double[padedw_], std::default_delete<double[]>());//����
		shared_ptr<double> p_polar_imag(new double[padedw_], std::default_delete<double[]>());
		shared_ptr<double> it_amp(new double[padedw_], std::default_delete<double[]>());//���
		for (int i = 0; i < trans_len_; i++)
		{
			//��ȷ���Ƿ����ʹ��memcpy
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

	//��ֵ��������
	void RadonLines::RadonPeaks(int numpeaks, double threshold, int nhoodx = 0, int nhoody = 0)
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


			//��ǰ���ֵ�����������ڵľ��Σ�������������
			int p1_x = maxp[1] - (nhoody - 1) / 2;
			int p1_y = maxp[0] - (nhoodx - 1) / 2;

			int p2_x = maxp[1] + (nhoody - 1) / 2 + 1;
			int p2_y = maxp[0] + (nhoodx - 1) / 2 + 1;

			//����ǰλ����֮ǰ�鵽�ķ�ֵλ�����ص����򲻼�¼��ǰ����
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


			//��-1
			if (p1_x < 0) p1_x = 0;
			if (p1_y < 0) p1_y = 0;
			if (p2_x > w - 1) p2_x = w - 1;
			if (p2_y > h - 1) p2_y = h - 1;
			image_(Rect(Point(p1_x, p1_y), Point(p2_x + 1, p2_y + 1))) = -1;
		}
	}

	//ȡֱ��
	void RadonLines::GetResult(int &n, shared_ptr<double> rho, shared_ptr<double> the)
	{
		for (int i = 0; i < rho_.size() && i < n; i++)
		{
			rho.get()[i] = rho_.at(i);
			the.get()[i] = the_.at(i);
		}
		n = n < rho_.size() ? n : rho_.size();
	}


}