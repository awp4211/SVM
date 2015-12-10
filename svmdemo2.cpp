#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

using namespace cv;

void drawCross(Mat &img, Point center, Scalar color)
{
	int col = center.x > 2 ? center.x : 2;
	int row = center.y> 2 ? center.y : 2;

	line(img, Point(col - 2, row - 2), Point(col + 2, row + 2), color);
	line(img, Point(col + 2, row - 2), Point(col - 2, row + 2), color);
}

int newSvmTest(int rows, int cols, int testCount)
{
	if (testCount > rows * cols)
		return 0;

	Mat img = Mat::zeros(rows, cols, CV_8UC3);
	Mat testPoint = Mat::zeros(rows, cols, CV_8UC1);
	Mat data = Mat::zeros(testCount, 2, CV_32FC1);
	Mat res = Mat::zeros(testCount, 1, CV_32SC1);

	//Create random test points
	for (int i = 0; i< testCount; i++)
	{
		int row = rand() % rows;
		int col = rand() % cols;

		if (testPoint.at<unsigned char>(row, col) == 0)
		{
			testPoint.at<unsigned char>(row, col) = 1;
			data.at<float>(i, 0) = float(col) / cols;
			data.at<float>(i, 1) = float(row) / rows;
		}
		else
		{
			i--;
			continue;
		}

		if (row > (50 * cos(col * CV_PI / 100) + 200))
		{
			drawCross(img, Point(col, row), CV_RGB(255, 0, 0));
			res.at<unsigned int>(i, 0) = 1;
		}
		else
		{
			if (col > 200)
			{
				drawCross(img, Point(col, row), CV_RGB(0, 255, 0));
				res.at<unsigned int>(i, 0) = 2;
			}
			else
			{
				drawCross(img, Point(col, row), CV_RGB(0, 0, 255));
				res.at<unsigned int>(i, 0) = 3;
			}
		}

	}

	//Show test points
	imshow("dst", img);
	waitKey(0);

	/////////////START SVM TRAINNING//////////////////
	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;

	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);

	svm.train(data, res, Mat(), Mat(), param);

	for (int i = 0; i< rows; i++)
	{
		for (int j = 0; j< cols; j++)
		{
			Mat m = Mat::zeros(1, 2, CV_32FC1);
			m.at<float>(0, 0) = float(j) / cols;
			m.at<float>(0, 1) = float(i)   / rows;

			float ret = 0.0;
			ret = svm.predict(m);
			Scalar rcolor;

			switch ((int)ret)
			{
			case 1: rcolor = CV_RGB(100, 0, 0); break;
			case 2: rcolor = CV_RGB(0, 100, 0); break;
			case 3: rcolor = CV_RGB(0, 0, 100); break;
			}

			line(img, Point(j, i), Point(j, i), rcolor);
		}
	}

	imshow("dst", img);
	waitKey(0);

	//Show support vectors
	int sv_num = svm.get_support_vector_count();
	for (int i = 0; i< sv_num; i++)
	{
		const float* support = svm.get_support_vector(i);
		circle(img, Point((int)(support[0] * cols), (int)(support[1] * rows)), 5, CV_RGB(200, 200, 200));
	}

	imshow("dst", img);
	waitKey(0);

	return 0;
}

int main(int argc, char** argv)
{
	return newSvmTest(400, 600, 100);
}