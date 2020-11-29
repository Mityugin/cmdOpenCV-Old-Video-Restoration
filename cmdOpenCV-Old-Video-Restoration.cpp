#include <iostream>

#include <opencv2/dnn_superres.hpp>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"

using namespace cv;
using namespace cv::dnn;
using namespace std;

static float hull_pts[] = {
    -90., -90., -90., -90., -90., -80., -80., -80., -80., -80., -80., -80., -80., -70., -70., -70., -70., -70., -70., -70., -70.,
    -70., -70., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -60., -50., -50., -50., -50., -50., -50., -50., -50.,
    -50., -50., -50., -50., -50., -50., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -40., -30.,
    -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -30., -20., -20., -20., -20., -20., -20., -20.,
    -20., -20., -20., -20., -20., -20., -20., -20., -20., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10., -10.,
    -10., -10., -10., -10., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 10., 10., 10., 10., 10., 10., 10.,
    10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 10., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20., 20.,
    20., 20., 20., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 30., 40., 40., 40., 40.,
    40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 50., 50., 50., 50., 50., 50., 50., 50., 50., 50.,
    50., 50., 50., 50., 50., 50., 50., 50., 50., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60., 60.,
    60., 60., 60., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 70., 80., 80., 80.,
    80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 80., 90., 90., 90., 90., 90., 90., 90., 90., 90., 90.,
    90., 90., 90., 90., 90., 90., 90., 90., 90., 100., 100., 100., 100., 100., 100., 100., 100., 100., 100., 50., 60., 70., 80., 90.,
    20., 30., 40., 50., 60., 70., 80., 90., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -20., -10., 0., 10., 20., 30., 40., 50.,
    60., 70., 80., 90., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -40., -30., -20., -10., 0., 10., 20.,
    30., 40., 50., 60., 70., 80., 90., 100., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -50.,
    -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., -60., -50., -40., -30., -20., -10., 0., 10., 20.,
    30., 40., 50., 60., 70., 80., 90., 100., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90.,
    100., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -80., -70., -60., -50.,
    -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -90., -80., -70., -60., -50., -40., -30., -20., -10.,
    0., 10., 20., 30., 40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30.,
    40., 50., 60., 70., 80., 90., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70.,
    80., -110., -100., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100.,
    -90., -80., -70., -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., 80., -110., -100., -90., -80., -70.,
    -60., -50., -40., -30., -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -110., -100., -90., -80., -70., -60., -50., -40., -30.,
    -20., -10., 0., 10., 20., 30., 40., 50., 60., 70., -90., -80., -70., -60., -50., -40., -30., -20., -10., 0.
};

int main()
{

    string input_path = "C:\\TEMP\\Czar-part2.mp4";
    string output_path = "C:\\TEMP\\Czar-part2-super2.mp4";

    string algorithm = "fsrcnn";
    //string algorithm = "espcn";
    int scale = 2;

    string path = "C:\\TEMP\\Models\\FSRCNN_x2.pb";
    //string path = "C:\\TEMP\\Models\\ESPCN_x2.pb";

    VideoCapture input_video(input_path);
    int ex = static_cast<int>(input_video.get(CAP_PROP_FOURCC));
    Size S = Size((int)input_video.get(CAP_PROP_FRAME_WIDTH) * scale,
        (int)input_video.get(CAP_PROP_FRAME_HEIGHT) * scale);
    
    VideoWriter output_video;
    output_video.open(output_path, VideoWriter::fourcc('m', 'p', '4', 'v'), input_video.get(CAP_PROP_FPS), S, true);

	// Specify the paths for the 2 files
	string protoFile = "C:\\TEMP\\Models\\colorization_deploy_v2.prototxt";
	string weightsFile = "C:\\TEMP\\Models\\colorization_release_v2.caffemodel";
	//string weightsFile = "./models/colorization_release_v2_norebal.caffemodel";

	const int W_in = 224;
	const int H_in = 224;

	// Read the network into Memory

	auto net = readNetFromCaffe(protoFile, weightsFile);
    
	net.setPreferableBackend(DNN_BACKEND_CUDA);
	net.setPreferableTarget(DNN_TARGET_CUDA_FP16);


    if (!input_video.isOpened())
    {
        std::cerr << "Could not open the video." << std::endl;
        return -1;
    }

    cv::dnn_superres::DnnSuperResImpl sr;

    sr.readModel(path);
    sr.setModel(algorithm, scale);

    sr.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    sr.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);

    for (;;)
    {
        Mat frame, output_frame;
        
        input_video >> frame;

        if (frame.empty())
            break;

        sr.upsample(frame, output_frame);

        /*cvtColor(output_frame, grayimg, COLOR_BGR2GRAY);
        cv::Ptr<CLAHE> clahe = createCLAHE();
        clahe->setClipLimit(4);
        Mat dst;
        clahe->apply(grayimg, dst);
        cvtColor(dst, output_frame, COLOR_GRAY2BGR);*/

       // populate cluster centers as 1x1 convolution kernel
		int sz[] = { 2, 313, 1, 1 };
		const Mat pts_in_hull(4, sz, CV_32F, hull_pts);
		Ptr<dnn::Layer> class8_ab = net.getLayer("class8_ab");
		class8_ab->blobs.push_back(pts_in_hull);
		Ptr<dnn::Layer> conv8_313_rh = net.getLayer("conv8_313_rh");
		conv8_313_rh->blobs.push_back(Mat(1, 313, CV_32F, Scalar(2.606)));

        // Extract the L channel grayscale image and average it
        Mat lab, L, input;
        output_frame.convertTo(output_frame, CV_32F, 1.0 / 255);
        cvtColor(output_frame, lab, COLOR_BGR2Lab);
        extractChannel(lab, L, 0);
        resize(L, input, Size(W_in, H_in));
        input -= 50;
        
        // L channel image input to the network, forward calculation
        Mat inputBlob = blobFromImage(input);
        net.setInput(inputBlob);
        Mat result = net.forward();

        // a, b channel extracted from the network output
        Size siz(result.size[2], result.size[3]);
        Mat a = Mat(siz, CV_32F, result.ptr(0, 0));
        Mat b = Mat(siz, CV_32F, result.ptr(0, 1));
        resize(a, a, output_frame.size());
        resize(b, b, output_frame.size());

        // Channel merge into a color map
        Mat color, chn[] = { L, a, b };
        merge(chn, 3, lab);
        cvtColor(lab, color, COLOR_Lab2BGR);


		//cvtColor(color, output_frame, COLOR_BGR2GRAY);
		//cv::Ptr<CLAHE> clahe = createCLAHE();
		//clahe->setClipLimit(4);
		//Mat dst;
		//clahe->apply(output_frame, dst);
		//cvtColor(dst, color, COLOR_GRAY2BGR);

        output_video << color;
 
        namedWindow("color video", WINDOW_AUTOSIZE);
        imshow("color video", color);

        namedWindow("Original video", WINDOW_AUTOSIZE);
        imshow("Original video", frame);

        char c = (char)waitKey(25);
        if (c == 27)
            break;
    }

    input_video.release();
    output_video.release();

    return 0;
}
