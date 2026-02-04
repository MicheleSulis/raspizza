#include <iostream>
#include <vector>
#include <string>
#include <chrono>

#include "ModelInterpreter.h"
#include "CameraHandler.h"

#include <opencv2/opencv.hpp>


std::unique_ptr<ModelInterpreter> model_interpreter_ptr; // must be accessible from the callback

// This function will be called by CameraHandler when a new frame is ready:
void processFrameAndInfer (const CameraFrame &frame)
{
	if (!model_interpreter_ptr) {
		std::cerr << "Interpreter not initialized!" << std::endl;
		return;
	}

	// Input size of TFLite model
	int model_input_w = model_interpreter_ptr->getInputWidth();
	int model_input_h = model_interpreter_ptr->getInputHeight();

	// Converts CameraFrame to cv::Mat for resizing
	// Let's assume that CameraFrame::data is in BGR from CameraHandler
	cv::Mat original_image_bgr(frame.height, frame.width, CV_8UC3, (void*) frame.data.data());

	// Resize the image at the model input
	cv::Mat resized_image_bgr;
	cv::resize(original_image_bgr, resized_image_bgr, cv::Size(model_input_w, model_input_h));

	// Swap the color endianness. OpenCV uses BGR, but TFLite uses RGB:
	cv::cvtColor(resized_image_bgr, resized_image_bgr, cv::COLOR_BGR2RGB);

	// Perform inference
	auto start_infer = std::chrono::high_resolution_clock::now();
	std::vector<Detection> detections = model_interpreter_ptr->runInference(resized_image_bgr.data);
	auto end_infer = std::chrono::high_resolution_clock::now();
	auto infer_duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_infer - start_infer);
	std::cout << "Inference time: " << infer_duration.count() << " ms" << std::endl;

	std::vector<std::string> class_labels = model_interpreter_ptr->getClassLabels();
	int argmax = 0;
	double max_confidence = 0;
	std::cout << "detections.size(): " << detections.size() << std::endl;
	for (size_t i = 0; i < detections.size(); ++i) {
		std::cout << class_labels[detections[i].class_id] << ": " << detections[i].confidence << std::endl;
		if (detections[i].confidence > max_confidence) {
			max_confidence = detections[i].confidence;
			argmax = i;
		}
	}
	std::cout << "Object detected: " << class_labels[argmax] << std::endl << std::endl;

	// Show image:
	cv::imshow("Object (C++)", original_image_bgr);
	cv::waitKey(1);
}


int main ()
{
	const int camera_width  = 640;
	const int camera_height = 480;

	// Initialize the model interpreter
	model_interpreter_ptr = std::make_unique<ModelInterpreter>();
	if (!model_interpreter_ptr->init()) {
		std::cerr << "Failed to initialize Model's Interpreter." << std::endl;
		return -1;
	}

	// Initialize the camera handler with the callback
	CameraHandler camera_handler(processFrameAndInfer);
	if (!camera_handler.init(camera_width, camera_height)) {
		std::cerr << "Failed to initialize CameraHandler." << std::endl;
		return -1;
	}

	if (!camera_handler.start()) {
		std::cerr << "Failed to start camera handler." << std::endl;
		return -1;
	}

	std::cout << "Running... Press Enter to stop." << std::endl;
	{	// the main thread doesn't actually do anything...
		std::string line;
		std::getline(std::cin, line);
	}

	std::cout << "Stopping camera and cleaning up..." << std::endl;
	camera_handler.stop();

	std::cout << "Program terminated." << std::endl;
	return 0;
}
