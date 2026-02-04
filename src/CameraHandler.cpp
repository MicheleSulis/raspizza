#include "CameraHandler.h"
#include <iostream>
#include <sys/mman.h>

// OpenCV for potential image conversion if needed (e.g., YUV to BGR)
#include <opencv2/opencv.hpp>
#include <libcamera/libcamera.h>

using namespace libcamera;

CameraHandler::CameraHandler (std::function<void(const CameraFrame&)> callback) :
	frame_callback_(callback),
	camera_manager_(std::make_unique<CameraManager>()),
	stream_(nullptr)
{
}

CameraHandler::~CameraHandler ()
{
	if (camera_) {
		camera_->stop();
		camera_->release();
	}
	if (allocator_) allocator_->free(stream_);
	if (camera_manager_) camera_manager_->stop();
}

bool CameraHandler::init (unsigned width, unsigned height)
{
	if (camera_manager_->start() != 0) {
		std::cerr << "Failed to start camera manager." << std::endl;
		return false;
	}

	auto cameras = camera_manager_->cameras();
	if (cameras.empty()) {
		std::cerr << "No cameras found." << std::endl;
		camera_manager_->stop();
		return false;
	}

	// First available camera use
	camera_ = camera_manager_->get(cameras[0]->id());
	if (!camera_) {
		std::cerr << "Failed to get camera." << std::endl;
		camera_manager_->stop();
		return false;
	}

	if (camera_->acquire() != 0) {
		std::cerr << "Failed to acquire camera." << std::endl;
		camera_manager_->stop();
		return false;
	}

	// Configure camera stream
	std::unique_ptr<CameraConfiguration> config = camera_->generateConfiguration({StreamRole::StillCapture});

	// Set pixel format and resolution
	config->at(0).pixelFormat = formats::NV12; // NV12 is a semi-planar YUV format (Y plane, UV (interleaved) plane)
	config->at(0).size = {width, height};
	config->at(0).bufferCount = 1; // lowest possible for low latency, although libcamera seems to increase it to 4...

	if (config->validate() == CameraConfiguration::Invalid) {
		std::cerr << "Invalid camera configuration." << std::endl;
		camera_->release();
		camera_manager_->stop();
		return false;
	}

	if (camera_->configure(config.get()) != 0) {
		std::cerr << "Failed to configure camera." << std::endl;
		camera_->release();
		camera_manager_->stop();
		return false;
	}

	// Allocate buffers for frame capture
	stream_ = config->at(0).stream();
	allocator_ = std::make_unique<FrameBufferAllocator>(camera_);
	if (allocator_->allocate(stream_) < 0) {
		std::cerr << "Failed to allocate buffers." << std::endl;
		camera_->release();
		camera_manager_->stop();
		return false;
	}

	// Connect the callback for completed requests
	camera_->requestCompleted.connect(this, &CameraHandler::requestComplete);

	// Create requests by associating them with allocated buffers
	for (unsigned int i = 0; i < config->at(0).bufferCount; ++i) {
		std::unique_ptr<Request> request = camera_->createRequest();
		if (!request) {
			std::cerr << "Failed to create request." << std::endl;
			return false;
		}
		if (request->addBuffer(stream_, allocator_->buffers(stream_)[i++].get()) != 0) {
			std::cerr << "Failed to add buffer to request." << std::endl;
			return false;
		}
		requests_.push_back(std::move(request));
	}

	std::cout
		<< "Camera initialized: " << config->at(0).size.width << "×" << config->at(0).size.height
		<< " (" << config->at(0).pixelFormat.toString() << ")"
		<< ", " << config->at(0).bufferCount << " buffers"
		<< std::endl;

	return true;
}

bool CameraHandler::start ()
{
	if (camera_->start() != 0) {
		std::cerr << "Failed to start camera." << std::endl;
		return false;
	}
	for (auto &req : requests_) {
		if (camera_->queueRequest(req.get()) != 0) {
			std::cerr << "[CameraHandler] Failed to queue request after camera start." << std::endl;
			return false;
		}
	}
	return true;
}

void CameraHandler::stop ()
{
	camera_->stop();
}

// callback called by libcamera when a request is completed
void CameraHandler::requestComplete (Request *request)
{
	// prepare variables for future mmap:
	size_t total_buffer_length = 0;
	void *mem = MAP_FAILED;

	if (request->status() == Request::RequestComplete) {
		// request->buffers() is a map between the various streams and their buffers; it uses the first (and only) stream
		const FrameBuffer *buffer = request->buffers().begin()->second;
		const FrameBuffer::Plane &plane0 = buffer->planes()[0]; // Plan Y for NV12

		int stride = stream_->configuration().stride;
		PixelFormat pixel_format = stream_->configuration().pixelFormat;

		std::cout
			<< "Received " << pixel_format.toString() << " frame"
			<< " with " << buffer->planes().size()
			<< " planes:";
		for (const auto &plane : buffer->planes()) {
			std::cout << " " << plane.length << "@FD=" << plane.fd.get();
			total_buffer_length += plane.length;
		}
		std::cout << std::endl;

		// The length of the mmap must be the total length of all mapped planes
		// In the case of NV12, the second plane (UV) begins immediately after the first (Y) on the same FD
		mem = mmap(NULL, total_buffer_length, PROT_READ, MAP_SHARED, plane0.fd.get(), 0);
		if (mem == MAP_FAILED) {
			std::cerr << "Failed to mmap buffer! Errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
			goto bailout;
		}

		// Convert the image to RGB format via OpenCV
		cv::Mat bgr_image;
		if (pixel_format == libcamera::formats::NV12) {
			// --- Conversion from NV12 to BGR ---
			// Create the Mat NV12, using the total buffer height (Y + UV) and stride
			// The Mat must represent the entire YUV buffer
			int img_width  = stream_->configuration().size.width;
			int img_height = stream_->configuration().size.height;
			cv::Mat nv12_image(img_height + img_height / 2, img_width, CV_8UC1, mem, stride);
			cv::cvtColor(nv12_image, bgr_image, cv::COLOR_YUV2BGR_NV12);
		} else if (pixel_format == libcamera::formats::MJPEG) {
			// --- Conversion from MJPEG to BGR ---
			cv::Mat mjpeg_data(1, total_buffer_length, CV_8UC1, mem);
			bgr_image = cv::imdecode(mjpeg_data, cv::IMREAD_COLOR);
			if (bgr_image.empty()) {
				std::cerr << "Failed to decode MJPEG frame!" << std::endl;
				goto bailout;
			}
			std::cout << "Decoded  MJPEG frame: " << bgr_image.cols << "×" << bgr_image.rows << std::endl;
		} else {
			std::cerr << "Skipping unsupported frame." << std::endl;
			goto bailout;
		}

		// Pass frame to next stage:
		if (frame_callback_) {
			CameraFrame frame;
			frame.width  = bgr_image.cols;
			frame.height = bgr_image.rows;
			frame.data.assign(bgr_image.data, bgr_image.data + (frame.width * frame.height * bgr_image.channels()));
			frame_callback_(frame);
		}

	} else { // means request->status() != Request::RequestComplete
		std::cerr << "Request failed: " << request->status() << std::endl;
	}

bailout:
	if (mem != MAP_FAILED) munmap(mem, total_buffer_length);
	request->reuse(Request::ReuseBuffers);
	camera_->queueRequest(request);
}