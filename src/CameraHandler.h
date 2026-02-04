#ifndef CAMERA_HANDLER_H
#define CAMERA_HANDLER_H

#include <libcamera/libcamera.h>
#include <vector>
#include <functional>

// Forward declarations for libcamera
namespace libcamera {
	class CameraManager;
	class Camera;
	class StreamConfiguration;
	class FrameBufferAllocator;
	class Request;
}

// Structure for an acquired frame
struct CameraFrame {
	std::vector<uint8_t> data;
	int width;
	int height;
};

class CameraHandler
{
public:
	 CameraHandler (const std::function<void(const CameraFrame&)> callback);
	~CameraHandler ();

	bool init  (unsigned width, unsigned height);
	bool start ();  // Starts streaming
	void stop  ();  // Stops streaming

private:
	std::function<void(const CameraFrame&)> const frame_callback_;

	std::unique_ptr<libcamera::CameraManager> camera_manager_;
	std::shared_ptr<libcamera::Camera> camera_;
	std::unique_ptr<libcamera::FrameBufferAllocator> allocator_;
	libcamera::Stream* stream_;
	std::vector<std::unique_ptr<libcamera::Request>> requests_;

	void requestComplete (libcamera::Request* request); // callback from libcamera
};

#endif // CAMERA_HANDLER_H
