#ifndef MODEL_INTERPRETER_H
#define MODEL_INTERPRETER_H

#include <string>
#include <vector>
#include <memory>

// OpenCV:
#include <opencv2/opencv.hpp>

// TensorFlow Lite:
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/model.h"

// Structure for containing detection results
struct Detection
{
	int       class_id;
	float     confidence;
};

class ModelInterpreter
{
public:
	ModelInterpreter ();

	// Initialize TFLite interpreter
	bool init ();

	// Performs inference and returns detections
	std::vector<Detection> runInference (const uint8_t* image_data);

	// Model information retrival
	int getInputWidth  () const {return model_input_width_;}
	int getInputHeight () const {return model_input_height_;}
	const std::vector<std::string> &getClassLabels () const {return class_labels_;}

private:
	// Neural network handlement
	std::vector<std::string> class_labels_;
	std::unique_ptr<tflite::FlatBufferModel> model_;
	std::unique_ptr<tflite::Interpreter> interpreter_;
	
	// Model input details
	int model_input_width_    = 0;
	int model_input_height_   = 0;
	int model_input_channels_ = 0;
	TfLiteType model_input_type_ = kTfLiteNoType; // Only kTfLiteUInt8 and kTfLiteFloat32 are supported
	TfLiteType model_output_type_ = kTfLiteNoType;

	// Quantization details
	float model_input_scale_  = 0;
	int model_input_zero_	  = 0;
	float model_output_scale_ = 0;
	int model_output_zero_    = 0;
	float x_scale_ = 1;
	float y_scale_ = 1;
};

#endif // MODEL_INTERPRETER_H