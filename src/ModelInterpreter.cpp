#include <iostream>
#include <fstream>
#include <cmath> // For round

// TensorFlow Lite includes
#include "tensorflow/lite/kernels/register.h"

#include "ModelInterpreter.h"

ModelInterpreter::ModelInterpreter()
{
}

bool ModelInterpreter::init()
{
	const char *model_file = "model/my_model.tflite";
	const char *label_file = "model/labels.txt";

	// Load labels:
	std::ifstream file(label_file);
	if (file.is_open())
	{
		class_labels_.clear();
		for (std::string line; std::getline(file, line); class_labels_.push_back(line))
			;
		file.close();
	}
	else
	{
		std::cerr << "Failed to load labels from: " << label_file << std::endl;
		return false;
	}

	// Load model:
	model_ = tflite::FlatBufferModel::BuildFromFile(model_file);
	if (!model_)
	{
		std::cerr << "Failed to load model from: " << model_file << std::endl;
		return false;
	}

	// Use built-in operations:
	tflite::ops::builtin::BuiltinOpResolver resolver;

	// Build model interpreter:
	tflite::InterpreterBuilder builder(*model_, resolver);
	builder.SetNumThreads(4);

	if (builder(&interpreter_) != kTfLiteOk)
	{
		std::cerr << "Failed to build interpreter." << std::endl;
		return false;
	}

	if (interpreter_->AllocateTensors() != kTfLiteOk)
	{
		std::cerr << "Failed to allocate tensors." << std::endl;
		return false;
	}

	// Get input and output tensor details:
	const TfLiteIntArray *input_dims = interpreter_->input_tensor(0)->dims;
	if (input_dims->size != 4)
	{ // NHWC format
		std::cerr << "Invalid input tensor dimensions. Expected 4, got " << input_dims->size << std::endl;
		return false;
	}
	model_input_height_ = input_dims->data[1];
	model_input_width_ = input_dims->data[2];
	model_input_channels_ = input_dims->data[3];
	model_input_type_ = interpreter_->input_tensor(0)->type;

	std::cout
		<< "Model loaded successfully:\n"
		<< " Input     shape: " << model_input_width_ << "×" << model_input_height_ << "×" << model_input_channels_
		<< " Type: " << model_input_type_
		<< "\n";
	for (int i = 0; i < interpreter_->outputs().size(); ++i)
	{
		const TfLiteTensor *output_tensor = interpreter_->output_tensor(i);
		std::cout << " Output #" << i << " shape:";
		for (int j = 0; j < output_tensor->dims->size; ++j)
			std::cout << (j ? "×" : " ") << output_tensor->dims->data[j];
		std::cout << " Type: " << output_tensor->type << "\n";
	}

	return true;
}

std::vector<Detection> ModelInterpreter::runInference(const uint8_t *image_data)
{
	std::vector<Detection> detections;

	// Copy the image data into the input tensor
	// Data type must match the model (e.g., uint8_t or float32).
	if (model_input_type_ == kTfLiteUInt8)
	{
		uint8_t *input_tensor_ptr = interpreter_->typed_input_tensor<uint8_t>(0);
		std::memcpy(input_tensor_ptr, image_data, model_input_width_ * model_input_height_ * model_input_channels_);
	}
	else if (model_input_type_ == kTfLiteFloat32)
	{
		// If the model expects float32 and the input is uint8, the data must be scaled:
		float *input_tensor_ptr = interpreter_->typed_input_tensor<float>(0);
		for (int i = 0; i < model_input_width_ * model_input_height_ * model_input_channels_; ++i)
			input_tensor_ptr[i] = static_cast<float>(image_data[i]);
	}
	else
	{
		std::cerr << "Unsupported input tensor type: " << model_input_type_ << std::endl;
		return detections;
	}

	// Performs inference
	if (interpreter_->Invoke() != kTfLiteOk)
	{
		std::cerr << "Failed to invoke TFLite interpreter." << std::endl;
		return detections;
	}

	// Post-processing: retrieve output tensors
	TfLiteTensor *output_tensor = interpreter_->output_tensor(0);
	if (!output_tensor)
	{
		std::cerr << "Failed to get output tensor." << std::endl;
		return detections;
	}

	// Dimensions of the tensor output are: [1, 4]
	// output_tensor->dims->data[0] = 1 (batch)
	// output_tensor->dims->data[1] = 4 (class probabilities)
	model_output_type_ = output_tensor->type;
	int num_classes = output_tensor->dims->data[1];
	std::cout << "n classes: " << num_classes << std::endl;
	if (output_tensor->dims->size != 2 || output_tensor->dims->data[0] != 1)
	{
		std::cerr << "Unexpected output shape." << std::endl;
		return detections;
	}

	if (model_output_type_ == kTfLiteUInt8) {
		std::cout << "endtered in kTfLiteFloat8: " << std::endl;
		const uint8_t* raw_predictions_data = interpreter_->typed_output_tensor<uint8_t>(0);

		for (int class_id = 0; class_id < num_classes; ++class_id) {
			uint8_t qval = raw_predictions_data[class_id];
			float confidence = model_output_scale_ * (static_cast<int>(qval) - model_output_zero_);
			detections.push_back(Detection{class_id, confidence});
		}
	} else if (model_output_type_ == kTfLiteFloat32) {
		std::cout << "endtered in kTfLiteFloat32: " << std::endl;
		const float *raw_predictions_data = interpreter_->typed_output_tensor<float>(0);

		for (int class_id = 0; class_id < num_classes; ++class_id) {
			float confidence = raw_predictions_data[class_id];
			detections.push_back(Detection{class_id, confidence});
		}
	}
	
	return detections;
}
