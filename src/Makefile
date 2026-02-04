CXX      := g++
CXXFLAGS := -std=c++17 -Wall -g

INCLUDES := \
    -I/usr/local/include \
    -I/usr/include \
    -I/usr/include/opencv2 \
    -I/usr/include/libcamera

LIBDIRS := \
    -L/usr/local/lib \
    -L/usr/lib/tensorflow/lite

LIBS := \
    -lflatbuffers \
    -ltensorflowlite \
    -lopencv_core \
    -lopencv_imgproc \
    -lopencv_videoio \
    -lopencv_highgui \
    -lopencv_imgcodecs \
    -lcamera \
    -lcamera-base \
    -lpthread

SRCS   := main.cpp ModelInterpreter.cpp CameraHandler.cpp
OBJS   := $(SRCS:.cpp=.o)
TARGET := my_interpreter

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $@ $(LIBDIRS) $(LIBS) \
	    -Wl,-rpath=/usr/local/lib:/usr/lib/tensorflow/lite

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
