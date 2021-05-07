#include <vector>
#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
// #include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

torch::Tensor toTensor(cv::Mat img) {
    torch::Tensor tensor_image = torch::from_blob(img.data, { img.rows, img.cols }, torch::kUInt8);
    auto tensor_image_normed = (tensor_image / 255.0).sub_(0.5).div_(0.5);
    return tensor_image_normed;
};

int main(int argc, char** argv) {
  // load jit module
  auto module = torch::jit::load(argv[1]);

  // load input image
  auto image = cv::imread(argv[2], cv::COLOR_BGR2GRAY);

  // preprocessing
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(28, 28));
  auto input_tensor = toTensor(resized_image);
  input_tensor.unsqueeze_(0).unsqueeze_(0);

  // forward
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(input_tensor);
  torch::Tensor output = module.forward(inputs).toTensor();

  // get result
  int result = output.argmax().item<int>();
  std::cout << "The number is: " << result << std::endl;
  return 0;
}
