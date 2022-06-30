#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace torch::indexing;

class Yolo
{
public:
  Yolo(std::string model_path, float threshold) : m_model_path(model_path), m_threshold(threshold)
  {
    m_module = torch::jit::load(model_path);
  }

  torch::Tensor nms(torch::Tensor dets, float thres)
  {
    auto x1 = dets.index({Ellipsis, 0});
    auto y1 = dets.index({Ellipsis, 1});
    auto x2 = dets.index({Ellipsis, 2});
    auto y2 = dets.index({Ellipsis, 3});
    auto scores = dets.index({Ellipsis, 4});

    auto areas = (x2 - x1 + 1) * (y2 - y1 + 1);
    auto order = scores.argsort(-1, true);

    torch::Tensor i;
    vector<torch::Tensor> keep;

    while (order.sizes()[0] > 0)
    {
      i = order[0];
      keep.push_back(i);
      auto xx1 = torch::maximum(x1.index({i}), x1.index({order.index({Slice(1, None, None)})}));
      auto yy1 = torch::maximum(y1.index({i}), y1.index({order.index({Slice(1, None, None)})}));
      auto xx2 = torch::minimum(x2.index({i}), x2.index({order.index({Slice(1, None, None)})}));
      auto yy2 = torch::minimum(y2.index({i}), y2.index({order.index({Slice(1, None, None)})}));

      auto w = torch::maximum(torch::zeros_like(xx2), xx2 - xx1 + 1);
      auto h = torch::maximum(torch::zeros_like(yy2), yy2 - yy1 + 1);

      auto inter = w * h;
      auto ovr = inter / (areas.index({i}) + areas.index({order.index({Slice(1, None, None)})}) - inter);
      auto inds = torch::where(ovr <= thres)[0];
      order = order.index({inds + 1});
    }

    return torch::stack(torch::TensorList(keep));
  }

  vector<vector<int>> predict(char *png_buffer, size_t png_buffer_length)
  {
    vector<vector<int>> ret;
    auto image = cv::imdecode(cv::Mat(1, png_buffer_length, CV_8UC1, png_buffer), CV_LOAD_IMAGE_UNCHANGED);
    if (image.data == NULL) {
      return ret;
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    // preprocessing
    auto h_gain = 640.0 / image.size[0];
    auto w_gain = 640.0 / image.size[1];

    cv::resize(image, image, cv::Size(640, 640), cv::INTER_LINEAR);
    auto input_tensor = to_tensor(image);
    input_tensor.unsqueeze_(0);

    // inference
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_tensor);

    auto output = m_module.forward(inputs).toTuple()->elements()[0].toTensor().squeeze();
    // processing output to N x 6 where 6 is (cx, cy, w, h, confidence, class_num)
    vector<torch::Tensor> processed_output_vec;

    processed_output_vec.push_back(output.index({Ellipsis, 0}));
    processed_output_vec.push_back(output.index({Ellipsis, 1}));
    processed_output_vec.push_back(output.index({Ellipsis, 2}));
    processed_output_vec.push_back(output.index({Ellipsis, 3}));

    auto max_indices_output = torch::max(output.index({Ellipsis, Slice(5, None, None)}), 1);
    auto class_scores = std::get<0>(max_indices_output) * output.index({Ellipsis, 4});
    auto class_nums = std::get<1>(max_indices_output);

    processed_output_vec.push_back(class_scores);
    processed_output_vec.push_back(class_nums);

    auto processed_output = torch::stack(torch::TensorList(processed_output_vec), 1);
    auto filtered_output = processed_output.index({processed_output.index({Ellipsis, 4}) > m_threshold, Ellipsis});
    if (filtered_output.sizes()[0] == 0) {
      return ret;
    }

    filtered_output.index_put_({Ellipsis, 0}, filtered_output.index({Ellipsis, 0}) / w_gain);
    filtered_output.index_put_({Ellipsis, 1}, filtered_output.index({Ellipsis, 1}) / h_gain);
    filtered_output.index_put_({Ellipsis, 2}, filtered_output.index({Ellipsis, 2}) / w_gain);
    filtered_output.index_put_({Ellipsis, 3}, filtered_output.index({Ellipsis, 3}) / h_gain);

    // change coords from center x, center y, width, height to xyxy
    auto filtered_output_shadow = filtered_output.clone();
    filtered_output.index_put_({Ellipsis, 0}, (filtered_output_shadow.index({Ellipsis, 0}) - filtered_output_shadow.index({Ellipsis, 2}) / 2));
    filtered_output.index_put_({Ellipsis, 1}, (filtered_output_shadow.index({Ellipsis, 1}) - filtered_output_shadow.index({Ellipsis, 3}) / 2));
    filtered_output.index_put_({Ellipsis, 2}, (filtered_output_shadow.index({Ellipsis, 0}) + filtered_output_shadow.index({Ellipsis, 2}) / 2));
    filtered_output.index_put_({Ellipsis, 3}, (filtered_output_shadow.index({Ellipsis, 1}) + filtered_output_shadow.index({Ellipsis, 3}) / 2));

    auto inds = nms(filtered_output, 0.3);
    filtered_output = filtered_output.index({inds});

    for (auto i = 0; i < filtered_output.sizes()[0]; i++)
    {
      vector<int> box;
      box.push_back(filtered_output.index({i, 0}).item<int>());
      box.push_back(filtered_output.index({i, 1}).item<int>());
      box.push_back(filtered_output.index({i, 2}).item<int>());
      box.push_back(filtered_output.index({i, 3}).item<int>());
      box.push_back(filtered_output.index({i, 5}).item<int>());
      ret.push_back(box);
    }
    return ret;
  }

  torch::Tensor to_tensor(cv::Mat img)
  {
    auto tensor_image = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kUInt8);
    tensor_image = tensor_image.permute({2, 0, 1});
    auto tensor_image_normed = tensor_image / 255.0;
    return tensor_image_normed;
  };

private:
  float m_threshold;
  std::string m_model_path;
  torch::jit::Module m_module;
};

int main(int argc, char **argv)
{
  // example of read image from file then inference
  ifstream file_img(argv[2], ios::binary);
  file_img.seekg(0, std::ios::end);
  int buffer_length = file_img.tellg();
  file_img.seekg(0, std::ios::beg);

  // Read image data into memory
  char *buffer = new char[buffer_length];
  file_img.read(buffer, buffer_length);

  Yolo yolo(argv[1], 0.6f);
  auto preds = yolo.predict(buffer, buffer_length);

  auto output_image = cv::imread(argv[2], cv::COLOR_BGR2RGB);
  for (auto r : preds)
  {
    auto rec = cv::Rect(r[0], r[1], r[2] - r[0], r[3] - r[1]);
    cv::rectangle(output_image, rec, cv::Scalar(0, 255, 0), 2, 8, 0);
  }

  cv::imwrite("output.png", output_image);
  cout<<"See output.png"<<endl;
}
