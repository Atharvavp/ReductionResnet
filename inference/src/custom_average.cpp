#include <torch/torch.h>
#include <torch/script.h>
#include <list>
#include <iostream>



torch::Tensor custom_average(const torch::Tensor& t_64, const torch::Tensor& t_128, const torch::Tensor& t_256, const torch::Tensor& t_512) {
  // int64_t max_size = 512;
  torch::Tensor output = torch::zeros(t_512.sizes());
  // float* out = output.data_ptr<float>();
  auto out = output.accessor<float, 2>();

  const int64_t rows = t_512.size(0);
  const int64_t cols = t_512.size(1);

  // std::cout << rows << " " << cols;

  // const float* data_64 = t_64.data_ptr<float>();
  // const float* data_128 = t_128.data_ptr<float>();
  // const float* data_256 = t_256.data_ptr<float>();
  // const float* data_512 = t_512.data_ptr<float>();

  const auto data_64 = t_64.accessor<float, 2>();
  const auto data_128 = t_128.accessor<float, 2>();
  const auto data_256 = t_256.accessor<float, 2>();
  const auto data_512 = t_512.accessor<float, 2>();

  // for(auto r = 0; r < rows; ++r) {
  //   for(auto c = 0; c < cols; ++c ){
  //     output.index_put_({r, c}, (t_512.index({r, c}).data<float>()[0], t_256.index({r, c % 2}).data<float>()[0], t_128.index({r, c % 4}).data<float>()[0], t_64.index({r, c % 8}).data<float>()[0]) / 4);
  //   }
  // }

  for(auto r = 0; r < rows; ++r) {
    //std::cout << "Rows: " << r;
    for(auto c = 0; c < cols; ++c ){
      // std::cout << "Rows: " << r;
      out[r][c] = (data_64[r][c % 8] + data_128[r][c % 4] + data_256[r][c % 2] + data_512[r][c]) / 4;
    }
  }

  return output;
}

// int main() {
//   // torch::Tensor tensor = torch::eye(3);
//   // auto x = tensor.size(0);

//   auto t = custom_interleave_and_average(torch::full({50, 64}, 1.0), torch::full({50, 128}, 2.0), torch::full({50, 256}, 3.0), torch::full({50, 512}, 4.0));

//   std::cout << "\n" << t.sizes() << "\n";
//   std::cout << t.index({45, 439});


//   // int x = GetEmbeddings(tensor);
//   // std::cout << x << std::endl;
//   // std::cout << tensor.index({1}) << std::endl;
// }

static auto registry = torch::RegisterOperators("custom_namespace::custom_average", &custom_average);
