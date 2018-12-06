/**
 * MIT License
 *
 * Copyright (c) 2018 Prabhsimran Singh
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
using namespace Eigen;

#include "functions.hpp"

namespace nn {

class Dense {
  private:
    MatrixXf W;
    RowVectorXf b;

  public:
    explicit Dense(const int &, const int &);

    MatrixXf forward(const MatrixXf &);

    std::pair<MatrixXf, MatrixXf> backward(const MatrixXf &, const MatrixXf &);
};

Dense::Dense(const int &num_inputs, const int &num_outputs) {
    W = MatrixXf(num_inputs, num_outputs).setRandom() * F::glorot_uniform(num_inputs, num_outputs);
    b = RowVectorXf(num_outputs).setZero();
}

MatrixXf Dense::forward(const MatrixXf &inputs) {
    return (inputs * W) + b;
}

std::pair<MatrixXf, MatrixXf> Dense::backward(const MatrixXf &inputs, const MatrixXf &gradients) {
}
} // namespace nn