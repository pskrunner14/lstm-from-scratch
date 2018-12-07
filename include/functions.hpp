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
#include <unsupported/Eigen/MatrixFunctions>
using namespace Eigen;

namespace F {

inline float glorot_uniform(const int &n_in, const int &n_out) {
    return 2.0 / (n_in + n_out);
}

inline MatrixXf sigmoid(const MatrixXf &x) {
    return (1.0 + (-x).array().exp()).inverse().matrix();
}

inline MatrixXf sigmoid_derivative(const MatrixXf &x) {
    return x * (1 - x.array()).matrix();
}

inline MatrixXf relu(const MatrixXf &x) {
    return x.cwiseMax(0.0);
}

MatrixXf tanh(const MatrixXf &x) {
    return sigmoid(x);
}

MatrixXf log_softmax(const MatrixXf &x) {
    auto e = x.array().exp();
    return (e / e.sum()).log().matrix();
}
} // namespace F