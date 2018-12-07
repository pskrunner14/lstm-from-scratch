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

#include "lstm/network.hpp"
#include "util.hpp"

int main() {
    // load data
    std::vector<std::vector<std::string>> X_train = util::load_dataset("data/train/src.txt");
    std::vector<std::vector<std::string>> y_train = util::load_dataset("data/train/tgt.txt");
    std::vector<std::vector<std::string>> X_dev = util::load_dataset("data/dev/src.txt");
    std::vector<std::vector<std::string>> X_dev = util::load_dataset("data/dev/tgt.txt");
    std::vector<std::vector<std::string>> X_test = util::load_dataset("data/test/src.txt");
    std::vector<std::vector<std::string>> X_test = util::load_dataset("data/test/tgt.txt");

    nn::LSTMCell lstm(128, 64, 64);
    MatrixXf input(64, 64);
    input.setRandom();

    MatrixXf h = lstm.forward(input);
    std::cout << h.rows() << "x" << h.cols() << std::endl;

    for (auto const &num : X_train[0]) {
        std::cout << num << std::endl;
    }

    return 0;
}