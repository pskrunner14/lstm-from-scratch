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

#include "functions.hpp"

class LSTMCell {
  private:
    // input operations' weights
    Eigen::MatrixXf Wxi;
    Eigen::MatrixXf Wxf;
    Eigen::MatrixXf Wxc;
    Eigen::MatrixXf Wxo;

    // hidden operations' weights
    Eigen::MatrixXf Whi;
    Eigen::MatrixXf Whf;
    Eigen::MatrixXf Whc;
    Eigen::MatrixXf Who;

    // cell operations' weights
    Eigen::MatrixXf Wci;
    Eigen::MatrixXf Wcf;
    Eigen::MatrixXf Wco;

    // bias terms for operations
    Eigen::RowVectorXf bi;
    Eigen::RowVectorXf bf;
    Eigen::RowVectorXf bc;
    Eigen::RowVectorXf bo;

  protected:
    Eigen::MatrixXf h; // hidden state
    Eigen::MatrixXf c; // cell state

    friend class LSTM;

  public:
    explicit LSTMCell(const int &, const int &);

    Eigen::MatrixXf &forward(const Eigen::MatrixXf &);

    void backward();
};

/**
 * LSTMCell Constructor.
 * 
 * @param hidden_size the size of the hidden state and cell state.
 * @param batch_size the size of batch used during training (for vectorization purposes).
 */
LSTMCell::LSTMCell(const int &hidden_size, const int &batch_size) {
}

/**
 * LSTMCell Forward Pass.
 * 
 * @param xt the input vector at time-step t.
 * @returns the output i.e. the hidden state at time-step t.
 */
Eigen::MatrixXf &LSTMCell::forward(const Eigen::MatrixXf &xt) {
    // it: input gate
    Eigen::MatrixXf it = F::sigmoid((Wxi * xt) + (Whi * h) + (Wci * c) + bi);
    // ft: forget gate
    Eigen::MatrixXf ft = F::sigmoid((Wxf * xt) + (Whf * h) + (Wcf * c) + bf);
    // update cell state
    c = (ft * c) + (it * F::tanh((Wxc * xt) + (Whc * h) + bc));
    // ot: output gate
    Eigen::MatrixXf ot = F::sigmoid((Wxo * xt) + (Who * h) + (Wco * c) + bo);
    // update hidden state
    h = ot * F::sigmoid(c);
    return h;
}

/**
 * LSTMCell Backward Pass.
 * 
 * @param xt the input vector at time-step t.
 * @returns the output i.e. the hidden state at time-step t.
 */
void LSTMCell::backward() {
}

class LSTM {
};