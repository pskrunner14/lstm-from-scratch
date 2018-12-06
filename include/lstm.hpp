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

struct LSTMState {
    // hidden state
    MatrixXf h;
    // cell state
    MatrixXf c;
};

class LSTMCell {
  private:
    // operations' weights
    MatrixXf Wi;
    MatrixXf Wf;
    MatrixXf Wc;
    MatrixXf Wo;

    // operations' biases
    RowVectorXf bi;
    RowVectorXf bf;
    RowVectorXf bc;
    RowVectorXf bo;

  protected:
    LSTMState states;

    friend class LSTMNetwork;

  public:
    explicit LSTMCell(const int &, const int &);

    void forward(const MatrixXf &);

    std::pair<MatrixXf, MatrixXf> backward(const MatrixXf &, const MatrixXf &);
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
 */
void LSTMCell::forward(const MatrixXf &xt) {
    // concatenate ht-1 and xt
    MatrixXf zt(states.h.rows(), xt.cols() + xt.cols());
    zt << states.h, xt;
    // it: input gate
    MatrixXf it = F::sigmoid((Wi * zt) + bi);
    // ft: forget gate
    MatrixXf ft = F::sigmoid((Wf * zt) + bf);
    // update cell state
    MatrixXf ct_update = F::tanh((Wc * zt) + bc);
    states.c = (ft * states.c) + (it * ct_update);
    // ot: output gate
    MatrixXf ot = F::sigmoid((Wo * zt) + bo);
    // update hidden state
    states.h = ot * F::sigmoid(states.c);
}

/**
 * LSTMCell Backward Pass.
 * 
 * @param xt the input vector at time-step t.
 * @returns the output i.e. the hidden state at time-step t.
 */
std::pair<MatrixXf, MatrixXf> LSTMCell::backward(const MatrixXf &inputs, const MatrixXf &gradients) {
}

class LSTMNetwork {
};
} // namespace nn