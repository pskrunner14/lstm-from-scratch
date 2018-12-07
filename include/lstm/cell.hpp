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
#include "layers.hpp"
#include "lstm/network.hpp"

namespace nn {

struct LSTMState {
    // hidden state
    MatrixXf h;
    // cell state
    MatrixXf c;
};

class LSTMCell {
  private:
    int batch_size;
    int hidden_size;
    int embedding_dim;

    Dense i2h;
    Dense h2h;

  protected:
    LSTMState states;

    friend class LSTMNetwork;

  public:
    explicit LSTMCell(const int &, const int &, const int &);

    MatrixXf &forward(const MatrixXf &);

    // MatrixXf backward(const MatrixXf &, const MatrixXf &);
};

/**
 * LSTMCell Constructor.
 * 
 * @param hidden_size the size of the hidden state and cell state.
 * @param batch_size the size of batch used during training (for vectorization purposes).
 */
LSTMCell::LSTMCell(const int &hidden_size, const int &embedding_dim, const int &batch_size)
    : i2h(Dense(embedding_dim, 4 * hidden_size)), h2h(Dense(hidden_size, 4 * hidden_size)) {

    this->hidden_size = hidden_size;
    this->embedding_dim = embedding_dim;
    this->batch_size = batch_size;

    states = LSTMState{MatrixXf(batch_size, hidden_size).setRandom() * F::glorot_uniform(batch_size, hidden_size),
                       MatrixXf(batch_size, hidden_size).setRandom() * F::glorot_uniform(batch_size, hidden_size)};
}

/**
 * LSTMCell Forward Pass.
 * 
 * @param xt the input vector at time-step t.
 * @returns the next hidden state for input into next lstm layer.
 */
MatrixXf &LSTMCell::forward(const MatrixXf &xt) {
    // i2h + h2h = [it_pre, ft_pre, ot_pre, x_pre]
    MatrixXf preactivations = i2h.forward(xt) + h2h.forward(states.h);
    // all pre sigmoid gates chunk
    MatrixXf pre_sigmoid_chunk = preactivations.block(0, 0, batch_size, 3 * hidden_size);
    // compute sigmoid on gates chunk
    MatrixXf all_gates = F::sigmoid(pre_sigmoid_chunk);
    // compute c_in (x_transform) i.e. information vector
    MatrixXf x_pre = preactivations.block(0, 3 * hidden_size, batch_size, hidden_size);
    MatrixXf x_transform = F::tanh(x_pre);
    // single out all the gates
    MatrixXf it = all_gates.block(0, 0, batch_size, hidden_size);
    MatrixXf ft = all_gates.block(0, hidden_size, batch_size, hidden_size);
    MatrixXf ot = all_gates.block(0, 2 * hidden_size, batch_size, hidden_size);
    // update cell state
    MatrixXf c_forget = ft.cwiseProduct(states.c);
    MatrixXf c_input = it.cwiseProduct(x_transform);
    states.c = c_forget + c_input;
    // compute next hidden state
    MatrixXf c_transform = F::tanh(states.c);
    states.h = ot.cwiseProduct(c_transform);
    return states.h;
}

/**
 * LSTMCell Backward Pass.
 * 
 * @param inputs the input vector given at time-step t.
 * @param gradients the gradients from upper layers computed using chain rule.
 * @returns the output i.e. the hidden state at time-step t.
 */
// MatrixXf LSTMCell::backward(const MatrixXf &inputs, const MatrixXf &gradients) {
// }
} // namespace nn