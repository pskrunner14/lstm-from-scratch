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
#include "lstm/cell.hpp"
#include "util.hpp"

namespace nn {

std::string SOS_TOKEN = "~";
std::string EOS_TOKEN = "#";

class LSTMNetwork {
  private:
    size_t n_layers;
    int batch_size;
    int embedding_dim;
    int n_tokens;
    int hidden_size;

    Embedding embedding;
    std::vector<LSTMCell> layers;
    Dense out;

    std::vector<std::vector<LSTMState>> states;

    friend class util::Trainer;

  public:
    explicit LSTMNetwork(const size_t &, const int &, const int &, const int &, const int &);

    MatrixXf operator()(const MatrixXf &);

    MatrixXf forward(const MatrixXf &);

    void backward(const MatrixXf &);
};

LSTMNetwork::LSTMNetwork(const size_t &n_layers, const int &hidden_size, const int &n_tokens, const int &embedding_dim, const int &batch_size)
    : embedding(Embedding(n_tokens, embedding_dim)), out(Dense(hidden_size, n_tokens)) {

    this->n_layers = n_layers;
    this->hidden_size = hidden_size;
    this->n_tokens = n_tokens;
    this->embedding_dim = embedding_dim;
    this->batch_size = batch_size;

    // initial layer (embedding -> hidden)
    layers.push_back(LSTMCell(hidden_size, embedding_dim, batch_size));
    for (size_t i = 1; i < n_layers; i++) {
        // rest of the layers (hidden -> hidden)
        layers.push_back(LSTMCell(hidden_size, hidden_size, batch_size));
    }
}

MatrixXf LSTMNetwork::operator()(const MatrixXf &inputs) {
    return forward(inputs);
}

MatrixXf LSTMNetwork::forward(const MatrixXf &inputs) {
    std::vector<LSTMState> t_states;
    MatrixXf output = embedding(inputs);
    for (size_t i = 0; i < layers.size(); i++) {
        output = layers[i](output);
        t_states.push_back(layers[i].state);
    }
    states.push_back(t_states);
    MatrixXf logits = out(output);
    return F::log_softmax(logits);
}

// void LSTMNetwork::backward(const MatrixXf &loss_grad) {

//     states.clear();
// }
} // namespace nn