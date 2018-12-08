#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
using namespace Eigen;

#include "lstm/network.hpp"
#include "util.hpp"

int main() {
    // load data
    std::vector<std::vector<std::string>> X_train, y_train, X_dev, y_dev, X_test, y_test;
    util::load_dataset(X_train, "data/train/src.txt");
    util::load_dataset(y_train, "data/train/tgt.txt");
    util::load_dataset(X_dev, "data/dev/src.txt");
    util::load_dataset(y_dev, "data/dev/tgt.txt");
    util::load_dataset(X_test, "data/test/src.txt");
    util::load_dataset(y_test, "data/test/tgt.txt");

    // create vocabs
    std::map<std::string, int> word_to_idx;
    std::map<int, std::string> idx_to_word;
    util::generate_vocabs(X_train, word_to_idx, idx_to_word);

    std::cout << "Vocabulary (idx: token) with " << word_to_idx.size() << " tokens" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    for (const auto &p : idx_to_word) {
        std::cout << p.first << ": " << p.second << std::endl;
    }

    // create LSTM network
    // nn::LSTMCell lstm(128, 64, 64);
    nn::LSTMNetwork lstm(4, 128, 10, 32, 64);
    MatrixXf input(64, 10);
    input.setRandom();

    // train the model
    MatrixXf probs = lstm(input);
    std::cout << probs.rows() << "x" << probs.cols() << std::endl;

    return 0;
}