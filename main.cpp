#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <limits>
#include <map>
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
    std::vector<std::vector<std::string>> y_dev = util::load_dataset("data/dev/tgt.txt");
    std::vector<std::vector<std::string>> X_test = util::load_dataset("data/test/src.txt");
    std::vector<std::vector<std::string>> y_test = util::load_dataset("data/test/tgt.txt");

    // create vocabs
    std::pair<std::map<std::string, int>, std::map<int, std::string>> vocabs = util::generate_vocabs(X_train);
    std::map<std::string, int> word_to_idx = vocabs.first;
    std::map<int, std::string> idx_to_word = vocabs.second;

    std::cout << "Vocabulary (idx: token) with " << word_to_idx.size() << " tokens" << std::endl;
    std::cout << "--------------------------------------" << std::endl;
    for (const auto &p : idx_to_word) {
        std::cout << p.first << ": " << p.second << std::endl;
    }

    // create LSTM network
    nn::LSTMCell lstm(128, 64, 64);
    MatrixXf input(64, 64);
    input.setRandom();

    // train the model
    MatrixXf h = lstm.forward(input);
    std::cout << h.rows() << "x" << h.cols() << std::endl;

    return 0;
}