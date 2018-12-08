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
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/string_file.hpp>
#include <boost/timer/timer.hpp>

namespace util {

/**
 * Load Dataset.
 * 
 * Loads a dataset by reading file into memory.
 * 
 * @param filename the name of the file containing data.
 * @returns the list of lists of tokens by line.
 */
std::vector<std::vector<std::string>> load_dataset(const std::string &filename) {
    std::string line;
    std::vector<std::vector<std::string>> contents;

    boost::filesystem::ifstream file(filename);
    while (std::getline(file, line)) {
        std::vector<std::string> strs;
        boost::split(strs, line, boost::is_any_of(" "));
        contents.push_back(strs);
    }
    return contents;
}

/**
 * Generate Vocabs.
 * 
 * Generates token-to-id and id-to-token vocabularies using training corpus.
 * 
 * @param corpus the corpus containing sequences of tokens.
 * @returns the pair of token-to-id and id-to-token maps.
 */
std::pair<std::map<std::string, int>, std::map<int, std::string>> generate_vocabs(const std::vector<std::vector<std::string>> &corpus) {
    std::set<std::string> tokens;
    std::map<std::string, int> word_to_idx;
    std::map<int, std::string> idx_to_word;

    for (const auto &line : corpus) {
        for (const auto &word : line) {
            tokens.insert(word);
        }
    }

    int index = 0;
    for (const auto &token : tokens) {
        word_to_idx[token] = index;
        idx_to_word[index] = token;
        index++;
    }
    return std::pair<std::map<std::string, int>, std::map<int, std::string>>(word_to_idx, idx_to_word);
}
} // namespace util