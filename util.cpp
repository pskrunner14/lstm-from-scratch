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

#include <boost/filesystem.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/string_file.hpp>
#include <boost/timer/timer.hpp>

std::vector<std::string> load_dataset(const std::string &filename) {
    std::string str;
    std::vector<std::string> contents;

    boost::filesystem::ifstream file(filename);
    while (std::getline(file, str)) {
        contents.push_back(str);
    }

    return contents;
}