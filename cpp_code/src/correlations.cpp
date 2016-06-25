#include "args.h"

#include "../Eigen/Core"
#include "../Eigen/Dense"
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_MxNf;

template<class Matrix>
void calculate_rdc(Matrix &M, Matrix &data) {
    // Do square indices here for omp optimization
#pragma omp parallel for
    for(int m=0; m < data.rows(); ++m) {
        for(int n=0; n < data.rows(); ++n) {
            if(n <= m) {
                // TODO: Use colidx as groups for comparison
                // TODO: Write rdc_eval()
                //M(m,n) = rdc_eval(data.row(m), data.row(n));
            }
        }
    }
}

template <class Matrix>
void load_data(std::string &input_file, Matrix &M) {
    std::ifstream ifs(input_file);
    if(!ifs) {
        std::cerr << "Unable to open input matrix file" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string line;
    std::string elem;
    int m = 0;
    while(std::getline(ifs, line)) {
        int n = 0;
        std::istringstream linestream(line);
        while(std::getline(linestream, elem, ',')) {
            //TODO: Fix this to actually work
            M(m,n) = std::atof(elem.c_str());
            n++;
        }
        m++;
        std::cout << M.row(m) << std::endl;
    }
}

int main(int argc, const char *argv[]) {
    struct cmd_args args;
    args = parse_command_line(argc, argv);

    Matrix_MxNf kingry(11599,108);
    load_data(args.infile, kingry);
//
//    std::cout << kingry.rows() << ' ' << kingry.cols() << std::endl;
//
//    // TODO: call the calculate_rdc() function
//    // TODO: write a function for pearson

    return 0;
}