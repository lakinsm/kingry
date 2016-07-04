#include "args.h"

#include "../Eigen/Core"
#include "../Eigen/Dense"
#include "../Eigen/QR"
#include <string>
#include <cstring>
#include <iostream>
#include <sstream>
#include <fstream>
#include <random>
#include <cmath>

template<typename Vector>
std::vector<float> rank(const Vector& v)
{
    std::vector<std::size_t> w(v.size());
    std::iota(begin(w), end(w), 0);
    std::sort(begin(w), end(w),
              [&v](std::size_t i, std::size_t j) { return v[i] < v[j]; });

    std::vector<float> r(w.size());
    for (std::size_t n, i = 0; i < w.size(); i += n)
    {
        n = 1;
        while (i + n < w.size() && v[w[i]] == v[w[i+n]]) ++n;
        for (std::size_t k = 0; k < n; ++k)
        {
            r[w[i+k]] = i + (n + 1) / 2.0; // average rank of n tied values
            // r[w[i+k]] = i + 1;          // min
            // r[w[i+k]] = i + n;          // max
            // r[w[i+k]] = i + k + 1;      // random order
        }
    }
    return r;
}

typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> Matrix_MxNf;

inline float Sin(float &x) {
    return std::sin(x);
}

template<class Matrix>
float rdc_eval(Matrix &x, Matrix &y) {
    float res;
    int k = 20;
    float s = (float)1 / (float)6;

    //RNG for random normal
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d(0,1);
    Matrix_MxNf randX(2, k);
    Matrix_MxNf randY(2, k);

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < k; ++j) {
            randX(i, j) = d(gen);
            randY(i, j) = d(gen);
        }
    }

    //std::cout << randX << std::endl;

    // Step 1: Calculate the rank vectors with bias column
    std::vector<float> vec_x;
    std::vector<float> vec_y;

    for(int i = 0; i < x.rows(); ++i) {
        vec_x.push_back(x(i));
        vec_y.push_back(y(i));
    }

    //std::cout << "check" << std::endl;

    std::vector<float> r_x = rank(vec_x);
    std::vector<float> r_y = rank(vec_y);

    //std::cout << r_x.size() << std::endl;
    //std::cout << r_y.size() << std::endl;

    //std::cout << "check2" << std::endl;

    Matrix_MxNf s1_x(vec_x.size(), 2);
    Matrix_MxNf s1_y(vec_x.size(), 2);

    for(int i = 0; i < r_x.size(); ++i) {
        s1_x(i, 0) = r_x[i] / r_x.size();
        s1_x(i, 1) = 1;
        s1_y(i, 0) = r_y[i] / r_y.size();
        s1_y(i, 1) = 1;
    }

    //std::cout << s1_x << std::endl;
    //std::cout << s1_y << std::endl;

    // Step 2: Multiply s1 matrices by coeff and random normal matrix
    Matrix_MxNf s2_x(x.rows()+1, k);
    Matrix_MxNf s2_y(y.rows()+1, k);
    s2_x = (s/2) * (s1_x * randX);
    s2_y = (s/2) * (s1_y * randY);


    // Step 3: Apply sine transformation
    s2_x.unaryExpr(&Sin);
    s2_y.unaryExpr(&Sin);

    //std::cout << s2_x << std::endl;
    std::cout << s2_x.rows() << ' ' << s2_x.cols() << std::endl;

    // Step 4: Canonical correlation using QR decomposition and the SVD
    Eigen::HouseholderQR<Matrix_MxNf> qrx(s2_x);
    Eigen::HouseholderQR<Matrix_MxNf> qry(s2_y);
    Matrix_MxNf Qx;
    Matrix_MxNf Qy;
    Qx = qrx.householderQ();
    Qy = qry.householderQ();
    Matrix_MxNf resQ(Qx.cols(), Qy.cols());
    Qx.transposeInPlace();
    resQ = Qx * Qy;

    Eigen::JacobiSVD<Matrix_MxNf> svd(resQ, Eigen::ComputeThinU | Eigen::ComputeThinV);
    res = svd.singularValues()(0);
    return res;
}

template<class Matrix>
void calculate_rdc(Matrix &M, Matrix &data) {
    // Do square indices here for openMP optimization
#pragma omp parallel for
    for(int m=0; m < data.rows(); ++m) {
        for(int n=0; n < data.rows(); ++n) {
            if(n <= m) {
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
            M(m,n) = std::atof(elem.c_str());
            n++;
        }
        m++;
        //std::cout << M.row(m) << std::endl;
    }
}

int main(int argc, const char *argv[]) {
    struct cmd_args args;
    args = parse_command_line(argc, argv);

    Matrix_MxNf kingry(11599,108);
    Matrix_MxNf results(11599,11599);
    load_data(args.infile, kingry);

    std::cout << kingry.rows() << ' ' << kingry.cols() << std::endl;
    //std::cout << kingry.row(11598) << std::endl;

    Matrix_MxNf subset(11599,30);
    std::vector<int> slung = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29};
    std::vector<int> sspleen = {54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83};
    std::vector<int> llung = {0,1,2,3,4,5,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53};
    std::vector<int> lspleen = {54,55,56,57,58,59,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107};

    // Testing
    Matrix_MxNf test1(100, 1);
    Matrix_MxNf test2(100, 1);
    for(int i = 0; i < 100; ++i) {
        test1(i, 0) = i;
        test2(99-i, 0) = i;
    }

    //std::cout << test1 << std::endl;
    //std::cout << test2 << std::endl;

    float f = rdc_eval(test1, test2);

    std::cout << f << std::endl;

    std::exit(0);

    //TODO: link calculate_rdc to rdc_eval so that the below functions work

    for( int i = 0; i < 30; ++i ) {
        subset.col(i) = kingry.col(slung[i]);
    }
    calculate_rdc(results, subset);
    //output_results(M);

    for( int i = 0; i < 30; ++i ) {
        subset.col(i) = kingry.col(sspleen[i]);
    }
    calculate_rdc(results, subset);
    //output_results(M);

    for( int i = 0; i < 30; ++i ) {
        subset.col(i) = kingry.col(llung[i]);
    }
    calculate_rdc(results, subset);
    //output_results(M);

    for( int i = 0; i < 30; ++i ) {
        subset.col(i) = kingry.col(lspleen[i]);
    }
    calculate_rdc(results, subset);
    //output_results(M);

    // TODO: write a function for pearson

    return 0;
}