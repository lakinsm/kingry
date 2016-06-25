#include "args.h"

#include "../Eigen/Core"
#include "../Eigen/Dense"

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
                M(m,n) = rdc_eval(data.row(m), data.row(n));
            }
        }
    }
}

int main(int argc, const char *argv[]) {
    struct cmd_args args;
    args = parse_command_line(argc, argv);

    // TODO: Load initial data matrix here
    // TODO: call the calculate_rdc() function
    // TODO: write a function for pearson

    return 0;
}