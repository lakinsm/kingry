#ifndef ARGS_H
#define ARGS_H

#include <iostream>
#include <string>
#include <vector>

struct cmd_args {
    std::string infile;
    std::string metric;
};

static void usage() {
    std::cout << "\nUsage:\n\tcorrelations <infile_path> <metric>" << std::endl;
    std::cout << "Metrics: pearson, rdc" << std::endl << std::endl;
    exit(EXIT_FAILURE);
}

static inline cmd_args
parse_command_line(int argc, const char *argv[]) {
    struct cmd_args args;
    std::vector<std::string> arg_list(argv, argv+argc);
    if(arg_list[2] != "pearson" && arg_list[2] != "rdc")
        usage();
    args.infile = arg_list[1];
    args.metric = arg_list[2];
    return args;
}

#endif // ARGS_H