// This SVM interface is a wrapper around libsvm for ease of use in our setup.
// Besides being a wrapper around libsvm, we've also modified libsvm such that
// the linear kernel is overwritten by our choice of the histogram intersection
// kernel or the chi-squared kernel.  We won't be using the linear kernel for this 
// project, as it does not achieve the same accuracy that the two replacement kernels
// achieve on our data.

#ifndef SVMINTERFACE_H
#define SVMINTERFACE_H

#include "svm.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <errno.h>
#include <iostream>
#include <vector>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

class SVMInterface
{
public:
	SVMInterface();
	~SVMInterface();
	void trainModel(std::string training_file, std::string model_file_name);
	//void trainModelProb(std::string training_file);
	double testModel(std::string testing_file, std::string model_file_name, std::string output_file);
	double testModelTRECVID(std::string testing_file, std::string model_file_name);
	//double testModelProb(std::string testing_file);
	bool classifyInstance(std::string instance, int label, float label_probability);

private:
	void setParameters(svm_parameter *param, bool regressor);
	void read_problem(const std::string filename);
	double predict(FILE *input, FILE *output);
	double predictAndReturnData(FILE *input, std::vector<int> &example_labels, std::vector<std::vector<int> > class_labels, std::vector<std::vector<double> > &probs, std::vector<int> &best_labels);
	static char* readline(FILE *input);
	void exit_input_error(int line_num);

	struct svm_problem prob; // set by read_problem
	struct svm_parameter param;
	struct svm_model *model;
	struct svm_model *model2;
	struct svm_node *x_space;
	struct svm_node *x;

	static int max_line_len;
	static char *line;
	int predict_probability;
	int max_nr_attr; // this is dumb. no way.
};

#endif