
#include "SVMInterface.h"

SVMInterface::SVMInterface()
{
	line = NULL;
	max_line_len = 1024;
	predict_probability = 0;
	max_nr_attr = 64; // this is dumb. no way.
}

void SVMInterface::exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

char* SVMInterface::readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

// we don't need to do all the file parsing here!
// just read in the data into x[0] and send it off for classification!
// svm_predict_probability(model,x,prob_estimates); 
// that returns a double, which is the classification.
double SVMInterface::predictAndReturnData(FILE *input, std::vector<int> &example_labels, std::vector<std::vector<int> > class_labels, std::vector<std::vector<double> > &probs, std::vector<int> &best_labels)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	double returned_accuracy = -1;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;

	if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
		{
			std::vector<int> labels_for_current_instance;
			// print the initial line showing the label header
			// and allocate space for probability estimates and label estimates.
			int *labels=(int *) malloc(nr_class*sizeof(int));

			// i can not explain why svm_get_labels does not print the labels in order...
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));

			for(j=0;j<nr_class;j++)
			{
				labels_for_current_instance.push_back(labels[j]);
			}
			free(labels);

			class_labels.push_back(labels_for_current_instance);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		// strtok up to first space... just grabs the first part, the label.
		label = strtok(line," \t\n");//(char *)example_labels[i];//strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		//strtod is string to double.
		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		// now read in remainder of feature:value pairs.
		while(1)
		{
			// need more attribute space.  gets allocated here.
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			// idx is the feature id.
			idx = strtok(NULL,":");
			// val is the feature value.
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			// strtol is string to long.  stores the feature id into the long value x[i].index
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index; // maximum feature index for this instance.

			errno = 0;
			// store value as a double in x[i].value
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		// we denote the end of the features by setting the index after the last feature index to -1.
		x[i].index = -1;

		// prob_estimates holds the probability estimates for each class...
		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			std::vector<double> probability_estimates;
			// x holds all of the feature/instance pairs.
			predict_label = svm_predict_probability(model,x,prob_estimates);
			best_labels.push_back(predict_label);
			for(j=0;j<nr_class;j++)
			{
				probability_estimates.push_back(prob_estimates[j]);
			}
			probs.push_back(probability_estimates);
		}
		else
		{
			predict_label = svm_predict(model,x);
		}

		// correctly labeled.
		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n",error/total);
		printf("Squared correlation coefficient = %g (regression)\n",
		       ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
		       ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
		       );
	}
	else
	{
		returned_accuracy = (double) correct / total * 100;
		printf("Accuracy = %g%% (%d/%d) (classification)\n",
		       (double)correct/total*100,correct,total);
	}
	if(predict_probability)
		free(prob_estimates);

	return returned_accuracy;
}

double SVMInterface::predict(FILE *input, FILE *output)
{
	int correct = 0;
	int total = 0;
	double error = 0;
	double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
	double returned_accuracy = -1;

	int svm_type=svm_get_svm_type(model);
	int nr_class=svm_get_nr_class(model);
	double *prob_estimates=NULL;
	int j;

	if(predict_probability)
	{
		if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
			printf("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
		else
		{
			// print the initial line showing the label header
			// and allocate space for probability estimates and label estimates.
			int *labels=(int *) malloc(nr_class*sizeof(int));

			// i can not explain why svm_get_labels does not print the labels in order...
			svm_get_labels(model,labels);
			prob_estimates = (double *) malloc(nr_class*sizeof(double));
			fprintf(output,"labels");		
			for(j=0;j<nr_class;j++)
				fprintf(output," %d",labels[j]);
			fprintf(output,"\n");
			free(labels);
		}
	}

	max_line_len = 1024;
	line = (char *)malloc(max_line_len*sizeof(char));
	while(readline(input) != NULL)
	{
		int i = 0;
		double target_label, predict_label;
		char *idx, *val, *label, *endptr;
		int inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0

		// strtok up to first space... just grabs the first part, the label.
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(total+1);

		//strtod is string to double.
		target_label = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(total+1);

		// now read in remainder of feature:value pairs.
		while(1)
		{
			// need more attribute space.  gets allocated here.
			if(i>=max_nr_attr-1)	// need one more for index = -1
			{
				max_nr_attr *= 2;
				x = (struct svm_node *) realloc(x,max_nr_attr*sizeof(struct svm_node));
			}

			// idx is the feature id.
			idx = strtok(NULL,":");
			// val is the feature value.
			val = strtok(NULL," \t");

			if(val == NULL)
				break;
			errno = 0;
			// strtol is string to long.  stores the feature id into the long value x[i].index
			x[i].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x[i].index <= inst_max_index)
				exit_input_error(total+1);
			else
				inst_max_index = x[i].index; // maximum feature index for this instance.

			errno = 0;
			// store value as a double in x[i].value
			x[i].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(total+1);

			++i;
		}
		// we denote the end of the features by setting the index after the last feature index to -1.
		x[i].index = -1;

		// prob_estimates holds the probability estimates for each class...
		if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
		{
			// x holds all of the feature/instance pairs.
			predict_label = svm_predict_probability(model,x,prob_estimates);
			fprintf(output,"%g",predict_label);
			for(j=0;j<nr_class;j++)
				fprintf(output," %g",prob_estimates[j]);
			fprintf(output,"\n");
		}
		else
		{
			predict_label = svm_predict(model,x);
			fprintf(output,"%g\n",predict_label);
		}

		// correctly labeled.
		if(predict_label == target_label)
			++correct;
		error += (predict_label-target_label)*(predict_label-target_label);
		sump += predict_label;
		sumt += target_label;
		sumpp += predict_label*predict_label;
		sumtt += target_label*target_label;
		sumpt += predict_label*target_label;
		++total;
	}
	if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
	{
		printf("Mean squared error = %g (regression)\n",error/total);
		printf("Squared correlation coefficient = %g (regression)\n",
		       ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
		       ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
		       );
	}
	else
	{
		returned_accuracy = (double) correct / total * 100;
		printf("Accuracy = %g%% (%d/%d) (classification)\n",
		       (double)correct/total*100,correct,total);
	}
	if(predict_probability)
		free(prob_estimates);

	return returned_accuracy;
}

// read in a problem (in svmlight format)
//void SVMInterface::read_problem(const char *filename)
void SVMInterface::read_problem(const std::string filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename.c_str(),"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0) 
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
	//std::cout << " deleting lots" << std::endl;
	//delete fp;
	//std::cout << "deleted fp and now endptr" << std::endl;
	//delete endptr;
	//std::cout << "deleted endptr and now idx" << std::endl;
	//delete idx;
	//std::cout << " deleted up to idx but not val" << std::endl;
	delete val;
	delete label;
	//std::cout << " deleted all" << std::endl;
}

void SVMInterface::setParameters(svm_parameter *param, bool regressor)
{
	if (regressor)
	{
		param->svm_type = EPSILON_SVR;
		param->probability = 1;
	}
	else
	{
		param->svm_type = C_SVC;
		param->probability = 0;
	}
	param->kernel_type = LINEAR;
	//param->degree = 3;
	param->gamma = 1/double(600);	// or 1/num_features
	param->coef0 = 0;
	param->nu = 0.5;
	param->cache_size = 100;
	param->C = 1;
	param->eps = 1e-3;
	param->p = 0.1;
	param->shrinking = 1;
	param->nr_weight = 0;
	param->weight_label = NULL;
	param->weight = NULL;
}

/*
void SVMInterface::trainModelProb(std::string training_data_file)
{
	const char *error_msg;
	const std::string model_file_name = "C:/data/kth/chris/trained_svm.model";

	read_problem(training_data_file);
	setParameters(&param, false);
	param.probability = 1;
	error_msg = svm_check_parameter(&prob, &param);

	std::cout << "training model" << std::endl;
	model = svm_train(&prob, &param);
	svm_save_model(model_file_name.c_str(), model);
	std::cout << "Model trained." << std::endl;
}
*/

void SVMInterface::trainModel(std::string training_data_file, std::string model_file_name)
{
	const char *error_msg;

	read_problem(training_data_file);
	setParameters(&param, false);
	error_msg = svm_check_parameter(&prob, &param);

	std::cout << "training model" << std::endl;
	model = svm_train(&prob, &param);
	svm_save_model(model_file_name.c_str(), model);
	std::cout << "Model trained." << std::endl;

	std::cout << "deleting..";
	//delete error_msg;
	std::cout << " error msg" << std::endl;
}

/*
double SVMInterface::testModelProb(std::string testing_data_file)
{
	const std::string model_file_name = "C:/data/TRECVID/svm/model.svm";
	const std::string output_file = "C:/data/TRECVID/svm/responses.txt";
	predict_probability = 1;

	FILE *testing_data, *output;
	testing_data = fopen(testing_data_file.c_str(), "r");
	output = fopen(output_file.c_str(), "w");

	model = svm_load_model(model_file_name.c_str());
	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	double accuracy = predict(testing_data, output);
	
	fclose(output);

	return accuracy;
}
*/

bool SVMInterface::classifyInstance(std::string instance, int label, float label_probability)
{
	return false;
}

double SVMInterface::testModelTRECVID(std::string testing_file, std::string model_file_name)
{
	std::string output_file = testing_file;
	output_file.append(".responses");

	FILE *testing_data, *output;
	testing_data = fopen(testing_file.c_str(), "r");
	output = fopen(output_file.c_str(), "w");

	model = svm_load_model(model_file_name.c_str());
	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	double accuracy = predict(testing_data, output);
	
	fclose(output);

	return accuracy;
}

double SVMInterface::testModel(std::string testing_data_file, std::string model_file_name, std::string output_file)
{
	FILE *testing_data, *output;
	testing_data = fopen(testing_data_file.c_str(), "r");
	output = fopen(output_file.c_str(), "w");

	model = svm_load_model(model_file_name.c_str());
	x = (struct svm_node *) malloc(max_nr_attr * sizeof(struct svm_node));
	double accuracy = predict(testing_data, output);
	
	fclose(output);
	fclose(testing_data);
	//delete output;
	//delete testing_data;
	
	svm_free_and_destroy_model(&model);
	svm_destroy_param(&param);

	return accuracy;
}

SVMInterface::~SVMInterface()
{
	delete x_space;
	delete x;
}

int SVMInterface::max_line_len = 1024;
char *SVMInterface::line = NULL;