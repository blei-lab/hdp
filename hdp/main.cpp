#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "hdp.h"
#define VERBOSE true

gsl_rng * RANDOM_NUMBER;

void print_usage_and_exit()
{
    printf("\nC++ implementation of Gibbs sampling (with split-merge) for hierarchical Dirichlet process.\n");
    printf("Authors: {chongw, blei}@cs.princeton.edu, Computer Science Department, Princeton University.\n");
    printf("usage:\n");
    printf("      hdp [options]\n");
    printf("      general parameters:\n");
    printf("      --algorithm:      train or test, not optional\n");
    printf("      --data:           data file, in lda-c format, not optional\n");
    printf("      --directory:      save directory, not optional\n");
    printf("      --max_iter:       the max number of iterations, default 1000\n");
    printf("      --save_lag:       the saving lag, default 100 (-1 means no savings for intermediate results)\n");
    printf("      --random_seed:    the random seed, default from the current time\n");
    printf("      --init_topics:    the initial number of topics, default 0\n");

    printf("\n      training parameters:\n");
    printf("      --gamma_a:        shape for 1st-level concentration parameter, default 1.0\n");
    printf("      --gamma_b:        scale for 1st-level concentration parameter, default 1.0\n");
    printf("      --alpha_a:        shape for 2nd-level concentration parameter, default 1.0\n");
    printf("      --alpha_b:        scale for 2nd-level concentration parameter, default 1.0\n");
    printf("      --sample_hyper:   sample 1st and 2nd-level concentration parameter, yes or no, default \"no\"\n");
    printf("      --eta:            topic Dirichlet parameter, default 0.5\n");
    printf("      --split_merge:    try split-merge or not, yes or no, default \"no\"\n");
    printf("      --restrict_scan:  number of intermediate scans, default 5 (-1 means no scan)\n");

    printf("\n      testing parameters:\n");
    printf("      --saved_model:    path for saved model, not optional\n");

    printf("\nexamples:\n");
    printf("      ./hdp --algorithm train --data data --directory train_dir\n");
    printf("      ./hdp --algorithm test --data data --saved_model saved_model --directory test_dir\n");
    printf("\n");
    exit(0);
}

int main(int argc, char** argv)
{
    if (argc < 2 || !strcmp(argv[1], "-help") || !strcmp(argv[1], "--help") ||
            !strcmp(argv[1], "-h") || !strcmp(argv[1], "--usage"))
    {
        print_usage_and_exit();
    }

    double gamma_a = 1.0, gamma_b = 1.0, alpha_a = 1.0, alpha_b = 1.0, eta = 0.5;
    int max_iter = 1000, save_lag = 100, init_topics = 0;
    bool sample_hyperparameter = false;

    bool split_merge = false;
    int num_restricted_scan = 5;

    time_t t;
    time(&t);
    long seed = (long) t;

    char* directory = NULL;;
    char* algorithm = NULL;;
    char* data_path = NULL;
    char* model_path = NULL;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--algorithm"))        algorithm = argv[++i];
        else if (!strcmp(argv[i], "--data"))        data_path = argv[++i];
        else if (!strcmp(argv[i], "--max_iter"))    max_iter = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--save_lag"))    save_lag = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--init_topics")) init_topics = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--directory"))   directory = argv[++i];
        else if (!strcmp(argv[i], "--random_seed")) seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gamma_a"))     gamma_a = atof(argv[++i]);
        else if (!strcmp(argv[i], "--gamma_b"))     gamma_b = atof(argv[++i]);
        else if (!strcmp(argv[i], "--alpha_a"))     alpha_a = atof(argv[++i]);
        else if (!strcmp(argv[i], "--alpha_b"))     alpha_b = atof(argv[++i]);
        else if (!strcmp(argv[i], "--eta"))         eta = atof(argv[++i]);
        else if (!strcmp(argv[i], "--restrict_scan")) num_restricted_scan = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--saved_model")) model_path = argv[++i];
        else if (!strcmp(argv[i], "--split_merge"))
        {
           ++i;
            if (!strcmp(argv[i], "yes") ||  !strcmp(argv[i], "YES"))
                split_merge = true;
        }
        else if (!strcmp(argv[i], "--sample_hyper"))
        {
           ++i;
            if (!strcmp(argv[i], "yes") ||  !strcmp(argv[i], "YES"))
                sample_hyperparameter = true;
        }
        else
        {
            printf("%s, unknown parameters, exit\n", argv[i]);
            exit(0);
        }
    }

    if (algorithm == NULL || directory == NULL || data_path == NULL)
    {
        printf("Note that algorithm, directory and data are not optional!\n");
        exit(0);
    }

    if (VERBOSE && !strcmp(algorithm, "train"))
    {
        printf("\nProgram starts with following parameters:\n");

        printf("algorithm:          = %s\n", algorithm);
        printf("data_path:          = %s\n", data_path);
        printf("directory:          = %s\n", directory);

        printf("max_iter            = %d\n", max_iter);
        printf("save_lag            = %d\n", save_lag);
        printf("init_topics         = %d\n", init_topics);
        printf("random_seed         = %d\n", seed);
        printf("gamma_a             = %.2f\n", gamma_a);
        printf("gamma_b             = %.2f\n", gamma_b);
        printf("alpha_a             = %.2f\n", alpha_a);
        printf("alpha_b             = %.2f\n", alpha_b);
        printf("eta                 = %.2f\n", eta);
        printf("#restricted_scans   = %d\n", num_restricted_scan);
        if (model_path != NULL)
        printf("saved model_path    = %s\n", model_path);
        if (split_merge)
        printf("split-merge         = yes\n");
        else
        printf("split-merge         = no\n");
        if (sample_hyperparameter)
        printf("sampling hyperparam = yes\n");
        else
        printf("sampling hyperparam = no\n");
    }

    if (!dir_exists(directory))
        mkdir(directory, S_IRUSR | S_IWUSR | S_IXUSR);

    // allocate the random number structure
    RANDOM_NUMBER = gsl_rng_alloc(gsl_rng_taus);
    gsl_rng_set(RANDOM_NUMBER, (long) seed); // init the seed

    if (!strcmp(algorithm, "train"))
    {
        // read data
        corpus * c = new corpus();
        c->read_data(data_path);

        // read hyperparameters

        hdp_hyperparameter * hdp_hyperparam = new hdp_hyperparameter();
        hdp_hyperparam->setup_parameters(gamma_a, gamma_b,
                                         alpha_a, alpha_b,
                                         max_iter, save_lag,
                                         num_restricted_scan,
                                         sample_hyperparameter,
                                         split_merge);

        hdp * hdp_instance = new hdp();

        hdp_instance->setup_state(c, eta, init_topics,
                                  hdp_hyperparam);

        hdp_instance->run(directory);

        // free resources
        delete hdp_instance;
        delete c;
        delete hdp_hyperparam;
    }

    if (!strcmp(algorithm, "test"))
    {
        corpus* c = new corpus();
        c->read_data(data_path);

        hdp_hyperparameter * hdp_hyperparam = new hdp_hyperparameter();
        hdp_hyperparam->setup_parameters(gamma_a, gamma_b,
                                         alpha_a, alpha_b,
                                         max_iter, save_lag,
                                         num_restricted_scan,
                                         sample_hyperparameter,
                                         split_merge);

        hdp * hdp_instance = new hdp();
        hdp_instance->load(model_path);
        hdp_instance->setup_state(c, hdp_hyperparam);
        hdp_instance->run_test(directory);

        delete hdp_hyperparam;
        delete hdp_instance;
        delete c;
    }
    gsl_rng_free(RANDOM_NUMBER);
}
