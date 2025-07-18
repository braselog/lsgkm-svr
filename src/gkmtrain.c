/* gkmtrain.c
 *
 * Copyright (C) 2015 Dongwon Lee
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <getopt.h>

#include "libsvm_gkm.h"

#ifdef __cplusplus
extern "C" {
#endif

#include "rbf_cuda.h"

#ifdef __cplusplus
}
#endif

#define CLOG_MAIN
#include "clog.h"

///////////////////////////////////////////////////////////////////////////////
void print_usage_and_exit()
{
    printf(
            "\n"
            "Usage: gkmtrain [options] <file1> <file2> <outprefix> [covariates]\n"
            "\n"
            " train gkm-SVM using libSVM\n"
            "\n"
            "Arguments:\n"
            " file1: if classification: positive sequence file (FASTA format). If regression: all sequences file (FASTA format)\n"
            " file2: if classification: negative sequence file (FASTA format). If regression: corresponding labels (one per line)\n"
            " outprefix: prefix of output file(s) <outprefix>.model.txt or\n"
            "            <outprefix>.cvpred.txt\n"
            " covariates: optional tab-separated file with covariates (one row per sequence)\n"
            "\n"
            "Options:\n"
            " -y <0 ~ 4>   set svm type (default: 0 C_SVC)\n"
            "                0 -- C_SVC\n"
            "                1 -- NU_SVC (untested)\n"
            "                2 -- ONE_CLASS (untested)\n"
            "                3 -- EPSILON_SVR\n"
            "                4 -- NU_SVR (untested)\n"
            " -t <0 ~ 9>   set kernel function (default: 4 wgkm)\n"
            "              NOTE: RBF kernels (3, 5 and 6) work best with -c 10 -g 2\n"
            "                0 -- gapped-kmer\n"
            "                1 -- estimated l-mer with full filter\n"
            "                2 -- estimated l-mer with truncated filter (gkm)\n"
            "                3 -- (truncated l-mer) gkm + RBF (gkmrbf)\n"
            "                4 -- (truncated l-mer) gkm + center weighted (wgkm)\n"
            "                     [weight = max(M, floor(M*exp(-ln(2)*D/H)+1))]\n"
            "                5 -- (truncated l-mer) gkm + center weighted + RBF (wgkmrbf)\n"
            "                6 -- gapped-kmer + RBF\n"
            "                7 -- Multiple Kernel Learning: GKM + RBF on covariates\n"
            "                8 -- Multiple Kernel Learning: GKM only (ignores covariates)\n"
            "                9 -- Multiple Kernel Learning: RBF only on covariates\n"
            " -l <int>     set word length, 3<=l<=12 (default: 11)\n"
            " -k <int>     set number of informative column, k<=l (default: 7)\n"
            " -d <int>     set maximum number of mismatches to consider, d<=4 (default: 3)\n"
            " -g <float>   set gamma for RBF kernel. -t 3 or 5 or 6 only (default: 1.0)\n"
            " -G <float>   set gamma for RBF kernel on covariates in MKL. -t 7 or 9 only (default: 1.0)\n"
            " -M <int>     set the initial value (M) of the exponential decay function\n"
            "              for wgkm-kernels. max=255, -t 4 or 5 only (default: 50)\n"
            " -H <float>   set the half-life parameter (H) that is the distance (D) required\n"
            "              to fall to half of its initial value in the exponential decay\n"
            "              function for wgkm-kernels. -t 4 or 5 only (default: 50)\n"
            " -W <float>   set initial weight for GKM kernel in MKL. -t 7 only (default: 0.5)\n"
            " -R <float>   set initial weight for RBF kernel in MKL. -t 7 only (default: 0.5)\n"
            " -I <int>     set max iterations for MKL optimization (default: 100)\n"
            " -E <float>   set convergence tolerance for MKL optimization (default: 1e-6)\n"
            " -N           if set, normalize individual kernels in MKL before combining\n"
            " -c <float>   set the regularization parameter SVM-C (default: 1.0)\n"
            " -e <float>   set the precision parameter epsilon (default: 0.001)\n"
            " -p <float>   set the SVR epsilon (default: 0.1)\n"
            " -w <float>   set the parameter SVM-C to w*C for the positive set (default: 1.0)\n"
            " -m <float>   set cache memory size in MB (default: 100.0)\n"
            "              NOTE: Large cache signifcantly reduces runtime. >4Gb is recommended\n" 
            " -s           if set, use the shrinking heuristics\n"
            " -x <int>     set N-fold cross validation mode (default: no cross validation)\n"
            " -i <int>     run i-th cross validation only 1<=i<=ncv (default: all)\n"
            " -r <int>     set random seed for shuffling in cross validation mode (default: 1)\n"
            " -v <0 ~ 4>   set the level of verbosity (default: 2)\n"
            "                0 -- error msgs only (ERROR)\n"
            "                1 -- warning msgs (WARN)\n"
            "                2 -- progress msgs at coarse-grained level (INFO)\n"
            "                3 -- progress msgs at fine-grained level (DEBUG)\n"
            "                4 -- progress msgs at finer-grained level (TRACE)\n"
            "-T <1|4|16>   set the number of threads for parallel calculation, 1, 4, or 16\n"
            "                 (default: 1)\n"
            "\n"
            "GPU Memory Management Options:\n"
            " --gpu-diag    print detailed GPU memory diagnostics and optimization stats\n"
            " --gpu-tune    perform auto-tuning of batch sizes before training\n" 
            " --gpu-reserve <MB>  reserve GPU memory for other operations (default: 0)\n"
            "\n");

	exit(0);
}

void read_problem_svc(const char *file1, const char *file2);
void read_problem_svr(const char *file1, const char *file2);
void read_problem_svc_with_covariates(const char *file1, const char *file2, const char *cov_file);
void read_problem_svr_with_covariates(const char *file1, const char *file2, const char *cov_file);
void do_cross_validation(const char *filename);

static struct svm_parameter param;
static struct svm_problem prob;        // set by read_problem
static struct svm_model *model;
static int cross_validation;
static int icv;
static int nr_fold;

static char *line = NULL;
static int max_line_len;

// this function was copied from libsvm & slightly modified 
static char* readline(FILE *input)
{
    if(fgets(line,max_line_len,input) == NULL)
        return NULL;

    while(strrchr(line,'\n') == NULL)
    {
        max_line_len *= 2;
        line = (char *) realloc(line, (size_t) max_line_len);
        int len = (int) strlen(line);
        if(fgets(line+len,max_line_len-len,input) == NULL)
            break;
    }
    
    //remove CR ('\r') or LF ('\n'), whichever comes first
    line[strcspn(line, "\r\n")] = '\0';

    return line;
}

int main(int argc, char** argv)
{
    char model_file_name[1024];
    char cvpred_file_name[1024];
    const char *error_msg;

    int verbosity = 2;
    int nthreads = 1;
	int rseed = 1;
    int tmpM;
    char *covariate_file = NULL;
    
    // GPU memory management options
    int gpu_diagnostics = 0;
    int gpu_auto_tune = 0;
    int gpu_reserve_mb = 0;

    /* Initialize the logger */
    if (clog_init_fd(LOGGER_ID, 1) != 0) {
        fprintf(stderr, "Logger initialization failed.\n");
        return 1;
    }

    clog_set_fmt(LOGGER_ID, LOGGER_FORMAT);
    clog_set_level(LOGGER_ID, CLOG_INFO);

	if(argc == 1) { print_usage_and_exit(); }

    // default values
    param.svm_type = C_SVC;
    param.kernel_type = EST_TRUNC_PW;
    param.L = 11;
    param.k = 7;
    param.d = 3;
    param.M = 50;
    param.H = 50;
    param.gamma = 1.0;
    param.rbf_gamma = 1.0;
    param.gkm_weight = 0.5;
    param.rbf_weight = 0.5;
    param.mkl_iterations = 100;
    param.mkl_tolerance = 1e-6;
    param.normalize_kernels = 0;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.shrinking = 0;
    param.nr_weight = 0;
    param.weight_label = (int *) malloc(sizeof(int)*1);
    param.weight = (double *) malloc(sizeof(double)*1);
    param.p = 0.1; //for SVR
    param.probability = 0; //not used
    param.nu = 0.5; //not used
    cross_validation = 0;
    icv = 0;

    // Long options for GPU management
    static struct option long_options[] = {
        {"gpu-diag", no_argument, 0, 1000},
        {"gpu-tune", no_argument, 0, 1001},
        {"gpu-reserve", required_argument, 0, 1002},
        {0, 0, 0, 0}
    };

	int c;
	int option_index = 0;
	while ((c = getopt_long(argc, argv, "y:t:l:k:d:g:G:M:H:W:R:I:E:c:e:p:w:m:x:i:r:sv:T:N", 
                           long_options, &option_index)) != -1) {
		switch (c) {
            case 'y':
                param.svm_type = atoi(optarg);
                break;
            case 't':
                param.kernel_type = atoi(optarg);
                break;
            case 'l':
                param.L = atoi(optarg);
                break;
            case 'k':
                param.k = atoi(optarg);
                break;
            case 'd':
                param.d = atoi(optarg);
                break;
            case 'g':
                param.gamma = atof(optarg);
                break;
            case 'G':
                param.rbf_gamma = atof(optarg);
                break;
            case 'M':
                tmpM = atoi(optarg);
                if (tmpM > 255) {
                    clog_warn(CLOG(LOGGER_ID), "maximum M is 255. M is set to default value [%d].", param.M);
                } else {
                    param.M = (uint8_t) tmpM;
                }
                break;
            case 'H':
                param.H = atof(optarg);
                break;
            case 'W':
                param.gkm_weight = atof(optarg);
                break;
            case 'R':
                param.rbf_weight = atof(optarg);
                break;
            case 'I':
                param.mkl_iterations = atoi(optarg);
                break;
            case 'E':
                param.mkl_tolerance = atof(optarg);
                break;
            case 'N':
                param.normalize_kernels = 1;
                break;
			case 'c':
				param.C = atof(optarg);
				break;
			case 'e':
				param.eps = atof(optarg);
				break;
			case 'p':
				param.p = atof(optarg);
				break;
			case 'w':
                param.nr_weight = 1;
                param.weight_label[0] = 1;
                param.weight[0] = atof(optarg);
				break;
            case 'm':
                param.cache_size = atof(optarg);
                break;
			case 'x':
                cross_validation = 1;
                nr_fold = atoi(optarg);
                if(nr_fold < 2) {
                    fprintf(stderr,"n-fold cross validation: n must >= 2\n");
                    print_usage_and_exit();
                }
				break;
			case 'i':
				icv = atoi(optarg);
				break;
			case 'r':
				rseed = atoi(optarg);
				break;
            case 's':
                param.shrinking = 1;
                break;
            case 'v':
                verbosity = atoi(optarg);
                break;
            case 'T':
                nthreads = atoi(optarg);
                break;
            case 1000:  // --gpu-diag
                gpu_diagnostics = 1;
                break;
            case 1001:  // --gpu-tune
                gpu_auto_tune = 1;
                break;
            case 1002:  // --gpu-reserve
                gpu_reserve_mb = atoi(optarg);
                break;
			default:
                if (c != 0) {
                    fprintf(stderr,"Unknown option: -%c\n", c);
                } else {
                    fprintf(stderr,"Unknown option: %s\n", long_options[option_index].name);
                }
                print_usage_and_exit();
		}
	}

    if (argc - optind < 3 || argc - optind > 4) {
        fprintf(stderr,"Wrong number of arguments [%d]. Expected 3 or 4.\n", argc - optind);
        print_usage_and_exit();
    }

	int index = optind;
	char *file1 = argv[index++];
	char *file2 = argv[index++];
	char *outprefix = argv[index++];
	if (argc - optind == 4) {
        covariate_file = argv[index];
    }

    switch(verbosity) 
    {
        case 0:
            clog_set_level(LOGGER_ID, CLOG_ERROR);
            break;
        case 1:
            clog_set_level(LOGGER_ID, CLOG_WARN);
            break;
        case 2:
            clog_set_level(LOGGER_ID, CLOG_INFO);
            break;
        case 3:
            clog_set_level(LOGGER_ID, CLOG_DEBUG);
            break;
        case 4:
            clog_set_level(LOGGER_ID, CLOG_TRACE);
            break;
        default:
            fprintf(stderr, "Unknown verbosity: %d\n", verbosity);
            print_usage_and_exit();
    }

    gkmkernel_set_num_threads(nthreads);

    clog_info(CLOG(LOGGER_ID), "Arguments:");
    clog_info(CLOG(LOGGER_ID), "  file1 = %s", file1);
    clog_info(CLOG(LOGGER_ID), "  file2 = %s", file2);
    clog_info(CLOG(LOGGER_ID), "  outprefix = %s", outprefix);
    if (covariate_file) {
        clog_info(CLOG(LOGGER_ID), "  covariates = %s", covariate_file);
    }

    clog_info(CLOG(LOGGER_ID), "Parameters:");
    clog_info(CLOG(LOGGER_ID), "  svm-type = %d", param.svm_type);
    clog_info(CLOG(LOGGER_ID), "  kernel-type = %d", param.kernel_type);
    clog_info(CLOG(LOGGER_ID), "  L = %d", param.L);
    clog_info(CLOG(LOGGER_ID), "  k = %d", param.k);
    clog_info(CLOG(LOGGER_ID), "  d = %d", param.d);
    if (param.kernel_type == EST_TRUNC_RBF || param.kernel_type == GKM_RBF || param.kernel_type == EST_TRUNC_PW_RBF) {
        clog_info(CLOG(LOGGER_ID), "  gamma = %g", param.gamma);
    }
    if (param.kernel_type == MKL_GKM_RBF || param.kernel_type == MKL_RBF_ONLY) {
        clog_info(CLOG(LOGGER_ID), "  rbf_gamma = %g", param.rbf_gamma);
    }
    if (param.kernel_type == MKL_GKM_RBF) {
        clog_info(CLOG(LOGGER_ID), "  gkm_weight = %g", param.gkm_weight);
        clog_info(CLOG(LOGGER_ID), "  rbf_weight = %g", param.rbf_weight);
        clog_info(CLOG(LOGGER_ID), "  mkl_iterations = %d", param.mkl_iterations);
        clog_info(CLOG(LOGGER_ID), "  mkl_tolerance = %g", param.mkl_tolerance);
        clog_info(CLOG(LOGGER_ID), "  normalize_kernels = %s", param.normalize_kernels ? "yes" : "no");
    }
    clog_info(CLOG(LOGGER_ID), "  C = %g", param.C);
    if (param.nr_weight == 1) {
    clog_info(CLOG(LOGGER_ID), "  w = %g", param.weight[0]);
    }
    clog_info(CLOG(LOGGER_ID), "  eps (convergence) = %g", param.eps);
    clog_info(CLOG(LOGGER_ID), "  p (SVR epsilon) = %g", param.p);
    clog_info(CLOG(LOGGER_ID), "  shrinking = %s", param.shrinking ? "yes" : "no");

    sprintf(model_file_name,"%s.model.txt", outprefix);

    if (cross_validation) {
        srand((unsigned int) rseed);
        clog_info(CLOG(LOGGER_ID), "random seed is set to %d", rseed);

        if (icv) {
            /* save CV results to this file if -x option is set */
            sprintf(cvpred_file_name,"%s.cvpred.%d.txt", outprefix, icv); 
        } else {
            sprintf(cvpred_file_name,"%s.cvpred.txt", outprefix); 
        }
    } 

    gkmkernel_init(&param);

    // GPU memory management and diagnostics
    if (gpu_reserve_mb > 0) {
        extern cuda_context_t g_cuda_context;
        if (g_cuda_context.is_initialized) {
            size_t reserve_bytes = (size_t)gpu_reserve_mb * 1024 * 1024;
            cuda_reserve_memory(&g_cuda_context, reserve_bytes);
            clog_info(CLOG(LOGGER_ID), "Reserved %d MB of GPU memory for other operations", gpu_reserve_mb);
        }
    }
    
    if (gpu_diagnostics) {
        extern cuda_context_t g_cuda_context;
        if (g_cuda_context.is_initialized) {
            clog_info(CLOG(LOGGER_ID), "=== GPU Memory Diagnostics ===");
            cuda_memory_diagnostics(&g_cuda_context);
            cuda_print_optimization_stats(&g_cuda_context);
        } else {
            clog_warn(CLOG(LOGGER_ID), "GPU not initialized - cannot show diagnostics");
        }
    }
    
    if (gpu_auto_tune) {
        extern cuda_context_t g_cuda_context;
        if (g_cuda_context.is_initialized) {
            clog_info(CLOG(LOGGER_ID), "Performing GPU batch size auto-tuning...");
            // Use typical covariate count for tuning (estimate based on L and k)
            int est_covariates = (param.L - param.k + 1) * (1 << (2 * param.k));
            if (est_covariates > 100000) est_covariates = 100000; // Cap at reasonable value
            
            int optimal_batch = cuda_auto_tune_batch_size(&g_cuda_context, est_covariates, 1000, 50000);
            if (optimal_batch > 0) {
                clog_info(CLOG(LOGGER_ID), "Auto-tuning completed. Optimal batch size: %d", optimal_batch);
            } else {
                clog_warn(CLOG(LOGGER_ID), "Auto-tuning failed");
            }
        } else {
            clog_warn(CLOG(LOGGER_ID), "GPU not initialized - cannot perform auto-tuning");
        }
    }

    max_line_len = 1024;
    line = (char *) malloc(sizeof(char) * ((size_t) max_line_len));
    if (param.svm_type == C_SVC || param.svm_type == NU_SVC) {
        if (covariate_file) {
            read_problem_svc_with_covariates(file1, file2, covariate_file);
        } else {
            read_problem_svc(file1, file2);
        }
    } else {
        if (covariate_file) {
            read_problem_svr_with_covariates(file1, file2, covariate_file);
        } else {
            read_problem_svr(file1, file2);
        }
    }

    error_msg = svm_check_parameter(&prob,&param);
    if(error_msg) {
        clog_error(CLOG(LOGGER_ID), error_msg);
        exit(1);
    }

    if(cross_validation) {
        do_cross_validation(cvpred_file_name);
    }
    else {
        model = svm_train(&prob,&param);

        clog_info(CLOG(LOGGER_ID), "save SVM model to %s", model_file_name);

        if(svm_save_model(model_file_name,model)) {
            clog_error(CLOG(LOGGER_ID), "can't save model to file %s", model_file_name);
            exit(1);
        }
        svm_free_and_destroy_model(&model);
    }

    int i;
    for (i=0; i<prob.l; i++) {
        gkmkernel_delete_object(prob.x[i].d);
    }

    svm_destroy_param(&param);
    free(prob.y);
    free(prob.x);
    free(line);

	return 0;
}

void read_labels_regression(const char *filename)
{
    FILE *fp = fopen(filename,"r");

    if(fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open labels file");
        exit(1);
    }

    clog_info(CLOG(LOGGER_ID), "reading labels");

    int iseq = 0; //index of the sequence
    char *dummyptr;
    while (readline(fp)) {
        if (iseq > prob.l) {
            clog_error(CLOG(LOGGER_ID), "error occured while reading sequence file (%d >= %d).\n", iseq, prob.l);
            exit(1);
        }
        prob.y[iseq] = strtod(line, &dummyptr);
        ++iseq;
    }
    
    clog_info(CLOG(LOGGER_ID), "done reading labels");

    fclose(fp);
}

void read_fasta_file_regression(const char *filename)
{
    FILE *fp = fopen(filename,"r");

    if(fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open fasta file - regression");
        exit(1);
    }

    int iseq = -1; //index of the sequence
    char seq[MAX_SEQ_LENGTH];
    char sid[MAX_SEQ_LENGTH];
    int seqlen = 0;
    sid[0] = '\0';
    while (readline(fp)) {
        if (iseq >= prob.l) {
            clog_error(CLOG(LOGGER_ID), "error occured while reading sequence file (%d >= %d).\n", iseq, prob.l);
            exit(1);
        }

        if (line[0] == '>') {
            if (((iseq % 1000) == 0)) {
                clog_info(CLOG(LOGGER_ID), "reading... %d", iseq);
            }

            if (iseq >= 0) {
                prob.x[iseq].d = gkmkernel_new_object(seq, sid, iseq);
            }
            ++iseq;

            seq[0] = '\0'; //reset sequence
            seqlen = 0;
            char *ptr = strtok(line," \t\r\n");
            if (strlen(ptr) >= MAX_SEQ_LENGTH) {
                clog_error(CLOG(LOGGER_ID), "maximum sequence id length is %d.\n", MAX_SEQ_LENGTH-1);
                exit(1);
            }
            strcpy(sid, ptr+1);
        } else {
            if (seqlen < MAX_SEQ_LENGTH-1) {
                if ((((size_t) seqlen) + strlen(line)) >= MAX_SEQ_LENGTH) {
                    clog_warn(CLOG(LOGGER_ID), "maximum sequence length allowed is %d. The first %d nucleotides of %s will only be used (Note: You can increase the MAX_SEQ_LENGTH parameter in libsvm_gkm.h and recompile).", MAX_SEQ_LENGTH-1, MAX_SEQ_LENGTH-1, sid);
                    int remaining_len = MAX_SEQ_LENGTH - seqlen - 1;
                    line[remaining_len] = '\0';
                }
                strcat(seq, line);
                seqlen += (int) strlen(line);
            }
        }
    }

    //last one
    prob.x[iseq].d = gkmkernel_new_object(seq, sid, iseq);

    clog_info(CLOG(LOGGER_ID), "reading... done");

    fclose(fp);
}

void read_fasta_file_classification(const char *filename, int offset, int label)
{
    FILE *fp = fopen(filename,"r");

    if(fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open fasta file - classification");
        exit(1);
    }

    int iseq = -1;
    char seq[MAX_SEQ_LENGTH];
    char sid[MAX_SEQ_LENGTH];
    int seqlen = 0;
    sid[0] = '\0';
    while (readline(fp)) {
        if (iseq >= prob.l) {
            clog_error(CLOG(LOGGER_ID), "error occured while reading sequence file (%d >= %d).\n", iseq, prob.l);
            exit(1);
        }

        if (line[0] == '>') {
            if (((iseq % 1000) == 0)) {
                clog_info(CLOG(LOGGER_ID), "reading... %d", iseq);
            }

            if (iseq >= 0) {
                prob.y[offset + iseq] = label;
                prob.x[offset + iseq].d = gkmkernel_new_object(seq, sid, offset + iseq);
            }
            ++iseq;

            seq[0] = '\0'; //reset sequence
            seqlen = 0;
            char *ptr = strtok(line," \t\r\n");
            if (strlen(ptr) >= MAX_SEQ_LENGTH) {
                clog_error(CLOG(LOGGER_ID), "maximum sequence id length is %d.\n", MAX_SEQ_LENGTH-1);
                exit(1);
            }
            strcpy(sid, ptr+1); } else {
            if (seqlen < MAX_SEQ_LENGTH-1) {
                if ((((size_t) seqlen) + strlen(line)) >= MAX_SEQ_LENGTH) {
                    clog_warn(CLOG(LOGGER_ID), "maximum sequence length allowed is %d. The first %d nucleotides of %s will only be used (Note: You can increase the MAX_SEQ_LENGTH parameter in libsvm_gkm.h and recompile).", MAX_SEQ_LENGTH-1, MAX_SEQ_LENGTH-1, sid);
                    int remaining_len = MAX_SEQ_LENGTH - seqlen - 1;
                    line[remaining_len] = '\0';
                }
                strcat(seq, line);
                seqlen += (int) strlen(line);
            }
        }
    }

    //last one
    prob.y[offset + iseq] = label;
    prob.x[offset + iseq].d = gkmkernel_new_object(seq, sid, offset + iseq);

    clog_info(CLOG(LOGGER_ID), "reading... done");

    fclose(fp);
}

int count_sequences(const char *filename)
{
    FILE *fp = fopen(filename,"r");
    int nseqs = 0;

    if(fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open file - count sequences");
        exit(1);
    }

    //count the number of sequences for memory allocation
    while(readline(fp)!=NULL) {
        if (line[0] == '>') {
            ++nseqs;
        }
    }
    fclose(fp);
    
    return nseqs;
}

void read_problem_svc(const char *file1, const char *file2)
{
    int n1 = count_sequences(file1);
    int n2 = count_sequences(file2);
    prob.l = n1+n2;

    prob.y = (double *) malloc (sizeof(double) * ((size_t) prob.l));
    prob.x = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) prob.l));

    clog_info(CLOG(LOGGER_ID), "reading %d sequences from %s", n1, file1);
    read_fasta_file_classification(file1, 0, 1);

    clog_info(CLOG(LOGGER_ID), "reading %d sequences from %s", n2, file2);
    read_fasta_file_classification(file2, n1, -1);
}

void read_problem_svr(const char *file1, const char *file2)
{
    int n1 = count_sequences(file1);
    prob.l = n1;

    prob.y = (double *) malloc (sizeof(double) * ((size_t) prob.l));
    prob.x = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) prob.l));

    read_fasta_file_regression(file1);
    read_labels_regression(file2);
}

void do_cross_validation(const char *filename)
{
    double *target = (double *) malloc(sizeof(double) * ((size_t) prob.l));

    svm_cross_validation(&prob,&param,nr_fold,icv,target,filename);

    //TODO: calculate AUC here
    free(target);
}

void read_covariates(const char *filename, int start_index, int num_sequences)
{
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        clog_error(CLOG(LOGGER_ID), "can't open covariate file %s", filename);
        exit(1);
    }

    clog_info(CLOG(LOGGER_ID), "reading covariates from %s", filename);

    int seq_idx = 0;
    int first_line = 1;
    int num_covariates = 0;
    
    while (readline(fp) && seq_idx < num_sequences) {
        if (seq_idx + start_index >= prob.l) {
            clog_error(CLOG(LOGGER_ID), "error: covariate file has more rows than sequences");
            exit(1);
        }
        
        // Count number of covariates from first line
        if (first_line) {
            char *line_copy = strdup(line);
            char *token = strtok(line_copy, "\t");
            while (token != NULL) {
                num_covariates++;
                token = strtok(NULL, "\t");
            }
            free(line_copy);
            first_line = 0;
            clog_info(CLOG(LOGGER_ID), "found %d covariates", num_covariates);
        }
        
        // Allocate memory for covariates if not already done
        if (prob.x[start_index + seq_idx].d->covariates == NULL) {
            prob.x[start_index + seq_idx].d->covariates = (double *) malloc(sizeof(double) * num_covariates);
            prob.x[start_index + seq_idx].d->num_covariates = num_covariates;
        }
        
        // Parse covariates from current line
        char *token = strtok(line, "\t");
        int cov_idx = 0;
        while (token != NULL && cov_idx < num_covariates) {
            prob.x[start_index + seq_idx].d->covariates[cov_idx] = atof(token);
            token = strtok(NULL, "\t");
            cov_idx++;
        }
        
        if (cov_idx != num_covariates) {
            clog_error(CLOG(LOGGER_ID), "error: inconsistent number of covariates at line %d", seq_idx + 1);
            exit(1);
        }
        
        seq_idx++;
    }
    
    if (seq_idx != num_sequences) {
        clog_error(CLOG(LOGGER_ID), "error: covariate file has %d rows but expected %d", seq_idx, num_sequences);
        exit(1);
    }
    
    fclose(fp);
    clog_info(CLOG(LOGGER_ID), "done reading covariates");
}

void read_problem_svc_with_covariates(const char *file1, const char *file2, const char *cov_file)
{
    int n1 = count_sequences(file1);
    int n2 = count_sequences(file2);
    prob.l = n1 + n2;

    prob.y = (double *) malloc(sizeof(double) * ((size_t) prob.l));
    prob.x = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) prob.l));

    clog_info(CLOG(LOGGER_ID), "reading %d sequences from %s", n1, file1);
    read_fasta_file_classification(file1, 0, 1);

    clog_info(CLOG(LOGGER_ID), "reading %d sequences from %s", n2, file2);
    read_fasta_file_classification(file2, n1, -1);

    read_covariates(cov_file, 0, prob.l);
}

void read_problem_svr_with_covariates(const char *file1, const char *file2, const char *cov_file)
{
    int n1 = count_sequences(file1);
    prob.l = n1;

    prob.y = (double *) malloc(sizeof(double) * ((size_t) prob.l));
    prob.x = (union svm_data *) malloc(sizeof(union svm_data) * ((size_t) prob.l));

    read_fasta_file_regression(file1);
    read_labels_regression(file2);
    read_covariates(cov_file, 0, prob.l);
}
