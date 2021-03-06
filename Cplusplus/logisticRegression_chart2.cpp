#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <ctime>
#include <cstdlib>

using namespace std;

typedef long int lint;

double min(double a, double b){
	return (a<b) ? a:b;
}

double max(double a, double b){
	return (a>b) ? a:b;
}

void Xw(double *w, double *x_val, lint *x_colind, lint *x_rowptr, lint l, double *z){	//Computes Xw into z
	lint i, j;
	for(i = 0; i < l; ++i){
		z[i] = 0;
		for(j = x_rowptr[i]; j < x_rowptr[i+1]; ++j){
			z[i] += x_val[j]*w[x_colind[j]];
		}
	}
}

void wX(double *w, double *x_val, lint *x_colind, lint *x_rowptr, lint l, lint n, double *z){	//Computes wX into z
	lint i, j;
	for(i = 0; i < n; ++i){
		z[i] = 0;
	}
	for(i = 0; i < l; ++i){
		for(j = x_rowptr[i]; j < x_rowptr[i+1]; ++j){
			z[x_colind[j]] += x_val[j]*w[i];
		}
	}
}

double norm2(double *g, int n){	//Returns the euclidean norm of g, credits: Sven Hammarling, Nag Ltd.
	lint i;
	double absgi, scale = 0, ssq = 1, temp;
	for(i = 0; i < n; ++i){
		if (g[i] != 0){
			absgi = fabs(g[i]);
			if (scale < absgi){
				temp = scale / absgi;
				ssq = ssq * (temp * temp) + 1.0;
				scale = absgi;
			}
			else{
				temp = absgi / scale;
				ssq += temp * temp;
			}
		}
	}
	return scale * sqrt(ssq);
}

double norm_inf(int n, double *g){	//Returns infinite norm of g
	double max = 0;
	lint i;
	for(i = 0; i < n; ++i){
		if(fabs(g[i]) > max){
			max = fabs(g[i]);
		}
	}
	return max;
}

double obj_fun(double *w, double *x_val, lint *x_colind, lint *x_rowptr, lint l, lint n, double *y, double C, double *z){
	double nll = 0, reg = 0;
	lint i;

	for (i = 0; i < l; ++i){
		double yz = y[i]*z[i];
		if (yz >= 0)
		        nll += C * log(1 + exp(-yz));
		else
		        nll += C * (-yz + log(1 + exp(yz)));	//To avoid overflow, we don't directly compute log(1 + exp(-yz))
	}

	for (i = 0; i < n; ++i){
		reg += w[i]*w[i];
	}

	return reg/2.0 + nll;
}

void grad_f_D(int l, int n, double *w, double *x_val, lint *x_colind, lint *x_rowptr, double *y, double C, double *z, double *g, double *D){	//Computes gradient of f into g, and also D
	lint i;

	double *sigma = (double *)malloc((l)*sizeof(double));

	for(i = 0; i < l; ++i){
		sigma[i] = 1 / (1 + exp(-y[i] *z[i]));
		D[i] = sigma[i] * (1 - sigma[i]);
		sigma[i] = C * (sigma[i] - 1) *y[i];
	}

	wX(sigma, x_val, x_colind, x_rowptr, l, n, g);

	for(i = 0; i < n; ++i){
		g[i] = w[i] + g[i];
	}

	free(sigma);
}

void H_d(lint l, lint n, double *x_val, lint *x_colind, lint *x_rowptr, double *D, double C, double *d, double *Hd){	//Computes Hessian*d into Hd
	lint i;
	double *wa = (double *)malloc((l)*sizeof(double));

	for(i = 0; i < l; ++i){
		wa[i] = 0;
	}
	Xw(d, x_val, x_colind, x_rowptr, l, wa);
	for(i=0;i<l;i++){
		wa[i] = C * D[i] * wa[i];
	}

	wX(wa, x_val, x_colind, x_rowptr, l, n, Hd);	
	for(i=0;i<n;i++){
		Hd[i] = d[i] + Hd[i];
	}
	free(wa);
}

double Dot(double *d, double *Hd, lint n){	//Returns dot product for Hd and d
	double dot = 0;
	lint i = 0;
	while(i < n){
		dot += Hd[i]*d[i];
		++i;
	}
	return dot;
}

void vec_add(double *s, double alpha, double *d, lint n){	//Updates s = s + alpha*d
	lint i = 0;
	while(i < n){
		s[i] += alpha * d[i];
		++i;
	}
}

void vec_scale(double *d, double beta, lint n){	//Updates d = beta*d
	lint i = 0;
	while(i < n){
		d[i] = beta * d[i];
		++i;
	}
}

void cgp(double delta, double *g, double *D, double C, double *x_val, lint *x_colind, lint *x_rowptr, lint l, lint n, double *y, double *s, double *r){	//Conjugate Gradient procedure for approximately solving the trust region sub-problem, finds s and r
	double zai = 0.1;	//Page 635
	double tol = zai * norm2(g, n);
	double *d = (double *)malloc((n)*sizeof(double));
	double *Hd = (double *)malloc((n)*sizeof(double));	//Stores Hessian * d
	double rTr, alpha, one = 1, rnewTrnew, beta;
	lint i;

	for (i = 0; i < n; ++i){
		s[i] = 0;
		r[i] = -g[i];
		d[i] = r[i];
	}
	rTr = Dot(r, r, n);
	while(1){
		// printf("%d\n", k);	//debug
		if(norm2(r, n) <= tol){
			break;
		}
		H_d(l, n, x_val, x_colind, x_rowptr, D, C, d, Hd);
		alpha = rTr/Dot(d, Hd, n);
		vec_add(s, alpha, d, n);
		// printf("%lf %lf\n", norm2(s, n), delta);	//debug
		if(norm2(s, n) > delta){
			vec_add(s, -alpha, d, n);
			double std = Dot(s, d, n);
			double sts = Dot(s, s, n);
			double dtd = Dot(d, d, n);
			double delta_sq = delta*delta;
			double rad = sqrt(std * std + dtd * (delta_sq-sts));
			if (std >= 0)
				alpha = (delta_sq - sts)/(std + rad);
			else
				alpha = (rad - std)/dtd;	//Both alphas are same, just to save computation
			vec_add(s, alpha, d, n);
			vec_add(r, -alpha, Hd, n);
			break;
		}
		vec_add(r, -alpha, Hd, n);
		rnewTrnew = Dot(r, r, n);
		beta = rnewTrnew/rTr; 
		vec_scale(d, beta, n);
		vec_add(d, one, r, n);			
		rTr = rnewTrnew;
	}
	free(d);
	free(Hd);
}

void tra(double *w, double *x_val, lint *x_colind, lint *x_rowptr, lint l, lint n, double *y, double C, double *time_chart, int mult){	//Trust Region Algorithm for losgistic regression
	double eta0 = 0.0001, eta1 = 0.25, eta2 = 0.75;	//Page 635
	double sigma1 = 0.25, sigma2 = 0.5, sigma3 = 4;	//Page 635
	double eps = 0.001; //Page 639

	double f;
	double *z = (double *)calloc(l, sizeof(double));	//Xw, initiallized to 0
	double *g = (double *)malloc((n)*sizeof(double));	//grad of f
	double *s = (double *)malloc((n)*sizeof(double));	
	double *r = (double *)malloc((n)*sizeof(double));	//used in cgp
	double *D = (double *)malloc((l)*sizeof(double));	//used to obtain hessian in cgp
	double *w_new = (double *)malloc((n)*sizeof(double));
	double inf_norm;
	int flag = 1;
	double delta, one = 1, gs, fnew, nume, deno, snorm, alpha;
	lint time_now = time(0);

	lint i;
	for(i = 0; i < n; ++i){
		w[i] = 0; 	//Page 639
	}

	Xw(w, x_val, x_colind, x_rowptr, l, z);
	f = obj_fun(w, x_val, x_colind, x_rowptr, l, n, y, C, z);
	grad_f_D(l, n, w, x_val, x_colind, x_rowptr, y, C, z, g, D);

	inf_norm = norm_inf(n, g);
	time_chart[time(0) - time_now + mult] = inf_norm;
	if(inf_norm < eps){
		flag = 0;	//Page 639
	}

	delta = norm2(g, n);	// Page 635
	// printf("%lf\n", delta); // debug
	// printf("%lf\n", f);	// debug

	int max_iter = 1000;
	int k = 0;
	if(flag){
		while(k < max_iter){
			inf_norm = norm_inf(n, g);
			time_chart[time(0) - time_now + mult] = inf_norm;
			if(inf_norm < eps){
				break;
			}
			// printf("norm g is %lf\n", norm_inf(n, g));	// debug
			cgp(delta, g, D, C, x_val, x_colind, x_rowptr, l, n, y, s, r);
			memcpy(w_new, w, sizeof(double) * n);
			vec_add(w_new, one, s, n);

			gs = Dot(g, s, n);
			deno = -0.5 * (gs - Dot(s, r, n));	//Using the difinition of r, finding the denominator of rho
			
			Xw(w_new, x_val, x_colind, x_rowptr, l, z);
			fnew = obj_fun(w_new, x_val, x_colind, x_rowptr, l, n, y, C, z);
			nume = f - fnew;	//Numerator of rho
			// printf("nume is %lf\n", nume);	// debug
			// printf("deno is %lf\n", deno);	// debug

			snorm = norm2(s, n);
			if (k == 1){
				delta = min(delta, snorm);
			}
			   
			if (fnew - f - gs <= 0)
				alpha = sigma3;
			else
				alpha = max(sigma1, -0.5 * (gs / (fnew - f - gs)));

			/* Update the trust region bound according to the ratio
			of actual to predicted reduction. Direcly lifted from the paper's code*/
			if (nume < eta0 * deno)
				delta = min(max(alpha, sigma1) * snorm, sigma2 * delta);
			else if (nume < eta1 * deno)
				delta = max(sigma1 * delta, min(alpha * snorm, sigma2 * delta));
			else if (nume < eta2 * deno)
				delta = max(sigma1 * delta, min(alpha * snorm, sigma3 * delta));
			else
				delta = max(delta, min(alpha * snorm, sigma3 * delta));
				      
			if (nume > eta0 * deno){
				++k;
				memcpy(w, w_new, sizeof(double) * n);
				f = fnew;
				grad_f_D(l, n, w, x_val, x_colind, x_rowptr, y, C, z, g, D);
			}
			// printf("%lf\n", delta); // debug
			// printf("%lf\n", f);	// debug
		}
	}
	free(z);
	free(g);
	free(s);
	free(r);
	free(D);
	free(w_new);
}

void data_parameters(lint *l, lint *n, lint *num_elements, const char *fileName){
	FILE *fp = fopen(fileName,"r");
	(*num_elements) = 0;
	(*l) = 0;
	(*n) = 0;
	int index;
	double value;
	while(1){
		fscanf(fp,"%lf",&value);
		while(1){
			int c;
			do{
				c = getc(fp);
				if(c == '\n'){
					goto out;
				}
				if(c == EOF){
					goto eof;
				}
			}while(isspace(c));
			ungetc(c, fp);
			fscanf(fp,"%d:%lf",&index, &value);
			++(*num_elements);
		}
out:
		if(index > (*n)){
			(*n) = index;
		}
		++(*l);
	}
eof:
	++(*n);									//Augmenting for bias term
	*num_elements = *num_elements + *l;		//Augmenting for bias term
	fclose(fp);
}

void read_data(double *x_val, lint *x_colind, lint *x_rowptr, double *y, lint l, lint n, const char *fileName){
	FILE *fp = fopen(fileName,"r");
	int index;
	double value;
	lint i, j = 0;
	for(i = 0; i < l; ++i){
		x_rowptr[i] = j;
		fscanf(fp,"%lf",&y[i]);
		while(1){
			int c;
			do{
				c = getc(fp);
				if(c=='\n'){
					goto out;
				}
			}while(isspace(c));
			ungetc(c,fp);
			fscanf(fp,"%d:%lf",&index, &value);
			x_val[j] = value;
			x_colind[j] = index-1;
			++j;
		}
out:
		x_val[j] = 1;		//Augmenting for bias
		x_colind[j] = n-1;
		++j;
	}
	x_rowptr[i] = j;
	fclose(fp);
}

double accuracy(double *w, double *x_val, lint *x_colind, lint *x_rowptr, lint l, lint n, double *y){
	double one = 1, accu = 0;
	double *z = (double *)calloc(l, sizeof(double));
	Xw(w, x_val, x_colind, x_rowptr, l, z);
	lint i;
	for(i = 0; i < l; ++i){
		if(z[i] > 0){
			if(one/(1 + exp(-z[i])) > 0.5 && y[i] == 1){
				++accu;
			}
		}
		else{
			if(one/(1 + exp(z[i])) > 0.5 && y[i] == -1){
				++accu;
			}
		}
	}
	return 100*accu/l;
}

void CV_chart2(double *C, const char *fileName[4]){
	lint l, n, num_elements, num_test_elements;
	lint num_train_elements, l_test, l_train, train_stop;
	lint train_row_yet, test_row_yet, num_train_yet, num_test_yet;
	int cv_folds = 5, num_files = 4;
	int i, j, cv_iter, k;
	double *time_chart = (double *)malloc((num_files*2000)*sizeof(double));
	srand ( unsigned ( time(0) ) );	//Assigning a random seed

	for(i = 0; i < num_files*2000; ++i){
		time_chart[i] = 0;
	}

	for(i = 0; i < num_files; ++i){
		if(i > 2){
			cv_folds = 2;
		}
		data_parameters(&l, &n, &num_elements, fileName[i]);

		double *w = (double *)malloc((n)*sizeof(double));
		double *y = (double *)malloc((l)*sizeof(double));
		double *x_val = (double *)malloc((num_elements)*sizeof(double));
		lint *x_colind = (lint *)malloc((num_elements)*sizeof(lint));
		lint *x_rowptr = (lint *)malloc((l+1)*sizeof(lint));

		read_data(x_val, x_colind, x_rowptr, y, l, n, fileName[i]);

		vector<int> randomized;
		for(j = 0; j < l; ++j){
			randomized.push_back(j);
		}
		random_shuffle ( randomized.begin(), randomized.end() );
		for(j = 0; j < cv_folds; ++j){
			sort (randomized.begin() + j*l/cv_folds, randomized.begin() + (j+1)*l/cv_folds);
		}

		vector<int>::iterator it;

		for(cv_iter = 0; cv_iter < 1; ++cv_iter){
			num_test_elements = 0;
			
			for (it = randomized.begin() + cv_iter*l/cv_folds; it < randomized.begin() + (cv_iter+1)*l/cv_folds; ++it){
    			num_test_elements += (x_rowptr[(*it)+1] - x_rowptr[*it]);
			}
			
			l_test = (cv_iter+1)*l/cv_folds - cv_iter*l/cv_folds;
    		double *y_test = (double *)malloc((l_test)*sizeof(double));
			double *x_val_test = (double *)malloc((num_test_elements)*sizeof(double));
			lint *x_colind_test = (lint *)malloc((num_test_elements)*sizeof(lint));
			lint *x_rowptr_test = (lint *)malloc((l_test + 1)*sizeof(lint));
			
			l_train = l - l_test;
			num_train_elements = num_elements - num_test_elements;
    		double *y_train = (double *)malloc((l_train)*sizeof(double));
			double *x_val_train = (double *)malloc((num_train_elements)*sizeof(double));
			lint *x_colind_train = (lint *)malloc((num_train_elements)*sizeof(lint));
			lint *x_rowptr_train = (lint *)malloc((l_train + 1)*sizeof(lint));
			
			train_row_yet = 0;
			test_row_yet = 0;
			num_train_yet = 0;
			num_test_yet = 0;
			
			for(j = 0; j < *(randomized.begin() + cv_iter*l/cv_folds); ++j){
				x_rowptr_train[train_row_yet] = num_train_yet;
				y_train[train_row_yet] = y[j];
				for(k = x_rowptr[j]; k < x_rowptr[j+1]; ++k){
					x_val_train[num_train_yet] = x_val[k];
					x_colind_train[num_train_yet] = x_colind[k];
					++num_train_yet;
				}
				++train_row_yet;
			}
			
			for (it = randomized.begin() + cv_iter*l/cv_folds; it < randomized.begin() + (cv_iter+1)*l/cv_folds; ++it){
				x_rowptr_test[test_row_yet] = num_test_yet;
				y_test[test_row_yet] = y[*it];
				for(k = x_rowptr[*it]; k < x_rowptr[(*it)+1]; ++k){
					x_val_test[num_test_yet] = x_val[k];
					x_colind_test[num_test_yet] = x_colind[k];
					++num_test_yet;
				}
				++test_row_yet;
				if(it == (randomized.begin() + (cv_iter+1)*l/cv_folds - 1)){
					train_stop = l;
				}
				else{
					train_stop = *(it + 1);
				}
				for(j = (*it)+1; j < train_stop; ++j){
					x_rowptr_train[train_row_yet] = num_train_yet;
					y_train[train_row_yet] = y[j];
					for(k = x_rowptr[j]; k < x_rowptr[j+1]; ++k){
						x_val_train[num_train_yet] = x_val[k];
						x_colind_train[num_train_yet] = x_colind[k];
						++num_train_yet;
					}
					++train_row_yet;
				}
			}
			
			x_rowptr_test[test_row_yet] = num_test_yet;
			x_rowptr_train[train_row_yet] = num_train_yet;

			j = 2;
			tra(w, x_val_train, x_colind_train, x_rowptr_train, l_train, n, y_train, C[j], time_chart, i*2000);

			free(y_test);
			free(x_val_test);
			free(x_colind_test);
			free(x_rowptr_test);
			free(y_train);
			free(x_val_train);
			free(x_colind_train);
			free(x_rowptr_train);
		}
		free(x_val);
		free(x_colind);
		free(x_rowptr);
		free(y);
		free(w);
	}
	for(i = 0; i < num_files*2000; ++i){
		if(time_chart[i] != 0){
			printf("%d %lf\n",i ,time_chart[i]);
		}
	}
}

int main(){
	double C[4] = {0.25, 1, 4, 16};
	const char *fileName[4] = {"a9a","real-sim.svml","news20.binary","rcv1_test.binary"};

	CV_chart2(C, fileName);

	return 0;
}
