#include <malloc.h>
#include <assert.h>

#include "mex.h"

void 
indsort(int N, int *feasible, int nf, double *f, 
				int *ind_N, int *ind_S, int *ind_S_count)
{
int i, j, k;
double *f1, *f2;
int *i1, *i2;
int dom_less, dom_more;

	/* Initalize all output variables */
	for(i=0; i<N; i++) {
		ind_N[i] = 0;
		ind_S_count[i] = 0;
		for(j=0; j<N; j++) {
			ind_S[i*N+j] = -1;
		}
	}

	/* NDS fill routine */
	for(i=0; i<N; i++) {
		/* id1 = feasible[i];  -- ids not used */
		f1 = &f[i*nf];
		for(j=0; j<N; j++) {
			if(i == j) continue;
			/* id2 = feasible[j]; -- ids not used */
			f2 = &f[j*nf];
			dom_less = 0;
			dom_more = 0;
			for(k=0; k<nf; k++) {
				if(f1[k] <= f2[k]) dom_less++;
				if(f1[k] >= f2[k]) dom_more++;
			}
			if(dom_less == nf && dom_more < nf) {
				i1 = &ind_S[i*N];
				/* i1[(int)ind_S_count[i]] = (double)id2;  -- ids not used */
				i1[(int)ind_S_count[i]] = (double)j;
				ind_S_count[i]++;
			} else if(dom_more == nf && dom_less < nf) {
				ind_N[i]++;
			}
		}
	}
}


void 
fsort(int N, int *ind_N, int *ind_S, int *ind_S_count,
			double *front_S, double *front_S_count)
{
int i, j, p, q;
int front;
double *cur_front_S, *next_front_S;

	/* initialize front variables */
	for(i=0; i<N; i++) front_S_count[i] = 0;
	for(i=0; i<N*N; i++) front_S[i] = -1;

	/* Create front 1 */
	for(i=0; i<N; i++) {
		if(ind_N[i] == 0) {
			front_S[(int)front_S_count[0]] = (double)i;
			front_S_count[0]++;
		}
	}

	/* Create rest of the fronts */
	front = 0;
	while(front_S_count[front] > 0 && front_S_count[front] < N) {
		cur_front_S = &front_S[front*N];
		next_front_S = &front_S[(front+1)*N];
		for(i=0; i<(int)front_S_count[front]; i++) {
			p = (int)cur_front_S[i];
			assert(p <= N);
			for(j=0; j<ind_S_count[p]; j++) {
				q = (int)ind_S[p*N+j];
				assert(q <= N); 

				ind_N[q]--;
				if(ind_N[q] == 0) {
					next_front_S[(int)front_S_count[front+1]] = (double)q;
					front_S_count[front+1]++;
				}
			}
		}
		front++;
		if(front == N) break;
	}
}



void 
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
int i;
int mrows, ncols;
int N, nf;
int *feasible;
double *f, *ptr;
int *ind_N, *ind_S, *ind_S_count;
double *front_S, *front_S_count;

	/* Check if proper number of arguments */
	if(nrhs != 2) {
		mexErrMsgTxt("Two Inputs are required.");
	} else if(nlhs != 2) {
		mexErrMsgTxt("Three outputs are returned.");
	}

	/* First input is the feasible indices */
	mrows = mxGetM(prhs[0]);
	ncols = mxGetN(prhs[0]);
	if(!mxIsDouble(prhs[0]) || !(mrows == 1)) {
		mexErrMsgTxt("First input must be a column array of integers.");
	}
	N = ncols;
	ptr = (double *)mxGetPr(prhs[0]);
	feasible = malloc(N * sizeof(int));
	for(i=0; i<N; i++) feasible[i] = (int)ptr[i];

	/* Second input is the objective array */
	mrows = mxGetM(prhs[1]);
	ncols = mxGetN(prhs[1]);
	if(!mxIsDouble(prhs[1]) || !(ncols == N)) {
		mexErrMsgTxt("Second input must be an array of doubles.");
	}
	nf = mrows;
	f = (double *)mxGetPr(prhs[1]);

	/* Create temporary variables */
	ind_N = (int *)malloc(N * sizeof(int));
	ind_S = (int *)malloc(N * N * sizeof(int));
	ind_S_count = (int *)malloc(N * sizeof(int));

	assert(ind_N != NULL);
	assert(ind_S != NULL);
	assert(ind_S_count != NULL);

	/* Create output variables */
	plhs[0] = mxCreateDoubleMatrix(N, N, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(N, 1, mxREAL);

	front_S = (double *)mxGetPr(plhs[0]);
	front_S_count = (double *)mxGetPr(plhs[1]);

	indsort(N, feasible, nf, f, ind_N, ind_S, ind_S_count);
	fsort(N, ind_N, ind_S, ind_S_count, front_S, front_S_count);

	free(ind_N);
	free(ind_S);
	free(ind_S_count);
}
