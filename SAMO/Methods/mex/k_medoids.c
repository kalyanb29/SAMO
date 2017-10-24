#include <stdio.h>
#include <malloc.h>

#include "cluster.h"
#include "mex.h"


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
int mrows, ncols, npts, nx, i, j, k, count;
double *x, **X, **dist, *weight;
int **mask, *id;
double error;
int ifound;
unsigned int seed;

	/* Check if proper number of arguments */
	if(nrhs != 3) {
		mexErrMsgTxt("Three inputs are required. (data, k, seed)");
	}
	if(nlhs != 1) {
		mexErrMsgTxt("One output is returned. (ids)");
	}

	/* First input is data to be clustered */
	if(!mxIsDouble(prhs[0])) {
		mexErrMsgTxt("First input must a double array.");
	}
	mrows = mxGetM(prhs[0]);
	ncols = mxGetN(prhs[0]);
	npts = mrows;
	nx = ncols;
	x = (double *)mxGetPr(prhs[0]);

	/* Second input is the 'k' for clusters */
	mrows = mxGetM(prhs[1]);
	ncols = mxGetN(prhs[1]);
	if(mrows != 1 || ncols != 1) {
		mexErrMsgTxt("Second input must be a scalar.");
	}
	k = (int)*(double *)mxGetPr(prhs[1]);
	if(k <= 1) {
		plhs[0] = mxCreateDoubleMatrix(npts, 1, mxREAL);
		x = (double *)mxGetPr(plhs[0]);
		for(i=0; i<npts; i++) x[i] = 1;
		return;
	}

	seed = (int)*(double *)mxGetPr(prhs[2]);
	set_seed(seed);

	X = (double **)malloc(npts * sizeof(double *));
	mask = (int **)malloc(npts * sizeof(int *));
	for(i=0; i<npts; i++) {
		X[i] = (double *)malloc(nx * sizeof(double));
		mask[i] = (int *)malloc(nx * sizeof(int));
	}
	id = (int *)malloc(npts * sizeof(int));
	weight = (double *)malloc(nx * sizeof(double));

	/* Copy data */
	count = 0;
	for(j=0; j<nx; j++) {
		for(i=0; i<npts; i++) {
			X[i][j] = x[count];
			mask[i][j] = 1;
			count++;
		}
		weight[j] = 1;
	}

	dist = distancematrix(npts, nx, X, mask, weight, 'e', 0);
	if(dist == NULL) {
		mexErrMsgTxt("Error in cluster.c:distancematrix");
	}
	kmedoids(k, npts, dist, 3, id, &error, &ifound);
	if(ifound <= 0) {
		mexErrMsgTxt("Error in cluster.c:kmedoids");
	}

	/* Create output variables */
	plhs[0] = mxCreateDoubleMatrix(npts, 1, mxREAL);

	x = (double *)mxGetPr(plhs[0]);
	for(i=0; i<npts; i++) x[i] = id[i]+1;

	for(i=0; i<npts; i++) {
		free(X[i]);
		free(mask[i]);
	}
	free(X);
	free(mask);
	free(id);
	free(weight);
	/*
	for(i=0; i<k; i++) {
		free(dist[i]);
	}
	free(dist);
	*/

}
