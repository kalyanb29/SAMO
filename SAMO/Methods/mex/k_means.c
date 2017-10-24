#include <stdio.h>
#include <malloc.h>

#include "cluster.h"
#include "mex.h"


void
mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
int mrows, ncols, npts, nx, i, j, k, count;
double *x, **X, **Centroid, *weight;
int **mask, **cmask, *id;
double error;
int ifound;
unsigned int seed;

	/* Check if proper number of arguments */
	if(nrhs != 3) {
		mexErrMsgTxt("Three Inputs are required (data, k, seed).");
	} 
	if(nlhs != 2) {
		mexErrMsgTxt("Two Outputs are returned (ids, centroids).");
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

		plhs[1] = mxCreateDoubleMatrix(1, nx, mxREAL);
		x = (double *)mxGetPr(plhs[1]);
		for(i=0; i<nx; i++) x[i] = 0;

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

	Centroid = (double **)malloc(k * sizeof(double *));
	cmask = (int **)malloc(k * sizeof(int *));
	for(i=0; i<k; i++) {
		Centroid[i] = (double *)malloc(nx * sizeof(double));
		cmask[i] = (int *)malloc(nx * sizeof(int));
	}

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

	kcluster(k, npts, nx, X, mask, weight, 0, 3, 'a', 'e', id, &error, &ifound);
	if(ifound <= 0) {
		mexErrMsgTxt("error in cluster.c");
	}
	getclustercentroids(k, npts, nx, X, mask, id, Centroid, cmask, 0, 'a');

	/* Create output variables */
	plhs[0] = mxCreateDoubleMatrix(npts, 1, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(k, nx, mxREAL);

	x = (double *)mxGetPr(plhs[0]);
	for(i=0; i<npts; i++) x[i] = id[i]+1;

	x = (double *)mxGetPr(plhs[1]);
	count = 0;
	for(j=0; j<nx; j++) {
		for(i=0; i<k; i++) {
			x[count] = Centroid[i][j];
			count++;
		}
	}

	for(i=0; i<npts; i++) {
		free(X[i]);
		free(mask[i]);
	}
	free(X);
	free(mask);
	free(id);
	free(weight);
	for(i=0; i<k; i++) {
		free(Centroid[i]);
		free(cmask[i]);
	}
	free(Centroid);
	free(cmask);
}
