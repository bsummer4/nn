/*
	This creates a random weights file with a given number of nodes
	for each layer.  See README for a description of the file format.
	All weights will have random values between -0.05 and 0.05.
*/

#include <stdio.h>
#include <stdlib.h>
#include <err.h>

int main (int argc, char *argv[]) {
	if (4!=argc) errx(1, "%s num-inputs num-hidden num-outputs", *argv);
	{char *s=getenv("seed"); if (s) srand48(atol(s));}
	int iho[3]; // input, hidden, output
	for (int ii=0; ii<3; ii++) {
		if (!(iho[ii] = atoi(argv[ii+1])))
			errx(1, "Invalid argument %s", argv[ii+1]); }
	{	double iho_d[3] = {iho[0], iho[1], iho[2]};
		fwrite(iho_d, sizeof(iho_d), 1, stdout); }
	const int i=iho[0], h=iho[1], o=iho[2], nweights = (i+1)*h + (h+1)*o;
	for (int ii=0; ii<nweights; ii++) {
		double d = drand48()/50 - 0.01;
		fwrite(&d, sizeof(d), 1, stdout); }
	return 0; }
