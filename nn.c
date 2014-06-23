/*
	This runs the neural net.  We accept commands from standard input,
	and do what we are told.  The commands we accept are:

		- training on -- We don't modify the weights unless
		                 this has been sent.
		- training off -- Revert the above
		- epoch -- Run the neural net for one epoch, and print
		  the overall network error.
		- networkerror -- Print the total network error for the
		  last epoch
		- nodeerror -- Print the total error-per-node for the
		  last epoch.
		- accuracy -- Print the percent of correct classifications for each
		  possible classification.
		- totalaccuracy -- Print the percent of correct
		  classifications overall.
		- Maybe some more for interactive debugging.

	# Implementation Note
	- The weights and data files are mapped into memory with mmap().
	- Multiple instances may share the same weights file.

	# TODO Some Simplification
	- The weights, and delta-weights are in two matrixes.  Think of them
	  this way.
	- The outputs are three vectors.  Think of them this way.
*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <err.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
const double learnrate = 0.1;

/* Utillities */
#define SI static inline
#define ITER(VAR,FROM,TO) for (int VAR=FROM; VAR<TO; VAR++)
#define FORII(TO) ITER(ii,0,TO)
#define FORJJ(TO) ITER(jj,0,TO)
#define FORKK(TO) ITER(kk,0,TO)
SI size_t filelen (int fd) { struct stat s; fstat(fd, &s); return s.st_size; }
SI double sigmoid (double x) { return 1.0 / (1.0 + pow(M_E, -x)); }
SI bool streq (char *a, char *b) { return !strcmp(a, b); }
SI void shuffle (int *ar, int n) {
	for (int i,tmp; n; ar++, n--) {
		i = rand() % n;
		if (i) tmp = *ar, *ar = ar[i], ar[i] = tmp; }}

/*
	# Heap Management
	Since we only need a fixed-size heap, we can do our entire
	heap allocation with a single sbrk() call.
*/
static char *heap;
static size_t heapsize = 0;
#define BADHEAP "couldn't allocate memory"
const void* badbrk = (void*) -1;
void startalloc () { if (badbrk==(heap=sbrk(0))) err(1, BADHEAP); }
void endalloc () { if (badbrk==sbrk(heapsize)) err(1, BADHEAP); }
void *alloc (size_t nbytes) {
	const size_t w=sizeof(void*), x=nbytes%w, pad=w-x;
	nbytes += pad;
	void *result = heap+heapsize;
	heapsize += nbytes;
	return result; }

enum { INPUT, HIDDEN, OUTPUT };
double *Wts;
char *Data;
double *Outputs, *Dwts;
bool Train = false;
int *Perm;
struct { double network, *node; } Error;
struct { int *total, *correct; } Accuracy;

/* Convience */
int Nin, Nhid, Nout, Nsets;
char *Sample;
int Lsizes[3];

/*
	Returns an array of weights corresponding to the inputs to the
	specified node.
*/
double *getwts (int layer, int node) {
	switch (layer) {
	case HIDDEN: return Wts + node*(Nin+1);
	case OUTPUT: return Wts + Nhid*(Nin+1) + node*(Nhid+1); }
	errx(254, "Internal Error"); }

double *getdwt (int layer, int node) {
	switch (layer) {
	case HIDDEN: return Dwts + node*(Nin+1);
	case OUTPUT: return Dwts + Nhid*(Nin+1) + node*(Nhid+1); }
	errx(254, "Internal Error"); }

double *getout (int layer, int node) {
	switch (layer) {
	case INPUT: return Outputs + node;
	case HIDDEN: return Outputs + Nin + node;
	case OUTPUT: return Outputs + Nin + Nhid + node; }
	return NULL; /* Satisfies stupid compiler */ }

/*	Returns an array of inputs for the specified node.  This does *not*
	include the bias node!  */
double *getins (int layer, int node) {
	switch (layer) {
	case HIDDEN: return Outputs;
	case OUTPUT: return Outputs + Nin; }
	errx(254, "Internal Error"); }
	
double expected (int i) { return (double) Sample[Nin + i]; }
SI void calcoutput (int layer, int node) {
	if (layer == INPUT) {
		*getout(layer, node) = (double) Sample[node];
		return; }
	int ninputs = (layer==HIDDEN) ? Nin : Nhid;
	double *in=getins(layer, node),
	       *out=getout(layer, node),
	       *wt=getwts(layer, node);
	*out = 0;
	for (int ii=0; ii<ninputs; ii++) *out += wt[ii]*in[ii];
	*out += wt[ninputs]; /* bias */ }

void calcoutputs () {
	for (int layer=INPUT; layer <= OUTPUT; layer++)
		ITER (node, 0, Lsizes[layer]) {
			calcoutput(layer, node);
			if (layer != HIDDEN) continue;
			double *o = getout(layer, node);
			*o = sigmoid(*o); }
	double total = 0.0;
	FORII (Nout) total += pow(M_E, *getout(OUTPUT, ii));
	FORII (Nout) {
		double *o = getout(OUTPUT, ii);
		*o = pow(M_E, *o) / total; }}

/* Update the globals 'Error' and 'Accuracy' for the last run.  */
void calcerror () {
	FORII(Nout) {
		double nodeerror = pow(expected(ii)-*getout(OUTPUT, ii), 2);
		Error.node[ii] += nodeerror;
		Error.network += nodeerror;
		int class=-1;
		FORII(Nout)
			if (expected(ii)) { class=ii; break; }
		if (-1==class) continue;
		Accuracy.total[class]++;
		if (*getout(OUTPUT, class) > 0.9)
			Accuracy.correct[class]++; }}

void dotrain () {
	FORII(Nout) {
		double outout=*getout(OUTPUT, ii), expect=expected(ii);
		FORJJ(Nhid+1) {
			double *dwt, hidout;
			dwt = getdwt(OUTPUT,ii)+jj;
			hidout = (jj==Nhid)?1:(*getout(HIDDEN, jj));
			*dwt = learnrate * hidout * (expect-outout); }}
	FORII(Nhid) {
		double hidout = *getout(HIDDEN, ii);
		FORJJ(Nin+1) {
			double *dwt = getdwt(HIDDEN,ii)+jj;
			double inout = (jj==Nin)?1:*getout(INPUT, jj);
			double sum = 0.0;
			FORKK(Nout) {
				double outout = *getout(OUTPUT, kk);
				sum += getwts(OUTPUT, kk)[ii] * (expected(kk)-outout); }
			*dwt = learnrate * sum * hidout * (1.0-hidout) * inout; }}

	FORII(Nout) {
		double *ws=getwts(OUTPUT, ii), *dws=getdwt(OUTPUT, ii);
		FORJJ(Nhid+1) ws[jj] += dws[jj]; }
	FORII(Nhid) {
		double *ws = getwts(HIDDEN, ii), *dws=getdwt(HIDDEN, ii);
		FORJJ(Nin+1) ws[jj] += dws[jj]; }}

SI void epoch () {
	shuffle(Perm, Nsets);
	Error.network = 0.0;
	FORII(Nout) {
		Error.node[ii] = 0.0;
		Accuracy.total[ii] = 0;
		Accuracy.correct[ii] = 0; }

	ITER (set, 0, Nsets) {
		Sample = (Perm[set] * (Nin+Nout)) + Data;
		calcoutputs();
		calcerror();
		if (Train) dotrain(); }

	Error.network /= 2;
	FORII(Nout) Error.node[ii] /= 2; }

SI void networkerror () { printf("%lf\n", Error.network); }
SI void nodeerror () { FORII(Nout) printf("%lf\n", Error.node[ii]); }

/* For debugging */
void printexpected () {
	FORII(Nout) printf("%lf\n", expected(ii)); }

SI void output () {
	FORII(Nout) printf("%lf\n", *getout(OUTPUT, ii)); }

SI void totalaccuracy () {
	double t=0.0, c=0.0;
	FORII(Nout) t+=Accuracy.total[ii], c+=Accuracy.correct[ii];
	printf("%lf\n", 100 * c/t); }

SI void accuracy () {
	FORII(Nout) {
		double t = (double) Accuracy.total[ii];
		double c = (double) Accuracy.correct[ii];
		printf("%lf\n", 100 * c/t); }}

SI void weights () {
	for (int layer=HIDDEN; layer<=OUTPUT; layer++)
		ITER(node, 0, Lsizes[layer]) {
			double *w = getwts(layer, node);
			printf("Weights for inputs to %s layer, node %d:",
			       (layer==HIDDEN)?"hidden":"output", node);
			for (int ii=0, col=0; ii<Lsizes[layer-1]; ii++, col=(col+1)%5)
				printf("\t%s%lf", col?" ":"\n", w[ii]);
			printf("\nbias: %lf\n", w[Lsizes[layer-1]]); }}

SI bool handleinput() {
	static char buf[80];
	buf[78] = '\0';
	if (!fgets(buf, 80, stdin)) return false;
	if (buf[78] != '\0' && buf[78] != '\n')
		errx(2, "An command line is too long");
	buf[strlen(buf)-1] = '\0'; // remove trailing newline
	if (*buf=='\0') return true;
	else if (streq(buf, "epoch")) epoch();
	else if (streq(buf, "weights")) weights();
	else if (streq(buf, "networkerror")) networkerror();
	else if (streq(buf, "nodeerror")) nodeerror();
	else if (streq(buf, "accuracy")) accuracy();
	else if (streq(buf, "totalaccuracy")) totalaccuracy();
	else if (streq(buf, "done")) exit(0);
	else if (streq(buf, "outputs")) output();
	else if (streq(buf, "expected")) printexpected();
	else if (streq(buf, "training on")) Train = true;
	else if (streq(buf, "training off")) Train = false;
	else { warnx("Invalid command: %s", buf); return true; }
	puts("done");
	fflush(stdout);
	return true; }

char *dfile;
double *wfile;
SI void mapinfiles(char *wts, char *data) {
	const size_t dhdr=2, whdr=3*sizeof(double);

	int d,w;
	char *noopen="Can't open '%s'";
	char *toosmall="'%s' isn't big enough to store a header";
	if (-1 == (d=open(data, O_RDONLY))) err(1, noopen, data);
	if (-1 == (w=open(wts, O_RDWR))) err(1, noopen, wts);
	int dlen = filelen(d), wlen = filelen(w);
	if (dlen < dhdr) errx(2, toosmall, data);
	if (wlen < whdr) errx(2, toosmall, wts);

	dfile = mmap(NULL, dlen-dhdr, PROT_READ, MAP_PRIVATE, d, 0);
	if (MAP_FAILED == dfile) err(1, "mmap");
	wfile = mmap(NULL, wlen-whdr, PROT_READ|PROT_WRITE, MAP_SHARED, w, 0);
	if (MAP_FAILED == wfile) err(1, "mmap");
	if (wfile[0] != (double) dfile[0] || wfile[2] != (double) dfile[1])
		errx(2, "'%s' and '%s' have incompatible headers", data, wts);
	Lsizes[0] = Nin = (int) wfile[0];
	Lsizes[1] = Nhid = (int) wfile[1];
	Lsizes[2] = Nout = (int) wfile[2];

	char *badheader = "The size of '%s' is not consistaint "
	                  "with it's header";
	if ((dlen-2) % (Nin+Nout)) errx(2, badheader, data);
	int numweights = (Nin+1)*Nhid + (Nhid+1)*Nout;
	if (wlen != sizeof(double)*(3 + numweights))
		errx(2, badheader, wts);

	Nsets = (dlen-2) / (Nin+Nout);
	Data = dfile+2; Wts = wfile+3; /* Ignore the headers */ }

int main (int argc, char *argv[]) {
	if (argc != 3) errx(1, "usage: %s weights-file data-file", *argv);
	{char* s=getenv("seed"); if (s) srand(atoi(s)); }
	mapinfiles(argv[1], argv[2]);

	// Allocate all heap space for the program.
	int numweights = (Nin+1)*Nhid + (Nhid+1)*Nout;
	int numnodes = Nin + Nout + Nhid;
	startalloc();
	Dwts = alloc(numweights * sizeof(double));
	Outputs = alloc(numnodes * sizeof(double));
	Perm = alloc(Nsets * sizeof(int));
	Error.node = alloc(Nout * sizeof(double));
	Accuracy.total = alloc(Nout * sizeof(int));
	Accuracy.correct = alloc(Nout * sizeof(int));
	endalloc();

	FORII(Nsets) Perm[ii]=ii;
	while (handleinput());
	return 0; }
