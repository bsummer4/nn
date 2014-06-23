/*
	This parses all input data files and then outputs the binary
	repersentation to stdout.  See README for the file format.

	We assume that the input file is correct unless it causes us
	problems.  We also don't check if the output stream is closed
	prematurly (we don't care).
*/

#include <stdio.h>
#include <err.h>
#include <stdbool.h>
#define SI static inline

SI char getbit (FILE *f, char *fn) {
	int i;
	if (!fscanf(f, "%d", &i)) errx(1, "Invalid number (file %s)", fn);
	if (!i || i==1) return i;
	errx(1, "All numbers must be 1 or 0, not %d (file %s)", i, fn); }

SI void grabfile (char *fn) {
	FILE *f;
	if (!(f = fopen(fn, "r"))) err(1, "Couldn't open %s", fn);
	for (int remain=12*8+10; remain; remain--) {
		char b = getbit(f, fn);
		fwrite(&b, 1, 1, stdout); }
	fclose(f); }

int main (int argc, char *argv[]) {
	char inputs_outputs[] = {12*8, 10};
	fwrite(&inputs_outputs, 1, 2, stdout);
	while (*++argv) grabfile(*argv);
	return 0; }
