CC = gcc
CFLAGS = -O3 -std=gnu99 -Wall -pedantic -lm
DATA = learn.bin test.bin validate.bin weights

<config.mk


all:V: $DATA nn init parse train writeup
train:V: weights.trained
graphs:V: node-error.eps network-error.eps
writeup:V: writeup.pdf

tidy:V:
	rm -f writeup.pdf nn init parse *.o weights *.plot o.* *.eps *.dvi *.ps \
	  *.aux *.log table.tex *.bin

clean:V: tidy
	rm -f weights.trained *.pdf


o.%: %.c
	$CC $CFLAGS $stem.c -o $target

%.bin:D: o.parse
	./o.parse $datadir/$stem-grid/* > $target


nn init parse: o.nn o.init o.parse
	cp -f o.nn nn
	cp -f o.parse parse
	cp -f o.init init

table.tex weights.trained:D: o.nn weights $DATA
	./train.tcl weights.trained .plot table.tex

weights:D: o.init
	./o.init $inputs $hidden $outputs > $target

writeup.pdf: graphs table.tex
	latex writeup.tex
	dvips writeup.dvi
	ps2pdf writeup.ps

node-error.eps: train
	gnuplot <<!
	set terminal postscript
	set output "node-error.eps"
	set xlabel "Epoch Number"
	set ylabel "Node Error"
	set nokey
	plot "node-error.0.plot" with lines, \\
	     "node-error.1.plot" with lines, \\
	     "node-error.2.plot" with lines, \\
	     "node-error.3.plot" with lines, \\
	     "node-error.4.plot" with lines, \\
	     "node-error.5.plot" with lines, \\
	     "node-error.6.plot" with lines, \\
	     "node-error.7.plot" with lines, \\
	     "node-error.8.plot" with lines, \\
	     "node-error.9.plot" with lines
	!

network-error.eps: train
	gnuplot <<!
	set terminal postscript
	set output "network-error.eps"
	set xlabel "Epoch Number"
	set ylabel "Network Error"
	plot "network-error.learn.plot" with linespoints, \\
	     "network-error.vali.plot" with linespoints
	!
