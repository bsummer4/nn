#!/usr/bin/tclsh8.6

# This runs 'nn' until it converges, and collects information on the
# process.  We fork off two version of nn; One for the training data and
# one for the validation data.  The result of running this program is that
# a set of files are created.
#
# 	- A trained version of the weights file
# 	- Some one-point-per-line data files with the error information
# 	- A latex table with the accuracy information

proc send {f ls} { foreach l $ls {puts $f $l}; flush $f }
proc waitfor f {
	set result [list]
	while 1 {
		if {[eof $f]} { return $result }
		gets $f line
		if {[string equal $line done]} { return $result }
		lappend result $line }}

set vali [dict create net [list] node [list]]
set learn [dict create net [list] node [list]]
set net_e [list]
set node_e [list]
set epochnum 0
proc trainset {l v} {
	for {set i 0} {$i<1} {incr i} {
		send $l {epoch nodeerror networkerror}
		waitfor $l
		dict lappend ::learn node [waitfor $l]
		dict lappend ::learn net [waitfor $l] }
	send $v {epoch nodeerror networkerror totalaccuracy}
	waitfor $v
	dict lappend ::vali node [waitfor $v]
	dict lappend ::vali net [waitfor $v]
	puts stderr "$::epochnum -- [waitfor $v]%"
	incr ::epochnum
	lindex [dict get $::vali net] end }

proc train {l v} {
	set best [list [trainset $l $v]]
	set decreasing_for 0
	set ves [list]
	while 1 {
		set ve [trainset $l $v]
		lappend ves $ve
		set vel [llength $ves]
		if {$ve < $best} {
			set best $ve
			set decreasing_for 0
			puts stderr "$::epochnum -- new best"
			exec cp weights $::resultfile } \
		else { incr decreasing_for }
		if {$vel<150} continue
		if {$vel>10000} break
		if {(double($decreasing_for)/double($vel)) > .33} break }}

proc . x { return $x }
proc usage {} {
	puts stderr "usage: $::argv0 result-file \[graph-suffix tex-table\]"
	puts "If only 1 argument is, then no graphs or tables are created.  "
	exit 0 }

switch -- [llength $argv] {
	1 {
		lassign $argv resultfile
		set graphs 0 }
	3 {
		lassign $argv resultfile suffix table
		set graphs 1 }
	default usage }

exec cp weights weights.init
set l [open "| ./o.nn weights learn.bin" r+]
set v [open "| ./o.nn weights validate.bin" r+]
send $l {{training on}}; waitfor $l
train $l $v
exec mv weights.init weights

if {!$graphs} exit

# Validation Network-Error Plots
set vnd [open network-error.vali$suffix w]
foreach e [dict get $::vali net] { puts $vnd $e }
close $vnd

# Learning Network-Error Plots
set tnd [open network-error.learn$suffix w]
foreach e [dict get $::learn net] { puts $tnd $e }
close $tnd

# Learning Error-per-Node Plots
set outputs {0 1 2 3 4 5 6 7 8 9}
foreach o $outputs { set ne.$o [open node-error.$o$suffix w] }
foreach e [dict get $::learn node] {
	foreach o $outputs {
		puts [set ne.$o] [lindex $e $o] }}

# Accuracy Tables
set v [open "| ./o.nn $resultfile test.bin" r+]
send $v {epoch accuracy totalaccuracy}
waitfor $v
set nodeacc [waitfor $v]
set totalacc [waitfor $v]
close $v

set tf [open $table w]
puts $tf {\begin{center}}
puts $tf {  \begin{tabular}{ | l | c | }}
puts $tf "    \\hline Overall & $totalacc\\% \\\\"
foreach c {A C D E F G H L P R} a $nodeacc {
	puts $tf "    \\hline $c & $a\\% \\\\" }
puts $tf {    \hline}
puts $tf {  \end{tabular}}
puts $tf {\end{center}}
close $tf
