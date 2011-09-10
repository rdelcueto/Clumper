# Clumper Makefile
package = Clumper
version = 0.0.1alpha
tarname = $(package)-dist
dist-dir = $(tarname)-$(version)

include ./Makefile.inc

all clean:
	cd ./src && make $@

.PHONY: all
