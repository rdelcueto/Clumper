# Clumper Makefile
package = Clumper
version = 1.0rc
tarname = $(package)-dist
dist-dir = $(tarname)-$(version)

include ./Makefile.inc

all clean:
	cd ./src && make $@

.PHONY: all
