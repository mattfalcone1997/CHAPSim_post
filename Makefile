
.PHONY : clean  scripts
clean : 
	rm -r bin/*
scripts : 
	cp src/* bin
all : clean scripts
