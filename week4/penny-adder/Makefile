all: penny01 penny02 penny03 penny04 penny05 penny06 penny07 penny08 penny09
penny01: penny01.c 
	cc -o penny01 penny01.c  
penny02: penny02.c time_diff.c time_diff.h
	cc -o penny02 penny02.c time_diff.c  -pthread
penny03: penny03.c time_diff.c time_diff.h
	cc -o penny03 penny03.c time_diff.c  -pthread
penny04: penny04.c time_diff.c time_diff.h
	cc -o penny04 penny04.c time_diff.c  -pthread
penny05: penny05.c time_diff.c time_diff.h
	cc -o penny05 penny05.c time_diff.c  -pthread
penny06: penny06.c time_diff.c time_diff.h
	cc -o penny06 penny06.c time_diff.c  -pthread
penny07: penny07.c time_diff.c time_diff.h
	cc -o penny07 penny07.c time_diff.c  -pthread
penny08: penny08.c time_diff.c time_diff.h
	cc -o penny08 penny08.c time_diff.c  -pthread
penny09: penny09.c time_diff.c time_diff.h
	cc -o penny09 penny09.c time_diff.c  -pthread
clean:
	rm -f penny01 penny01 penny02 penny03 penny04 penny05 penny06 penny07 penny08 penny09 *~ a.out
