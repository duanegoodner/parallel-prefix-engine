# build tester.cpp and sort.cpp

CC=mpiCC
CFLAGS=-I.

all: test_sort

tester.o: tester.cpp test_sort.o helper.o
	$(CC) -c -o tester.o $< $(CFLAGS)

test_sort.o: sort.cpp helper.o
	$(CC) -c -o test_sort.o $< $(CFLAGS)

helper.o: helper.cpp
	$(CC) -c -o helper.o $< $(CFLAGS)

test_sort: tester.o
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o test_sort
