# build tester.cpp and prefix_sum.cpp

CC=mpiCC
CFLAGS=-I.

all: test_prefix_sum

tester.o: tester.cpp test_prefix_sum.o
	$(CC) -c -o tester.o $< $(CFLAGS)

test_prefix_sum.o: prefix_sum.cpp
	$(CC) -c -o test_prefix_sum.o $< $(CFLAGS)

test_prefix_sum: tester.o
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	rm -f *.o test_prefix_sum
