ROOTLIBS = $(shell root-config --libs) -lMinuit -lMLP -lXMLIO -lTMVA -lGenVector -lEG

CC = g++
CFLAGS = -g -std=c++1y

TARGET = processBaby

all: $(TARGET)

processBaby: ScanChain.h ScanChain.cpp processBaby.cpp
	$(CC) $(CFLAGS) -I${ROOTSYS}/include -o processBaby processBaby.cpp $(ROOTLIBS)

