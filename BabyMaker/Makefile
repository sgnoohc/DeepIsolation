ROOTLIBS = $(shell root-config --libs) -lMinuit -lMLP -lXMLIO -lTMVA -lGenVector -lEG
FASTJETCFG = $(shell /cvmfs/cms.cern.ch/slc6_amd64_gcc630/external/fastjet/3.1.0/bin/fastjet-config --cxxflags --libs)

CC = g++
CFLAGS = -g -std=c++1y

TARGET = processBaby

all: $(TARGET)

processBaby: ScanChain.h ScanChain.cpp processBaby.cpp
	$(CC) $(CFLAGS) -I${ROOTSYS}/include -o processBaby processBaby.cpp $(ROOTLIBS) $(FASTJETCFG)

