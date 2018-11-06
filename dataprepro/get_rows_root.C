#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderArray.h"
#include <glob.h>
#include <vector>
#include <iostream>
#include <string>
using namespace std;

inline vector<string> glob(const string& pat){
    using namespace std;
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}

void printEventcount(TString infileM){
    TFile *fileM = TFile::Open(infileM); //open the files
    TTreeReader eventReader("Events", fileM); //create a TTreeReader, a TTreeReaderArray/Value for every leaf
    cout << infileM << " has " << eventReader.GetEntries(0) << " events\n";
}

void getEventCounts(string globStr="superstar/*_S_*.root"){

    vector<string> filenames;
    filenames = glob(globStr);

    for (int i = 0; i < filenames.size(); i++){
        printEventcount(filenames[i]);
    }

}
