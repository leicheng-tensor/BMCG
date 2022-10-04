#include "mex.h"   
    void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])   
    {  
    int i;   
    i=mxGetScalar(prhs[0]); //get input parameter  
    if(i==1)   
      mexPrintf("hello,world!\n");   
    else   
      mexPrintf("大家好！！！！\n");   
    }  