
1) Download and install LMAFIT:
http://lmafit.blogs.rice.edu/

2) We compare the performance of LMAFIT with "APGL (NNLS)", FPCA and Optspace.
These three codes can be downloaded from:
http://www.math.nus.edu.sg/~mattohkc/NNLS.html
http://www.columbia.edu/~sm2756/FPCA.htm 
http://www.stanford.edu/~raghuram/optspace/code.html

In particular, comparison with APGL is needed in:
Table 4.2, Table 4.3, Table 4.4, Table 4.5 , Table 4.6 and Fig. 4.5, 
Table 4.7 and Fig. 4.6

Comparison with FPCA is needed in:
Fig.  4.3, Table 4.1, Table 4.4

Comparison with Optspace is needed in:
Table 4.1

Please download and install these codes before running the testing scripts on 
the corresponding figures and tables.

3) Please download:
 i)  testing scripts: perf-release.zip
 ii) testing data matrices: perf-release-data.zip

Unzip "perf-release.zip" and "perf-release-data.zip", then add their MATLAB path 

4) Please see "run_all.m" for an example on running our scripts

-----------------------------------
Fig.  4.1: 
run: plot_LMAFIT_rank_estimate.m

Fig.  4.2: 
run: plot_LMAFIT_residual

Fig.  4.3:  
comparison with APGL and FPCA
first run: Test_MC_PhasePlot.m, Test_MC_PhasePlot_FPCA.m, Test_MC_PhasePlot_APGL.m,
then plot figures by using "plot_phase_MC.m"
    
Table 4.1:
comparison with APGL, FPCA and OptSpace
run: Test_MatComp_LMAFIT_Table_diff_FR_vary_perf.m

The results of C version of OptSpace is generated by:
Test_MatComp_LMAFIT_Table_diff_FR_vary_optspace.m
(Compile OptSpace_C/testing/test, and copy it to perf-release)

Table 4.2:
comparison with APGL
run: runRandomMatComp_perf.m with "scenario = 'noiseless';"

Table 4.3:
comparison with APGL and FPCA
run: runRandomMatComp_perf.m with "scenario = 'noisy';"

Table 4.4: 
comparison with APGL
run: Test_LMAFIT_Table_ill_cond.m
power-law decaying: sig_meth = 1;  
exponential decaying: sig_meth = 3;  

Table 4.5 
comparison with APGL
run: runRealMatComp_perf_wen.m;   
preprocessing data: readRealMatComp.m


Table 4.6 and Fig. 4.5
comparison with APGL
run: Test_LMAFIT_image.m

Table 4.7 and Fig. 4.6
comparison with APGL
run: videocodes/testMCvideoSamp.m
