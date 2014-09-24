## Setup

In order for the demo to run correctly:

1. Download the VOC validation data from the following source ( https://filebox.ece.vt.edu/~vittal/embr/voctest50data.tar ) and copy the 'voctest50data' direcory into the 'embr/mbr_intseg' directory.
2. Download and install TRW-S from the following source ( http://www.robots.ox.ac.uk/%7Eojw/files/imrender_v2.4.zip ).
3. Download 'DivMBest_intseg.m' from the following source (link to DivMBest repository) and place the file in the 'embr/mbr_intseg' directory. The 'DivMBest_intseg.m' is used to generate the Diverse M-Best solutions for each image in the 'voctest50data'. Also, 'DivMBest_intseg.m' uses TRW-S to perform the inference.
4. The results from the paper can be replicated by running the demo script 'demo_intseg.m'
