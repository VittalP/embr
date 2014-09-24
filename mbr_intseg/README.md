## Setup

In order for the demo to run correctly:

1. Download the VOC validation data from the following source () and copy the 'voctest50data' direcory into the 'mbr_intseg' directory.
2. Download and install TRW-S from the following source ().
3. Download 'DivMBest_intseg.m' from the following source and place the file in the 'mbr_intseg' directory. The 'DivMBest_intseg.m' is used to generate the Diverse M-Best solutions for each image in the 'voctest50data'. Also, 'DivMBest_intseg.m' uses TRW-S to perform the inference.
4. The results from the paper can be replicated by running the demo script 'demo_intseg.m'
