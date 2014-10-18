## Setup

In order to run the demo correctly:

1. Download the DivMBest pose code from the following source ( https://github.com/batra-mlp-lab/divmbest/tree/master/pose_estimation ) and place it in embr/mbr_pose directory.
2. Follow the instructions in the README present in the above location.
3. The results presented in the publiction can now be replicated by running the 'demo_pose_estimation.m' script.

## Downloads

Pre-computed DivMBest solutions for all the images in the PARSE dataset can be downloaded from the following source ( https://filebox.ece.vt.edu/~vittal/data/DivMBest_PARSE.zip ).
After downloading, unzip the tarball, create a directory called 'cache' within ./embr/embr_pose , and copy the files into it.


## Acknowledgements

We thank Yang and Ramanan for releasing the code accompanying the following publication.


        @article{yang2013articulated,
          title={Articulated human detection with flexible mixtures of parts},
          author={Yang, Yi and Ramanan, Deva},
          journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
          volume={35},
          number={12},
          pages={2878-2890},
          year={2013},
          publisher={IEEE}
        }
