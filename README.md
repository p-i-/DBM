Reworking of: "Learning Deep Boltzmann Machines" by Ruslan Salakhutdinov
              http://www.cs.toronto.edu/~rsalakhu/DBM.html
          By: pi@pipad.org
        Date: October 2016


Original license:
  Permission is granted for anyone to copy, use, modify, or distribute this
  program and accompanying programs and documents for any purpose, provided
  this copyright notice is retained and prominently displayed, along with
  a note saying that the original programs are available from our web page.

  The programs and documents are distributed without any warranty, express or
  implied.  As the programs were written for research purposes only, they have
  not been tested to the degree that would be advisable in any important
  application.  All use of these programs is entirely at the user's own risk. 


Usage:
  main bypassToStage % = 0(default), 1, 2, 3, 4

  Stages:
    0 : load MNIST data from remote website, decompress & process
    1 : pretrain L1 (first hidden layer)
    2 : pretrain L2 (second hidden layer)
    3 : finalise DBM
    4 : fine tune using back propagation

NOTES:
  I had a go at reworking this code as I found the original impenetrable and I thought it may
  improve my MatLab and ML skills. Features of the rework include:

    * simple consistent conventions for variable names, e.g.
        W01 represents the weight matrix from levels 0 (visible) to 1 (first hidden).
    * automatically download MNIST data from Internet
    * structuring of source tree, initial&generated data is kept separate from code
    * use of functions to promote sanity & keep the number of active variables down to a minimum
    * use of auxiliary functions to improve legibility
    * console outputs progress in a human readable manner

  Pretty much every single idea has been unpicked and reconstructed.
  however, I have a very incomplete understanding of the underlying technology.
  While I have fixed several bugs I suspect I have introduced a couple more.
  During one particular edit I was able to get startling training accuracy
  (<1000 misclassifications for stage 3, typically it generates ~4k)
  however MatLab crashed and I lost my work and I was unable remember the steps
  necessary to recreate. Vexing!

  If any ML expert can (help me) clean this up, I am most grateful.
  I can generally be found on ##machinelearning on IRC (Freenode server)

SOURCES:
  I'm not quite sure if this code pertains to exactly one academic paper. 
  Plausible candidates I have found are:

  - [DBM09] Salakhutdinov/Hinton (2009) -- Deep Boltzmann Machines
    http://www.cs.toronto.edu/~fritz/absps/dbm.pdf
    Section 3.1: Greedy Layerwise Pretraining of DBMs

  - [ELoDBM] Salakhutdinov/LaRochelle 2010 -- Efficient Learning of Deep Boltzmann Machines
    http://www.cs.cmu.edu/~rsalakhu/papers/dbmrec.pdf
    Section 2.2: Greedy Pretraining of DBM?s <-- Fig 2 & Algo 1
    ^ This is EXACTLY what our pretraining steps are doing.

  - [ELP4DBM] Salakhutdinov & Hinton (2010) -- An Efficient Learning Procedure for Deep Boltzmann Machines
      2 versions: 
        * MIT-CSAIL-TR-2010-037 (Aug 4 2010) <-- I USED THIS ONE
        * http://www.utstat.toronto.edu/~rsalakhu/papers/neco_DBM.pdf (2012)
      ~30 pages(!)
      Fig 8 (left) appears to be this code.

  Possibly helpful resources:
    - http://deeplearning.net/tutorial/rbm.html
    - Hinton Coursera vid: 14 - 2 - Discriminative learning for DBNs [9 mins]
