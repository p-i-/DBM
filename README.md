# DBM
Reworking of: [Learning Deep Boltzmann Machines](http://www.cs.toronto.edu/~rsalakhu/DBM.html) by Ruslan Salakhutdinov

by **π** (`pi~pipad!org`)  
October 2016

## Original license:
>   Permission is granted for anyone to copy, use, modify, or distribute this
>   program and accompanying programs and documents for any purpose, provided
>   this copyright notice is retained and prominently displayed, along with
>   a note saying that the original programs are available from our web page.
> 
>   The programs and documents are distributed without any warranty, express or
>   implied.  As the programs were written for research purposes only, they have
>   not been tested to the degree that would be advisable in any important
>   application.  All use of these programs is entirely at the user's own risk. 


## Usage
    main [bypassToStage]

* Stages:
    * 0: *(default)* Load MNIST data from remote website, decompress & process
    * 1: Pretrain L1 (first hidden layer)
    * 2: Pretrain L2 (second hidden layer)
    * 3: Finalise DBM
    * 4: Fine tune using back propagation

Code runs to completion on MatLab 2016A on OSX.

## About the rewrite
I reworked this code as I found the original impenetrable and I wanted to improve my MatLab and ML skills. I think this code may serve as a template for exploring other NN designs. 

Features of the rework include:

* Automatically download MNIST data from Internet.
* Optionally bypass stages for an efficient debugging cycle.
* Simple consistent conventions for succinct naming of variables, e.g. `W01` represents the weight matrix from levels `0` (visible) to `1` (first hidden).  `W10` would go the other direction and hence be the transpose of `W01`. My experience is that e.g. `b1` can be spotted at a glance everywhere on a visible page, whereas something like `hidbiaswakephase` needs to actually be read&deciphered. It's much easier for my visual cortex to spot patterns with short variable names.  I am aware that conventional wisdom is in favour of long descriptive variable names; I break this rule consciously to bring clarity. I suspect that the original author (Ruslan) had a blueprint in front of him that he was transcribing to code. However without that blueprint the code is really hard to follow.
* Structuring of source tree, data is kept separate from code (in `/data`). The original code had everything in the same folder including generated temporary datafiles.
* I've inserted appropriate functions to promote sanity & keep the number of active variables down to a minimum (i.e. using the stack). The original didn't even use functions, creating a cluttered mess in the global store.
* I've used auxiliary functions to allow, e.g. `binarySampleFrom( sigmoid( foo ) )` which makes things much more legible.
* Console outputs are now human readable, rather than a zillion Lines flashing before your eyes.
* I've rewritten pretty much every little piece of machinery, frequently finding cleaner ways of doing things. 

Pretty much every single idea has been unpicked and reconstructed.  However, I still have a very incomplete understanding of the underlying process.  While I have fixed several bugs I suspect I have introduced a couple more (although it appears to perform at least as well as the original).  During one particular edit I was able to get startling training accuracy (<1000 misclassifications for stage 3, typically it generates ~4k) however MatLab crashed and I lost my work and I was unable remember the steps necessary to recreate. Vexing! I have no idea if it was a statistical anomaly.

If any ML expert can (help me) clean this up, I am most grateful.  I can generally be found on ##machinelearning on IRC (Freenode server).

## SOURCES
  I'm not quite sure if this code pertains to exactly one academic paper.  Plausible candidates I have found are:

  - [DBM09] [Deep Boltzmann Machines](http://www.cs.toronto.edu/~fritz/absps/dbm.pdf) (Salakhutdinov/Hinton 2009)  
    See Section 3.1: Greedy Layerwise Pretraining of DBMs

  - [ELoDBM] [Efficient Learning of Deep Boltzmann Machines](http://www.cs.cmu.edu/~rsalakhu/papers/dbmrec.pdf) (Salakhutdinov/LaRochelle 2010)  
    See Section 2.2: Greedy Pretraining of DBM?s <-- Fig 2 & Algo 1  
    ^ This is EXACTLY what our pretraining steps are doing.  

  - [ELP4DBM] [An Efficient Learning Procedure for Deep Boltzmann Machines](http://www.utstat.toronto.edu/~rsalakhu/papers/neco_DBM.pdf) (Salakhutdinov & Hinton 2010)  
    2 versions:

       - MIT-CSAIL-TR-2010-037 (Aug 4 2010) <-- I USED THIS ONE  
       -  (2012) 

    ~30 pages(!).  Fig 8 (left) appears to be this code.

**Possibly helpful resources:**

  - [http://deeplearning.net/tutorial/rbm.html]()
  - Hinton Coursera vid: 14 - 2 - Discriminative learning for DBNs [9 mins]
