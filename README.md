# Clair - Yet another deep neural network based variant caller  
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)  
Contact: Ruibang Luo  
Email: rbluo@cs.hku.hk  

***

## Introduction
This is Clair v1, which was made available in Jan 2019. Please visit [v2, Nov 2019](https://github.com/HKU-BAL/Clair) for the latest version.

__Clair__ is the successor of [Clairvoyante](https://github.com/aquaskyline/Clairvoyante).

## Usage
```shell
git clone https://github.com/HKU-BAL/Clair.git
cd Clair
./clair.py
```

## Models
Please download models from [here](http://www.bio8.cs.hku.hk/clair/models/).

Folder | Tech | Sample used | Depth used
--- | :---: | :---: | :---: |
pacbio/rsii | PacBio RSII | HG001,2,3,4 | "Full depth of each sample" x {0.1,0.2,0.4,0.6,0.8,1.0} |
pacbio/ccs | PacBio CCS 15k | HG002 | ~28-fold x {0.1, 0.2 ..., 0.9} |
ont/r94 | ONT R9.4 (no flip-flop) | HG001 | ~37-fold x {0.1, 0.2 ..., 0.9} |
ont/r94-flipflop | ONT R9.4 (flip-flop) | HG001 | ~150-fold x 0.7<sup>{0,1 ..., 9}</sup>
