# QLY - A deep neural network based variant caller
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
Contact: Ruibang Luo
Email: rbluo@cs.hku.hk

***

## Installation
```shell
git clone --depth=1 https://github.com/HKU-BAL/QLY.git
cd QLY
curl http://www.bio8.cs.hku.hk/trainedModels.tbz | tar -jxf -
```

## Introduction
Identifying the variants of DNA sequences sensitively and accurately is an important but challenging task in the field of genomics. This task is particularly difficult when dealing with Single Molecule Sequencing, the error rate of which is still tens to hundreds of times higher than Next Generation Sequencing. With the increasing prevalence of Single Molecule Sequencing, an efficient variant caller will not only expedite basic research but also enable various downstream applications.


## Prerequisition
### Basics
Make sure you have Tensorflow ≥ 1.0.0 installed, the following commands install the lastest CPU version of Tensorflow:

```shell
pip install tensorflow
pip install blosc
pip install intervaltree
pip install numpy
```

To check the version of Tensorflow you have installed:

```shell
python -c 'import tensorflow as tf; print(tf.__version__)'
```

To do variant calling using trained models, CPU will suffice. QLY uses all available CPU cores by default in `call_var_qly.py`, use 4 threads by default in `qly-callVarBam.py`, and can be controlled using the parameter `--threads`. To train a new model, a high-end GPU along with the GPU version of Tensorflow is needed. To install the GPU version of tensorflow:

```shell
pip install tensorflow-gpu
```

QLY was written in Python2 (tested on Python 2.7.10 in Linux and Python 2.7.13 in MacOS). It can be translated to Python3 using "2to3" just like other projects.

### Performance of GPUs in model training
Equiptment | Seconds per Epoch per 11M Variant Tensors |
:---: |:---:|
Tesla V100 | 90 |
GTX 1080 Ti | 170 |
GTX 980 | 350 |
GTX Titan | 520 |
Tesla K40 (-ac 3004,875) | 580 |
Tesla K40 | 620 |
Tesla K80 (one socket) | 600 |
GTX 680 | 780 |
Intel Xeon E5-2680 v4 28-core | 2900


### Speed up with PyPy
Without a change to the code, using PyPy python interpreter on some tensorflow independent modules such as `qly-dataPrepScripts/ExtractVariantCandidates.py` and `qly-dataPrepScripts/CreateTensor.py` gives a 5-10 times speed up. Pypy python interpreter can be installed by apt-get, yum, Homebrew, MacPorts, etc. If you have no root access to your system, the official website of Pypy provides a portable binary distribution for Linux. Following is a rundown extracted from Pypy's website (pypy-5.8 in this case) on how to install the binaries.

```shell
wget https://bitbucket.org/squeaky/portable-pypy/downloads/pypy-5.8-1-linux_x86_64-portable.tar.bz2
tar -jxf pypy-5.8-1-linux_x86_64-portable.tar.bz2
cd pypy-5.8-linux_x86_64-portable/bin
./pypy -m ensurepip
./pip install -U pip wheel intervaltree blosc
# Use pypy as an inplace substitution of python to run the scripts in dataPrepScripts/
```

If you can use apt-get or yum in your system, please install both `pypy` and `pypy-dev` packages. And then install the pip for pypy.

```shell
sudo apt-get install pypy pypy-dev
wget https://bootstrap.pypa.io/get-pip.py
sudo pypy get-pip.py
sudo pypy -m pip install blosc
sudo pypy -m pip install intervaltree
```

To guarantee a good user experience, pypy must be installed to run `qly-callVarBam.py` (call variants from BAM), and `qly-callVarBamParallel.py` that generate parallelizable commands to run `qly-callVarBam.py`.
Tensorflow is optimized using Cython thus not compatible with `pypy`. For the list of scripts compatible to `pypy`, please refer to the **Folder Stucture and Program Descriptions** section.
*Pypy is an awesome Python JIT intepreter, you can donate to [the project](https://pypy.org).*

***

## Quick Start with Variant Calling
You have a slow way and a quick way to get some demo variant calls. The slow way generates required files from VCF and BAM files. The fast way downloads the required files.
### Download testing dataset
#### I have plenty of time

```shell
wget 'http://www.bio8.cs.hku.hk/testingData.tar'
tar -xf testingData.tar
cd qly-dataPrepScripts
sh PrepDataBeforeDemo.sh
```

#### I need some results now

```shell
wget 'http://www.bio8.cs.hku.hk/training.tar'
tar -xf training.tar
```

### Call variants
#### Call variants from at known variant sites using a BAM file and a trained model

```shell
cd training
python ../QLY/qly-callVarBam.py \
       --chkpnt_fn ../trainedModels/fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e-3.epoch500 \
       --bam_fn ../testingData/chr21/chr21.bam \
       --ref_fn ../testingData/chr21/chr21.fa \
       --bed_fn ../testingData/chr21/chr21.bed \
       --call_fn chr21_calls.vcf \
       --ctgName chr21
less chr21_calls.vcf
```

#### Call variants from the tensors of candidate variant and a trained model

```shell
cd training
python ../QLY/call_var_qly.py --chkpnt_fn ../trainedModels/fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e-3.epoch500 --tensor_fn tensor_can_chr21 --call_fn tensor_can_chr21.vcf
less tensor_can_chr21.vcf
```

***

## How to call variant directly from BAM

### Input example
| Input | File name |
| :---: | :---: |
| BAM | `GIAB_v3.2.2_Illumina_50x_GRCh38_HG001.bam` |
| Reference Genome | `GRCh38_full_analysis_set_plus_decoy_hla.fa` |
| BED for where to call variants | `GRCh38_high_confidence_interval.bed` |

* If no BED file was provided, QLY will call variants on the whole genome

### Commands
```shell
python QLY/qly-callVarBamParallel.py \
       --chkpnt_fn trainedModels/fullv3-illumina-novoalign-hg001+hg002-hg38/learningRate1e-3.epoch500 \
       --ref_fn GRCh38_full_analysis_set_plus_decoy_hla.fa \
       --bam_fn GIAB_v3.2.2_Illumina_50x_GRCh38_HG001.bam \
       --bed_fn GRCh38_high_confidence_interval.bed \
       --sampleName HG001 \
       --output_prefix hg001 \
       --threshold 0.125 \
       --minCoverage 4 \
       --tensorflowThreads 4 \
       > commands.sh
export CUDA_VISIBLE_DEVICES=""
cat commands.sh | parallel -j4
vcfcat hg001*.vcf | vcfstreamsort | bgziptabix hg001.vcf.gz
```

* `parallel -j4` will run 4 commands in parallel. Each command using at most `--tensorflowThreads 4` threads. `vcfcat`, `vcfstreamsort` and `bgziptabix` are a part of **vcflib**.
* If you don't have `parallel` installed on your computer, try `awk '{print "\""$0"\""}' commands.sh | xargs -P4 -L1 sh -c`.
* `CUDA_VISIBLE_DEVICES=""` makes GPUs invisible to QLY so it will use CPU only. Please notice that unless you want to run `commands.sh` in serial, you cannot use GPU because one running copy of QLY will occupy all available memory of a GPU. While the bottleneck of `qly-callVarBam.py` is at the CPU only `CreateTensor.py` script, the effect of GPU accelerate is insignificant (roughly about 15% faster). But if you have multiple GPU cards in your system, and you want to utilize them in variant calling, you may want split the `commands.sh` in to parts, and run the parts by firstly `export CUDA_VISIBLE_DEVICES="$i"`, where `$i` is an integer from 0 identifying the seqeunce of the GPU to be used.

***

## VCF Output Format
`QLY/call_var_qly.py` outputs variants in VCF format with version 4.1 specifications.
QLY can predict the exact length of insertions and deletions shorter than or equal to 4bp. For insertions and deletions with a length between 5bp to 15bp, callVar guesses the length from input tensors. The indels with guessed length are denoted with a `LENGUESS` info tag. Although the guessed indel length might be incorrect, users can still benchmark QLY's sensitivity by matching the indel positions to other callsets. For indels longer than 15bp, `call_var_qly.py` outputs them as SV without providing an alternative allele. To fit into a different usage scenario, QLY allows users to extend its model easily to support exact length prediction on longer indels by adding categories to the model output. However, this requires additional training data on the new categories. Users can also increase the length limit from where an indel is outputted as a SV by increasing the parameter flankingBaseNum from 16bp to a higher value. This extends the flanking bases to be considered with a candidate variant.

***

## Build a Model
### Quick start with a model training demo

```
wget 'http://www.bio8.cs.hku.hk/testingData.tar'
tar -xf testingData.tar
cd QLY
python demoRun.py
```


## Folder Stucture and Program Descriptions
*You can also run the program to get the parameter details.*

`qly-dataPrepScripts/` | Data Preparation Scripts. Outputs are gzipped unless using standard output. Scripts in this folder are compatible with `pypy`.
--- | ---
`ExtractVariantCandidates.py`| Extract the position of variant candidates. Input: BAM; Reference FASTA. Important options: --threshold "Minimum alternative allelic fraction to report a candidate"; --minCoverage "Minimum coverage to report a candidate".
`GetTruth.py`| Extract the variants from a truth VCF. Input: VCF.
`CreateTensor.py`| Create tensors for candidates or truth variants. Input: A candidate list; BAM; Reference FASTA. Important option: --considerleftedge "Count the left-most base-pairs of a read for coverage even if the starting position of a read is after the starting position of a tensor. Enable if you are 1) using reads shorter than 100bp, 2) using a tensor with flanking length longer than 16bp, and 3) you are using amplicon sequencing or other sequencing technologies, in which reads starting positions are random is not a basic assumption".
`PairWithNonVariants.py`| Pair truth variant tensors with non-variant tensors. Input: Truth variants tensors; Candidate variant tensors. Important options: --amp x "1-time truth variants + x-time non-variants".
`ChooseItemInBed.py`| Helper script. Output the items overlapping with input the BED file.
`CountNumInBed.py`| Helper script. Count the number of items overlapping with the input BED file.
`param.py`| Global parameters for the scripts in the folder.
`PrepDataBeforeDemo.sh`| A **Demo** showing how to prepare data for model training.
`PrepDataBeforeDemo.pypy.sh`| The same demo but using pypy in place of python. `pypy` is highly recommended. It's easy to install, and makes the scripts run 5-10 times faster.
`CombineMultipleDatasetsForTraining.sh`| Use chr21 and chr22 to exemplify how to maunually combine multiple datasets for model training.
`CombineMultipleDatasetsForTraining.py`| A helper script for combining multiple datasets. You still need to run PairWithNonVariants.py and tensor2Bin.py after this script.


`QLY/` | Model Training and Variant Caller Scripts. Scripts in this folder are NOT compatible with `pypy`. Please run with `python`.
--- | ---
`call_var_qly.py `| Call variants using candidate variant tensors.
`qly-callVarBam.py` | Call variants directly from a BAM file.
`qly-callVarBamParallel.py` | Generate `qly-callVarBam.py` commands that can be run in parallel. A BED file is required to specify the regions for variant calling. `--refChunkSize` set the genome chuck size per job.
`demoRun.py` | A **Demo** showing how to train a model from scratch.
`evaluate.py` | Evaluate a model in terms of base change, zygosity, variant type and indel length.
`param.py` |  Hyperparameters for model training and other global parameters for the scripts in the folder.
`tensor2Bin.py` |  Create a compressed binary tensors file to facilitate and speed up future usage. Input: Mixed tensors by PairWithNonVariants.py; Truth variants by GetTruth.py and a BED file marks the high confidence regions in the reference genome.
`train.py` |  Training a model using adaptive learning rate decay. By default, the learning rate will decay for three times. Input a binary tensors file created by Tensor2Bin.py is highly recommended.
`trainNonstop.py` |  Helper script. Train a model continuously using the same learning rate and l2 regularization lambda.
`trainWithoutValidationNonstop.py` | Helper script. Train a model continuously using the same learning rate and l2 regularization lambda. Take all the input tensors as training data and do not calculate loss in the validation data.
`calTrainDevDiff.py` | Helper script. Calculate the training loss and validation loss on a trained model.
`getTensorAndLayerPNG.py` | Create high resolution PNG figures to visualize input tensor, layer activations and output.
`getEmbedding.py` | Prepare a folder readable by Tensorboard for visualizing predicted results.
`utils_qly.py` | Helper functions to the network.


*GIAB provides a BED file that marks the high confidence regions in the reference. The models perform better by using only the truth variants in these regions for training. If you don't have such a BED file, you can use a BED file that covers the whole genome.*

***



## About Setting the Alternative Allele Frequency Cutoff

Different from model training, in which all genome positions are candidates but randomly subsampled for training, variant calling using a trained model will require the user to define a minimal alternative allele frequency cutoff for a genome position to be considered as a candidate for variant calling. For all sequencing technologies, the lower the cutoff, the lower the speed. Setting a cutoff too low will increase the false positive rate significantly, while too high will increase the false negative rate significantly. The option `--threshold` controls the cutoff in these three scripts `qly-callVarBam.py`, `qly-callVarBamParallel.py` and `ExtractVariantCandidates.py`. The suggested cutoff is listed below for different sequencing technologies. A higher cutoff will increase the accuracy of datasets with poor sequencing quality, while a lower cutoff will increase the sensitivity in applications like clinical research. Setting a lower cutoff and further filter the variants by their quality is also a good practice.

Sequencing Technology | Alt. AF Cutoff |
:---: |:---:|
Illumina | 0.125 |
PacBio P6-C4 | 0.2 |
ONT R9.4 | 0.25 |

***

## About the Testing Data
The testing dataset 'testingData.tar' includes:
1) the Illumina alignments of chr21 and chr22 on GRCh38 from [GIAB Github](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/data/NA12878/NIST_NA12878_HG001_HiSeq_300x/NHGRI_Illumina300X_novoalign_bams/HG001.GRCh38_full_plus_hs38d1_analysis_set_minus_alts.300x.bam), downsampled to 50x.
2) the truth variants v3.3.2 from [GIAB](ftp://ftp-trace.ncbi.nlm.nih.gov/giab/ftp/release/NA12878_HG001/NISTv3.3.2/GRCh38).

***

## Limitations
### On variants with two alternative alleles (GT: 1/2)

QLY network can only output one of the two possible alternative alleles at a position. We will further extend the network to support genome variants with two alternative alleles.

### On training

In rare cases, the model training will stuck early at a local optimal and cannot be further optimized without a higher learning rate. As we observed ,the problem only happens at the very beginning of model training, and can be predicated if the loss remains stable in the first few training epochs.
