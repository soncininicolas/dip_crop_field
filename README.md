# Digital Image Processing of Crop Fields
This repository contains code for the final project of the Digital Image Processing course.
This course was part of the PhD in Computer Science (Doctorado en Informática) of the "Universidad Nacional de Rosario", Argentina.
The course was led by Dr. Juan Carlos Gómez and Dr. Gonzalo Daniel Sad.

This project aims to do both image segmentation for an agricultural field as well as crop row detection.
Sample images are taken from the following datasets:
 - The Rosario Dataset: TBD REFERENCE
 - Zavalla 2023 (unpublished): TBD REFERENCE 
 - FieldSAFE: TBD REFERENCE (only image segmentation, no crop rows are present)


## Usage

The following code was tested under the Ubuntu 20.04 Linux distribution.
Quick start:
```bash
sudo apt install python3-venv && \
python3 -m venv venv && \
source venv/bin/activate && \
pip3 install -r requirements && \
python3 TBD -h
```


### Requirements
You should have a version of Python3 installed, if not refer to https://www.python.org/downloads/.

We make use of the virtual environments provided by python3-venv, which can be installed using aptitude:
`sudo apt install python3-venv`
and then an environment can be created by performing:
`python3 -m venv venv`
which can be activated by performing:
`source venv/bin/activate`
and after you're done with this tool can be deactivated by doing
`deactivate`

Required packages can be installed by activating the environment and doing:
`pip3 install -r requirements.txt`
which should take a while as it's installing all packages to the virtual environment.

Once everything is done you can show the script help with:
`python3 TBD -h`
and you can run an example with
`python3 TBD ...`

Prepared examples can be run by invoking the scripts in the `scripts/` folder.


## License
The code is licensed under The MIT License.
A copy of the license should be available in the LICENSE file.
For more information please refer to https://opensource.org/license/mit
