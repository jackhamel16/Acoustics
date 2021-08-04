# Acoustic Scattering 

This code utilizes an integral equation formulation to compute equivalent acoustic sources on a geometry's surface induced by a defined acoustic wave excitation.  

## Running

Running this code is best done via running julia from the command line/prompt.  This software is developed on Windows and to execute I would run the command: `julia run.jl input_file.txt` where `input_file.txt` is a text file containing all information about the problem the user desires to run.  Examples of this file are in the examples directory.

### Solvers

Currently, users can solve the sound-soft Integral Equation (IE), the normal derivative of the sound-soft IE and the sound-soft combined field IE.  Additionally, the code can compute the decomposed Wigner Smith (WS) time-delay matrix for the geometry and build an excitation to excite a desired WS mode and solve the sound-soft IE with that excitation.

### Acceleration

Currently, the code is only capable of using the Adaptive Cross Approximation to compress the Z matrix and accelerate the entire computation for the sound-soft IE.

## Documentation

There is detailed documentation in the documentation directory about the algorithms in the code in AcousticsIE.docx (constantly a work in progress) and details on verification work and testing in the sub-directories.

## Notes

Make sure Juno's working directory is the project directory.
